# Ler arquivos `.csv` e imagens do Dataset

import pandas as pd
import tensorflow as tf

from pathlib import Path

def read_csv(csv_path, dataset_root):
    df = pd.read_csv(csv_path)
    if df.shape[1] >= 2:
        filename_col = df.columns[0]
        label_col = df.columns[1]
    else:
        raise ValueError(f"CSV {csv_path} must have at least 2 columns: filename,label")

    filepaths = []
    for fn in df[filename_col].astype(str).tolist():
        p = Path(fn)
        if p.exists():
            filepaths.append(str(p))
        else:
            candidate = Path(dataset_root) / fn
            if candidate.exists():
                filepaths.append(str(candidate))
            else:
                found = None
                for sub in ['with-stone', 'without-stone']:
                    cand2 = Path(dataset_root) / sub / fn
                    if cand2.exists():
                        found = cand2
                        break
                if found:
                    filepaths.append(str(found))
                else:
                    raise FileNotFoundError(f"Could not resolve path for {fn} (csv: {csv_path})")

    labels = df[label_col].astype(int).tolist()
    return filepaths, labels

print('Arquivos `.csv` prontos')

def make_dataset(filepaths, labels, img_size=(384, 384), batch_size=32, shuffle=True, augment=False, seed=123, zoom_out_factor=0.2, contrast_factor=0.2, brightness_delta=0.2):
    AUTOTUNE = tf.data.AUTOTUNE

    def _load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1]
        img = tf.image.resize(img, img_size)
        return img, tf.cast(label, tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((list(filepaths), list(labels)))
    if shuffle:
        try:
            buffer = min(10000, len(filepaths))
        except Exception:
            buffer = 1000
        ds = ds.shuffle(buffer_size=buffer, seed=seed)

    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)

    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomZoom(height_factor=(0.0, zoom_out_factor), width_factor=(0.0, zoom_out_factor)),
            tf.keras.layers.RandomRotation(rotation_factor),
            tf.keras.layers.RandomContrast(contrast_factor),
            tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=brightness_delta)),
        ], name="data_augmentation")
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

print('Imagens de treino e teste prontas')
