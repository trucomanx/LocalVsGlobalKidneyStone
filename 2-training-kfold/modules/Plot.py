# Apresentar imagens de treino e teste

import numpy as np
import matplotlib.pyplot as plt

def plot_dataset_images(dataset, class_names, n=32, title='Samples', seed=123):
    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    ds = dataset.unbatch()
    if seed is not None:
        ds = ds.shuffle(buffer_size=1000, seed=seed)
    ds = ds.take(n)

    images = []
    labels = []
    for img, lbl in ds:
        images.append(img.numpy())
        labels.append(int(lbl.numpy()))

    actual_n = len(images)
    if actual_n == 0:
        raise ValueError(f"Dataset didn't returned images. Verify if the dataset is correct.")
    cols = int(math.sqrt(actual_n))
    cols = max(1, cols)
    rows = math.ceil(actual_n / cols)

    plt.figure(figsize=(cols * 1.8, rows * 1.8))
    for i, (im, lbl) in enumerate(zip(images, labels)):
        ax = plt.subplot(rows, cols, i+1)
        if im.dtype == np.float32 or im.dtype == np.float64:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0, 255
        if im.ndim == 3 and im.shape[2] == 1:
            im = im.squeeze(axis=2)
            ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.axis('off')
        if class_names is not None and lbl < len(class_names):
            ax.set_title(class_names[lbl], fontsize=8)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
