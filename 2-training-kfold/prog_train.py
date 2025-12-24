import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetV2S,
    MobileNetV3Large,
    ResNet50,
    DenseNet121,
    ConvNeXtTiny
)
from tensorflow.keras.applications import (
    efficientnet_v2,
    mobilenet_v3,
    resnet,
    densenet,
    convnext
)

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Save results
# ============================
def save_history_and_plots( history,
                            output_dir,
                            prefix
                        ):
    """
    history    : objeto retornado por model.fit()
    output_dir : diretório onde salvar
    prefix     : ex: 'fold0-stage1'
    """

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # HISTÓRICO → DataFrame
    # -------------------------
    hist_df = pd.DataFrame(history.history)
    hist_df["epoch"] = hist_df.index + 1

    csv_path = os.path.join(output_dir, f"history-{prefix}.csv")
    hist_df.to_csv(csv_path, index=False)

    # -------------------------
    # PLOT: Accuracy
    # -------------------------
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["accuracy"], label="train")
    plt.plot(hist_df["epoch"], hist_df["val_accuracy"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)

    acc_plot_path = os.path.join(output_dir, f"accuracy-{prefix}.png")
    plt.savefig(acc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # PLOT: Validation Loss
    # -------------------------
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["loss"], label="train")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss vs Epochs")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(output_dir, f"val-loss-{prefix}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] CSV salvo em: {csv_path}")
    print(f"[OK] Plots salvos em:")
    print(f"     {acc_plot_path}")
    print(f"     {loss_plot_path}")


# ============================
# callbacks
# ============================
def get_keras_model_name(output_dir, label):
    return os.path.join(output_dir, f"best-model-{label}.keras")
    
def get_callbacks(output_dir, patience,label):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=get_keras_model_name(output_dir, label),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
# ============================
# preprocess
# ============================

def get_preprocess_fn(model_type):
    if model_type == "EfficientNetV2S":
        return efficientnet_v2.preprocess_input
    elif model_type == "MobileNetV3Large":
        return mobilenet_v3.preprocess_input
    elif model_type == "ResNet50":
        return resnet.preprocess_input
    elif model_type == "DenseNet121":
        return densenet.preprocess_input
    elif model_type == "ConvNeXtTiny":
        return convnext.preprocess_input
    else:
        raise ValueError
    
# ============================
# DATASET
# ============================
def load_dataset_from_csv(  csv_path,
                            base_path,
                            img_size,
                            model_type,
                            batch_size,
                            shuffle=True,
                            num_classes=2,
                            augment=False,
                            augmentation=None
                        ):
                        
    df = pd.read_csv(csv_path)

    filepaths = df["filepath"].apply(lambda p: os.path.join(base_path, p)).values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    preprocess_fn = get_preprocess_fn(model_type)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    def _load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)

        img = tf.cast(img, tf.float32)

        if augment and augmentation is not None:
            img = augmentation(img, training=True)

        img = preprocess_fn(img)

        label = tf.one_hot(label, num_classes)
        return img, label

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ============================
# plot dataset
# ============================
def save_dataset_grid_raw(
    dataset,
    output_path,
    grid_rows=4,
    grid_cols=4,
    class_names=None,
    max_batches=1
):
    """
    Visualiza imagens exatamente como entram no modelo
    (após augmentation + preprocess_input)

    dataset      : tf.data.Dataset
    output_path  : caminho para salvar a imagem
    grid_rows    : número de linhas
    grid_cols    : número de colunas
    class_names  : lista opcional de nomes das classes
    max_batches  : quantos batches percorrer
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_images = grid_rows * grid_cols
    images = []
    labels = []

    for batch_idx, (x_batch, y_batch) in enumerate(dataset):
        for i in range(x_batch.shape[0]):
            images.append(x_batch[i].numpy())
            labels.append(y_batch[i].numpy())

            if len(images) >= num_images:
                break

        if batch_idx + 1 >= max_batches or len(images) >= num_images:
            break

    images = np.array(images)
    labels = np.array(labels)
    labels_idx = np.argmax(labels, axis=1)

    plt.figure(figsize=(grid_cols * 3, grid_rows * 3))

    for i in range(num_images):
        plt.subplot(grid_rows, grid_cols, i + 1)

        img = images[i]

        # ⚠️ SEM normalização — mostra exatamente o tensor
        plt.imshow(img)
        plt.axis("off")

        label = labels_idx[i]
        title = class_names[label] if class_names else f"Label: {label}"
        plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Grade RAW salva em: {output_path}")

# ============================
# Data augmentation
# ============================
def get_augmentation(my_seed=42):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=my_seed),
        layers.RandomZoom(
            height_factor=(-0.1, 0.0),
            width_factor=(-0.1, 0.0),
            fill_mode="constant",
            fill_value=0.0
        ),
        layers.RandomBrightness(0.05),
        #layers.GaussianNoise(0.005),
    ], name="data_augmentation")
# ============================
# MODELO
# ============================
def build_model(model_type="EfficientNetV2S", num_classes=2):

    if model_type == "EfficientNetV2S":
        Backbone = EfficientNetV2S
        image_sz = (224, 224)

    elif model_type == "MobileNetV3Large":
        Backbone = MobileNetV3Large
        image_sz = (224, 224)

    elif model_type == "ConvNeXtTiny":
        Backbone = ConvNeXtTiny
        image_sz = (224, 224)

    elif model_type == "ResNet50":
        Backbone = ResNet50
        image_sz = (224, 224)

    elif model_type == "DenseNet121":
        Backbone = DenseNet121
        image_sz = (224, 224)

    else:
        raise ValueError(f"Backbone não suportado: {model_type}")

    backbone = Backbone(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_sz, 3)
    )

    backbone.trainable = False

    inputs = layers.Input(shape=(*image_sz, 3))
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, backbone, image_sz


# ============================
# MÉTRICAS
# ============================
def compute_metrics(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return acc, prec, tnr, rec, f1, cm

# ============================
# TREINAMENTO POR SUB-DATASET
# ============================
def train_subdataset(   patch_dataset_path, 
                        output_result_path, 
                        model_type, 
                        batch_size=32,
                        epochs_stage_1 = 200, 
                        epochs_stage_2 = 200, 
                        learning_rate_stage_1 = 1e-3, 
                        learning_rate_stage_2 = 1e-4, 
                        early_stop_patience = 100, 
                        num_folds = 5,
                        my_seed = 42):
                        
    tf.random.set_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
                        
    os.makedirs(output_result_path, exist_ok=True)

    metrics = {
        "accuracy": [],
        "precision": [],
        "true-negative-rate": [],
        "recall": [],
        "f1-score": []
    }

    for fold in range(num_folds):
        print(f"\n=== Fold {fold} ===")

        model, backbone, input_img_sz = build_model(model_type=model_type)
        
        fold_path = os.path.join(output_result_path,f"fold{fold}")
        os.makedirs(fold_path,exist_ok=True)
            
        # ---------
        # STAGE 0: load dataset
        # ---------
        
        train_csv = os.path.join(patch_dataset_path, f"train{fold}.csv")
        val_csv = os.path.join(patch_dataset_path, f"val{fold}.csv")
        
        augmentation = get_augmentation(my_seed=my_seed)

        train_ds = load_dataset_from_csv(
            train_csv,
            patch_dataset_path,
            input_img_sz,
            model_type,
            batch_size,
            shuffle=True,
            augment=True,
            augmentation=augmentation
        )

        val_ds = load_dataset_from_csv(
            val_csv,
            patch_dataset_path,
            input_img_sz,
            model_type,
            batch_size,
            shuffle=False,
            augment=False
        )

        # ---------
        # STAGE 1: backbone congelado
        # ---------
        model.compile(
            optimizer=Adam(learning_rate_stage_1),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks_s1 = get_callbacks(fold_path, early_stop_patience,"stage1")

        results=model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage_1,
            callbacks=callbacks_s1,
            verbose=1
        )
        
        save_history_and_plots(
            history=results,
            output_dir=fold_path,
            prefix=f"stage1"
        )
        
        save_dataset_grid_raw(
            train_ds,
            os.path.join(fold_path,f"dataset-train-stage1.png"),
            grid_rows=4,
            grid_cols=4,
            class_names=None,
            max_batches=1
        )

        # ---------
        # STAGE 2: backbone destravado
        # ---------
        backbone.trainable = True
        for layer in backbone.layers:
            # Congela todas as BatchNorm
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        # Agora congela as camadas iniciais (ex: 20 primeiras)
        for layer in backbone.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate_stage_2),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks_s2 = get_callbacks(fold_path, early_stop_patience,"stage2")
        
        results=model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage_2,
            callbacks=callbacks_s2,
            verbose=1
        )
        
        save_history_and_plots(
            history=results,
            output_dir=fold_path,
            prefix=f"stage2"
        )
        
        save_dataset_grid_raw(
            train_ds,
            os.path.join(fold_path,f"dataset-train-stage2.png"),
            grid_rows=4,
            grid_cols=4,
            class_names=None,
            max_batches=1
        )

        # ---------
        # LOAD BEST MODEL
        # ---------

        best_model_path = get_keras_model_name(fold_path, "stage2")
        
        model = tf.keras.models.load_model(best_model_path)
        
        # ---------
        # AVALIAÇÃO
        # ---------
        y_true_all = []
        y_pred_all = []

        for x_batch, y_batch in val_ds:
            preds = model.predict(x_batch, verbose=0)
            y_true_all.append(y_batch.numpy())
            y_pred_all.append(preds)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        acc, prec, tnr, rec, f1, cm = compute_metrics(y_true_all, y_pred_all)

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["true-negative-rate"].append(tnr)
        metrics["recall"].append(rec)
        metrics["f1-score"].append(f1)

        cm_path = os.path.join(fold_path, f"val-confusion-matrix-stage2.json")
        with open(cm_path, "w") as f:
            json.dump(cm.tolist(), f, indent=4)

    # ---------
    # SALVA DADOS DE TREINO
    # ---------
    
    variables = {
        "model": model_type,
        "image_size": list(input_img_sz),
        "epochs_stage_1": epochs_stage_1,
        "epochs_stage_2": epochs_stage_2,
        "batch_size": batch_size,
        "learning_rate_stage_1": learning_rate_stage_1, 
        "learning_rate_stage_2": learning_rate_stage_2, 
        "early_stop_patience": early_stop_patience, 
        "my_seed": my_seed
    }

    variable_path = os.path.join(output_result_path, "variables.json")
    with open(variable_path, "w") as f:
        json.dump(variables, f, indent=4)

    # ---------
    # SALVA MÉTRICAS GERAIS
    # ---------
    metrics_path = os.path.join(output_result_path, "val-metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nTreinamento finalizado.")
    print("Métricas salvas em:", metrics_path)

# ============================
# EXEMPLO DE USO
# ============================
if __name__ == "__main__":
    patch_dataset_path = "/caminho/para/dataset-64"
    output_result_path = "/caminho/para/outputs/dataset-64"
    model_type = "EfficientNetV2S"
    
    train_subdataset(patch_dataset_path, output_result_path, model_type)

