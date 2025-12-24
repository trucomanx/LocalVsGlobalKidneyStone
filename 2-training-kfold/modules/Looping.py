from modules.Dataset import read_csv, make_dataset
from modules.Model import build_model
from modules.Plot import plot_dataset_images
from modules.Metrics import compute_metrics_and_confmat

# Looping de treino
#
# Aqui se aciona o pipeline: para cada subdataset (arquivo de imagem do Dataset que esteja citado no `.csv`), ele lê os arquivos em questão, treina o modelo, salva o melhor modelo e grava as métricas em um arquivo `.json`.

import json
import math
from pathlib import Path

from sklearn.utils.class_weight import compute_class_weight

def process_dataset(dataset_root, 
                    output_root, 
                    model_name='EfficientNetV2S', 
                    epochs=12, 
                    batch_size=32, 
                    lr=None, 
                    folds=5, 
                    early_stop=4, 
                    seed=123, 
                    zoom_out_factor=0.02, 
                    contrast_factor=0.01, 
                    brightness_delta=0.05):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    potential = list(dataset_root.glob('train0.csv'))
    subdatasets = []
    if potential:
        subdatasets = [dataset_root]
    else:
        for sub in dataset_root.iterdir():
            if sub.is_dir() and (sub / 'train0.csv').exists():
                subdatasets.append(sub)

    if not subdatasets:
        raise FileNotFoundError(f"No train0.csv found in {dataset_root} or its immediate subfolders.")

    for sub in subdatasets:
        print(f"\n--- Processing subdataset: {sub} ---")

        metrics_agg = { 'accuracy': [], 'precision': [], 'true_negative_rate': [], 'recall': [], 'f1_score': [] }

        existing_folds = []
        for fold in range(folds):
            fold_out = output_root / f"fold{fold}"
            metrics_file = fold_out / "metrics.json"
            if fold_out.exists() and metrics_file.exists():
                print(f"Fold {fold} já processado. Pulando.")
                existing_folds.append(fold)

        for fold in range(folds):
            if fold in existing_folds:
                continue  # Pula folds já processados

            train_csv = sub / f"train{fold}.csv"
            val_csv = sub / f"val{fold}.csv"
            if not train_csv.exists() or not val_csv.exists():
                print(f"Skipping fold {fold}: missing {train_csv} or {val_csv}")
                continue

            fold_out = output_root / f"fold{fold}"
            fold_out.mkdir(parents=True, exist_ok=True)

            print(f"\nFold {fold}: reading csvs")
            train_files, train_labels = read_csv(train_csv, sub)
            val_files, val_labels = read_csv(val_csv, sub)

            try:
                classes = np.unique(train_labels)
                cw = compute_class_weight('balanced', classes=classes, y=train_labels)
                class_weight = {int(c): float(w) for c,w in zip(classes,cw)}
            except Exception as e:
                print("Warning: could not compute class weights, proceeding without. Error:", e)
                class_weight = None

            model, img_size = build_model(model_name=model_name)
            
            train_ds = make_dataset(train_files, train_labels, img_size=img_size, batch_size=batch_size, shuffle=False, augment=True, seed=seed, zoom_out_factor=zoom_out_factor, contrast_factor=contrast_factor, brightness_delta=brightness_delta)
            val_ds = make_dataset(val_files, val_labels, img_size=img_size, batch_size=batch_size, shuffle=False, augment=False)

            class_names = ['normal', 'stone']

            print("Apresentando imagens de treino")
            plot_dataset_images(train_ds, class_names=class_names, n=36, title='Train samples', seed=seed)

            print("Apresentando imagens de validação")
            plot_dataset_images(val_ds, class_names=class_names, n=36, title='Validation samples', seed=seed)



            best_model_path = fold_out / f"best_model_fold{fold}.keras"
            cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(str(best_model_path), monitor='val_loss', save_best_only=True)
            cb_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop, restore_best_weights=False)
            cb_rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

            if lr is not None:
                tf.keras.backend.set_value(model.optimizer.lr, lr)

            print(f"Training fold {fold} (epochs={epochs}, batch_size={batch_size})")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[cb_checkpoint, cb_early, cb_rlr],
                class_weight=class_weight,
                verbose=2
            )

            if best_model_path.exists():
                print("Loading best model from", best_model_path)
                model = tf.keras.models.load_model(str(best_model_path))

            y_probs = model.predict(val_ds, verbose=0)
            y_probs = np.asarray(y_probs).reshape(-1)
            y_true = np.array(val_labels)

            metrics, conf = compute_metrics_and_confmat(y_true, y_probs)
            print(f"Fold {fold} metrics:", metrics)

            for k in metrics_agg.keys():
                metrics_agg[k].append(metrics[k])

            metrics_file = fold_out / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            cm_file = fold_out / "confusion_matrix.json"
            with open(cm_file, 'w') as f:
                json.dump(conf, f, indent=2)

            print(f"Saved fold {fold} metrics to {metrics_file} and confusion matrix to {cm_file}")

        agg_file = output_root / 'val-metrics-agg.json'
        with open(agg_file, 'w') as f:
            json.dump(metrics_agg, f, indent=2)
        print(f"Saved aggregated metrics to {agg_file}")

print('Looping de treino pronto')
