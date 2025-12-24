from modules.Looping import process_dataset

# Treinamento do modelo EfficientNetV2
#
# Este código tem como objetivo treinar um classificador binário (cálculo renal: 1 / normal: 0) usando backbones EfficientNetV2 e lê os arquivos `train{i}.csv` / `val{i}.csv`.

import argparse
import tensorflow as tf

from pathlib import Path

print('TensorFlow version:', tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs disponíveis: {gpus}")
else:
    print("Nenhuma GPU encontrada.")

# Configurações

MODEL_NAME = 'EfficientNetV2S'
EPOCHS = 12
BATCH_SIZE = 32
LR = None  # e.g. 1e-4
FOLDS = 5
EARLY_STOP = EPOCHS * 0.2
SEED = 123
ZOOM_OUT = 0.2
CONTRAST = 0.2
BRIGHTNESS = 0.2

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print('Pasta do Dataset:', DATASET_ROOT)
print('Pasta para o Output:', OUTPUT_ROOT)

# ## Definição do modelo
# https://github.com/leondgarse/keras_efficientnet_v2

def main():
    parser = argparse.ArgumentParser(description="EfficientNetV2 training - command line arguments")
    
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset (required)")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory (required)")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model name to be trained")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate (e.g. 1e-4). If not provided, optimizer's learning rate will not be changed")
    parser.add_argument("--folds", type=int, default=FOLDS, help="Number of folds for cross-validation")
    parser.add_argument("--early_stop", type=int, default=int(EARLY_STOP), help="EarlyStopping patience in epochs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--zoom_out_factor", type=float, default=ZOOM_OUT, help="Zoom-out factor (positive, e.g. 0.2)")
    parser.add_argument("--contrast_factor", type=float, default=CONTRAST, help="Contrast adjustment factor (e.g. 0.2)")
    parser.add_argument("--brightness_delta", type=float, default=BRIGHTNESS, help="Brightness delta (e.g. 0.2)")
    
    args = parser.parse_args()

    dataset_path = Path(DATASET_ROOT)
    
    # Verificar se o diretório do dataset existe
    if not dataset_path.exists():
        print(f"Erro: Diretório do dataset não encontrado: {dataset_root}")
        return
    
    # Verificar se existem arquivos CSV de treino
    def find_train_csvs(dataset_path):
        dataset_path = Path(dataset_path)
        return sorted(list(dataset_path.glob('**/train*.csv'))) # Busca arquivos train*.csv em dataset_root e em subpastas

    train_csvs = find_train_csvs(dataset_path)
    print("Arquivos train*.csv encontrados:")

    if len(train_csvs) == 0:
        print(f"No train*.csv found in {dataset_path} or subfolders. Check DATASET_ROOT.")
        return

    for csv in train_csvs:
        print("  ", csv)

    # Começa o treinamento. Lembre-se que os arquivos .csv e as imagens devem estar alocadas corretamente no caminho indicado pela variável `DATASET_ROOT`.
    process_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        folds=args.folds,
        early_stop=args.early_stop,
        seed=args.seed,
        zoom_out_factor=args.zoom_out_factor,
        contrast_factor=args.contrast_factor,
        brightness_delta=args.brightness_delta
    )

if __name__ == "__main__":
    main()
