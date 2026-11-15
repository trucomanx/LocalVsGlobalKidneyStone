# Crossvalidation



## ⚙️ Requirements

Clone the repository and install dependencies with:

```bash
pip install -r requirements.txt
```

## Option 1) Using script

You need first to generate the source dataset dir in [../1-dataset-processing](../1-dataset-processing).

```bash
./crossval-size.sh <OUTPUT_DIR> <BACKBONE_NAME> <PATCH_SIZE> <BATCH_SIZE>
```

BACKBONE_NAME can be "EfficientNetV2S", "MobileNetV3Large", "ResNet50", ...

**Example:**

```bash
./crossval-size.sh "output" "ResNet50" 48 64
./crossval-size.sh "output" "ResNet50" 56 64
./crossval-size.sh "output" "ResNet50" 64 64
./crossval-size.sh "output" "ResNet50" 72 64
./crossval-size.sh "output" "ResNet50" 80 64
./crossval-size.sh "output" "ResNet50" 86 64
./crossval-size.sh "output" "ResNet50" 94 64
```

## Option 2) Using google Colab

Upload the [crossval-size.ipynb](crossval-size.ipynb) Jupyter file to Google Colab.
