# Crossvalidation


## Using script

You need first to generate the source dataset dir in (../1-dataset-processing)[../1-dataset-processing].

```bash
./crossval-size.sh <OUTPUT_DIR> <BACKBONE_NAME> <PATCH_SIZE>
```

**Example:**

```bash
./crossval-size.sh "output" "ResNet50" 48
./crossval-size.sh "output" "ResNet50" 56
./crossval-size.sh "output" "ResNet50" 64
./crossval-size.sh "output" "ResNet50" 72
./crossval-size.sh "output" "ResNet50" 80
./crossval-size.sh "output" "ResNet50" 86
./crossval-size.sh "output" "ResNet50" 94
```
