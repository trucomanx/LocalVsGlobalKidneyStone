# Kidney Stone Image Patch Processing

This repository contains Python scripts for **image preprocessing and dataset generation** for kidney stone classification.  
The code splits input medical images into **patches** (sub-images) and generates structured datasets for training machine learning models.

---

## 📌 Features
- Split images and corresponding label masks into **overlapping patches**.
- Organize dataset into two categories:
  - `with-stone` (positive patches containing stone regions).
  - `without-stone` (negative patches without stones).
- Save dataset patches to disk with custom patch sizes.
- Generate **K-Fold CSV files** for machine learning training and validation.
- Configurable patch size via command line arguments.

---

## 📂 Project Structure

```
./
│
├── prog_generate.py                     # Main entry point (argparse-based)
├── dp_utils/
│   ├── image_processing.py      # Image splitting, patch generation, saving
│   └── csv_processing.py        # Dataset CSV and K-Fold generation
│
├── data/
│   ├── dataset-source/          # Input dataset (raw images + labels)
│   │   ├── stone/
│   │   │   ├── image/*.tif
│   │   │   └── label/*.tif
│   │   └── normal/
│   │       ├── image/*.jpg
│   │       └── label/*.jpg
│   │
│   └── dataset-article/         # Output datasets (generated patches)
│
└── README.md
```

---

## ⚙️ Requirements

Clone the repository and install dependencies with:

```bash
pip install -r requirements.txt
```

---
## ▶️ Installation


### Download dataset source
Download the dataset from Kaggle, first generate the file `~/.kaggle/kaggle.json`, then

```
./download-source.sh "dataset-source/stone" 
```

### Generate dataset <PATCH_SIZE>
Run the prog_generate script with a patch size:

```bash
python prog_generate.py -ds <PATH_TO_SOURCE> -do <PATH_TO_OUTPUT> -ps <PATCH_SIZE>
```

Example:

``bash
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 48
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 56
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 64
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 72
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 80
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 88
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 96
```


### Description
This will:
1. Split all images in `dataset-source/` into **patches**.
2. Save them into:
   ```
   dataset-article/dataset-<PATCH_SIZE>/with-stone/
   dataset-article/dataset-<PATCH_SIZE>/without-stone/
   ```
3. Generate **5-fold train/val CSV files** inside the same folder:
   - `train0.csv`, `val0.csv`
   - `train1.csv`, `val1.csv`
   - `train2.csv`, `val2.csv`
   - `train3.csv`, `val3.csv`
   - `train4.csv`, `val4.csv`

---

## 📊 Output Example

For a dataset generated with `-ps 224`, the output directory looks like:

```
dataset-article/dataset-224/
│
├── with-stone/
│   ├── imagestone0-patch0.png
│   ├── imagestone0-patch1.png
│   └── ...
│
├── without-stone/
│   ├── imagenormal0-patch0.png
│   ├── imagenormal0-patch1.png
│   └── ...
│
├── train0.csv
├── val0.csv
├── train1.csv
├── val1.csv
└── ...
```

Each CSV file has the format:

```csv
filepath,label
with-stone/imagestone0-patch0.png,1
without-stone/imagenormal0-patch0.png,0
...,...
```
---


