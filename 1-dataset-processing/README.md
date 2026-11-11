# Kidney Stone Image Patch Processing

This repository contains Python scripts for **image preprocessing and dataset generation** for kidney stone classification.  
The code splits input medical images into **patches** (sub-images) and generates structured datasets for training machine learning models.

---

## ⚙️ Requirements

Clone the repository and install dependencies with:

```bash
pip install -r requirements.txt
```

---
## ▶️ Installation


### 1 - Download dataset source
Download the dataset from Kaggle, first generate the file `~/.kaggle/kaggle.json`, then

```
./download-source.sh "dataset-source/stone" 
```

### 2 - Generate dataset <PATCH_SIZE>
Run the prog_generate script with a patch size:

```bash
python prog_generate.py -ds <PATH_TO_SOURCE> -do <PATH_TO_OUTPUT> -ps <PATCH_SIZE>
```

Example:

```bash
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 48
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 56
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 64
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 72
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 80
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 88
python ./src/prog_generate.py -ds "dataset-source" -do "dataset-article" -ps 96
```


#### Description
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

#### 📊 Output Example

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
## 📊 Generate statistics


```bash
./genera-stats.sh "dataset-source" "dataset-stats"
```

