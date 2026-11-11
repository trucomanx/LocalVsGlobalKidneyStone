# Kidney Stone Image Patch Processing

This repository contains Python scripts for **image preprocessing and dataset generation** for kidney stone classification.  
The code splits input medical images into **patches** (sub-images) and generates structured datasets for training machine learning models.


## ⚙️ Requirements

Clone the repository and install dependencies with:

```bash
pip install -r requirements.txt
```


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
The description of the output dataset can be seen in [README.patch_size.md](README.patch_size.md)

## 📊 Generate statistics


```bash
./genera-stats.sh "dataset-source" "dataset-stats"
python3 ./src/prog_stat_report.py \
    --input_dir "dataset-stats/jsons" \
    --output_dir "dataset-stats"
```

