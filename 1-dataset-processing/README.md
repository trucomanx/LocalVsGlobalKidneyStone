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

The generation of a specific dataset with a patch size can be seen in [README.generate.md](README.generate.md). However, you don't need to take this step here.
The description of the output dataset can be seen in [README.patch_size.md](README.patch_size.md).

## 📊 Generate statistics

If you wish to generate all dataset statistics from the source, use the next code:

```bash
./genera-stats.sh "dataset-source" "dataset-stats"
python3./src/prog_stat_report.py \
    --input_dir "dataset-stats/jsons" \
    --output_dir "dataset-stats"
```

**You will need a lot of disk space**
