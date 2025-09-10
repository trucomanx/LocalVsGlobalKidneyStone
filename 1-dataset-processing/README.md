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
project/
│
├── main.py                     # Main entry point (argparse-based)
├── utils/
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

## ⚙️ Installation

Clone the repository and install dependencies with:

```bash
pip install -r requirements.txt
```


---

## ▶️ Usage

Run the main script with a patch size:

```bash
# Args: --patch_size or -ps
python main.py -ps <PATCH_SIZE>

```

---
## ▶️ Dataset generation

This example generates only one dataset.

```bash
python main.py -ps 224
```

This example generates the entire set of datasets in Ubuntu's `bash`.

```bash
for ps in $(seq 64 16 224); do
    python3 main.py -ps $ps
done
```

### Description
This will:
1. Split all images in `data/dataset-source/` into **patches**.
2. Save them into:
   ```
   data/dataset-article/dataset-*/with-stone/
   data/dataset-article/dataset-*/without-stone/
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
data/dataset-article/dataset-224/
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

## 📜 License
MIT License. Free to use and modify.
