# Description of generated dataset <PATCH_SIZE>

## Description
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

