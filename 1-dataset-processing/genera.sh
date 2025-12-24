#!/bin/bash

BASE_DIR="data/dataset-article"
DEST_DIR="/media/fernando/INFORMATION/0-DATASET/KIDNEY"


if [ ! -d "$DEST_DIR" ]; then
    echo "❌ Errot: The directory $DEST_DIR does not exist."
    exit 1
fi

#: <<'COMMENT'
#for size in $(seq 96 8 344); do
for size in $(seq 96 8 104); do
    echo "=== Processing size $size ==="

    # 1. Executa o comando para gerar o dataset
    python prog_generate.py -ps "$size"

    DATASET_DIR="$BASE_DIR/dataset-$size"
    ZIP_FILE="dataset-$size.zip"

    # 2. Compacta a pasta
    if [ -d "$DATASET_DIR" ]; then
        echo "Compacting $DATASET_DIR in $ZIP_FILE..."
        zip -rq "$ZIP_FILE" "$DATASET_DIR"

        # 3. Remove a pasta
        echo "Removing directory $DATASET_DIR..."
        rm -rf "$DATASET_DIR"

        # 4. Move o zip para o destino
        echo "Moving $ZIP_FILE to $DEST_DIR..."
        mv "$ZIP_FILE" "$DEST_DIR/"
    else
        echo "❌ Pasta $DATASET_DIR não encontrada, pulando..."
    fi

    echo
done

echo "✅ Process completed!"
#COMMENT

python3 utils/dataset_report.py --input_dir $DEST_DIR  --output_dir $DEST_DIR

