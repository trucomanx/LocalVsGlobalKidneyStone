#!/bin/bash

SRC_DIR="${1:-dataset-source}"
DEST_DIR="${2:-dataset-stats}"
GENERATE_ZIP="${3:---zip}"

################################################################################
TMP_DIR="temp_dir"

mkdir -p "$DEST_DIR"

if [ ! -d "$SRC_DIR" ]; then
    echo "❌ Error: The directory $SRC_DIR does not exist."
    exit 1
fi

for size in $(seq 96 8 104); do
    echo "=== Processing size $size ==="

    # 1. Generate dataset
    python3 ./src/prog_generate.py -ds "$SRC_DIR" -do "$TMP_DIR" -ps "$size"

    SUBDATASET_NAME="dataset-$size"

    if [ -d "$TMP_DIR/$SUBDATASET_NAME" ]; then

        # 2. Copy report.json
        mkdir -p "$DEST_DIR/jsons/"
        cp -p "$TMP_DIR/$SUBDATASET_NAME/report.json" \
              "$DEST_DIR/jsons/$SUBDATASET_NAME.json"

        # 3. Generate ZIP only if requested
        if [ "$GENERATE_ZIP" = "--zip" ]; then

            cd "$TMP_DIR" || exit 1

            ZIP_FILE="$SUBDATASET_NAME.zip"

            echo "Compacting $SUBDATASET_NAME in $ZIP_FILE..."
            zip -rq "$ZIP_FILE" "$SUBDATASET_NAME"

            cd .. || exit 1

            echo "Moving $ZIP_FILE to $DEST_DIR/zips..."
            mkdir -p "$DEST_DIR/zips/"
            mv "$TMP_DIR/$ZIP_FILE" "$DEST_DIR/zips/"

        fi

        # 4. Remove temporary dataset directory
        echo "Removing directory $SUBDATASET_NAME..."
        rm -rf "$TMP_DIR/$SUBDATASET_NAME"

    else
        echo "❌ Directory $TMP_DIR/$SUBDATASET_NAME not found, skipping..."
    fi

    echo
done

echo "✅ Processing completed!"

# 5. Remove temporary directory
rm -rf "$TMP_DIR"

# 6. Generate consolidated report
python3 ./src/prog_dataset_report.py \
    --input_dir "$DEST_DIR/jsons" \
    --output_dir "$DEST_DIR"

