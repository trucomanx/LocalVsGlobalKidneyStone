#!/bin/bash

OUTPUT_DIR=${1-output}
BACKBONE_NAME=${2:-ResNet50}
PATCH_SIZE=${3:-64}
BATCH_SIZE=$(4:-64)

################################################################################

DATASET_SOURCE_DIR="../1-dataset-processing/dataset-source"
PATCH_ROOT_DIR="dataset-article"

DATASET_PATCH_DIR="$PATCH_ROOT_DIR/dataset-$PATCH_SIZE"

#
python ../1-dataset-processing/src/prog_generate.py \
    -ds $DATASET_SOURCE_DIR \
    -do $PATCH_ROOT_DIR \
    -ps $PATCH_SIZE

python prog_train.py \
    --patch-dataset-path "$DATASET_PATCH_DIR" \
    --output-result-path "$OUTPUT_DIR/$BACKBONE_NAME/dataset-$PATCH_SIZE" \
    --model-type "$BACKBONE_NAME" \
    --batch-size $BATCH_SIZE \
    --epochs-stage-1 32 \
    --epochs-stage-2 32 \
    --learning-rate-stage-1 0.001 \
    --learning-rate-stage-2 0.0001 \
    --early-stop-patience 16 \
    --num-folds 5 \
    --my-seed 42 \
    --class-json-path "$DATASET_PATCH_DIR/report.json"

rm -r -f $PATCH_ROOT_DIR
