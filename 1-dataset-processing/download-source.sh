#!/bin/bash

OUTDIR="${1:-dataset-source/stone}" 

kaggle datasets download -d murillobouzon/kssd2025-kidney-stone-segmentation-dataset -p tmpdat/ --unzip

mkdir -p $OUTDIR
cp -rf tmpdat/data/image $OUTDIR
cp -rf tmpdat/data/label $OUTDIR
rm -rf tmpdat/
