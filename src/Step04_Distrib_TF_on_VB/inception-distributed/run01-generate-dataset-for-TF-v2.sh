#!/bin/bash

DIR_BASE="$HOME/data/data_Camelyon_5Cls"
DIR_TRAIN="${DIR_BASE}/train/"
DIR_VALIDATION="${DIR_BASE}/validation/"
DIR_OUTPUT="${DIR_BASE}/data-tf/"
LABELS_FILE="${DIR_BASE}/labels.txt"

runpy="inception/data/build_image_data.py"

mkdir -p "$DIR_OUTPUT"

##bazel build inception/build_image_data

##bazel-bin/inception/build_image_data \
python ${runpy} \
  --train_directory="${DIR_TRAIN}" \
  --validation_directory="${DIR_VALIDATION}" \
  --output_directory="${DIR_OUTPUT}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=2 \
  --validation_shards=1 \
  --num_threads=1