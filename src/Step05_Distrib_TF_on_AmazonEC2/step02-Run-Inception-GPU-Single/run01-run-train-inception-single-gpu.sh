#!/bin/bash

DIR_BASE="/data/camelyon16/dataset_PINK_300k"
DIR_OUTPUT="${DIR_BASE}/tf-data/"
LABELS_FILE="${DIR_BASE}/labels.txt"
DIR_TRAIN_MODEL="${DIR_BASE}/tf-model-train/"

LOG="${DIR_BASE}/log-train-$(date +%Y-%m-%d:%H:%M).txt"

export PYTHONPATH="${PWD}:$PYTHONPATH"

runpy="inception/histology_train.py"

python ${runpy} \
--batch_size=16 \
--data_dir="${DIR_OUTPUT}" \
--train_dir="${DIR_TRAIN_MODEL}"
## > $LOG 2>&1

