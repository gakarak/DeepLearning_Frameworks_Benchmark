#!/bin/bash

DIR_BASE="$HOME/data/data_Camelyon_5Cls"
DIR_TRAIN="${DIR_BASE}/train/"
DIR_VALIDATION="${DIR_BASE}/validation/"
DIR_OUTPUT="${DIR_BASE}/data-tf/"
LABELS_FILE="${DIR_BASE}/labels.txt"

DIR_TRAIN_MODEL="${DIR_BASE}/model-tf-train/"

export PYTHONPATH="${PWD}:$PYTHONPATH"

runpy="inception/histology_train.py"

##bazel build inception/imagenet_train

##bazel-bin/inception/imagenet_train --num_gpus=1 --batch_size=16 --data_dir="${DIR_OUTPUT}" --train_dir="${DIR_TRAIN_MODEL}"
##bazel-bin/inception/imagenet_train --batch_size=16 --data_dir="${DIR_OUTPUT}" --train_dir="${DIR_TRAIN_MODEL}"

python ${runpy} \
--batch_size=16 \
--data_dir="${DIR_OUTPUT}" \
--train_dir="${DIR_TRAIN_MODEL}"


