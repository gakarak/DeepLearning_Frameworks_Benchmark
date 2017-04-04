#!/bin/bash

DIR_BASE="$HOME/data/data_Camelyon_5Cls"
DIR_TRAIN="${DIR_BASE}/train/"
DIR_VALIDATION="${DIR_BASE}/validation/"
DIR_OUTPUT="${DIR_BASE}/data-tf/"
LABELS_FILE="${DIR_BASE}/labels.txt"

DIR_TRAIN_MODEL="${DIR_BASE}/model-tf-train/"
DIR_EVAL_MODEL="${DIR_BASE}/model-tf-eval/"

runpy="inception/histology_eval.py"

mkdir -p "${DIR_EVAL_MODEL}"

export PYTHONPATH="${PWD}:$PYTHONPATH"

##bazel-bin/inception/imagenet_train --batch_size=4 --data_dir="${DIR_OUTPUT}" --train_dir="${DIR_TRAIN_MODEL}"

##bazel-bin/inception/imagenet_eval --num_examples=500 --num_gpus=1 --batch_size=8 --data_dir="${DIR_OUTPUT}" --checkpoint_dir="${DIR_TRAIN_MODEL}" --eval_dir="${DIR_EVAL_MODEL}" --run_once
python ${runpy} \
\ ##--num_examples=500 \
--num_gpus=1 \
--batch_size=16 \
--data_dir="${DIR_OUTPUT}" \
--checkpoint_dir="${DIR_TRAIN_MODEL}" \
--eval_dir="${DIR_EVAL_MODEL}" \
--run_once
