#!/bin/bash

DIR_BASE="$HOME/data/data_Camelyon_5Cls"
DIR_TRAIN="${DIR_BASE}/train/"
DIR_VALIDATION="${DIR_BASE}/validation/"
DIR_OUTPUT="${DIR_BASE}/data-tf/"
LABELS_FILE="${DIR_BASE}/labels.txt"

DIR_TRAIN_MODEL="${DIR_BASE}/model-tf-train/"

export PYTHONPATH="${PWD}:$PYTHONPATH"
runpy="inception/histology_distributed_train.py"

# Run()
CUDA_VISIBLE_DEVICES=''
bazel-bin/inception/imagenet_distributed_train \
--batch_size=4 \
--data_dir="${DIR_OUTPUT}" \
--job_name='worker' \
--task_id=1 \
--ps_hosts='ubun1:2220' \
--worker_hosts='ubun1:2222,ubun2:2222'



