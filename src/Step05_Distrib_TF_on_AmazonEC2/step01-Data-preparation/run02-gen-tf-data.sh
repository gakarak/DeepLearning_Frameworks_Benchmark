#!/bin/bash

wdir="/data/camelyon16/dataset_PINK_300k"

fnLabels="${wdir}/labels.txt"
fnTrain="${wdir}/idx-train.txt"
fnVal="${wdir}/idx-val.txt"
foutDir="${wdir}/tf-data"

runpy="${wdir}/build_image_data_v2.py"

python ${runpy} \
  --train_idx_path="${fnTrain}" \
  --validation_idx_path="${fnVal}" \
  --output_directory="${foutDir}" \
  --labels_file="${fnLabels}" \
  --train_shards=2 \
  --validation_shards=1 \
  --num_threads=1
