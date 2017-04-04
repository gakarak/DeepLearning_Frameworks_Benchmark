#!/bin/bash

wdir="/home/opuser/data/subdataset_PINK_50000"

fnLabels="${wdir}/labels.txt"
fnTrain="${wdir}/idx-train.txt"
fnVal="${wdir}/idx-val.txt"
foutDir="${wdir}/tf-data"

mkdir -p $foutDir

runpy="${wdir}/build_image_data_v2.py"

python ${runpy} \
  --train_idx_path="${fnTrain}" \
  --validation_idx_path="${fnVal}" \
  --output_directory="${foutDir}" \
  --labels_file="${fnLabels}" \
  --train_shards=8 \
  --validation_shards=2 \
  --num_threads=2
