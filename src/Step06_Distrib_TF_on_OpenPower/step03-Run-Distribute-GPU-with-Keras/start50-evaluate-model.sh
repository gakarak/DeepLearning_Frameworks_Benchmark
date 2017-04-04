#!/bin/bash

runpy='run02_tf_keras_distrib_opower_evaluate_model_v1.py'

batchSize=72

# Validation dataset path
pathCSV='/home/opuser/data/camelyon16/subdataset_PINK_10000_JPEG_Val/idx-cls.txt'

pathDirModel='/home/opuser/data/camelyon16/log-TaskOPower-1PS-8WK-1GPU/train_logs_tfkeras_job-worker-PS1-WS8-idx0'

# Select last CheckPoint file
pathModel=`ls -1 ${pathDirModel}/*.ckpt* | grep -v 'meta' | sort -n | tail -n 1`

echo "
-------
BatchSize : ${batchSize}
PathCSV   : ${pathCSV}
PathModel : ${pathModel}
-------
"

stdbuf -i0 -o0 -e0 python ${runpy} --batch_size=${batchSize} --path_csv=${pathCSV} --path_model=${pathModel}
