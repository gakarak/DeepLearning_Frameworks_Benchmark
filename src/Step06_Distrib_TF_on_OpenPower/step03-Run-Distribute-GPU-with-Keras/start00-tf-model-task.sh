#!/bin/bash

function usage {
    echo -e "
Usage: $(basename $0) {taskType:ps/worker}
\t{taskId:ps-Id/worker-Id}
\t{TaskName: to create train results dir}
\t{batchSize} {numEpochs}
\t{ps-hosts}
\t{worker-hosts}
\t{GPU-Idx}
"
    exit 1
}

function errorParam {
    pname="$@"
    echo "Incrorrect value of parameter ${pname}"
    echo "------"
    usage
}

################################
if [ -z "$1" ]; then
    usage
fi

source $HOME/bin/set-cuda.sh

DIR_BASE="/home/opuser/data/camelyon16/subdataset_PINK_50000_JPEG_Train"
IDX_CSV="${DIR_BASE}/idx-cls.txt"
DIR_WORK=`dirname $0`

DIR_OUTPUT="${DIR_BASE}/tf-data/"
LABELS_FILE="${DIR_BASE}/labels.txt"
taskType=$1
taskId=$2
taskName=$3
batchSize=$4
numEpochs=$5
psHosts="${6}"
workerHosts="${7}"
idxGPU=$8

hostName=`hostname`

echo -e "DEBUG:
\tDIR_BASE  = [${DIR_BASE}]
\tHost: ${hostName}
---
\ttaskType  = [${taskType}]
\ttaskId    = [${taskId}]
\ttaskName  = [${taskName}]
\tbatchSize = [${batchSize}]
\tnumIter   = [${numEpochs}]
\tidxGPU    = [${idxGPU}]
"

DIR_TRAIN_MODEL="${DIR_BASE}/tf-model-train-distrib-${taskName}/"
DIR_LOG="${DIR_BASE}/../log-${taskName}"
if [ "${taskType}" == "worker" ]; then
    rm -rf "${DIR_TRAIN_MODEL}"
fi
mkdir -p ${DIR_LOG}

################################
cd ${DIR_WORK}
export PYTHONPATH="${PWD}:$PYTHONPATH"
runpy="run01_tf_keras_distrib_opower_v1.py"

# Log
outLog="${DIR_LOG}/log-${taskType}-${hostName}-${taskId}-${batchSize}.txt"

export CUDA_VISIBLE_DEVICES=${idxGPU}

# Run()
python ${runpy} \
--job_name="${taskType}" \
--num_epochs="${numEpochs}" \
--batch_size=${batchSize} \
--path_train_dataset_idx="${IDX_CSV}" \
--task_index=${taskId} \
--ps_hosts="${psHosts}" \
--worker_hosts="${workerHosts}" > ${outLog} 2>&1
