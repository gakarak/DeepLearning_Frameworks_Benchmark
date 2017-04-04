#!/bin/bash

bidx='idx-cls.txt'
pathDirTrain="/home/opuser/data/camelyon16/subdataset_PINK_50000_JPEG_Train"
pathDirVal="/home/opuser/data/camelyon16/subdataset_PINK_10000_JPEG_Val"

pathTrainCSV="${pathDirTrain}/${bidx}"
pathValCSV="${pathDirVal}/${bidx}"

pathTrainMean="${pathTrainCSV}-meanval.txt"

#############
runpy="run00_build_mean_data_v0.py"

echo "*** Generate train-mean:"
python ${runpy} ${pathTrainCSV}

cp ${pathTrainMean} ${pathDirVal}/

uid=${USER}

for ii in `seq 1 9`
do
    tnode="node${ii}"
    echo "(copy) --> ( ${uid}@${tnode} )"
    scp $pathTrainMean ${uid}@${tnode}:${pathDirTrain}/
    scp $pathTrainMean ${uid}@${tnode}:${pathDirVal}/
done
