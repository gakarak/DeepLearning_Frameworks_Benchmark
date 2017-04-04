#!/bin/bash

DIR_BASE="/home/opuser/data/camelyon16/subdataset_PINK_50000_JPEG_Train"
DIR_WORK="${PWD}"
SSH_USER=`whoami` # On Amazon EC2 default is "ubuntu"

numEpochs=10
batchSize=64
numGPU=1
startPortPS="2220"
startPortWK="2230"
listPS="node1"
##listWK="node1"
listWK="node2 node3 node4 node5 node6 node7 node8 node9"

################################################
numPS=`echo $listPS | wc -w`
numWK=`echo $listWK | wc -w`

TaskName="TaskOPower-${numPS}PS-${numWK}WK-${numGPU}GPU"

echo "#PS=${numPS}, #WK=${numWK}"

# (*1) Prepare batch WK-Hosts:
listWKj=""
listWKg=""
idxWK="0"
for nn in ${listWK}
do
    portWK="${startPortWK}"
    for gg in `seq 1 ${numGPU}`
    do
	echo "WK: ${nn} -> ${portPS} <- ($idxPS), gpu=[${gg}]"
	if [ -z "${lstWKj}" ]; then
	    lstWKj="${nn}:${portWK}|${idxWK}"
	    lstWKg="${nn}:${portWK}"
	else
	    lstWKj="${lstWKj},${nn}:${portWK}|${idxWK}"
	    lstWKg="${lstWKg},${nn}:${portWK}"
	fi
	((idxWK++))
	((portWK++))
    done
done
echo "WK-List: --> [${lstWKj}]"
echo "WK-List-Global: --> [${lstWKg}]"

DIR_TRAIN_MODEL="${DIR_BASE}/tf-model-train-distrib-${TaskName}/"
DIR_LOG="${DIR_BASE}/../log-${TaskName}"
hostName=`hostname`
# (*2) Collect logs
for ii in `echo ${lstWKj} | sed 's/\,/\ /g'`
do
    tidx=`echo $ii | cut -d\| -f2`
    thost=`echo $ii | cut -d\: -f1`
    if [ "${hostName}" != "${thost}" ]; then
	echo "run WK-Task on [${thost}] with id=${tidx}"
	mkdir -p ${DIR_LOG}
	echo "::: [${DIR_LOG}/]"
	scp ${SSH_USER}@${thost}:${DIR_LOG}/*.txt ${DIR_LOG}/
	tdir="${DIR_BASE}/../train_logs_tfkeras_job-worker-PS${numPS}-WS${numWK}-idx${tidx}"
	todir="${DIR_LOG}/$(basename ${tdir})"
	mkdir -p ${todir}
	scp ${SSH_USER}@${thost}:${tdir}/* ${todir}/
    else
	echo "skip host [${thost}] ..."
    fi
done

