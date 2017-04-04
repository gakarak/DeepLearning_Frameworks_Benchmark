#!/bin/bash

DIR_BASE="/data/camelyon16/dataset_PINK_300k"
DIR_WORK="${PWD}"
SSH_USER=`whoami` # On Amazon EC2 default is "ubuntu"

numIter=100
batchSize=24
numGPU=1
startPortPS="2220"
startPortWK="2230"
listPS="node1g4"
listWK="node1g4 node2g4 node3g4 node4g4 node5g4"
##listWK="node1g4 node2g4 node3g4 node4g4"
##listWK="node1g4 node2g4 node3g4"
##listWK="node1g4 node2g4"

################################################
numPS=`echo $listPS | wc -w`
numWK=`echo $listWK | wc -w`

TaskName="Task4g-${numPS}PS-${numWK}WK-${numGPU}GPU"

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

DIR_LOG="${DIR_BASE}/log-${TaskName}"
hostName=`hostname`
# (*2) Collect logs
for ii in `echo ${lstWKj} | sed 's/\,/\ /g'`
do
    tidx=`echo $ii | cut -d\| -f2`
    thost=`echo $ii | cut -d\: -f1`
    if [ "${hostName}" != "${thost}" ]; then
	echo "run WK-Task on [${thost}] with id=${tidx}"
	echo "${DIR_LOG}/"
	scp ${SSH_USER}@${thost}:${DIR_LOG}/*.txt ${DIR_LOG}/
    else
	echo "skip host [${thost}] ..."
    fi
done

