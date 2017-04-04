#!/bin/bash

DIR_BASE="/data/camelyon16/dataset_PINK_300k"
DIR_WORK="${PWD}"
SSH_USER=`whoami` # On Amazon EC2 default is "ubuntu"

shTask="${DIR_WORK}/run00-inceptiont-task.sh"

numIter=100
batchSize=24
numGPU=1
startPortPS="2220"
startPortWK="2230"
listPS="node1g1"
listWK="node1g1 node2g1 node3g1 node4g1 node5g1"

################################################
numPS=`echo $listPS | wc -w`
numWK=`echo $listWK | wc -w`

TaskName="Task1g-${numPS}PS-${numWK}WK-${numGPU}GPU"

echo "#PS=${numPS}, #WK=${numWK}"

################################################
# (1.1) Prepare batch PS-Hosts:
listPSj=""
listPSg=""
idxPS="0"
for nn in ${listPS}
do
    portPS="${startPortPS}"
    echo "PS: ${nn} -> ${portPS} <- ($idxPS)"
    if [ -z "${lstPSj}" ]; then
	lstPSj="${nn}:${portPS}|${idxPS}"
	lstPSg="${nn}:${portPS}"
    else
	lstPSj="${lstPSj},${nn}:${portPS}|${idxPS}"
	lstPSg="${lstPSj},${nn}:${portPS}"
    fi
    ((idxPS++))
    ((portPS++))
done
echo "PS-List: --> [${lstPSj}]"
echo "PS-List-Global: --> [${lstPSg}]"

# (1.2) Prepare batch WK-Hosts:
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

################################################
# (2.1) Run batch PS-Tasks:
for ii in `echo ${lstPSj} | sed 's/\,/\ /g'`
do
    tidx=`echo $ii | cut -d\| -f2`
    thost=`echo $ii | cut -d\: -f1`
    echo "run PS-Task on [${thost}] with id=${tidx}"
    ssh ${SSH_USER}@${thost} "( ${shTask} ps ${tidx} ${TaskName} ${batchSize} ${numIter} ${lstPSg} ${lstWKg} >/dev/null 2>&1 ) &"
    ##ssh ${SSH_USER}@${thost} "${shTask} ps ${tidx} ${TaskName} ${batchSize} ${numIter} ${lstPSg} ${lstWKg}"
    ##${shTask} ps ${tidx} ${TaskName} ${batchSize} ${numIter} ${lstPSg} ${lstWKg}
done

# (2.2) Run batch WK-Tasks:
for ii in `echo ${lstWKj} | sed 's/\,/\ /g'`
do
    tidx=`echo $ii | cut -d\| -f2`
    thost=`echo $ii | cut -d\: -f1`
    echo "run WK-Task on [${thost}] with id=${tidx}"
    ssh ${SSH_USER}@${thost} "( ${shTask} worker ${tidx} ${TaskName} ${batchSize} ${numIter} ${lstPSg} ${lstWKg} >/dev/null 2>&1 ) &"
    ##ssh ${SSH_USER}@${thost} "${shTask} worker ${tidx} ${TaskName} ${batchSize} ${numIter} ${lstPSg} ${lstWKg}"
done

