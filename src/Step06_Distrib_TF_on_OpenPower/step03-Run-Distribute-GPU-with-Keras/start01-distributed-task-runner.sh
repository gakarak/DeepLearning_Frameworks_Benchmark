#!/bin/bash

DIR_BASE="/home/opuser/data/camelyon16/subdataset_PINK_50000_JPEG_Train/idx-cls.txt"
DIR_WORK="${PWD}"
SSH_USER=`whoami` # On Amazon EC2 default is "ubuntu", on OpenPower: "opuser"

shTask="${DIR_WORK}/start00-tf-model-task.sh"

numEpochs=4
batchSize=72
numGPU=1
startPortPS="2220"
startPortWK="2230"
listPS="node1"
listWK="node2 node3 node4 node5 node6 node7 node8 node9"

################################################
numPS=`echo $listPS | wc -w`
numWK=`echo $listWK | wc -w`

TaskName="TaskOPower-${numPS}PS-${numWK}WK-${numGPU}GPU"

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
    idxGPU=0
    for gg in `seq 1 ${numGPU}`
    do
	echo "WK: ${nn} -> ${portPS} <- ($idxPS), gpu=[${gg}]"
	if [ -z "${lstWKj}" ]; then
	    lstWKj="${nn}:${portWK}|${idxWK}|${idxGPU}"
	    lstWKg="${nn}:${portWK}"
	else
	    lstWKj="${lstWKj},${nn}:${portWK}|${idxWK}|${idxGPU}"
	    lstWKg="${lstWKg},${nn}:${portWK}"
	fi
	((idxWK++))
	((portWK++))
	((idxGPU++))
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
    ssh ${SSH_USER}@${thost} "( ${shTask} ps ${tidx} ${TaskName} ${batchSize} ${numEpochs} ${lstPSg} ${lstWKg} -1 >/dev/null 2>&1 ) &"
done

sleep 2

# (2.2) Run batch WK-Tasks:
for ii in `echo ${lstWKj} | sed 's/\,/\ /g'`
do
    tidx=`echo $ii | cut -d\| -f2`
    tidxGPU=`echo $ii | cut -d\| -f3`
    thost=`echo $ii | cut -d\: -f1`
    echo "run WK-Task on [${thost}] with id=${tidx}, GPU=${tidxGPU}"
    ssh ${SSH_USER}@${thost} "( ${shTask} worker ${tidx} ${TaskName} ${batchSize} ${numEpochs} ${lstPSg} ${lstWKg} ${tidxGPU} >/dev/null 2>&1 ) &"
done

