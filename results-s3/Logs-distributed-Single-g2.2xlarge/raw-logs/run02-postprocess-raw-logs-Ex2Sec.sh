#!/bin/bash

str="4,6,9,10,12"

strId='Ex2Sec'
idx='10'

for ll in `ls -1d log-Task*/`
do
    echo "---> ($ll) <---"
    cnt="0"
    numWK=`ls -1 ${ll}/*worker*.txt | wc -l`
    for ii in `ls -1 ${ll}/*worker*.txt | sort -n`
    do
	ftmp="${ii}.tmp-${strId}"
	if [ "${cnt}" == "0" ]; then
	    :> $ftmp
	    for kk in `cat $ii | grep 'step [0-9]' | grep 'INFO:tensorflow:Worker' | tail -n3 | sed 's/(/\ /g' | sed 's/)/\ /g' | cut -d\  -f4,6,${idx} | sed 's/\:\ /\ /g' | sed 's/\,//' | sed 's/\ /|/g'`
	    do
		ttime0=`echo $kk | cut -d\| -f1`
		tstep=`echo $kk | cut -d\| -f2`
		tval=`echo $kk | cut -d\| -f3`
		ttime=$(($(date -d "${ttime0}" +%s%N)/1000000))
		##echo "--> (${ttime}, ${tstep}, ${tval})"
		echo "${ttime}, ${numWK}, ${tstep}, ${tval}" >> $ftmp
	    done
	else
	    cat $ii | grep 'step [0-9]' | grep 'INFO:tensorflow:Worker' | tail -n3 \
	    | sed 's/(/\ /g' | sed 's/)/\ /g' \
	    | cut -d\  -f${idx} | sed 's/\:\ /\ /g' | sed 's/\,//' | sed 's/\ /, /g' > $ftmp
	fi
	((cnt++))
    done
done

for ll in `ls -1d log-Task*/`
do
    tidx=`basename $ll`
    echo "===> (${tidx}) <==="
    fout="${tidx}-${strId}.txt"
    lstFn=""
    for ii in `ls -1 ${ll}/*worker*.txt | sort -n`
    do
	ftmp="${ii}.tmp-${strId}"
	lstFn="${lstFn} ${ftmp}"
    done
    paste -d ,  $lstFn > $fout
done



