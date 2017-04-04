#!/bin/bash

uid="${USER}"

##wdir=`dirname $PWD`
##tdir=`basename $PWD`
wdir="$PWD"
tdir="$PWD"

for ii in `seq 1 9`
do
    ##scp -r ./*.sh ubuntu@node${ii}g4:/data/camelyon16/dataset_PINK_300k-train-n3g4/
    scp -r ${tdir}/* ${uid}@node${ii}:${wdir}/
done
