#!/bin/bash

wdir=`dirname $PWD`
tdir=`basename $PWD`

for ii in `seq 2 5`
do
    ##scp -r ./*.sh ubuntu@node${ii}g4:/data/camelyon16/dataset_PINK_300k-train-n3g4/
    scp -r ../${tdir} ubuntu@node${ii}g4:${wdir}/
done
