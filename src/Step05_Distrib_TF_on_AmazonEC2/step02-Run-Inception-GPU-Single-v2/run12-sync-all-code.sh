#!/bin/bash

wdir=`dirname $PWD`
tdir=`basename $PWD`

for ii in `seq 2 5`
do
    scp -r ../${tdir} ubuntu@node${ii}g1:${wdir}/
done
