#!/bin/bash

for ii in `seq 1 5`
do
    ssh ubuntu@node${ii}g1 "rm -rf /data/camelyon16/dataset_PINK_300k/log-Task*"
done
