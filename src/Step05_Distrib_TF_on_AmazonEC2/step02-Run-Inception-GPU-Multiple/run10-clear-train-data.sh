#!/bin/bash

for ii in `seq 1 3`
do
##    ssh ubuntu@node${ii}g4 "rm -rf /data/camelyon16/dataset_PINK_300k/tf-model-train-distrib"
    ssh ubuntu@node${ii}g4 "rm -rf /data/camelyon16/dataset_PINK_300k/tf-model-train-distrib*"
done
