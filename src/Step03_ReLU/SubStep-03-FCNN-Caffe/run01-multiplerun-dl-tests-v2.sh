#!/bin/bash

##runpy="run01_Caffe_MNIST_TrainTest_Generator.py"
runpy="run01_Caffe_MNIST_TrainTest_Generator_ReLU.py"

numReps=10
sizeBatch=128

numEpoch=1
python $runpy 64 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 128 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 512 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=5
python $runpy 64 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 128 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 512 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=10
python $runpy 64 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 128 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 512 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 1024 ${sizeBatch} ${numEpoch} ${numReps}
