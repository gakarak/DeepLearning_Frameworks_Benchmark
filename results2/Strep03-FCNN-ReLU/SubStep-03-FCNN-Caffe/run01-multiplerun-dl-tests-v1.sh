#!/bin/bash

##runpy="run01_Caffe_MNIST_TrainTest_Generator.py"
runpy="run01_Caffe_MNIST_TrainTest_Generator_ReLU.py"

numReps=10
sizeBatch=128

numEpoch=1
python $runpy 100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100:100:100 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=5
python $runpy 100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100:100:100 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=10
python $runpy 100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
python $runpy 100:100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
