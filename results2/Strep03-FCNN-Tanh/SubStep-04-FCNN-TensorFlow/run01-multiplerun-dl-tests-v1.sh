#!/bin/bash

runpy="run01_Tensorflow_MNIST_TrainTest.py"

numReps=5
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
