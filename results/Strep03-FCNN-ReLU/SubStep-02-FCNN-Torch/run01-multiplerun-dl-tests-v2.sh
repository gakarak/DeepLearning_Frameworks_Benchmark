#!/bin/bash

##runLua="run01_Torch_MNIST_TrainTest_v2.lua"
runLua="run01_Torch_MNIST_TrainTest_v2_ReLU.lua"

numReps=10
sizeBatch=128

numEpoch=1
th $runLua 64 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 128 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 512 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=5
th $runLua 64 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 128 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 512 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=10
th $runLua 64 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 128 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 512 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 1024 ${sizeBatch} ${numEpoch} ${numReps}
