#!/bin/bash

runLua="run01_Torch_MNIST_TrainTest_v2.lua"

numReps=10
sizeBatch=128

numEpoch=1
th $runLua 100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100:100:100 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=5
th $runLua 100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100:100:100 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=10
th $runLua 100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
th $runLua 100:100:100:100 ${sizeBatch} ${numEpoch} ${numReps}
