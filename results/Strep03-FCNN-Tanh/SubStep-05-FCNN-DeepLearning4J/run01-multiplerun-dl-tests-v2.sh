#!/bin/bash

runJava="-cp target/DeepAltorosD4J-1.0-SNAPSHOT.jar by.grid.imlab.Deeplearning4J_FCNN_MNIST"

numReps=5
sizeBatch=128

numEpoch=1
java $runJava 64 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 128 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 512 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=5
java $runJava 64 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 128 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 512 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=10
java $runJava 64 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 128 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 512 ${sizeBatch} ${numEpoch} ${numReps}
java $runJava 1024 ${sizeBatch} ${numEpoch} ${numReps}
