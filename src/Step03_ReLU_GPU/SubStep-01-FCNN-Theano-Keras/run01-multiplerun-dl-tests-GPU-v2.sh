#!/bin/bash

##runpy="run01_Keras_MNIST_TrainTest.py"
runpy="run01_Keras_MNIST_TrainTest_ReLU.py"


numReps=10
sizeBatch=128

numEpoch=1
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 64 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 128 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 512 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=5
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 64 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 128 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 512 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 1024 ${sizeBatch} ${numEpoch} ${numReps}

numEpoch=10
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 64 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 128 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 512 ${sizeBatch} ${numEpoch} ${numReps}
THEANO_FLAGS=floatX=float32,device=gpu python $runpy 1024 ${sizeBatch} ${numEpoch} ${numReps}
