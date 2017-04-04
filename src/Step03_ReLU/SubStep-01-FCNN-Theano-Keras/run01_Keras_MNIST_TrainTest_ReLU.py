#!/usr/bin/python

import os
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils.visualize_util import plot as kerasPlot
import time
#
from tensorflow.examples.tutorials.mnist import input_data

paramBatchSize=128
paramEpochs=10
paramReps=10

if __name__=='__main__':
    # (0) Parse command-line arguments:
    if len(sys.argv)<2:
        print "Usage: %s {L1:L2:L3:...:Ln} {batchSize} {#Epochs} {numReps}" % os.path.basename(sys.argv[0])
        sys.exit(1)
    arrParam=[int(ii) for ii in sys.argv[1].split(':')]
    if len(arrParam)<1:
        print "Error: incorrect number of params... exit"
        sys.exit(1)
    if len(sys.argv) > 4:
        paramBatchSize  = int(sys.argv[2])
        paramEpochs     = int(sys.argv[3])
        paramReps       = int(sys.argv[4])
    # (1) Read dataset:
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # (2) Build model:
    numParam = len(arrParam)
    sizeInput=mnist.train.images.shape[1]
    sizeOutput=mnist.train.labels.shape[1]
    model = Sequential()
    model.add(Dense(output_dim=arrParam[0], input_dim=sizeInput, init="glorot_uniform"))
    model.add(Activation("relu"))
    for ii in xrange(1,numParam):
        model.add(Dense(output_dim=arrParam[ii], input_dim=arrParam[ii-1], init="glorot_uniform"))
        model.add(Activation("relu"))
    model.add(Dense(output_dim=sizeOutput, input_dim=arrParam[numParam-1], init="glorot_uniform"))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
    # Debug:
    modelName='ModelFCN-Keras-p%s-b%d-e%d' % (sys.argv[1], paramBatchSize, paramEpochs)
    kerasPlot(model, to_file='%s.png' % modelName)
    # (3) Train & test model:
    foutLog='%s-Log.txt' % modelName
    with open(foutLog,'w') as f:
        f.write('model, timeTrain, timeTest, acc\n')
        for ii in xrange(paramReps):
            # train-stage
            t0=time.time()
            model.fit(mnist.train.images, mnist.train.labels, nb_epoch=paramEpochs, batch_size=paramBatchSize, show_accuracy=False, verbose=0)
            dtTrain=time.time()-t0
            t0=time.time()
            retACC=model.evaluate(mnist.test.images, mnist.test.labels, show_accuracy=True)
            dtTest=time.time()-t0
            tsrt = '%s, %0.3f, %0.3f, %0.3f' % (modelName, dtTrain, dtTest, retACC[1])
            print tsrt
            f.write('%s\n' % tsrt)
