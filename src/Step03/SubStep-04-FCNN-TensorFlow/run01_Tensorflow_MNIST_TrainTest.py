#!/usr/bin/python

import os
import sys
import time
#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

###########################
def buildFCNN(varX, arrLayersSizes, numInput=784, numOutput=10):
    numLayers=len(arrLayersSizes)
    layer_x = tf.nn.tanh(
        tf.add(
            tf.matmul(varX, tf.Variable(tf.random_normal([numInput, arrLayersSizes[0]]))),
            tf.Variable(tf.random_normal([arrLayersSizes[0]]))
        ))
    for ii in xrange(1,numLayers):
        layer_x= tf.nn.tanh(
            tf.add(
                tf.matmul(layer_x, tf.Variable(tf.random_normal([arrLayersSizes[ii-1], arrLayersSizes[ii]]))),
                tf.Variable(tf.random_normal([arrLayersSizes[ii]]))
            ))
    return tf.matmul(layer_x, tf.Variable(tf.random_normal([arrLayersSizes[-1],numOutput]))) + tf.Variable(tf.random_normal([numOutput]))

paramBatchSize=128
paramEpochs=10
paramReps=10

###########################
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
    x=tf.placeholder("float", [None, sizeInput])
    y=tf.placeholder("float", [None, sizeOutput])
    modelFCNN=buildFCNN(x, arrLayersSizes=arrParam, numInput=sizeInput, numOutput=sizeOutput)
    # (3) Configure optimizer:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(modelFCNN, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    init = tf.initialize_all_variables()
    # (4) Configure test-functions:
    correct_prediction = tf.equal(tf.argmax(modelFCNN, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # (5) Run train & test model:
    modelName = 'ModelFCN-Tensorflow-p%s-b%d-e%d' % (sys.argv[1], paramBatchSize, paramEpochs)
    foutLog = '%s-Log.txt' % modelName
    with open(foutLog,'w') as f:
        f.write('model, timeTrain, timeTest, acc\n')
        for rr in xrange(paramReps):
            with tf.Session() as sess:
                sess.run(init)
                t0 = time.time()
                for epoch in range(paramEpochs):
                    numBatch=int(mnist.train.num_examples/paramBatchSize)
                    for ii in range(numBatch):
                        batch_xs, batch_ys = mnist.train.next_batch(paramBatchSize)
                        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                dtTrain = time.time() - t0
                t0 = time.time()
                retACC=sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                dtTest = time.time() - t0
                tsrt = '%s, %0.2f, %0.2f, %0.2f' % (modelName, dtTrain, dtTest, retACC)
                print tsrt
                f.write('%s\n' % tsrt)
