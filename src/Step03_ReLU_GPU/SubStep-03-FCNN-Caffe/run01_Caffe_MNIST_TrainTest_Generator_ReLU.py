#!/usr/bin/python

import os
import sys
import time
import glob
#

paramBatchSize=128
paramEpochs=10
paramReps=10

def generateSolver(modelName, parNumTrain, parSizeBatch, parNumEpoch):
    numIter=parNumEpoch*parNumTrain/parSizeBatch
    tstr="""
net: "network-%s.prototxt"
test_iter: 100
test_interval: 50000000
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
lr_policy: "inv"
gamma: 0.0001
power: 0.75
display: %d
max_iter: %d
snapshot: %d
snapshot_prefix: "%s"
solver_mode: CPU
""" % (modelName, numIter, numIter, numIter, modelName)
    return tstr

def generateModel(parDirTrainDB, parDirTestDB, parSizeBatch, parLayers):
    numParam = len(parLayers)
    sizeInput = 32*32
    sizeOutput = 10
    tstr="""
name: "FCNN"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "%s"
    batch_size: %d
    backend: LMDB
  }
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "%s"
    batch_size: 100
    backend: LMDB
  }
}
""" % (parDirTrainDB, paramBatchSize, parDirTestDB)
    tstrTmp="""
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "ip1"
  inner_product_param {
    num_output: %d
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "relu1"
}
""" % (parLayers[0])
    tstr='%s\n%s' % (tstr, tstrTmp)
    for ii in xrange(1, numParam):
        tstrTmp="""
layer {
  name: "ip%d"
  type: "InnerProduct"
  bottom: "relu%d"
  top: "ip%d"
  inner_product_param {
    num_output: %d
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu%d"
  type: "ReLU"
  bottom: "ip%d"
  top: "relu%d"
}
""" % (ii+1,ii,ii+1,parLayers[ii], ii+1,ii+1,ii+1)
        tstr='%s\n%s' % (tstr,tstrTmp)
    tstrTmp="""
layer {
  name: "ip%d"
  type: "InnerProduct"
  bottom: "relu%d"
  top: "ip%d"
  inner_product_param {
    num_output: %d
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip%d"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip%d"
  bottom: "label"
  top: "loss"
}
""" % (numParam+1, numParam, numParam+1, sizeOutput, numParam+1, numParam+1)
    tstr = '%s\n%s' % (tstr, tstrTmp)
    return tstr

if __name__=='__main__':
    # (0) Parse command-line arguments:
    if len(sys.argv)<2:
        print "Usage: %s {L1:L2:L3:...:Ln} {batchSize} {#Epochs} {numReps}" % os.path.basename(sys.argv[0])
        sys.exit(1)
    wdirTrain='mnist_train_lmdb'
    wdirTest ='mnist_test_lmdb'
    if not os.path.isdir(wdirTrain):
        print 'Error: incorrect train-db directory: [%s]' % wdirTrain
        sys.exit(1)
    if not os.path.isdir(wdirTest):
        print 'Error: incorrect test-db directory: [%s]' % wdirTest
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
    numTrain=60000
    numTest=10000
    modelName = 'ModelFCN-Caffe-p%s-b%d-e%d' % (sys.argv[1], paramBatchSize, paramEpochs)
    fileConfSolver='solver-%s.prototxt' % modelName
    fileConfModel ='network-%s.prototxt' % modelName
    with open(fileConfSolver,'w') as f:
        f.write('%s\n' % generateSolver(modelName, numTrain, paramBatchSize, paramEpochs))
    with open(fileConfModel, 'w') as f:
        f.write('%s\n' % generateModel(wdirTrain, wdirTest, paramBatchSize, arrParam))
    #####
    foutLog = '%s-Log.txt' % modelName
    with open(foutLog, 'w') as f:
        f.write('model, timeTrain, timeTest, acc\n')
        for ii in xrange(paramReps):
            # (1) Run training:
            t0 = time.time()
            strCommand="caffe train --solver=%s" % fileConfSolver
            print '(%d/%d) : run [%s]' % (ii, paramReps, strCommand)
            os.system(strCommand)
            dtTrain = time.time() - t0
            # (2) Run test:
            fnModel=glob.glob('%s*.caffemodel' % modelName)
            t0 = time.time()
            strRet=os.popen('caffe test -model %s -weights %s 2>&1' % (fileConfModel, fnModel[0])).read()
            dtTest = time.time() - t0
            retACC=float(strRet.splitlines()[-2].split('=')[-1])
            tsrt = '%s, %0.3f, %0.3f, %0.3f' % (modelName, dtTrain, dtTest, retACC)
            print tsrt
            f.write('%s\n' % tsrt)

