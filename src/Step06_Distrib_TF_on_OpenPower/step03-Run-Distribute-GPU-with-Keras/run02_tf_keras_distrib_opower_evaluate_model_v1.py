#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import numpy as np
import pandas as pd
import time
import datetime

import skimage.transform as sktf
import skimage.io as skio

from keras.models import Sequential
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras.layers import Convolution2D, Flatten, Dense, Activation, MaxPooling2D, ZeroPadding2D, Dropout
from keras.utils import np_utils
from tensorflow.contrib import slim

from keras.utils.visualize_util import plot as kplot

import math
import sys
import tempfile

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Ext params:
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Training batch size")
tf.app.flags.DEFINE_string("path_csv", "",
                           "Path to CSV with index-of-cls-images")
tf.app.flags.DEFINE_string("path_model", "",
                           "Path to TF-Serialyzed model file (*.ckpt)")

FLAGS = tf.app.flags.FLAGS

##############################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret

##############################################
def buildModel_VGG16_Orig(inpShape, numLabels):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=inpShape))
    model.add(Convolution2D(32, 3, 3, activation='relu',input_shape=inpShape))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numLabels, activation='softmax'))
    return model

def buildModel_VGG16_Mod(inpShape, numLabels):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu',input_shape=inpShape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    #
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(numLabels, activation='softmax'))
    return model

##############################################
class Batcher:
    meanPrefix='meanval.txt'
    pathCSV=None
    pathMeanVal=None
    lstPathImg=None
    lstIdxLbl=None
    uniqIdxLbl = None
    lstLabels=None
    uniqLabels=None
    numLabels=0
    numImg=0
    rangeIdx=None
    shapeImg=None
    numIterPerEpoch=0
    meanValue = 0.0
    meanImage = None
    imgScale  = 255.
    modelPrefix = None
    def __init__(self, pathCSV, isTheanoShape=False):
        self.isTheanoShape=isTheanoShape
        if not os.path.isfile(pathCSV):
            raise Exception('Cant fine CSV file [%s]' % pathCSV)
        self.pathCSV     = pathCSV
        self.pathMeanVal = '%s-%s' % (self.pathCSV, self.meanPrefix)
        self.wdir=os.path.dirname(self.pathCSV)
        tdata = pd.read_csv(self.pathCSV, sep=',')
        tlstPathImg = tdata['path'].as_matrix()
        tmplst=[os.path.join(self.wdir, ii) for ii in tlstPathImg]
        self.lstPathImg = np.array(tmplst)
        self.lstIdxLbl  = tdata['lblidx'].as_matrix()
        self.uniqIdxLbl = np.unique(self.lstIdxLbl)
        self.lstLabels  = tdata['label'].as_matrix()
        self.uniqLabels = np.unique(self.lstLabels)
        assert (len(self.uniqIdxLbl) == len(self.uniqLabels))
        # build correct correspondence Idx <-> Label Names and map[LabelName]=#LabelNames
        self.mapLabelsSizes={}
        tmpLabels=[]
        for ii in self.uniqIdxLbl.tolist():
            tmp=self.lstLabels[self.lstIdxLbl==ii]
            tlbl=np.unique(tmp)[0]
            self.mapLabelsSizes[tlbl] = len(tmp)
            tmpLabels.append(tlbl)
        self.uniqLabels=np.array(tmpLabels)
        #
        self.numImg = len(self.lstIdxLbl)
        self.numLabels  = len(self.uniqIdxLbl)
        self.arrIdxLbl2Cat = np_utils.to_categorical(self.lstIdxLbl, self.numLabels)
        #
        self.mapPath        = {}
        self.mapNumImg      = {}
        for kk in self.uniqIdxLbl.tolist():
            self.mapPath[kk]      = self.lstPathImg[self.lstIdxLbl==kk]
            self.mapNumImg[kk]    = len(self.mapPath[kk])
        #
        timg = skio.imread(self.lstPathImg[0])
        self.isRGB = (len(timg.shape)==3)
        if self.isTheanoShape:
            if len(timg.shape) < 3:
                self.shapeImg = tuple([1] + list(timg.shape))
            else:
                self.shapeImg = (timg.shape[2], timg.shape[0], timg.shape[1])
        else:
            if len(timg.shape) < 3:
                self.shapeImg = (timg.shape[0],timg.shape[1],1)
            else:
                self.shapeImg = timg.shape
        self.rangeIdx = range(self.numImg)
    def toString(self):
        self.checkIsInitialized()
        tstr = '#Images=%d, #Labels=%d, meanValue=%0.3f' % (self.numImg, self.numImg, self.meanValue)
        tstr2=''
        for kk,vv in self.mapLabelsSizes.items():
            tstr2='%s\t#%s = %d\n' % (tstr2, kk, vv)
        tstr = '%s\n%s' % (tstr, tstr2)
        return tstr
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def calcNumIterPerEpoch(self, batchSize):
        self.checkIsInitialized()
        return int(self.numImg/batchSize)
    def isInitialized(self):
        return (self.numImg>0) and (self.numLabels>0) and (self.wdir is not None)
    def checkIsInitialized(self):
        if not self.isInitialized():
            raise Exception('class Batcher() is not correctly initialized')
    def precalculateMean(self, maxNumberOfImage=4000, isRecalculateMean=False):
        self.checkIsInitialized()
        if os.path.isfile(self.pathMeanVal) and (not isRecalculateMean):
            print (':: found mean-value file, try to load from it [%s] ...' % self.pathMeanVal)
            tmp=np.loadtxt(self.pathMeanVal)
            self.meanValue = float(tmp[0])
        else:
            tmpListPath=np.random.permutation(self.lstPathImg)
            tmpNumImages=len(tmpListPath) - 1
            if tmpNumImages<=maxNumberOfImage:
                print (':: #Images=%d less than parameter [maxNumberOfImage=%d], cut to %d' % (tmpNumImages, maxNumberOfImage, tmpNumImages))
                maxNumberOfImage = tmpNumImages
            tmpListPath = tmpListPath[:maxNumberOfImage]
            self.meanImage = None
            for ppi, pp in enumerate(tmpListPath):
                tmp = skio.imread(pp).astype(np.float)/self.imgScale
                if self.meanImage is None:
                    self.meanImage = tmp
                else:
                    self.meanImage +=tmp
                if (ppi%500)==0:
                    print ('\t[%d/%d] ...' % (ppi, len(tmpListPath)))
            self.meanImage /=len(tmpListPath)
            self.meanValue = float(np.mean(self.meanImage))
            tmpStd = np.std(self.meanImage)
            tmpArray = np.array([self.meanValue, tmpStd])
            print (':: mean-value [%0.3f] saved to [%s]' % (self.meanValue, self.pathMeanVal))
            np.savetxt(self.pathMeanVal, tmpArray)
    def getBatchDataByRndIdx(self, rndIdx, isRemoveMean=True):
        parBatchSize = len(rndIdx)
        # rndIdx=np.random.permutation(self.rangeIdx)[:parBatchSize]
        dataX=np.zeros([parBatchSize] + list(self.shapeImg))
        dataY=self.arrIdxLbl2Cat[rndIdx,:]
        dataL=self.lstLabels[rndIdx]
        for ii,idx in enumerate(rndIdx):
            tpath = self.lstPathImg[idx]
            timg = skio.imread(tpath).astype(np.float)/self.imgScale
            if self.isTheanoShape:
                if self.isRGB:
                    timg = timg.transpose((2,0,1))
                else:
                    timg = timg.reshape(self.shapeImg)
            else:
                if not self.isRGB:
                    timg = timg.reshape(self.shapeImg)
            if isRemoveMean:
                dataX[ii] = timg - self.meanValue
            else:
                dataX[ii] = timg
        return (dataX, dataY, dataL)
    def getBatchData(self, parBatchSize=128, isRemoveMean=True):
        rndIdx=np.random.permutation(self.rangeIdx)[:parBatchSize]
        return self.getBatchDataByRndIdx(rndIdx=rndIdx, isRemoveMean=isRemoveMean)
    def buildModel(self):
        self.checkIsInitialized()
        return buildModel_VGG16_Mod(self.shapeImg, self.numLabels)

##############################################
def main(_):
    batchSize       = FLAGS.batch_size
    pathDatasetCSV  = FLAGS.path_csv
    pathModelTF     = FLAGS.path_model
    #
    if not os.path.isfile(pathDatasetCSV):
        raise Exception('Cant find Dataset Index file [%s]' % pathDatasetCSV)
    if not os.path.isfile(pathModelTF):
        raise Exception('Cant find TF Model file (*.ckpt-*) [%s]' % pathModelTF)
    if batchSize<1:
        raise Exception('Invalid Batch Size value [%s]' % batchSize)
    #
    batcherTrainTF = Batcher(pathCSV=pathDatasetCSV, isTheanoShape=True)
    batcherTrainTF.precalculateMean()
    print (batcherTrainTF)
    #
    dataShape   = batcherTrainTF.shapeImg
    numLabels   = batcherTrainTF.numLabels
    numImages   = batcherTrainTF.numImg
    listIndex   = range(numImages)
    splitListOfIndex    = split_list_by_blocks(listIndex, batchSize)
    numSplits   = len(splitListOfIndex)
    #
    with tf.device('/gpu:0'):
        # [*] Build Keras model:
        varX = tf.placeholder(tf.float32, [None, dataShape[0], dataShape[1], dataShape[2]])
        varY = tf.placeholder(tf.float32, [None, numLabels])
        #
        model = batcherTrainTF.buildModel()
        retY = model(varX)
        correct_prediction = tf.equal(tf.argmax(retY, 1), tf.argmax(varY, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, pathModelTF)
        arrTrainACC = []
        arrRetProb  = None
        for ii,ss in enumerate(splitListOfIndex):
            batch_xs, batch_ys, _ = batcherTrainTF.getBatchDataByRndIdx(ss)
            train_feed = {varX: batch_xs, varY: batch_ys}
            tret    = sess.run(retY, feed_dict=train_feed)
            if arrRetProb is None:
                arrRetProb = tret
            else:
                arrRetProb = np.concatenate((arrRetProb,tret))
            taccVal = sess.run(accuracy, feed_dict=train_feed)
            arrTrainACC.append(taccVal)
            print ('\t(eval) %d/%d : %0.3f' % (ii, numSplits, taccVal))
        maccTrain = np.mean(arrTrainACC)
        print ("Model [%s]: TRAIN Accuracy = %0.3f%%" % (os.path.basename(pathModelTF), 100. * maccTrain))
        # Export Class-Probabilities to file
        tdataCSV=pd.read_csv(batcherTrainTF.pathCSV)
        tdataCSV['probC0'] = arrRetProb[:, 0]
        tdataCSV['probC1'] = arrRetProb[:, 1]
        tpref=os.path.basename(os.path.dirname(pathModelTF)).split('_job-')[1]
        tfoutCSV = '%s-prob-%s.csv' % (batcherTrainTF.pathCSV,tpref)
        tdataCSV.to_csv(tfoutCSV, index=False)

if __name__ == "__main__":
    tf.app.run()
