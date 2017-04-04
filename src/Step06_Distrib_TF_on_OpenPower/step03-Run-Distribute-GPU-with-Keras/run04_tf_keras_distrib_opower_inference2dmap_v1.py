#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import glob
import time
import json

try:
   import cPickle as pickle
except:
   import pickle

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D,\
    Merge, ZeroPadding2D, Dropout
from keras import optimizers as opt
from keras.utils.visualize_util import plot as kplot

import skimage.io as skio
import skimage.transform as sktf

from keras.utils import np_utils

import numpy as np
import pandas as pd

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
    def __init__(self, pathCSV, isTheanoShape=True):
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
        self.modelPrefix = self.genPrefix()
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
    def precalculateMean(self, maxNumberOfImage=2000, isRecalculateMean=False):
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
    def evaluateModelOnBatch(self, model, parBatchSizeEval = 128):
        splitIdx = split_list_by_blocks(self.rangeIdx, parBatchSizeEval)
        numSplit = len(splitIdx)
        arrAcc  = []
        arrLoss = []
        for ii,lidx in enumerate(splitIdx):
            dataX, dataY, _ = self.getBatchDataByRndIdx(lidx)
            tret = model.test_on_batch(dataX, dataY)
            arrAcc.append(tret[1])
            arrLoss.append(tret[0])
            print ('[%d/%d] loss/acc = %0.3f/%0.2f%%' % (ii, numSplit, tret[0], 100.*tret[1]))
        meanAcc  = float(np.mean(arrAcc))
        meanLoss = float(np.mean(arrLoss))
        return (meanLoss, meanAcc)
    def predictOnModel(self, model, parBatchSizeEval = 128):
        splitIdx = split_list_by_blocks(self.rangeIdx, parBatchSizeEval)
        numSplit = len(splitIdx)
        retTot=None
        for ii, lidx in enumerate(splitIdx):
            dataX, _, _ = self.getBatchDataByRndIdx(lidx)
            tret = model.predict_on_batch(dataX)
            if retTot is None:
                retTot = tret
            else:
                retTot = np.concatenate((retTot,tret))
            if (ii%5)==0:
                print ('\tevaluate [%d/%d]' % (ii,numSplit))
        return retTot
    def buildModel(self):
        self.checkIsInitialized()
        return buildModel_VGG16_Mod(self.shapeImg, self.numLabels)
    def genPrefix(self):
        ret = "kmodel-cls-%s" % (time.strftime('%Y%m%d-%H%M%S'))
        return ret
    def exportModel(self, model, epochId):
        foutModel   = "%s-e%03d.json" % (self.modelPrefix, epochId)
        foutWeights = "%s-e%03d.h5" % (self.modelPrefix, epochId)
        foutModel   = '%s-%s' % (self.pathCSV, foutModel)
        foutWeights = '%s-%s' % (self.pathCSV, foutWeights)
        with open(foutModel, 'w') as f:
            str = json.dumps(json.loads(model.to_json()), indent=3)
            f.write(str)
        model.save_weights(foutWeights, overwrite=True)
        return foutModel

##############################################
def readImageAndReshape(fimg, isTheanoShape=True):
    timg = skio.imread(fimg).astype(np.float) / 255.
    # timg -= batcher.meanValue
    if isTheanoShape:
        if len(timg.shape) == 3:
            timg = timg.transpose((2, 0, 1))
        else:
            timg = timg.reshape((1, timg.shape[0], timg.shape[1]))
    else:
        if len(timg.shape) == 3:
            timg = timg.reshape((timg.shape[0], timg.shape[1], 1))
        else:
            pass
    # (0) Final reshape: for batch-processing
    timg = timg.reshape([1] + list(timg.shape))
    return timg

##############################################
def splitModel2CNNandFCNN(model, inpuImageShape, nameFlattenLayer='flatten_1'):
    inpShape = inpuImageShape
    #
    lstLayerNames = [[ii, ll.name] for ii, ll in enumerate(model.layers)]
    layerFlatten = [ii for ii in lstLayerNames if ii[1] == nameFlattenLayer][0]
    idxFlatten = layerFlatten[0]
    numLayers = len(lstLayerNames)
    numLayersCNN = idxFlatten + 0
    for ii in lstLayerNames:
        print (ii)
    print ('--------')
    print ('Flatten layer is %s, Flatten-index is %d' % (layerFlatten, idxFlatten))
    modelCNN = Sequential()
    modelFCNN = Sequential()
    # (1) Prepare CNN-part of Model
    print ('----[ CNN-Part ]----')
    for ii in range(numLayersCNN):
        tmpLayer = model.layers[ii]
        if ii == 0:
            if isinstance(tmpLayer, keras.layers.Convolution2D):
                newLayer = Convolution2D(nb_filter=tmpLayer.nb_filter,
                                         nb_row=tmpLayer.nb_row,
                                         nb_col=tmpLayer.nb_col,
                                         border_mode=tmpLayer.border_mode,
                                         input_shape=inpShape)
            elif isinstance(tmpLayer, keras.layers.ZeroPadding2D):
                newLayer = ZeroPadding2D(padding=tmpLayer.padding,
                                         input_shape=inpShape)
            else:
                raise Exception('Unsupported input CNN-Part-of-Model layer... [%s]' % type(tmpLayer))
        else:
            newLayer = tmpLayer
        modelCNN.add(newLayer)
        print ('\t:: CNN-Part of Model: load layer #%d/%d (%s)' % (ii, numLayersCNN, tmpLayer.name))
    # modelCNN.build(input_shape=inpShape)
    print (':: Load CNN-Part Model weights...')
    for ii in range(numLayersCNN):
        modelCNN.layers[ii].set_weights(model.layers[ii].get_weights())
    # (2) Prepare FCNN-part of Model
    print ('----[ FCNN-Part ]----')
    shapeFCNN = model.layers[numLayersCNN - 1].get_output_shape_at(0)[1:]
    lstIdxFNNLayers = range(numLayersCNN, numLayers)
    for i0, ii in enumerate(lstIdxFNNLayers):
        tmpLayer = model.layers[ii]
        if i0 == 0:
            newLayer = Flatten(input_shape=shapeFCNN)
        else:
            newLayer = tmpLayer
        modelFCNN.add(newLayer)
        print ('\t:: F*CNN-Part of Model: load layer #%d/%d (%s)' % (ii, len(lstIdxFNNLayers), tmpLayer.name))
    modelFCNN.build(input_shape=shapeFCNN)
    print (':: Load F*CNN-Part Model weights...')
    for i0, ii in enumerate(lstIdxFNNLayers):
        modelFCNN.layers[i0].set_weights(model.layers[ii].get_weights())
    #
    print ('--------------------')
    print ('::CNN:')
    for ii, ll in enumerate(modelCNN.layers):
        print ('\t%d : %s ---> %s' % (ii, ll.name, ll.get_output_shape_at(0)))
    print ('::F*CNN:')
    for ii, ll in enumerate(modelFCNN.layers):
        print ('\t%d : %s ---> %s' % (ii, ll.name, ll.get_output_shape_at(0)))
    return (modelCNN, modelFCNN)

##############################################
def loadModelFromJson(pathModelJson):
    if not os.path.isfile(pathModelJson):
        raise Exception('Cant find JSON-file [%s]' % pathModelJson)
    tpathBase = os.path.splitext(pathModelJson)[0]
    tpathModelWeights = '%s.h5' % tpathBase
    if not os.path.isfile(tpathModelWeights):
        raise Exception('Cant find h5-Weights-file [%s]' % tpathModelWeights)
    with open(pathModelJson, 'r') as f:
        tmpStr = f.read()
        model = keras.models.model_from_json(tmpStr)
        model.load_weights(tpathModelWeights)
        return model

##############################################
def buildProbMap(modelCNN, modelFCNN, pimg):
    retMapCNN = modelCNN.predict_on_batch(pimg)[0]
    plt.figure()
    for xx in range(40):
        plt.subplot(5,8,xx+1)
        plt.imshow(retMapCNN[xx])
        plt.axis('off')
    plt.axis('off')
    plt.show()

    inpShapeFCNN = modelFCNN.layers[0].get_input_shape_at(0)[1:]
    numLblFCNN = modelFCNN.layers[-1].get_output_shape_at(0)[1]
    nch = inpShapeFCNN[0]
    nrow = inpShapeFCNN[1]
    ncol = inpShapeFCNN[2]
    nrowCNN = retMapCNN.shape[1]
    ncolCNN = retMapCNN.shape[2]
    nrowCNN0 = nrowCNN - nrow + 1
    ncolCNN0 = ncolCNN - ncol + 1
    #
    batchSizeFCNNmax = 1024
    tretProb0 = None
    lstIdx = [[rr, cc] for rr in range(nrowCNN0) for cc in range(ncolCNN0)]
    splitLstIdx = split_list_by_blocks(lstIdx, batchSizeFCNNmax)
    for i0, lstPos in enumerate(splitLstIdx):
        tsizBatch = len(lstPos)
        tdataX = np.zeros((tsizBatch, nch, nrow, ncol))
        for j0, pp in enumerate(lstPos):
            tdataX[j0] = retMapCNN[:, pp[0]:pp[0] + nrow, pp[1]:pp[1] + ncol]
        tretFCNN = modelFCNN.predict_on_batch(tdataX)
        if tretProb0 is None:
            tretProb0 = tretFCNN
        else:
            tretProb0 = np.concatenate((tretProb0, tretFCNN))
    tretProb0R = tretProb0.reshape((nrowCNN0, ncolCNN0, numLblFCNN))
    retProb = np.zeros((nrowCNN, ncolCNN, numLblFCNN))
    retProb[:, :, 0] = 1.0
    tdr = (nrow - 1) / 2
    tdc = (ncol - 1) / 2
    retProb[tdr:nrowCNN0 + tdr, tdc:ncolCNN0 + tdc, :] = tretProb0R
    return retProb

##############################################
isTheanoShape   = True
# import tensorflow as tf

##############################################
if __name__ == '__main__':
    #
    if len(sys.argv)<4:
        print ('Usage: %s {/path/to/image} {/path/to/index.csv} {/path/to/model.json}')
        sys.exit(1)
    fimgDetect      = sys.argv[1]
    pathCSV         = sys.argv[2]
    pathModelJson   = sys.argv[3]
    if not os.path.isfile(fimgDetect):
        raise Exception('Cant find input image [%s], exit...' % fimgDetect)
    if not os.path.isfile(pathCSV):
        raise Exception('Cant find index file [%s], exit...' % pathCSV)
    if not os.path.isfile(pathModelJson):
        raise Exception('Cant find model json [%s], exit...' % pathModelJson)
    #
    batcher = Batcher(pathCSV=pathCSV)
    batcher.precalculateMean()
    print (batcher)
    timg  = readImageAndReshape(fimgDetect, isTheanoShape=True)
    timg -= batcher.meanValue
    #
    inpShape = timg.shape[1:]
    # with tf.device('/cpu:0'):
    model = loadModelFromJson(pathModelJson)
    #
    print (':: Input Model:')
    for ii, ll in enumerate(model.layers):
        print ('\t%d : %s ---> %s' % (ii, ll.name, ll.get_output_shape_at(0)))
    #
    t1 = time.time()
    modelCNN, modelFCNN = splitModel2CNNandFCNN(model, inpShape)
    #
    retProb=buildProbMap(modelCNN, modelFCNN, timg)
    dt = time.time() - t1
    print (':: dt = %0.3s' % dt)
    #
    retProbT=(retProb[:, :, 1] > (0.99 * retProb[:, :, 1].max())).astype(np.float)
    timg0=timg[0][0].copy()
    timgn=(timg0-timg0.min())/(timg0.max()-timg0.min())
    timgM=np.zeros( list(timgn.shape) + [3])
    timgM[:, :, 0] = sktf.resize(retProbT,timgn.shape)
    timgM[:, :, 1] = timgn
    #
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(retProb[:,:,1])
    plt.title('prob-map')
    plt.subplot(2, 2, 2)
    plt.imshow(retProbT)
    plt.title('prob-map thresholded: 0.95*max')
    plt.subplot(2, 2, 3)
    plt.imshow(timgn, cmap=plt.gray())
    plt.title('inp-image')
    plt.subplot(2, 2, 4)
    plt.imshow(timgM)
    plt.title('map-image')
    plt.show()
