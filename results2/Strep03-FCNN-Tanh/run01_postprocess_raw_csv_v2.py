#!/usr/bin/python

import sys
import os
import numpy as np
import pandas as pd
import glob


parBatchSize=128
dictTest={'Test1' : ['100','100:100','100:100:100','100:100:100:100'],
          'Test2' : ['64','128','512','1024']}
dictNumP={'Test1' : [80284,90384,100484,110584],
          'Test2' : [51664,102544,407824,814864]}
lstParEpoch=[1,5,10]
lstFrameworks=['Keras', 'Torch', 'Caffe', 'Tensorflow', 'Deeplearning4J']

########################
if __name__=='__main__':
    if len(sys.argv)<2:
        print 'Usage: %s {/path/to/dir/wit/results}' % sys.argv[0]
        sys.exit(1)
    wdir=sys.argv[1]
    if not os.path.isdir(wdir):
        print 'Error: cant find directory [%s]' % wdir
        sys.exit(1)
    strHeader='Model,archType,numL,numN,numP,numEpoch,timeTrainMean,timeTrainStd,timeTestMean,timTestStd,accMean,accStd'
    foutCSVTest = '%s/results-all-in-one.csv' % wdir
    with open(foutCSVTest,'w') as f:
        print strHeader
        f.write('%s\n' % strHeader)
        for kk,lstParLayers in dictTest.items():
            for ff in lstFrameworks:
                for ee in lstParEpoch:
                    for lli,ll in enumerate(lstParLayers):
                        numN = int(ll.split(':')[0])
                        numL = len(ll.split(':'))
                        strSearch='%s/*/ModelFCN-%s-p%s-b%d-e%d-Log.txt' % (wdir, ff, ll, parBatchSize, ee)
                        lstCSV=glob.glob(strSearch)
                        if len(lstCSV)<1:
                            print 'Error: cant find raw-data CSV file!!! [%s]' % strSearch
                        else:
                            fnCSV=lstCSV[0]
                            tdata=pd.read_csv(fnCSV)
                            tarr=tdata[[' timeTrain', ' timeTest', ' acc']].as_matrix()
                            dataMean=np.mean(tarr,0)
                            dataStd = np.std(tarr, 0)
                            strCSV = '%s,%s,%d,%d,%d,%d, %0.3f,%0.4f, %0.3f,%0.4f, %0.3f,%0.4f' % (ff,kk[-1],numL,numN,dictNumP[kk][lli],ee, dataMean[0],dataStd[0], dataMean[1],dataStd[1], dataMean[2],dataStd[2])
                            print strCSV
                            f.write('%s\n' % strCSV)
