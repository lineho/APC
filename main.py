import securityFile as sf
from database import DB
import emulator as emul
import beforeDataPreprocessing as bpp
import dataPreProcessing as pp
import faultDetection as fd
import faultClassification as fc
import test
import pandas as pd
import os
from time import sleep
import multiprocessing as mp
from multiprocessing import Process,Value, Array

data = DB(sf.db_host, sf.db_port, sf.db_user, sf.db_passwd, sf.db_dbname, sf.db_charset, sf.db_autocommit)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('create directory name: ', directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if __name__ == "__main__":

    #init variable
    processCount = sf.main_processCount
    tableName = sf.main_tableName
    dicToCheckSetVIDs = sf.main_dicToCheckSetVIDs
    dicToCheckSetVIDsVaule = sf.main_dicToCheckSetVIDsVaule
    dicToCheckActualVIDs = sf.main_dicToCheckActualVIDs

    createFolder('./data')
    createFolder('./data/df_raw')
    createFolder('./data/df')
    createFolder('./data/etc')

    #Before preprocessing (Manipulation step)
    ################################################################
    df_normalIntergrated = pd.DataFrame()
    listOfNormalAndAbnormal = [] #real label
    for runNum in range(1, processCount + 1): #raw data
        globals()['df_raw_{}'.format(runNum)] = pd.DataFrame()
    for runNum in range(1, processCount + 1): #manipulation data
        globals()['df_{}'.format(runNum)] = pd.DataFrame()
    for runNum in range(1, processCount + 1): #manipulation data
        globals()['listOffault_{}'.format(runNum)] = ()

    # Data Normal Abnormal Classification and Normal Data Integration Steps
    for runNum in range(1, processCount+1):
        globals()['df_raw_{}'.format(runNum)] = bpp.getEveryRun(data.connectDB(), data.db, runNum, tableName, sf.da_time, sf.da_mainprocess)
        isNormal, globals()['listOffault_{}'.format(runNum)] = bpp.distributeNormalAndAbnormal(globals()['df_raw_{}'.format(runNum)], dicToCheckSetVIDs, dicToCheckSetVIDsVaule)
        listOfNormalAndAbnormal.append(isNormal)
        if isNormal == 1:
            df_normalIntergrated = pd.concat([df_normalIntergrated, globals()['df_raw_{}'.format(runNum)]])

        globals()['df_raw_{}'.format(runNum)].to_excel('./data/df_raw/df_raw_{}.xlsx'.format(runNum))
        globals()['df_listOffault_{}'.format(runNum)] = pd.DataFrame(globals()['listOffault_{}'.format(runNum)])
        globals()['df_listOffault_{}'.format(runNum)].to_excel('./data/etc/df_listOffault_{}.xlsx'.format(runNum))

    # Data Manipulation Stage
    for runNum in range(1, processCount+1):
        if listOfNormalAndAbnormal[runNum-1] == 0:
            globals()['df_{}'.format(runNum)] = bpp.manipulationDB(runNum ,globals()['df_raw_{}'.format(runNum)],
                                                                   df_normalIntergrated, globals()['listOffault_{}'.format(runNum)],
                                                                   sf.main_dicToCheckSetVIDs, sf.main_dicToCheckActualVIDs)
        else:
            globals()['df_{}'.format(runNum)] = globals()['df_raw_{}'.format(runNum)]

        globals()['df_{}'.format(runNum)].to_excel('./data/df/df_{}.xlsx'.format(runNum))

    ################################################################

    #Preprocessing(train로 선진행)
    #Numeric Column Non-Numeric Separation by Only normal processing data & Delete Non-numeric
    nonNumericDataList = pp.isNumeric(df_normalIntergrated)
    for runNum in range(1, processCount+1):
        pass
    #VIF factor 삭제

    #Fault detection Algorithm
    #train

    #Preprocessing(test로 선진행)
    #Normal에 없는 열 전부 삭제

    #Fault detection
    #test

    #fault classification
    #교차비교(같은 FDC_list를 가졌음에도 불구하고 흔들리지 않는 것과 흔들리는 것의 차이)


    #test(오세창 박사님의 리그레션) (4단계: VID to set, OES to VID, OES to set, VID and OES to set)


