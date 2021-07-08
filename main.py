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
import copy
import pickle
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
    fc_tableName = sf.fc_tableName
    dicToCheckSetVIDs = sf.main_dicToCheckSetVIDs
    dicToCheckSetVIDsVaule = sf.main_dicToCheckSetVIDsVaule
    dicToCheckActualVIDs = sf.main_dicToCheckActualVIDs
    prcessParameters = sf.process_Parameter

    createFolder('./data')
    createFolder('./data/df_raw')
    createFolder('./data/df')
    createFolder('./data/vif')
    createFolder('./data/after_vif')
    createFolder('./data/after_vif_csv')
    createFolder('./data/fd')
    createFolder('./data/etc')

    df_normalIntergrated = pd.DataFrame()
    listOfNormalAndAbnormal = [] #real label
    list_abnormalDataRate = []
    list_usedSVID = []
    for runNum in range(1, processCount + 1): #raw data
        globals()['df_raw_{}'.format(runNum)] = pd.DataFrame()
    for runNum in range(1, processCount + 1): #manipulation data
        globals()['df_{}'.format(runNum)] = pd.DataFrame()
    for runNum in range(1, processCount + 1): #manipulation data
        globals()['listOffault_{}'.format(runNum)] = ()
    for runNum in range(1, processCount + 1): #manipulation data
        globals()['df_fd_{}'.format(runNum)] = pd.DataFrame()

    #Before preprocessing (Manipulation step)
    ################################################################
    # # Data Normal Abnormal Classification and Normal Data Integration Steps
    # for runNum in range(1, processCount+1):
    #     globals()['df_raw_{}'.format(runNum)] = bpp.getEveryRun(data.connectDB(), data.db, runNum, tableName, sf.da_time, sf.da_mainprocess)
    #     isNormal, globals()['listOffault_{}'.format(runNum)] = bpp.distributeNormalAndAbnormal(globals()['df_raw_{}'.format(runNum)], dicToCheckSetVIDs, dicToCheckSetVIDsVaule)
    #     listOfNormalAndAbnormal.append(isNormal)
    #     if isNormal == 1:
    #         df_normalIntergrated = pd.concat([df_normalIntergrated, globals()['df_raw_{}'.format(runNum)]])
    #     globals()['df_raw_{}'.format(runNum)].to_excel('./data/df_raw/df_raw_{}.xlsx'.format(runNum))
    #     globals()['df_listOffault_{}'.format(runNum)] = pd.DataFrame(globals()['listOffault_{}'.format(runNum)])
    #     globals()['df_listOffault_{}'.format(runNum)].to_excel('./data/etc/df_listOffault_{}.xlsx'.format(runNum))
    #
    # with open('listOfNormalAndAbnormal.pkl', 'wb') as f:
    #     pickle.dump(listOfNormalAndAbnormal, f)
    #
    # # Data Manipulation Stage
    # for runNum in range(1, processCount+1):
    #     if listOfNormalAndAbnormal[runNum-1] == 0:
    #         globals()['df_{}'.format(runNum)] = bpp.manipulationDB(runNum ,globals()['df_raw_{}'.format(runNum)],
    #                                                                df_normalIntergrated, globals()['listOffault_{}'.format(runNum)],
    #                                                                sf.main_dicToCheckSetVIDs, sf.main_dicToCheckActualVIDs)
    #     else:
    #         globals()['df_{}'.format(runNum)] = globals()['df_raw_{}'.format(runNum)]
    #
    #  ###### globals()['df_{}'.format(runNum)].to_excel('./data/df/df_{}.xlsx'.format(runNum))
    #
    # ################################################################
    #
    # #Preprocessing(train로 선진행)
    # #Numeric Column Non-Numeric Separation by Only normal processing data & Delete Non-numeric
    # df_normalIntergrated ,nonNumericDataList = pp.isNumeric(df_normalIntergrated)
    # for runNum in range(1, processCount+1):
    #     print("Start delete NonNumericm, run number: ", runNum)
    #     globals()['df_{}'.format(runNum)] = pp.deleteNonNumeric(globals()['df_{}'.format(runNum)], nonNumericDataList)
    #
    # listOfVID_all = df_normalIntergrated.columns.values.tolist()
    #
    # #normal에서 constant 지울 열 추출 (뒤 vif랑 반대개념으로 추출)
    # listOfConstant_all = copy.deepcopy(listOfVID_all)
    # print(listOfConstant_all)
    # for runNum in range(1, processCount + 1):
    #     if listOfNormalAndAbnormal[runNum - 1] == 1:
    #         print("get constant value by run number: ", runNum)
    #         listOfConstant_each = pp.getConstantValue(globals()['df_{}'.format(runNum)])
    #         listOfConstant_all = list(set(listOfConstant_all) & set(listOfConstant_each))
    # print("get listOfConstant_all:", len(listOfConstant_all),listOfConstant_all)
    #
    # #CONSTANT값 제거
    # for runNum in range(1, processCount+1):
    #     print("Start delete CONSTANT value, run number: ", runNum)
    #     globals()['df_{}'.format(runNum)] = pp.deleteConstantValue(globals()['df_{}'.format(runNum)], listOfConstant_all)
    #
    # globals()['df_{}'.format(runNum)].to_excel('./data/df/df_{}.xlsx'.format(runNum))
    #
    # #VIF factor 살리는 것 구하기(노멀)
    # listOfVIF_all = copy.deepcopy(listOfVID_all)
    # for runNum in range(1, processCount+1):
    #     if listOfNormalAndAbnormal[runNum-1] == 1:
    #         print("get VIF factor by run number: ", runNum)
    #         df_vif, listOfVIF_each = pp.getVIF(globals()['df_{}'.format(runNum)])
    #         listOfVIF_all = list(set(listOfVIF_all) & set(listOfVIF_each))
    #
    #         df_vif.to_excel('./data/vif/df_vif_{}.xlsx'.format(runNum))
    #
    # print("get listOfVIF_all:", len(listOfVIF_all),listOfVIF_all)
    #
    # #VIF factor 제외 삭제(전체)
    # for runNum in range(1, processCount+1):
    #     globals()['df_{}'.format(runNum)] = pp.removeMulticollinearityeachdata(globals()['df_{}'.format(runNum)] ,listOfVIF_all)
    #     globals()['df_{}'.format(runNum)] = globals()['df_{}'.format(runNum)].reindex(sorted(globals()['df_{}'.format(runNum)].columns), axis=1)
    #
    #     globals()['df_{}'.format(runNum)].to_excel('./data/after_vif/df_{}.xlsx'.format(runNum))
    #     globals()['df_{}'.format(runNum)].to_csv('./data/after_vif_csv/df_{}.csv'.format(runNum))
    #
    # df_normalIntergrated = pp.removeMulticollinearityeachdata(df_normalIntergrated, listOfVIF_all)
    # df_normalIntergrated = df_normalIntergrated.reset_index(drop=True)
    # df_normalIntergrated = df_normalIntergrated.reindex(sorted(df_normalIntergrated.columns), axis=1)
    #
    # df_normalIntergrated.to_excel('./data/after_vif/df_normalIntergrated.xlsx')
    # df_normalIntergrated.to_csv('./data/after_vif_csv/df_normalIntergrated.csv')

    ################################################################

    #Fault detection Algorithm
    #train

    ################################################################################
    #test: 데이터 불러오기
    print("data load")
    df_normalIntergrated = pd.read_excel('./data/after_vif/df_normalIntergrated.xlsx')
    df_normalIntergrated.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    df_normalIntergrated = df_normalIntergrated.set_index(['time'])

    for runNum in range(1, processCount + 1):
        globals()['df_{}'.format(runNum)] = pd.read_excel('./data/after_vif/df_{}.xlsx'.format(runNum))
        globals()['df_{}'.format(runNum)].rename(columns={"Unnamed: 0": "time"}, inplace=True)
        globals()['df_{}'.format(runNum)] = globals()['df_{}'.format(runNum)].set_index(['time'])

    with open('./listOfNormalAndAbnormal.pkl', 'rb') as f:
        listOfNormalAndAbnormal = pickle.load(f)
    print("complete data load")

    ######################################################################################
    #Fault detection
    #train
    df_normalIntergrated_labeled = fd.anomalyDetectionTrain(df_normalIntergrated, dicToCheckActualVIDs)
    df_normalIntergrated_labeled.to_excel('./data/fd/df_normalIntergrated_labeled.xlsx')

    # Fault detection

    for runNum in range(1, processCount+1):
        globals()['df_fd_{}'.format(runNum)] = fd.anomalyDetectionTest(globals()['df_{}'.format(runNum)], dicToCheckActualVIDs)

    #FD score
    for runNum in range(1, processCount + 1):
        if listOfNormalAndAbnormal[runNum - 1] == 1:
            globals()['df_{}'.format(runNum)]['real_labeled'] = 1
            globals()['df_{}'.format(runNum)].to_excel('./data/fd/df_fd_{}.xlsx'.format(runNum))
        else:
            globals()['df_{}'.format(runNum)]['real_labeled'] = -1
            globals()['df_{}'.format(runNum)].to_excel('./data/fd/df_fd_{}.xlsx'.format(runNum))

    for runNum in range(1, processCount+1):
        print("FD predict score, run number:s ", runNum)
        fd_score_each = fd.anomalyDetectionScore(globals()['df_{}'.format(runNum)])
        list_abnormalDataRate.append(fd_score_each)

    print("FD predict score, all")
    df_total = pd.DataFrame()
    for runNum in range(1, processCount+1):
        df_total = pd.concat([df_total, globals()['df_{}'.format(runNum)]], ignore_index=True)
    df_total.to_excel('./data/fd/df_allIntergrated_labeled.xlsx')

    fd_score_total = fd.anomalyDetectionScore(df_total)

    ######################################################################################
    #
    # #fault classification
    # #fdc_list database 불러와서 저장하기.
    # df_nameOfSVID, doubleList_nameOfSVID = fc.getNameOfSVID(data.connectDB(), fc_tableName)
    # resultWordAnalsis = fc.wordAnalsis(doubleList_nameOfSVID, prcessParameters) # emulator 써서 0부터 시작인거 알아야함.
    # for runNum in range(1, processCount+1):
    #     print("runNum:", runNum, ",abnormalDataRate is",list_abnormalDataRate[runNum-1], "%")
    #     if list_abnormalDataRate[runNum-1] <= 20 :
    #         print("This is a normal process, so FC don't proceed with the analysis.")
    #     else:
    #         print("Abnormal Process Occurrence!")

    #test(오세창 박사님의 리그레션) (4단계: VID to set, OES to VID, OES to set, VID and OES to set)


