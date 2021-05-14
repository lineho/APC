import DB
import pymysql
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine
import numpy as np
import xlsxwriter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import kmeans1d
from sklearn.cluster import KMeans
from collections import Counter
import sys

def isNumeric(db):
    print("Start train data isNumeric")

    cursor = db.cursor()
    print("DB Connect Success")
    cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name='normaldata';
                    """)
    countColNum = cursor.fetchone()
    countColNum = countColNum[0]
    # print(countColNum)

    sql_countRowNum = "SELECT COUNT(*) FROM normaldata"
    cursor.execute(sql_countRowNum)
    countRowNum = cursor.fetchone()
    countRowNum = countRowNum[0]
    # print(countRowNum)

    numericDataList = list()
    nonNumericDataList = list()

    # is Numeric function
    # Reference: https://stackoverflow.com/questions/5064977/detect-if-value-is-number-in-mysql
    for i in range(1, countColNum):
        sql_countIsNumeric =  "SELECT count(VID{VID_Num}) FROM normaldata WHERE CONCAT('',VID{VID_Num} * 1) = VID{VID_Num};".format(VID_Num=i)

        cursor.execute(sql_countIsNumeric)
        countIsNumeric = cursor.fetchone()
        countIsNumeric = countIsNumeric[0]

        if countRowNum-countIsNumeric == 0:
            numericDataList.append(i)
        else:
            k = "VID" + str(i)
            nonNumericDataList.append(k)

    return numericDataList,nonNumericDataList


def dummy_data(data, columns):
    print("Start dummy_data")
    for column in columns:
        #The process of creating dummy data (how to label)
        #data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)

        #Dummy data Delete the created column. (Duplicate Delete)
        data = data.drop(column, axis=1)

    return data

def extractDataByTrainProcess(numericDataList, nonNumericDataList):
    print("Start extractDataByNormalProcess")
    query = "SELECT * FROM normaldata ORDER BY time"
    dfNormalData = pd.read_sql_query(query, DB.connectDB())

    # Reference: https://hogni.tistory.com/9

    is_processing = dfNormalData['VID20'] == 'Processing'

    # Filter data that meets the conditions and store it in a new variable.
    dfNormalData_processing = dfNormalData[is_processing]

    is_time = dfNormalData_processing['VID21'] == 'Time'
    dfNormalData_processing = dfNormalData_processing[is_time]

    # Rearrange the Index and output the results.
    dfNormalData_processing.reset_index(drop=True, inplace=True)
    dfNormalData_processing_extractTime = dfNormalData_processing.drop(['time'], axis=1)
    dummy_columns = nonNumericDataList
    dfTrainData_processing_extractTime_nonumericalDummy = dummy_data(dfNormalData_processing_extractTime, dummy_columns)
    print(dfTrainData_processing_extractTime_nonumericalDummy.shape)

    dfTrainData_processing_extractTime_nonumericalDummy = dfTrainData_processing_extractTime_nonumericalDummy.loc[:, dfTrainData_processing_extractTime_nonumericalDummy.apply(pd.Series.nunique) != 1]
    print(dfTrainData_processing_extractTime_nonumericalDummy.shape)

    dfTrainData_processing_extractTime_nonumericalDummy.to_excel('dfTrainData_processing_extractTime_nonumericalDummy.xlsx')

    return dfTrainData_processing_extractTime_nonumericalDummy

def getMulticollinearity(dfTrainData_processing_extractTime_nonumericalDummy):
    print("Start get Multicollinearity")

    # scale multicollinearity
    vif = pd.DataFrame()
    tx = dfTrainData_processing_extractTime_nonumericalDummy
    tx = tx.astype('float', copy=False)
    tx = tx._get_numeric_data() #tx.corr(), tx.shape
    vif["VIF_factor"] = [variance_inflation_factor(tx.values, i) for i in range(tx.shape[1])]
    vif["features"] = tx.columns
    vif = vif.sort_values("VIF_factor").reset_index(drop=True)
    vif.to_excel('vif.xlsx')

    return vif

def removeMulticollinearityInfinite(vif):
    print("Start remove MulticollinearityInfinite")

    df_vif = vif
    df_vif = df_vif.dropna(axis=0)
    df_vif.reset_index(drop=True, inplace=True)
    df_vif.drop([df_vif.columns[0], df_vif.columns[1]], axis=1)
    df_vif = df_vif[df_vif['VIF_factor'] != np.inf]
    df_vif.to_excel('removeInfiniteVif.xlsx')
    #print(df_vif)

    return df_vif

def kmeansByMulticollinearity():
    print("Start kmeans By Multicollinearity")
    df = pd.read_excel('dfTrainData_processing_extractTime_nonumericalDummy.xlsx', header=0)
    df_vif = pd.read_excel('removeInfiniteVif.xlsx', header=0)

    df_vif.rename(columns={"Unnamed: 0": "index"}, inplace=True)
    df_vif['features'] = df_vif['features'].str.split("VID").str[1]
    df_vif = df_vif.set_index(['features'])
    df_vif = df_vif.drop(columns='index')
    df_vif.to_excel('prekmeansByMulticollinearity.xlsx')


    df_vif_top20 = df_vif.VIF_factor.sort_values(ascending=False)[:20]

    plt.barh(df_vif_top20.index, df_vif_top20)  # draw a horizontal histogram
    plt.title('Variance Inflation Factors')
    plt.ylabel("VIDs")
    plt.xlabel("VIF_factor")

    plt.tight_layout()
    plt.show()

def scikitOneDimensionalkMeans():
    print("Start kMeans")
    df = pd.read_excel('prekmeansByMulticollinearity.xlsx', header=0)

    x = df['VIF_factor'].tolist()
    x = df[['VIF_factor']]

    k_pick = 1
    memory_distortions = float('inf')
    breaker = False

    for i in range(1, 501, 20):
        distortions = []
        start = i
        for i in range(i, i+20):
            kmeans_plus = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
            kmeans_plus.fit(x)
            distortions.append(kmeans_plus.inertia_)
            if  min(distortions) <= memory_distortions:
                memory_distortions = min(distortions)
                k_pick += 1
                print("k pick",k_pick)
            else:
                breaker = True
                break
        if breaker == True:
            break
            #print('distortion : %.2f' % kmeans_plus.inertia_)
        plt.plot(range(start, start+20), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()
        plt.show()

    distortions = []
    for i in range(start, start+20):
        kmeans_plus = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        kmeans_plus.fit(x)
        distortions.append(kmeans_plus.inertia_)

        #print('distortion : %.2f' % kmeans_plus.inertia_)
    plt.plot(range(start, start+20), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.show()

    print("k value is ", k_pick)
    km = KMeans(n_clusters=k_pick, init="k-means++")
    km.fit(x)
    pred = km.fit_predict(x)
    print(x)
    print(pred)
    i = pred[0]

    print("Number of SVIDs to be used finally:", list(pred).count(i))
    zeroCountBykMeans = list(pred).count(i)

    df = df.loc[:zeroCountBykMeans-1, : ]
    df = df.set_index(['features'])
    df.to_excel('after_k_means.xlsx')

def traindbFinalFormToAnalyze():
    print("train db FinalForm To Analyze")
    vif = pd.read_excel('after_k_means.xlsx', header=0)
    df = pd.read_excel('dfTrainData_processing_extractTime_nonumericalDummy.xlsx', header=0)

    vif['features'] = 'VID' + vif['features'].astype(str)
    vifList = vif.features.tolist()

    df.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    df = df.set_index(['time'])

    for i in df:
        if i in vifList:
            continue
        else:
            del df[i]

    #print(df)
    df.to_excel('after_preprocessing_train_data.xlsx')

def extractDataByTestProcess(numericDataList, nonNumericDataList):
    print("Start extractDataByTestProcess")
    query = "SELECT * FROM fdc_data_20201022_raw ORDER BY time"
    dfTestData = pd.read_sql_query(query, DB.connectDB())

    # Reference: https://hogni.tistory.com/9
    is_processing = dfTestData['VID20'] == 'Processing'

    # Filter data that meets the conditions and store it in a new variable.
    dfTestData_processing = dfTestData[is_processing]

    is_time = dfTestData_processing['VID21'] == 'Time'
    dfTestData_processing = dfTestData_processing[is_time]

    # Rearrange the Index and output the results.
    dfTestData_processing.reset_index(drop=True, inplace=True)

    dfTestData_processing_extractTime = dfTestData_processing.drop(['time'], axis=1)
    dummy_columns = nonNumericDataList

    dfTestData_processing_extractTime_nonumericalDummy = dummy_data(dfTestData_processing_extractTime, dummy_columns)
    print(dfTestData_processing_extractTime_nonumericalDummy.shape)
    dfTestData_processing_extractTime_nonumericalDummy.to_excel('raw_dfTestData.xlsx')

    df_test = pd.read_excel('dfTrainData_processing_extractTime_nonumericalDummy.xlsx', header=0)

    df_test_list = df_test.columns.tolist()

    for i in dfTestData_processing_extractTime_nonumericalDummy:
        if i in df_test_list:
            continue
        else:
            del dfTestData_processing_extractTime_nonumericalDummy[i]

    print(dfTestData_processing_extractTime_nonumericalDummy.shape)

    dfTestData_processing_extractTime_nonumericalDummy.to_excel('dfTestData_processing_extractTime_nonumericalDummy.xlsx')

    return dfTestData_processing_extractTime_nonumericalDummy


def testdbFinalFormToAnalyze():
    print("test db FinalForm To Analyze")

    vif = pd.read_excel('after_k_means.xlsx', header=0)
    df_test = pd.read_excel('dfTestData_processing_extractTime_nonumericalDummy.xlsx', header=0)

    vif['features'] = 'VID' + vif['features'].astype(str)
    vifList = vif.features.tolist()

    df_test.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    df = df_test.set_index(['time'])

    for i in df:
        if i in vifList:
            continue
        else:
            del df[i]

    df.columns.tolist()

    df.to_excel('after_preprocessing_test_data.xlsx')