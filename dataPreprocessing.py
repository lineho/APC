import DB
import pymysql
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine
import numpy as np
import xlsxwriter

from sklearn.preprocessing import LabelEncoder

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

def dummy_data(data, columns):
    print("Start dummy_data")
    for column in columns:
        #The process of creating dummy data (how to label)
        #data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        #Dummy data Delete the created column. (Duplicate Delete)
        data = data.drop(column, axis=1)
    return data

def extractRawData(numericDataList, nonNumericDataList):
    print("Start extractRawData")
    query = "SELECT * FROM fdc_data_20201022_raw ORDER BY time"
    dfRawData20201022 = pd.read_sql_query(query, DB.connectDB())
    
    #Outputs the raw data received.
    #print("dfRawData20201022: ",dfRawData20201022)

    #Reference: https://hogni.tistory.com/9
    # Select the country column.
    # Compare the values and conditions in the column.
    # Assign the result to a new variable.
    is_processing = dfRawData20201022['VID20'] == 'Processing'

    # Filter data that meets the conditions and store it in a new variable.
    dfProcessingData20201022 = dfRawData20201022[is_processing]

    # Rearrange the Index and output the results.
    dfProcessingData20201022.reset_index(drop=True, inplace=True)

    dfProcessingData20201022_extractVID433_Y = dfProcessingData20201022.loc[:, ['VID433']]

    dfProcessingData20201022_extractVID433_X = dfProcessingData20201022.drop(['time','VID433'], axis=1)

    dummy_columns = nonNumericDataList
    dfProcessingData20201022_extractVID433_X_dummy = dummy_data(dfProcessingData20201022_extractVID433_X, dummy_columns)
    print(dfProcessingData20201022_extractVID433_X_dummy.shape)

    return dfProcessingData20201022_extractVID433_X_dummy, dfProcessingData20201022_extractVID433_Y

def isNumeric(db):
    print("Start isNumeric")

    cursor = db.cursor()
    print("DB Connect Success")
    cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name='fdc_data_20201022_raw';
                    """)
    countColNum = cursor.fetchone()
    countColNum = countColNum[0]

    # print(countColNum)

    sql_countRowNum = "SELECT COUNT(*) FROM fdc_data_20201022_raw"
    cursor.execute(sql_countRowNum)
    countRowNum = cursor.fetchone()
    countRowNum = countRowNum[0]
    # print(countRowNum)

    numericDataList = list()
    nonNumericDataList = list()

    # is Numeric function
    # https://stackoverflow.com/questions/5064977/detect-if-value-is-number-in-mysql
    for i in range(1, countColNum):
        sql_countIsNumeric =  "SELECT count(VID{VID_Num}) FROM fdc_data_20201022_raw WHERE CONCAT('',VID{VID_Num} * 1) = VID{VID_Num};".format(VID_Num=i)

        cursor.execute(sql_countIsNumeric)
        countIsNumeric = cursor.fetchone()
        countIsNumeric = countIsNumeric[0]

        #Subtract the character rows from the entire row and output them by VID, which means they are all numeric data when zeroed.
        #print("VID{VID_Num}: ".format(VID_Num=i), countRowNum-countIsNumeric)

        if countRowNum-countIsNumeric == 0:
            numericDataList.append(i)
        else:
            k = "VID" + str(i)
            nonNumericDataList.append(k)
    #print("numericDataList", numericDataList)
    #print("nonNumericDataList", nonNumericDataList)
    #print(len(numericDataList))
    #print(len(nonNumericDataList))

    return numericDataList,nonNumericDataList