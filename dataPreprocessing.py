import pandas as pd
import sys, os
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import numpy as np

warnings.filterwarnings(action='ignore')


def isNumeric(df):
    print("Start train data isNumeric")
    try:
        for column_name in df:
            df[column_name] = df[column_name].apply(pd.to_numeric, downcast = 'float' ,errors = 'coerce')
        listOfVID = df.columns[df.isna().any()].tolist()
        df = df.dropna(axis=1, how='all')

        return df, listOfVID

    except Exception as ex:
        print("Error: \n", ex)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def deleteNonNumeric(df, nonNumericDataList):
    for i in nonNumericDataList:
        df = df.drop(i, axis=1)
    print("Delete nonNumeric, After data shape is: " ,df.shape)

    return df

def getConstantValue(df):
    listOfConstant = []
    for column in df:
        # print(type(float(df[column])))
        df[column].apply(lambda x: float(x))
        if len(df[column].unique()) == 1:
            listOfConstant.append(column)
    return listOfConstant

def deleteConstantValue(df, listOfConstant_all):
    for i in listOfConstant_all:
        df = df.drop(i, axis=1)
    print("Delete Constant Value, After data shape is: " ,df.shape)
    return df

def getVIF(df):
    try:
        vif = pd.DataFrame()
        tx = df
        tx = tx.astype('float', copy=False)
        tx = tx._get_numeric_data() #tx.corr(), tx.shape
        vif["VIF_factor"] = [variance_inflation_factor(tx.values, i) for i in range(tx.shape[1])]
        vif["features"] = tx.columns
        vif = vif.sort_values("VIF_factor").reset_index(drop=True)
        df_vif = vif
        df_vif = df_vif.dropna(axis=0)
        df_vif.reset_index(drop=True, inplace=True)
        df_vif.drop([df_vif.columns[0], df_vif.columns[1]], axis=1)
        df_vif = df_vif[df_vif['VIF_factor'] != np.inf]
        # isVIFunder10 = df_vif['VIF_factor'] <= 10
        # df_vif = df_vif[isVIFunder10]
        df_vif_list = df_vif['features'].tolist()
        return df_vif, df_vif_list

    except Exception as ex:
        print("Error: \n", ex)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def removeMulticollinearityeachdata(df, listOfVIF_all):
    print("remove Multicollinearity each data")
    df_removeMulticollinearity = pd.DataFrame(df, columns=listOfVIF_all)
    print(df_removeMulticollinearity.shape)

    return df_removeMulticollinearity
