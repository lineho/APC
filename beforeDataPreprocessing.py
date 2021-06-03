import sys, os
import pandas as pd
import numpy as np

def getEveryRun(db, dbName ,runNum, tableName, da_time, da_mainprocess):
    print("Start getEveryRun(db:" ,dbName,"runNum:", runNum,", tableName:",tableName+str(runNum),")")

    try:
        print("Get DB number:", runNum)

        cursor = db.cursor()
        cursor.execute("""
                           SELECT COUNT(*) as cnt FROM {tableName}{Run_Num}
                           """.format(tableName=tableName, Run_Num=runNum))
        countRowNum = cursor.fetchone()
        countRowNum = countRowNum[0]
        analyticalNumbers = countRowNum//10
        countRowNum = analyticalNumbers * 10
        query = "SELECT * FROM {tableName}{Run_Num} LIMIT {countRowNum}".format(tableName=tableName, countRowNum = countRowNum,Run_Num=runNum)
        globals()['{tableName}{Run_Num}'.format(tableName=tableName, Run_Num=runNum)] = pd.read_sql_query(query, db)

        globals()['{tableName}{Run_Num}'.format(tableName=tableName, Run_Num=runNum)] = extractProcess(globals()['{tableName}{Run_Num}'.format(tableName=tableName, Run_Num=runNum)],da_time,da_mainprocess)

        return globals()['{tableName}{Run_Num}'.format(tableName=tableName, Run_Num=runNum)]

    except Exception as ex:
        print("Error: \n", ex)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def distributeNormalAndAbnormal(df, dicToCheckSetVIDs, dicToCheckSetVIDsVaule):
    try:
        print("Start distributeNormalAndAbnormal")
        isNormaldata = 0
        listOffault = []
        for key, value in dicToCheckSetVIDs.items():
            if (df[value].astype(int) == dicToCheckSetVIDsVaule[key]).any() == True:
                isNormaldata += 1
            else:
                listOffault.append(key)
        if isNormaldata == len(dicToCheckSetVIDs):
            return 1, listOffault
        else:
            return 0, listOffault

    except Exception as ex:
        print("Error: \n", ex)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def manipulationDB(runNum ,df_raw, df_normal, listOfFault, main_dicToCheckSetVIDs, main_dicToCheckActualVIDs):
    try:
        print("start Manipulation DB")
        print("runNum:", runNum, "will manipulation")

        df_normal = df_normal.copy()
        df_manipulate = df_raw.copy()

        for i in range(0, len(listOfFault)):
            globals()['listOfSet_{}'.format(main_dicToCheckSetVIDs[listOfFault[i]])] = ()
            globals()['listOfActual_{}'.format(main_dicToCheckActualVIDs[listOfFault[i]])] = ()

            globals()['listOfSet_{}'.format(main_dicToCheckSetVIDs[listOfFault[i]])] = df_normal[main_dicToCheckSetVIDs[listOfFault[i]]].tolist()
            globals()['listOfActual_{}'.format(main_dicToCheckActualVIDs[listOfFault[i]])] = df_normal[main_dicToCheckActualVIDs[listOfFault[i]]].tolist()

            df_raw[main_dicToCheckSetVIDs[listOfFault[i]]] = np.random.choice(globals()['listOfSet_{}'.format(main_dicToCheckSetVIDs[listOfFault[i]])], size=len(df_raw))
            df_raw[main_dicToCheckActualVIDs[listOfFault[i]]] = np.random.choice( globals()['listOfActual_{}'.format(main_dicToCheckActualVIDs[listOfFault[i]])], size=len(df_raw))

        return df_manipulate

    except Exception as ex:
        print("Error: \n", ex)

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


def extractProcess(df, da_time, da_mainprocess):
    try:
        print("extract process data")
        del df[da_time]
        df_delTime = df.copy()
        for i, j in da_mainprocess.items():
            df_delTime[i] = df_delTime[i].astype('str')
            isProcessing = df_delTime[i] == j
            df_delTime = df_delTime[isProcessing]
        df_delTime.reset_index(drop=True, inplace=True)
        return df_delTime

    except Exception as ex:
        print("Error: \n", ex)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
