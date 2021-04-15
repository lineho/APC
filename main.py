import DB
import dataPreprocessing
import prerandomForest
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process,Value, Array
from time import sleep
import xlsxwriter
import postPreprocessing
import prekMeans
import preXGBoost

DB.connectDB()

# main process
if __name__ == "__main__":

    # Preprocessing
    # SVID extraction step.
    ###################################################################
    # Separate numeric and character data.
    numericDataList, nonNumericDataList = dataPreprocessing.isNumeric(DB.connectDB())

    # Feature X,Y Extract for Machine Learning for SVID Selection.
    x,y = dataPreprocessing.extractRawData(numericDataList, nonNumericDataList)

    # Number of radomForrest runs
    runNum = 10

    # Use XGBooost or RandomFoerst
    # Run i to verify importace rate. i Run to remove extreme randomness.
    # Permutation importance makes it more accurate but takes longer to run.
    #prerandomForest.severalTimesRandomForest(x, y, runNum)
    preXGBoost.severalTimesrunXGBoost(x, y, runNum)

    # kMeans Clustering
    # The reason I saved it in Excel is to cut it off at this point.
    # You can make code with DB later.
    # Select SVID with somewhat higher importance rate by 1-D means. (k=2)
    prekMeans.readXlsx()
    listSelectedSVID = prekMeans.OneDimensionalkMeans(runNum)

    postPreprocessing.PostPreprocess(listSelectedSVID, x, y)
    ###################################################################
    
    # multiprocessing단계
    # The References is shown below.
    # https://m.blog.naver.com/PostView.nhn?blogId=keon9&logNo=221163226431&proxyReferer=https:%2F%2Fwww.google.com%2F
    ###################################################################
    # num = Value('d', 0.0)
    # p1 = mp.Process(name="SubProcess1", target=DB.realTimeEmulatorDB, args=(num,)) #Data Receiving Process
    # p2 = mp.Process(name="SubProcess2", target=DB.extractionProcessDB, args=(num,)) #Data Split and Analysis Process
    # p1.start()
    # print("Waiting First DB data")
    # p2.start()
    # p1.join()
    # p2.join()
    ###################################################################

