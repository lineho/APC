import DB
import dataPreprocessing
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process,Value, Array
from time import sleep
import test
import faultDetection
import faultClassification

DB.connectDB()

# main process
if __name__ == "__main__":

    # Preprocessing
    # Key SVID selection step.
    ###################################################################
    # Separate numeric and character data.
    numericDataList, nonNumericDataList = dataPreprocessing.isNumeric(DB.connectDB())

    # Feature X,Y Extract for Multicollinearity for SVID Selection.
    df = dataPreprocessing.extractDataByTrainProcess(numericDataList, nonNumericDataList)
    vif = dataPreprocessing.getMulticollinearity(df)
    vif = dataPreprocessing.removeMulticollinearityInfinite(vif)


    # kMeans Clustering: The reason I saved it in Excel is to cut it off at this point.
    dataPreprocessing.kmeansByMulticollinearity()
    dataPreprocessing.scikitOneDimensionalkMeans()
    dataPreprocessing.traindbFinalFormToAnalyze()

    #Create test data
    dataPreprocessing.extractDataByTestProcess(numericDataList, nonNumericDataList)
    dataPreprocessing.testdbFinalFormToAnalyze()

    # FD
    faultDetection.anomalytest()
    faultDetection.drawplot()

    # Befor FC
    dataPreprocessing.listClassification(DB.connectDB())

    # # # # FC
    faultClassification.removeNoise()
    faultClassification.randomForest()
    #


    
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
