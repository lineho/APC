import pymysql
from time import sleep
from datetime import datetime
import urllib
import os
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup

#DB realTimeEmulator Operational function.
def realTimeEmulatorDB(startExtration):

    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    print("DB Connect Success")

    # execute SQL query using execute() method.
    # ex) SELECT * FROM fdc_data_20201022_raw
    cursor.execute("SELECT COUNT(*) FROM fdc_data_20201022_raw")

    # Fetch a single row using fetchone() method. data format: tuple
    dataCount = cursor.fetchone()
    totalRowCount = dataCount[0]

    now_datetime = datetime.today()
    now_str_datetime = now_datetime.strftime('%Y_%m_%d')

    sql = "Create table IF NOT EXISTS fdc_virtual_data_{table_name}_raw like fdc_data_20201022_raw".format(table_name=now_str_datetime)
    cursor.execute(sql)

    #cursor.execute("Create Table fdc_data_")

    cursor.execute("SELECT * FROM fdc_data_20201022_raw")
    datas = cursor.fetchall()
    #datasAsPd = pd.DataFrame(datas)

    for data in datas:
        data = list(data)
        now_datetime = datetime.today()
        now_str_datetimes = now_datetime.strftime('%Y-%m-%d %H:%M:%S.%f') #Set to milliseconds and Sleep 0.1 for fast rotation. .%f
        data[0] = now_str_datetimes
        data = tuple(data)
        print(data)
        global startnum
        startExtration.value = 1
        insertSql = "INSERT INTO fdc_virtual_data_{table_name}_raw VALUES {data}".format(table_name=now_str_datetime, data=data)
        cursor.execute(insertSql)
        sleep(0.1) #1 is normal.
    # disconnect from server
    db.close()

def extractionProcessDB(startExtration):
    while startExtration.value == 0.0:
        if startExtration.value == 1:
            break
    print("Start Analyze!!!")
    # while startExtration.value == 1:
    #     print("Start Analyze!!!")
