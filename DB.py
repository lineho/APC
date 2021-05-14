import pymysql

#DB connect definition function.
def connectDB():
    # Global variable of db to make it available throughout.
    global db
    db = pymysql.connect(host='localhost', port=3307, user='root', passwd='1234', db='fdc_db_sf6', charset='utf8',
                         autocommit=True)
    return db

