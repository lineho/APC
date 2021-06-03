import pymysql

class DB:
    def __init__(self, db_host,db_port,db_user,db_passwd,db_dbname,db_charset,db_autocommit):
        self.host = db_host
        self.port = db_port
        self.user = db_user
        self.passwd = db_passwd
        self.db = db_dbname
        self.charset = db_charset
        self.autocommit = db_autocommit

    #DB connect definition function.
    def connectDB(self):
        # Global variable of db to make it available throughout.
        global db
        db = pymysql.connect(host=self.host, port=self.port, user=self.user,
                             passwd=self.passwd, db=self.db,
                             charset=self.charset, autocommit=self.autocommit)
        return db

