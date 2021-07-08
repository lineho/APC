import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

def getNameOfSVID(db, tableName):
    cursor = db.cursor()

    print("Start get SVID full name list")
    query = "SELECT * FROM {tableName}".format(tableName=tableName)

    df = pd.read_sql_query(query, db)
    del df['VID']
    del df['ENUMs']
    del df['DataType']
    del df['Units']
    df = df.set_index(['Index'])
    # print(df)
    sentence = []
    num = 0
    for i in df.index:
        val = df.loc[i, 'FullName']
        val = val.lower()
        sentence.append([])
        list_val = val.split()
        sentence[num].append(list_val)
        num += 1

    return df, sentence

def wordAnalsis(doubleList_nameOfSVID, prcessParameters):
    sentences = doubleList_nameOfSVID
    prcessParametersSentences = []

    i = 0
    while i < len(sentences):
        listSentences = sum(sentences[i], [])
        i += 1
        listWordAnalsis = list(set(prcessParameters).intersection(set(listSentences)))
        if len(listWordAnalsis) == 0:
            listWordAnalsis.append('etc')
        prcessParametersSentences.append(listWordAnalsis)

    adjacency_dict = {i: j for i, j in enumerate(prcessParametersSentences)}

    return adjacency_dict