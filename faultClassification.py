import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.inspection import permutation_importance
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split

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
    print(adjacency_dict)
    return adjacency_dict

def gbm(runNum, df_an, df_normal_labeled, dicToCheckActualVIDs):
    print("run number",runNum, ": start FC")
    df_toAnalyze = df_an.copy()
    df_toAnalyze['pred_labeled'] = -1
    df_normalIntergrated_labeled = df_normal_labeled.copy()
    df = pd.concat([df_toAnalyze, df_normalIntergrated_labeled])
    for value in dicToCheckActualVIDs.values():
        del df[value]
    Y_train = df['pred_labeled']
    Y_test = Y_train
    del df['pred_labeled']
    del df['real_labeled']

    X_train = df
    X_test = X_train


    xgb = XGBClassifier(n_estimators=100, learning_rate= 0.1, max_depth=10, eval_metric='mlogloss')
    xgb.fit(X_train, Y_train)
    # xgb_prd = xgb.predict(X_test)
    #
    # metrics(Y_test, xgb_prd)

    list_feature_column = list(df.columns.values)

    array_list_feature_column = np.array(list_feature_column)
    # #https://hwi-doc.tistory.com/entry/Feature-selection-feature-importance-vs-permutation-importance
    perm_importance = permutation_importance(xgb, X_test, Y_test)
    plt.figure(figsize=(8, 16))
    sorted_idx = perm_importance.importances_mean.argsort()
    # plt.barh(array_list_feature_column[sorted_idx], perm_importance.importances_mean[sorted_idx])
    # plt.xlabel("Process Number: "+str(runNum)+" Permutation Importance")
    # plt.savefig('./image/'+str(runNum)+'savefig_default.png', dpi=200)
    # plt.show()

    df_1 = pd.DataFrame(perm_importance.importances_mean[sorted_idx])
    df_2 = pd.DataFrame(array_list_feature_column[sorted_idx])
    df_importance = pd.concat([df_1, df_2], axis= 1)
    df_importance.columns = ['importance', 'VID']
    df_importance['VID'] = df_importance['VID'].astype(pd.StringDtype())
    df_importance['VID'] = df_importance['VID'].str.replace('VID', '')
    return df_importance

def totalclassification(df_total, dicToCheckActualVIDs):
    df = df_total.copy()
    for value in dicToCheckActualVIDs.values():
        del df[value]


    target = df['real_labeled']
    del df['real_labeled']
    del df['pred_labeled']

    data = df

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.8, shuffle=True, stratify=target,
                                                          random_state=34)

    xgb = XGBClassifier(n_estimators=100, learning_rate= 0.1, max_depth=10, eval_metric='mlogloss')
    xgb.fit(x_train, y_train)


    list_feature_column = list(df.columns.values)

    array_list_feature_column = np.array(list_feature_column)
    # #https://hwi-doc.tistory.com/entry/Feature-selection-feature-importance-vs-permutation-importance
    perm_importance = permutation_importance(xgb, x_test, y_test)
    plt.figure(figsize=(8, 16))
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(array_list_feature_column[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.savefig('./image/savefig_default.png', dpi=200)
    plt.show()
    df_1 = pd.DataFrame(perm_importance.importances_mean[sorted_idx])
    df_2 = pd.DataFrame(array_list_feature_column[sorted_idx])
    df_importance = pd.concat([df_1, df_2], axis= 1)
    df_importance.columns = ['importance', 'VID']
    df_importance['VID'] = df_importance['VID'].astype(pd.StringDtype())
    df_importance['VID'] = df_importance['VID'].str.replace('VID', '')
    return df_importance

def detectRootCause(prcessParameters, df_importance, resultWordAnalsis):
    df = pd.DataFrame(prcessParameters)
    df['importance'] = 0
    df.columns = ['root_cause', 'importance']

    for i in range(len(df_importance)):
        print(df_importance.iloc[i,1])
        print(df_importance.iloc[i,0])
        print(resultWordAnalsis[int(df_importance.iloc[i,1])-1])
        for part in resultWordAnalsis[int(df_importance.iloc[i,1])-1]:
            rowValue = df.loc[df['root_cause'] == part].index[0]
            df.loc[rowValue, 'importance'] += df_importance.iloc[i,0]

    print(df)
