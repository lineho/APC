import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

def randomForest():
    print("start random forest")
    #테스트 데이터란 정상 Full 비정상 10개만. (10초안에 받았다고 판정.)
    df = pd.read_excel('xlsx/test.xlsx', header=0)
    df.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    del df['time']
    y = df['label']
    df.drop(['label'], axis=1, inplace =True)
    X =df
    Xs = X
    forest = RandomForestClassifier(n_estimators=500)
    forest.fit(Xs, y)
    # Prediction
    y_pred = forest.predict(Xs)
    print(y_pred)
    print(y)
    # Verifying accuracy check
    print('accuracy :', metrics.accuracy_score(y, y_pred))
    ftr_importances_values = forest.feature_importances_
    ftr_importances_10 = pd.Series(ftr_importances_values, index=X.columns)
    ftr_importances = pd.DataFrame(ftr_importances_values, index = X.columns)
    print(ftr_importances)
    ftr_top10 = ftr_importances_10.sort_values(ascending=False)[:10]
    #plt.figure(figsize=(13,10))
    plt.title('Top 10 Feature Importances')
    plt.ylabel("SVID_Num")
    plt.xlabel("importanc rate")
    sns.barplot(x=ftr_top10, y=ftr_top10.index)
    plt.show()
    ftr_importances.to_excel('xlsx/ftr_importances.xlsx')


def removeNoise():
    df = pd.read_csv('xlsx/labeling_x_train.csv', header=0)
    df_label = df['label']
    df = df.set_index(['time'])
    list_df = list(df)
    print(list_df)
    del df['label']
    df = df.rolling(200).mean()
    df = pd.DataFrame(df)
    df = pd.concat([df, df_label], axis=1)
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)

    df.columns = list_df
    print(df)
    df.to_excel('xlsx/removeNoise_labeling_x_train.xlsx')
    df.to_csv('xlsx/removeNoise_labeling_x_train.csv')

def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    F1 = f1_score(y_test, y_pred, average='micro')
    # AUC = roc_auc_score(y_test, y_pred, multi_class="ovr",average='micro')

    #print('confusion:\n', confusion)
    print('\naccuracy: {:.4f}'.format(accuracy))
    print('precision: {:.4f}'.format(precision))
    print('recall: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    # print('AUC: {:.4f}'.format(AUC))

def synthesizeDataInNormalEveryTenSeconds(numberofManipulation):
    print("synthesize Data In Normal Every Ten Seconds")
    df_normal = pd.read_csv('xlsx/after_preprocessing_train_data.xlsx', header=0)
    print(df_normal)
    #새로운 것 불러오기
    #열 맞춰서 제거.
    #노말통 + 앱노말 10초씩 가져오기.
    #이것을 randomforest 돌리기. (랜덤포레스트 돌릴때 변수를 주어야 저장될때 알게 저장될 듯.)
    # 랜덤포레스트 돌린거 따로 FC 폴더에 집어넣어야될 듯

def Cross_validation():
    print("Cross_validation")
    #교차검증하는 방법 생각.
    #최종적으로 무엇이 문제인지도 표출해야함.

