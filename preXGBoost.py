from sklearn.model_selection import train_test_split
import xgboost as xgb ## XGBoost 불러오기
from xgboost import plot_importance ## Feature Importance를 불러오기 위함
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

def runXGBoost(x, y, num):
    #print("Start XGBoost: " + str(num))
    #tx = tx.astype('float', copy=False)
    #ty = ty.astype('float', copy= False)

    x = x.astype('float', copy=False)
    y = y.astype('float', copy= False)

    #x_train, y_train = tx, ty
    #x_test, y_test = x, y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    #print("x_train: ", len(x_train))
    #print("x_test: ", len(x_test))
    #print("y_train: ", len(y_train))
    #print("y_test: ", len(y_test))

    xgb_wrapper = XGBClassifier(n_estimators = 400, learning_rate = 0.1 , max_depth = 3, eval_metric="mlogloss")
    xgb_wrapper.fit(x_train, y_train)
    y_pred = xgb_wrapper.predict(x_test)
    get_clf_eval(y_test, y_pred)
    print(y_pred)
    print(y_test)
    #끊는 선 찾는 방법.
    # evals = [(x_test, y_test)]
    # xgb_wrapper.fit(x_train, y_train, early_stopping_rounds=100, eval_metric="mlogloss", eval_set=evals, verbose=True)
    # ws100_preds = xgb_wrapper.predict(x_test)
    # get_clf_eval(y_test, ws100_preds)
    # fig, ax = plt.subplots(figsize=(10, 12))
    # plot_importance(xgb_wrapper, ax=ax)
    # plt.show()

    # Verifying accuracy check
    ftr_importances_values = xgb_wrapper.feature_importances_
    ftr_importances_10 = pd.Series(ftr_importances_values, index=x_train.columns)
    ftr_importances = pd.DataFrame(ftr_importances_values, index = x_train.columns)
    #print(ftr_importances)
    ftr_top10 = ftr_importances_10.sort_values(ascending=False)[:10]

    #plt.figure(figsize=(13,10))
    plt.title('Top 10 Feature Importances')
    plt.ylabel("SVID_Num")
    plt.xlabel("importanc rate")
    sns.barplot(x=ftr_top10, y=ftr_top10.index)
    plt.show()

    return ftr_importances

def severalTimesrunXGBoost(x, y, num):
    num = num+1
    for i in range(1, num):
        if i == 1:
            df1 = runXGBoost(x, y, i)
            result = df1

        elif i == num-1:
            df2 = runXGBoost(x, y, i)
            result = pd.concat([result, df2], axis=1)
            print(result)
            # Excel Fileization
            result.to_excel('result.xlsx')

        else:
            df2 =  runXGBoost(x, y, i)
            result = pd.concat([result, df2], axis=1)


# 평가지표 출력하는 함수 설정
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