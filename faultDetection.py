from sklearn.svm import OneClassSVM
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import shap

def anomalyDetectionTrain(df, dicToCheckActualVIDs):
    x_train = df
    print("start anomaly train")
    for value in dicToCheckActualVIDs.values():
        del x_train[value]

    svm = OneClassSVM(kernel='rbf', gamma=100, nu=0.0068)

    svm.fit(x_train)
    svm.fit(x_train)
    print("complete svm fit")

    with open('trainSvm.pkl', 'wb') as f:
        pickle.dump(svm, f, protocol=pickle.HIGHEST_PROTOCOL)

    return x_train

def anomalyDetectionTest(df, dicToCheckActualVIDs):
    with open('trainSvm.pkl', 'rb') as f:
        clf_from_pickle = pickle.load(f)
    dfCopy = df.copy()
    for value in dicToCheckActualVIDs.values():
        del dfCopy[value]

    y_pred = clf_from_pickle.predict(dfCopy)
    print("complete svm y_pred")
    df['pred_labeled'] = y_pred

    return dfCopy

def anomalyDetectionScore(df):

    dic_pri = {1: 0 , -1 : 0}
    dic_real = to_frequency_table(df['real_labeled'])
    dic_pred = to_frequency_table(df['pred_labeled'])
    dic_real = {k: dic_pri.get(k, 0) + dic_real.get(k, 0) for k in set(dic_pri) | set(dic_real)}
    dic_pred = {k: dic_pri.get(k, 0) + dic_pred.get(k, 0) for k in set(dic_pri) | set(dic_pred)}
    print("real: ", dic_real)
    print("pred: ", dic_pred)

    y_true = df['real_labeled'].tolist()
    y_pred = df['pred_labeled'].tolist()
    confmat = confusion_matrix(y_true, y_pred, labels= [1, -1])
    print(confmat)
    print(classification_report(y_true, y_pred))
    print_metrics(y_true,y_pred, "Analysis result")
    abnormalDataRate = dic_pred[-1] / (dic_pred[1] + dic_pred[-1]) * 100
    return abnormalDataRate

def print_metrics(y, pred_y, title=None):
    print(title)
    print("Accuracy:", accuracy_score(y, pred_y))
    print('Recall:', recall_score(y, pred_y))
    print('Precision:', precision_score(y, pred_y))
    print('F1 score:', f1_score(y, pred_y))

def to_frequency_table(data):
    frequencytable = {}
    for key in data:
        if key in frequencytable:
            frequencytable[key] += 1
        else:
            frequencytable[key] = 1
    return frequencytable