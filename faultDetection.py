from sklearn.svm import OneClassSVM
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def anomalyDetectionTrain(df, dicToCheckActualVIDs):
    x_train = df
    print("start anomaly train")
    for value in dicToCheckActualVIDs.values():
        del x_train[value]

    svm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.038)
    svm.fit(x_train)
    svm.fit(x_train)
    print("complete svm fit")
    saved_model = pickle.dumps(svm)

    return x_train, saved_model

def anomalyDetectionTest(df, model, dicToCheckActualVIDs):
    clf_from_pickle = pickle.loads(model)
    for value in dicToCheckActualVIDs.values():
        del df[value]

    y_pred = clf_from_pickle.predict(df)
    print("complete svm y_pred")
    df['pred_labeled'] = y_pred

    return df

def anomalyDetectionScore(df):

    dic_pri = {1: 0 , -1 : 0}
    dic_real = to_frequency_table(df['real_labeled'])
    dic_pred = to_frequency_table(df['pred_labeled'])
    print("real: ",{k: dic_pri.get(k, 0) + dic_real.get(k, 0) for k in set(dic_pri) | set(dic_real)})
    print("pred: ",{k: dic_pri.get(k, 0) + dic_pred.get(k, 0) for k in set(dic_pri) | set(dic_pred)})

    y_true = df['real_labeled'].tolist()
    y_pred = df['pred_labeled'].tolist()
    confmat = confusion_matrix(y_true, y_pred, labels= [1, -1])
    print(confmat)
    print(classification_report(y_true, y_pred))
    print_metrics(y_true,y_pred, "분석결과")

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def print_metrics(y, pred_y, title=None):
    print(title)
    print("정확도:", accuracy_score(y, pred_y))
    print('재현율(recall):', recall_score(y, pred_y))
    print('정밀도(precision):', precision_score(y, pred_y))
    print('F1 score:', f1_score(y, pred_y))

def to_frequency_table(data):
    frequencytable = {}
    for key in data:
        if key in frequencytable:
            frequencytable[key] += 1
        else:
            frequencytable[key] = 1
    return frequencytable