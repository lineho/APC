from sklearn.svm import OneClassSVM
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from scipy import interpolate
import matplotlib.pyplot as plt

def anomalyDetectionTrain(df, dicToCheckActualVIDs):
    x_train = df.copy()
    print("start anomaly train")
    for value in dicToCheckActualVIDs.values():
        del x_train[value]

    svm = OneClassSVM(kernel='rbf', gamma=10000, nu=0.0068)

    svm.fit(x_train)
    #print("complete svm fit")

    with open('trainSvm.pkl', 'wb') as f:
        pickle.dump(svm, f, protocol=pickle.HIGHEST_PROTOCOL)

    return df

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

def anomalyDetectionScore(df, isAll):


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
    if isAll == 1:
        print_score(y_pred, y_true) #다훈 추가해달라는 부분

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def print_score(y_pred, y_test):

    clf_performance = confusion_matrix(y_pred, y_test)
    plot_confusion_matrix(clf_performance, classes=['1', '-1'], normalize=True)
    plt.figure()
    plt.show()