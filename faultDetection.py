import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import seaborn as sns



# MFC 고장이나 MFC 값을 알수 없는 상황으로 가정함.
def anomalytest():
    print("start anomaly train")
    x_train = pd.read_excel('after_preprocessing_train_data.xlsx', header=0)
    x_train = x_train.set_index(['time'])
    del x_train['VID434']

    x_test = pd.read_excel('after_preprocessing_test_data.xlsx', header=0)
    x_test = x_test.set_index(['time'])
    del x_test['VID434']

    y_actual = pd.read_excel('dfTestData_processing_extractTime_nonumericalDummy.xlsx', header=0)
    y_set = pd.read_excel('raw_dfTestData.xlsx', header=0)
    y_set = y_set['VID433']
    y_actual = y_actual['VID434']
    y_plot = pd.concat([y_set, y_actual], axis=1)

    svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.03)
    svm.fit(x_train)
    print("complete svm fit")
    y_pred = svm.predict(x_test)
    print("complete svm y_pred")
    print(y_pred)
    x_test['label'] = y_pred
    #get_clf_eval(y_test, y_pred)

    y_plot = pd.concat([y_plot, x_test['label']], axis=1)
    print(y_plot)
    y_plot.to_excel('y_plot.xlsx')

    x_train['label'] = 1
    x_train.to_excel('labeling_x_train.xlsx')
    x_train.to_csv('labeling_x_train.csv')

    x_test.to_excel('labeling_x_test.xlsx')
    x_test.to_csv('labeling_x_test.csv')



def drawplot():


    print("draw fault detection result")
    df = pd.read_excel('y_plot.xlsx', header=0)
    df.rename(columns={"Unnamed: 0": "time"}, inplace=True)

    #176인데 1인비율 176아닌데 -1인 비율 구하기
    print(len(df))
    print(df[(df['VID433'] == 176)].count())
    print(df[(df['VID433'] != 176)].count())
    print(len(df[(df['VID433'] == 176) &(df['label'] == 1)]))
    print(len(df[(df['VID433'] == 176) &(df['label'] != 1)]))
    print(len(df[(df['VID433'] != 176) &(df['label'] == 1)]))
    print(len(df[(df['VID433'] != 176) &(df['label'] != 1)]))


    idx_nm_3 = df[(df['VID433'] <= 150) | (df['VID434'] <= 150)  | (df['VID434'] >= 200)].index
    df = df.drop(idx_nm_3)

    sns.jointplot(x=df['VID433'], y=df['VID434'], data= df)
    sns.pairplot(df, hue='label', markers=["o", "s"])
    plt.suptitle("yyyy", y=1.02)
    plt.show()

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


