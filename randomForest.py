from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import permutation_importance
from sklearn.inspection import permutation_importance
import numpy as np

from sklearn.preprocessing import LabelEncoder
import mglearn

def randomForest(x, y, num):
    print("Start randomForest: "+str(num))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print("x_train: ", len(x_train))
    print("x_test: ", len(x_test))
    print("y_train: ", len(y_train))
    print("y_test: ", len(y_test))


    # Learning Progress
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(x_train, y_train)

    # Prediction
    y_pred = forest.predict(x_test)
    print(y_pred)
    print(y_test)

    # Verifying accuracy check
    print('accuracy :', metrics.accuracy_score(y_test, y_pred))
    num = num-1
    ftr_importances_values = forest.feature_importances_
    ftr_importances_20 = pd.Series(ftr_importances_values, index=x_train.columns)
    ftr_importances = pd.DataFrame(ftr_importances_values, index = x_train.columns)
    print(ftr_importances)
    ftr_top20 = ftr_importances_20.sort_values(ascending=False)[:20]

    plt.figure(figsize=(13,10))
    plt.title('Top 20 Feature Importances')
    sns.barplot(x=ftr_top20, y=ftr_top20.index)
    plt.show()

    # permutation importance (feat. Hyukjun)
    #ValueError: Image size of 3000x91000 pixels is too large. It must be less than 2^16 in each direction.
    # list_feature_column = list(x.columns.values)
    # array_list_feature_column = np.array(list_feature_column)
    # perm_importance = permutation_importance(forest, x_test, y_test)
    # fig = plt.figure(figsize=(30, 30))
    # sorted_idx = perm_importance.importances_mean.argsort()
    # print(array_list_feature_column[sorted_idx])
    # print(perm_importance.importances_mean[sorted_idx])
    # plt.barh(array_list_feature_column[sorted_idx], perm_importance.importances_mean[sorted_idx])
    # plt.xlabel("Permutation Importance")
    # plt.show()

    return ftr_importances

def severalTimesRandomForest(x, y, num):
    num = num+1
    for i in range(1, num):
        if i == 1:
            df1 = randomForest(x, y, i)
            result = df1

        elif i == num-1:
            df2 = randomForest(x, y, i)
            result = pd.concat([result, df2], axis=1)
            print(result)
            # Excel Fileization
            result.to_excel('result.xlsx')

        else:
            df2 = randomForest(x, y, i)
            result = pd.concat([result, df2], axis=1)
