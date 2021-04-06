import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import kmeans1d
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def readXlsx():
    df = pd.read_excel('result.xlsx', header=0)
    df.rename(columns={"Unnamed: 0": "SVID_Num"}, inplace=True)
    df['SVID_Num'] = df['SVID_Num'].str.split("VID").str[1]
    df = df.set_index(['SVID_Num'])
    #print(df)
    df = df[df != 0]
    df= df.dropna(axis=0)
    df.columns = np.arange(len(df.columns))+1

    df.to_excel('result1.xlsx')

    df.iloc[:,].plot(style=".")
    plt.ylabel("importanc rate")
    plt.legend(title='Number of\n executions')
    plt.show()

def OneDimensionalkMeans(num):
    print("kMeans")
    df = pd.read_excel('result1.xlsx', header=0)
    list_of_index = [item for item in df.iloc[:, 0]]
    #print(list_of_index)

    variable_ea = num
    for i in range(1, variable_ea + 1):
        globals()['Var_{}'.format(i)] = {}

    for i in range(1, num+1):

        df1 = df.iloc[:, i:i + 1]
        x = df1.values.reshape(-1, ).tolist()
        k = 2

        clusters, centroids = kmeans1d.cluster(x, k)
        #print(clusters)
        print("Count Key SVID:",clusters.count(1))
        #print(centroids)
        dic = dict(zip(list_of_index, clusters))

        for key, value in dic.items():
            if value != 0:
                globals()['Var_{}'.format(i)][key] = value

        # keySVID_List = list(dic1.keys())
        # print(keySVID_List)

    total_SVID_count = Counter(globals()['Var_{}'.format(1)])

    for i in range(2, num+1):
        total_SVID_count += Counter(globals()['Var_{}'.format(i)])

    total_SVID_count = dict(total_SVID_count)
    #print(total_SVID_count)

    select_SVID = {}
    for key, value in total_SVID_count.items():
        if value == num:
            select_SVID[key] = value
    listSelectSVID = list(select_SVID)
    print("Last selected key svid count", len(listSelectSVID))
    print(listSelectSVID)
