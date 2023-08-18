

from calendar import month
from copy import deepcopy
from email.header import Header
from lib2to3.pgen2 import driver
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.datasets import load_iris
import pandas as pd
import pandas as pd
import numpy as np
# 绘图
import seaborn as sns
import matplotlib.pyplot as plt


## 使用机器学习的内置函数计算，使用鸢尾花数据
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if False:
    #加载数据
    iris = load_iris()
    x = iris.keys()
    data_x = iris.data
    data_y = iris.target
    #print(data_y,data_x)
    print()

    #PCA方法训练并求取新维度的数据
    pca = PCA(n_components=2)
    pca = pca.fit(data_x)
    x_dr = pca.transform(data_x)

    #图形化显示
    plt.scatter(x_dr[data_y==0,0],x_dr[data_y==0,1],c='red',label=iris.target_names[0])
    plt.scatter(x_dr[data_y==1,0],x_dr[data_y==1,1],c='green',label=iris.target_names[1])
    plt.scatter(x_dr[data_y==2,0],x_dr[data_y==2,1],c='blue',label=iris.target_names[2])
    plt.legend
    plt.title("iris dataset")
    plt.show()
    print()
else:
    # 完整代码：
    # 数据处理
    df = pd.read_csv(r"pca_final_data.txt", encoding='gbk', index_col=0).reset_index(drop=True)
    # print(df)
    # Bartlett's球状检验
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    print(chi_square_value, p_value)
    # KMO检验
    # 检查变量间的相关性和偏相关性，取值在0-1之间；KOM统计量越接近1，变量间的相关性越强，偏相关性越弱，因⼦分析的效果越好。# 通常取值从0.6开始进⾏因⼦分析
    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo_model = calculate_kmo(df)
    print(kmo_all)
    # #标准化
    # #所需库
    # from sklearn import preprocessing
    # #进⾏标准化
    # df = preprocessing.scale(df)
    # print(df)
    # #求解系数相关矩阵
    # covX = np.around(np.corrcoef(df.T),decimals=3)
    # print(covX)
    # #求解特征值和特征向量
    # featValue, featVec=  np.linalg.eig(covX.T)  #求解系数相关矩阵的特征值和特征向量
    # print(featValue, featVec)
    #不标准化
    #均值
    def meanX(dataX):
        return np.mean(dataX,axis=0)#axis=0表⽰依照列来求均值。假设输⼊list,则axis=1
    average = meanX(df)
    print(average)
    #查看列数和⾏数
    m, n = np.shape(df)
    print(m,n)
    #均值矩阵
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    print(avgs)
    #去中⼼化
    data_adjust = df - avgs
    print(data_adjust)
    #协⽅差阵
    covX = np.cov(data_adjust.T)   #计算协⽅差矩阵
    print(covX)
    #计算协⽅差阵的特征值和特征向量
    featValue, featVec = np.linalg.eig(covX)  #求解协⽅差矩阵的特征值和特征向量print(featValue, featVec)
    #### 下⾯没有区分 #######
    #对特征值进⾏排序并输出降序
    featValue = sorted(featValue)[::-1]
    print(featValue)
    #绘制散点图和折线图
    # 同样的数据绘制散点图和折线图
    plt.scatter(range(1, df.shape[1] + 1), featValue)
    plt.plot(range(1, df.shape[1] + 1), featValue)
    # 显⽰图的标题和xy轴的名字
    # 最好使⽤英⽂，中⽂可能乱码
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()  # 显⽰⽹格
    plt.show()  # 显⽰图形
    #求特征值的贡献度
    gx = featValue/np.sum(featValue)
    print(gx)
    #求特征值的累计贡献度
    lg = np.cumsum(gx)
    print(lg)
    #选出主成分
    k=[i for i in range(len(lg)) if lg[i]<0.97]
    k = list(k)
    print(k)
    #选出主成分对应的特征向量矩阵
    selectVec = np.matrix(featVec.T[k]).T
    selectVe=selectVec*(-1)
    print(selectVec)
    #主成分得分
    finalData = np.dot(data_adjust,selectVec)
    print(finalData)
    #绘制热⼒图
    plt.figure(figsize = (7,5))
    ax = sns.heatmap(selectVec, annot=True, cmap="BuPu")
    # 设置y轴字体⼤⼩
    ax.yaxis.set_tick_params(labelsize=15)
    plt.title("Factor Analysis", fontsize="xx-large")
    # 设置y轴标签
    plt.ylabel("Sepal Width", fontsize="xx-large")
    # 显⽰图⽚
    plt.show()
    # 保存图⽚
    # plt.savefig("factorAnalysis", dpi=500)
    print()