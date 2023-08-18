import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#coding:utf-8
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
plt.rcParams['font.sans-serif']=['SimHei']
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller

#### 1.载入数据
# 载入为dataframe格式,data列为序列值，sdate列为日期

data = pd.read_csv("D:\yttest\Demo\plot.txt", encoding='gb18030')
print(data)
data2 = data[['dt', 'AverageTemperature']]
data3 = data[['AverageTemperature']]
print(data2['dt'][0])
# for i in range(len(data2['dt'])):
#     data2['dt'][i] = datetime.strptime(data2['dt'][i], '%Y-%m-%d')
#data = pd.read_csv("D:/yttest/data2.csv", encoding='gb18030', parse_dates=['dt'], index_col='dt',date_parser=dateparse)

train_results = sm.tsa.arma_order_select_ic(data3, ic=['aic', 'bic'], max_ar=8, max_ma=8)
 
AIC = train_results.aic_min_order  # 6,8
BIC = train_results.bic_min_order # 6,8

path="D:/yttest/Demo/res.txt"
#file = open("D:/yttest/Demo/res.txt",'w')
model=sm.tsa.arima.ARIMA(data3,order=(1,0,2)).fit()
with open(path, 'a') as fpw:
    fpw.write(str(model.forecast(105)))

#model=sm.tsa.arima.ARIMA(data3,order=(6,8,1)).fit()
#file.write(str(model.forecast(105)))
#tempModel.summary2()给出一份模型报告
