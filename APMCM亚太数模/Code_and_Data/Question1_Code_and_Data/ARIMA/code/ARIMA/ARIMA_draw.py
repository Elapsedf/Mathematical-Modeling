import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
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
plt.rcParams['axes.unicode_minus']=False # 解决负号不显示问题
data = pd.read_csv("D:/yttest/data2.csv", encoding='gb18030')
data2 = data[['dt', 'AverageTemperature']]
data3 = data[['AverageTemperature']]

model=sm.tsa.arima.ARIMA(data3,order=(5,2,2)).fit()  # 参数 1,0 append 再看参数的
resid = model.resid #赋值

import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
 
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()
 
jieci1ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid.values.squeeze(), lags=40, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
plt.show()