

from calendar import month
from copy import deepcopy
from email.header import Header
from lib2to3.pgen2 import driver
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA as sklearnPCA


df = pd.read_csv('newData.txt')
arr = df.to_numpy()
arr = arr.tolist()

# 把每一行的时间切割 -- 只留下年份
for i in range(0,len(arr)):
    arr[i][0] = str(arr[i][0])[0:4]
    
print()

year_name = [] # 年份
temp = [] # 温度
temp_tmp = [] # 求平均用
for i in range(0,len(arr)):
    if i > 0 and arr[i][0]!=arr[i-1][0]:
        # 此时到了新的一年
        year_name.append(arr[i-1][0])
        temp.append(mean(temp_tmp))
        temp_tmp = [] # 置空
    temp_tmp.append(arr[i][1])

f = open('new_pca_data_all.txt','w')
msg = 'year,AverageTemperature'
f.write(msg+'\n')
for i in range(0,len(year_name)):
    msg = f'{year_name[i]},{temp[i]}'
    f.write(msg+'\n'),
print()