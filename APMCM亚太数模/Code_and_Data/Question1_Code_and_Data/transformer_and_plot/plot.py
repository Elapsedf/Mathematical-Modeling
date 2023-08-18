


from calendar import month
from copy import deepcopy
from email.header import Header
from lib2to3.pgen2 import driver
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from scipy.interpolate import make_interp_spline



# 趋势图
begin_Date = [1997,3]
def processing_data(Date,months):
    newDate = deepcopy(Date)
    for i in range(0, months):
        # Data -> list[int]
        if(newDate[1] == 12):
            newDate[1] = 1
            newDate[0] += 1
        else:
            newDate[1] += 1
    return newDate

def smooth_xy(lx, ly):
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)

    x=x.flatten()

    y_smooth = make_interp_spline(x, y)(x_smooth)
    y_smooth=y_smooth.flatten()
    return [x_smooth, y_smooth]


df  = pd.read_csv('1997_2025.txt')
# df2 = pd.read_csv('2100.txt')
arr = df.to_numpy()
# arr2 = df.to_numpy()
x_number = df.to_numpy()[:,0]
y = df.to_numpy()[:,1]

# 初始化日期
x = [] 
for i in range(0,len(x_number)):
    str_tmp = processing_data(begin_Date,int(x_number[i])-1)
    x.append(f'{str_tmp[0]}.{str_tmp[1]}')
print()

for i in range(0,len(arr)):
    arr[i][0] = (float(x[i]))
print()
arr = np.array(arr)

x = arr[:,0]
y1 = arr[:,1]
# y2 = arr2[:,1]
plt.figure(figsize=(7, 5), dpi=100)
plt.plot(x, y1, c='orange', linestyle='--',label="1997~2025mean_temperature")
# plt.plot(x, y2, c='chocolate', linestyle=':',label="2100_mean_temperature")
plt.legend(loc='best')
plt.yticks(range(15,30, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("time_epoch", fontdict={'size': 16})
plt.ylabel("mean_temperature", fontdict={'size': 16})
plt.title("mean_temperature_plot", fontdict={'size': 20})
plt.show()