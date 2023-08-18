



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


# 改一下
# 改成2013-2025的



df = pd.read_csv('2013_2025.txt')
arr = df.to_numpy()
x = arr[:,0]
y1 = arr[:,1]
plt.figure(figsize=(7, 5), dpi=100)
plt.plot(x, y1, c='red', linestyle='--',label="mean_temperature")
plt.legend(loc='best')
plt.yticks(range(0, 35, 5))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("time_epoch", fontdict={'size': 16})
plt.ylabel("mean_temperature", fontdict={'size': 16})
plt.title("mean_temperature_plot", fontdict={'size': 20})
plt.show()