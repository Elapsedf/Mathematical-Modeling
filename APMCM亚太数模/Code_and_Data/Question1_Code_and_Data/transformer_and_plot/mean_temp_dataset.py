


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



df = pd.read_csv('new_pca_data_all.txt')
arr = df.to_numpy()
arr_list = arr.tolist()
year_name = arr[:,0].tolist()
year_name = list(set(year_name))
year_name.sort()
hash_table = dict()

for i in range(0,len(year_name)):
    hash_table[year_name[i]]=[]
for i in range(0,len(arr_list)):
    hash_table[arr_list[i][0]].append(arr_list[i][1])
# 存放结果
f = open(r'C:\Users\Administrator\Desktop\APMCM\plot\all_mean_data.txt','w')
msg = 'date,tem\n'
f.write(msg)
for i in hash_table:
    msg = f'{i},{mean(hash_table[i])}\n'
    f.write(msg)
print()