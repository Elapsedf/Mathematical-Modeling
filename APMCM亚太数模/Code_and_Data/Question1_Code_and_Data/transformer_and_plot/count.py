


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


df = pd.read_csv(r'C:\Users\Administrator\Desktop\APMCM\plot\v.txt')
arr = df.to_numpy().tolist()

hash_table = {}
for i in range(len(arr)-1,-1,-1):
    if arr[i][0] in hash_table:
        hash_table[arr[i][0]]+=1
    else:
        hash_table[arr[i][0]]=1

f = open(r'C:\Users\Administrator\Desktop\APMCM\plot\v_data.txt','w')
msg = 'year\n'
f.write(msg)
for i in hash_table:
    msg = f'{i},{hash_table[i]}\n'
    f.write(msg)
print()