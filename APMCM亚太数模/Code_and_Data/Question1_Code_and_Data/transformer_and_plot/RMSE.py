



from calendar import month
from copy import deepcopy
from email.header import Header
import imp
from lib2to3.pgen2 import driver
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA as sklearnPCA
import torch.nn as nn
import torch


df = pd.read_csv(r'C:\Users\Administrator\Desktop\APMCM\plot\RMSE1.txt')
df2 = pd.read_csv(r'C:\Users\Administrator\Desktop\APMCM\plot\RMSE2.txt')
arr1 = torch.tensor(df.to_numpy())
arr2 = torch.tensor(df2.to_numpy())

criterion = nn.MSELoss()

loss = criterion(arr1,arr2)
print()