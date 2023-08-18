from utils import fix_pythonpath_if_working_locally

fix_pythonpath_if_working_locally()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
from darts.metrics import mape, mse, mae
from darts import TimeSeries
from darts.models import TCNModel, RNNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, r2_score
from darts.utils.missing_values import fill_missing_values
from darts.datasets import AirPassengersDataset, SunspotsDataset, EnergyDataset
from sklearn.metrics import mean_squared_error
import xlrd
import csv
import os
import xlsxwriter
from xlutils.copy import copy
import openpyxl
from openpyxl.utils import get_column_letter,column_index_from_string
from openpyxl import Workbook

# Read data:


# We'll use the month as a covariate
""" month_series = datetime_attribute_timeseries(ts, attribute="month", one_hot=True)
scaler_month = Scaler()
month_series = scaler_month.fit_transform(month_series)
 """
train_path="/home/jack/Project/TCN_new/data_14-10/"
test_path="/home/jack/Project/TCN_new/data_25-62/25to62/"

# 读取Excel文件中的特定工作表
def get_data(file_path):
    df = pd.read_excel(train_path+file_path,index_col="timestamp")
    df.index = pd.to_datetime(df.index) 
    ts=df['value']
    ts=TimeSeries.from_series(ts)
    scaler=Scaler() 
    ts=scaler.fit_transform(ts)
    return ts, scaler
def cor_data(file_path):
    df = pd.read_excel(train_path+file_path,index_col="timestamp")
    df.index = pd.to_datetime(df.index) 
    ts=df['value']
    ts=TimeSeries.from_series(ts)
    return ts

ts,ts_scaler=get_data("14-10.xlsx")
cor1=cor_data("10-all.xlsx")
cor2=cor_data("14-all.xlsx")
cor3=cor_data("all-14.xlsx")
cor4=cor_data("all-10.xlsx")
cor_all=cor1.stack(cor2)
cor_all=cor_all.stack(cor3)
cor_all=cor_all.stack(cor4)
cor_scaler=Scaler()
cor_all=cor_scaler.fit_transform(cor_all)
train,test_val=ts.split_before(pd.Timestamp("20220910"))
train_cor,test_cor=cor_all.split_before(pd.Timestamp("20220910"))
""" train_14_o=get_data("train","14-other_train.xlsx")
train_o_10=get_data("train","other-10_train.xlsx")
train_o_14=get_data("train","other-14_train.xlsx") """

""" 
test_14_10=get_data("test","14-10_test.xlsx")
test_10_o=get_data("test","10-other_test.xlsx") """
""" test_14_o=get_data("test","14-other_test.xlsx")
test_o_14=get_data("test","other-14_test.xlsx")
test_o_10=get_data("test","other-10_test.xlsx") """


""" train_multivariate = merge_time_series([ts, train_10_o, ])
test_multivariate = merge_time_series([test_14_10,tst_o_14, test_10_o,test_14_o,test_o_10]) """
# Create training and validation sets:
""" train, val = ts.split_after(pd.Timestamp("19580801"))
train_month, val_month = month_series.split_after(pd.Timestamp("19580801")) """
model_air = TCNModel(
    input_chunk_length=35,
    output_chunk_length=31,
    n_epochs=500,
    dropout=0.1,
    dilation_base=2,
    weight_norm=True,
    kernel_size=5,
    num_filters=3,
    random_state=0,
)
model_air.fit(
    series=train,
    past_covariates=train_cor,
    val_series=test_val,
    val_past_covariates=test_cor,
    verbose=True,
)
backtest = model_air.historical_forecasts(
    series=ts,
    past_covariates=cor_all,
    start=0.7,
    forecast_horizon=31, 
    retrain=False,
    verbose=True,
)
pred=model_air.predict(series=ts,past_covariates=cor_all,n=30)
Mse=mse(ts,backtest)
Rmse=math.sqrt(Mse)
Mae=mae(ts,backtest)
#Mape=mape(ts,backtest)
print("Mse: ", Mse)
print("Rmse: ", Rmse)
print("Mae: ", Mae)
#print("Mape: ", Mape)

ts.plot(label="actual")
backtest.plot(label="backtest")
pred.plot(label="pred(H=30days)")
plt.legend()
fig = plt.figure(figsize=(10, 8))
plt.savefig('/home/jack/Project/TCN_new/25-62归一化.png')


ts=ts_scaler.inverse_transform(ts)
pred=ts_scaler.inverse_transform(pred)
backtest=ts_scaler.inverse_transform(backtest)
""" ts.plot(label="actual")
backtest.plot(label="backtest")
pred.plot(label="pred(H=31days)")
plt.savefig('/home/jack/Project/TCN_new/14-10实际运载量.png')
 """
""" ts.plot(label="actual")
backtest.plot(label="backtest")
pred.plot(label="pred(H=30days)")
plt.legend()
plt.savefig('/home/jack/Project/TCN_new/14-10归一化.png')
 """
# plot unnormalized data
""" ts_unscaled = ts_scaler.inverse_transform(ts)
ts_unscaled=np.array(ts_unscaled.values).reshape(-1,1).flatten()
pred_unscaled = ts_scaler.inverse_transform(pred)
pred_unscaled = np.array(pred_unscaled.values).reshape(-1,1).flatten()
backtest_unscaled = ts_scaler.inverse_transform(backtest)
backtest_unscaled=np.array(backtest_unscaled.values).reshape(-1,1).flatten() """

#放弃，输出成数在另外一个数据画！
""" fig, ax = plt.subplots()
ax.plot(ts_unscaled, label="actual")
ax.plot(backtest_unscaled, label="backtest")
ax.plot(pred_unscaled, label="pred(H=31days)")
ax.legend()
plt.savefig('/home/jack/Project/TCN_new/14-10实际运载量.png') """
pred=pred.pd_dataframe()
ts=ts.pd_dataframe()
backtest=backtest.pd_dataframe()
p_values=pred['value']
p_dates=pred.index
t_values=ts['value']
t_dates=ts.index
b_values=backtest['value']
b_dates=backtest.index
with open("/home/jack/Project/TCN_new/data_25-62/25to62/pred.txt", 'a+') as fpa:
    len=pred.__len__()
    for i in range(len):
        fpa.write(str(p_dates[i])+"\t"+str(p_values[i])+"\n")
with open("/home/jack/Project/TCN_new/data_25-62/25to62/ts.txt", 'a+') as fpa:
    len=ts.__len__()
    for i in range(len):
        fpa.write(str(t_dates[i])+"\t"+str(t_values[i])+"\n")
with open("/home/jack/Project/TCN_new/data_25-62/25to62/back.txt", 'a+') as fpa:
    len=backtest.__len__()
    for i in range(len):
        fpa.write(str(b_dates[i])+"\t"+str(b_values[i])+"\n")
