


import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Administrator\Desktop\APMCM\plot\data.txt')

arr = df.to_numpy()

f = open(r'newData.txt','w')
for i in range(0,len(arr)):
    # 对arr[0]作操作
    newStr = ''
    for j in range(0,len(arr[i][0])):
        if str.isdigit(arr[i][0][j]):
            newStr += arr[i][0][j]
    arr[i][0] = newStr
    msg = f'{arr[i][0]},{arr[i][1]},{arr[i][5]},{arr[i][6]}'
    f.write(msg+'\n')
print()