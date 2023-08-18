
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv(r'C:\Users\Administrator\Desktop\APMCM\plot\plot_trans.txt')

cap = cv.VideoCapture(0)
cpframe = df
while True:
    ret, frame = cap.read()
    cv.imshow("video", frame)
    if cv.waitKey(1) & 0xFF ==ord('q'):
        cpframe = frame
        break
 
img_gray = cv.cvtColor(cpframe, cv.COLOR_RGB2GRAY)
 
Y = np.arange(0, np.shape(img_gray)[0], 1)
X = np.arange(0, np.shape(img_gray)[1], 1)
X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, img_gray, cmap=cm.gist_rainbow)
plt.show()