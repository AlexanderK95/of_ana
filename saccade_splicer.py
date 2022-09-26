import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

def shiftToZero(a):
    return ((a+180)%360)-180

def derivative(arrX, arrY):
    out = np.zeros(arrX.size)
    for i in np.arange(1,arrX.size-1):
        out[i] = (arrY[i+1]-arrY[i-1]) / (arrX[i+1]-arrX[i-1])
    return out

gaze_df = pd.read_csv("Data/20220926-142032_2_gaze_hr.csv", sep=",")
speed = np.sqrt(gaze_df['rot.x'] ** 2 + gaze_df['rot.z'] ** 2)
# plt.plot(gaze_df['time'], speed)
# plt.plot(gaze_df['time'], gaze_df['rot.y'])
# plt.plot(gaze_df['time'], gaze_df['rot.z'])
# plt.plot(gaze_df['time'], gaze_df['rot.x'].map(shiftToZero))
# plt.plot(gaze_df['time'], gaze_df['rot.y'].map(shiftToZero))
# plt.plot(gaze_df['time'], gaze_df['rot.z'].map(shiftToZero))
# plt.show()

x = gaze_df['time'][0:500]
y1 = gaze_df['rot.x'][0:500]
y2 = gaze_df['rot.y'][0:500]
dy1 = derivative(x,y1)    #velocity
dy2 = derivative(x,y2)    #velocity
speed = np.sqrt(dy1**2 + dy2**2)
accel = derivative(x, speed)
# ddy = derivative(x,dy)  #acceleration
# dddy = derivative(x,ddy) #jerk
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,speed/30)
plt.plot(x,accel/3000)
# plt.plot(x,dy/10)
# plt.plot(x,ddy/1000)
# plt.plot(x,dddy/10000)
plt.show()

# print((gaze_df['time'].iloc[-1]-gaze_df['time'].iloc[0])/gaze_df['time'].size)