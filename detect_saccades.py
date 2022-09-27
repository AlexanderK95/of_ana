import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter
from scipy import signal



def derivative(arrX, arrY):
    out = np.zeros(arrX.size)
    for i in np.arange(1,arrX.size-1):
        out[i] = (arrY[i+1]-arrY[i-1]) / (arrX[i+1]-arrX[i-1])
    return out

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def splice(x, y, t, thresh_vel, thresh_acc, plot=False):
    x_ma = x.rolling(3).mean()
    x_ma = x.rolling(3).mean()
    y_ma = y.rolling(3).mean()
    y_ma = y.rolling(3).mean()

    dx = derivative(t, x_ma)
    dy = derivative(t, y_ma)

    speed = np.sqrt(dx**2 + dy**2)
    speed_ma = moving_average(speed, 3)

    accel = derivative(t, speed_ma)

    if plot:
        fig, axs = plt.subplots(3,1)
        axs[0].plot(t, x, c='cyan')
        axs[0].plot(t, x_ma, c='magenta')
        axs[0].plot(t, y, c='blue')
        axs[0].plot(t, y_ma, c='red')

        axs[1].plot(t, speed, c='cyan')
        axs[1].plot(t, speed_ma, c='magenta')

        axs[2].plot(t, accel)

        plt.show()