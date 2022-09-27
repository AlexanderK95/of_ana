import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter
from scipy import signal
from detect_saccades import splice
from detectors import saccade_detection

def shiftToZero(a):
    return ((a+180)%360)-180

def derivative(arrX, arrY):
    out = np.zeros(arrX.size)
    for i in np.arange(1,arrX.size-1):
        out[i] = (arrY[i+1]-arrY[i-1]) / (arrX[i+1]-arrX[i-1])
    return out

def derivative2(t, x):
    dx = np.diff(x)
    dt = np.diff(t)+0.001
    return dx/dt

gaze_df = pd.read_csv("Data/20220926-142032_5_gaze_hr.csv", sep=",")

thresh_vel = 400
thresh_acc = 10000
max_isi = 1

ppdh = 14.58
ppdv = 13.36

t = gaze_df['time']
x = gaze_df['rot.y']
y = gaze_df['rot.x']

# splice(x, y, t, thresh_vel, thresh_acc, True)
# sampleCnt = t.size*10

# x_rs = np.linspace(t.iloc[0], t.iloc[-1], sampleCnt, endpoint=True)

# y1_rs = signal.resample_poly(y1, up=10, down=1, padtype='line')
# y2_rs = signal.resample_poly(y2, up=10, down=1, padtype='line')


ssac, esac, vel, acc = saccade_detection(savgol_filter(x,5,2), savgol_filter(y,5,2), t, maxvel=thresh_vel, maxacc=thresh_acc, minlen=10/1000)
# ssac, esac, vel, acc = saccade_detection(x, y, t, maxvel=thresh_vel, maxacc=thresh_acc, minlen=10/1000)

isi = []
duration = []
amplitude = []
t0 = esac[1][1]
for i in np.arange(1,len(esac)):
    t1 = esac[i][0]
    if t1-t0 < max_isi and t1-t0 > 0:
        isi.append(t1-t0)
    t0 = esac[i][1]
    duration.append(esac[i][2])
    amplitude.append(np.sqrt((esac[i][3]-esac[i][5])**2 + (esac[i][4]-esac[i][6])**2))

dy1 = derivative(t,x)    #velocity
dy2 = derivative(t,y)    #velocity
# dy1 = derivative2(t,x)    #velocity
# dy2 = derivative2(t,x)    #velocity
speed = np.sqrt(dy1**2 + dy2**2)

# speed_filtered = savgol_filter(speed, 4, 2)

# accel_filtered = derivative(x, speed_filtered)
accel = derivative(t,speed)

# peaks = find_peaks(accel, height=threshold)[0]
# peaks_filtered = find_peaks(accel_filtered, height=threshold_filtered)[0]


isi = np.array(isi)*1000
duration = np.array(duration)*1000
amplitude = np.array(amplitude)

plt.figure()
plt.hist(isi, 10, density=True)
plt.title(f"ISI Distribution (#Saccades: {isi.size}, Mean: {isi.mean():0.2f} ms)")

plt.figure()
plt.hist(duration, 10, density=True)
plt.title(f"Duration Distribution (#Saccades: {duration.size}, Mean: {duration.mean():0.2f} ms)")

plt.figure()
plt.hist(amplitude, 10, density=True)
plt.title(f"Amplitude Distribution (#Saccades: {amplitude.size}, Mean: {amplitude.mean():0.2f} °)")


fig, axs = plt.subplots(3,1)
axs[0].plot(t,savgol_filter(x,5,2), color='blue')
axs[0].plot(t,x, color='cyan')
axs[0].plot(t,savgol_filter(y,5,2), color='red')
axs[0].plot(t,y, color='orange')
for sacc in esac:
    axs[0].axvline(sacc[0], color='g')
    axs[0].axvline(sacc[1], color='r')
axs[0].set_ylabel('x and y gaze coordinates [°]')

axs[1].plot(t, speed)
axs[1].plot(t, vel)
for sacc in esac:
    axs[1].axvline(sacc[0], color='g')
    axs[1].axvline(sacc[1], color='r')
axs[1].axhline(thresh_vel)
axs[1].set_ylabel('velocity [°/2]')

axs[2].plot(t, accel)
axs[2].plot(t, acc)
for sacc in esac:
    axs[2].axvline(sacc[0], color='g')
    axs[2].axvline(sacc[1], color='r')
axs[2].axhline(thresh_acc)
axs[2].set_ylabel('acceleration [°/s^2]')

fig.tight_layout()
plt.show()


pass
# print((gaze_df['time'].iloc[-1]-gaze_df['time'].iloc[0])/gaze_df['time'].size)