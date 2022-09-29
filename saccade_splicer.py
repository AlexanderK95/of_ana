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


### Settings ###
file_name = "20220926-142032_5"
save_folder = "Stimuli"
split_length = 0.3      # length of image sequences in seconds

thresh_vel = 400        # velocity threshold for saccade detection in degrees / s
thresh_acc = 10000      # acceleration threshold for saccade detection degrees / s²
max_isi = 1             # maximum time between saccades in seconds (isi above that will be excludes for the statistics)

shouldPlot = False

### Detecting Saccades ###
gaze_pos_df = pd.read_csv(f"Data/{file_name}_gaze_hr.csv", sep=",")

ppdh = 14.58
ppdv = 13.36

t = gaze_pos_df['time']
x = gaze_pos_df['rot.y']
y = gaze_pos_df['rot.x']


ssac, esac, vel, acc = saccade_detection(savgol_filter(x,5,2), savgol_filter(y,5,2), t, maxvel=thresh_vel, maxacc=thresh_acc, minlen=10/1000)
# ssac, esac, vel, acc = saccade_detection(x, y, t, maxvel=thresh_vel, maxacc=thresh_acc, minlen=10/1000)

isi = []
duration = []
amplitude = []
split_idx = []
t0 = esac[0][1]
for i in np.arange(1,len(esac)):
    t1 = esac[i][0]
    if t1-t0 < max_isi and t1-t0 > 0:
        isi.append(t1-t0)
    if t1-t0 > 1.3*split_length:
        split_idx.append(t0)
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


if shouldPlot:
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



### Splitting Videos in chunks and labeling with heading params ###

body_df = pd.read_csv(f"Data/{file_name}_body.csv", sep=",")
head_df = pd.read_csv(f"Data/{file_name}_head.csv", sep=",")
gaze_df = pd.read_csv(f"Data/{file_name}_gaze.csv", sep=",")

gaze_vid = f"Data/{file_name}_gaze_cam.mp4"
head_vid = f"Data/{file_name}_head_cam.mp4"

cap_head = cv.VideoCapture(cv.samples.findFile(head_vid))
cap_gaze = cv.VideoCapture(cv.samples.findFile(gaze_vid))

FRAME_COUNT_HEAD = int(cap_head.get(cv.CAP_PROP_FRAME_COUNT))
FRAME_COUNT_GAZE = int(cap_head.get(cv.CAP_PROP_FRAME_COUNT))

FRAME_WIDTH = int(cap_head.get(cv.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap_head.get(cv.CAP_PROP_FRAME_HEIGHT))

assert FRAME_COUNT_HEAD == FRAME_COUNT_GAZE, "Number of frames in head and gaze video do not match!"
assert FRAME_COUNT_HEAD == gaze_df.shape[0], "Number of frames does not math the number of data points in the csv files!"

t_vid = gaze_df['time']
split_idx_hr = np.array(split_idx)
split_idx = []
for t in split_idx_hr:
    tmpArr = np.absolute(t_vid-t)
    split_idx.append(tmpArr.argmin())



print((gaze_df['time'].iloc[-1]-gaze_df['time'].iloc[0])/gaze_df['time'].size)
avg_fd = (gaze_df['time'].iloc[-1]-gaze_df['time'].iloc[0])/gaze_df['time'].size
num_frames = np.ceil(split_length/avg_fd)

if shouldPlot:
    plt.figure()
    plt.plot(t)
    plt.plot(np.arange(0,t.size,t.size/t_vid.size),t_vid)
    plt.show()

headingData = pd.DataFrame(columns=["file_head", "file_gaze", "velX", "velY", "VelZ", "pitch", "yaw", "roll"])

saccNr = 0
for i in tqdm(np.arange(int(FRAME_COUNT_HEAD))):
    retHead, frameHead = cap_head.read()
    retGaze, frameGaze = cap_gaze.read()

    start = split_idx[saccNr] - num_frames
    end = split_idx[saccNr]

    if i == start:
        time0 = head_df["time"][i]
        pos0 = np.array([head_df["pos.x"][i], head_df["pos.y"][i], head_df["pos.z"][i]])
        rot0 = np.array([head_df["rot.x"][i], head_df["rot.y"][i], head_df["rot.z"][i]])

        file_head = f"{save_folder}/head_centered/{file_name}_head_{saccNr}.avi"
        file_gaze = f"{save_folder}/retina_centered/{file_name}_gaze_{saccNr}.avi"
        out_head = cv.VideoWriter(file_head, cv.VideoWriter_fourcc('M','J','P','G'), num_frames, (FRAME_WIDTH,FRAME_HEIGHT))
        out_gaze = cv.VideoWriter(file_gaze, cv.VideoWriter_fourcc('M','J','P','G'), num_frames, (FRAME_WIDTH,FRAME_HEIGHT))

        out_head.write(frameHead)
        out_gaze.write(frameGaze)

    if i > start and i < end:
        out_head.write(frameHead)
        out_gaze.write(frameGaze)

    if i == end:
        time1 = head_df["time"][i]
        pos1 = np.array([head_df["pos.x"][i], head_df["pos.y"][i], head_df["pos.z"][i]])
        rot1 = np.array([head_df["rot.x"][i], head_df["rot.y"][i], head_df["rot.z"][i]])
        delta_t = time1 - time0
        vel = (pos1 - pos0) / delta_t
        rot_diff = rot1 - rot0

        new_row = pd.Series({"file_head": file_head, "file_gaze": file_gaze, "velX": vel[0], "velY": vel[1], "VelZ": vel[2], "pitch": rot_diff[0], "yaw": rot_diff[1], "roll": rot_diff[2]})
        headingData = pd.concat([headingData, new_row.to_frame().T], ignore_index=True)

        out_head.release()
        out_gaze.release()
        saccNr += 1
        if saccNr == len(split_idx):
            break

headingData.to_csv(f"{save_folder}/{file_name}.csv")