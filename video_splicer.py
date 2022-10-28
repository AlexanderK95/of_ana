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
file_name = "20221005-170121_2"
save_folder = "Stimuli_ws"
split_length = 0.3      # length of image sequences in seconds

numFrames = 8

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
split_idx = np.arange(numFrames*3, FRAME_COUNT_HEAD - (FRAME_COUNT_HEAD%numFrames), numFrames+1)

print((gaze_df['time'].iloc[-1]-gaze_df['time'].iloc[0])/gaze_df['time'].size)
avg_fd = (gaze_df['time'].iloc[-1]-gaze_df['time'].iloc[0])/gaze_df['time'].size
num_frames = np.ceil(split_length/avg_fd)

headingData = pd.DataFrame(columns=["file_head", "file_gaze", "velX", "velY", "velZ", "pitch", "yaw", "roll"])

spliceNr = 0

num_frames = numFrames

for i in tqdm(np.arange(int(FRAME_COUNT_HEAD))):
    retHead, frameHead = cap_head.read()
    retGaze, frameGaze = cap_gaze.read()

    start = split_idx[spliceNr] - num_frames
    end = split_idx[spliceNr]

    # print([start, end])

    if i == start:
        time0 = head_df["time"][i]
        pos0 = np.array([head_df["pos.x"][i], head_df["pos.y"][i], head_df["pos.z"][i]])
        rot0 = np.array([head_df["rot.x"][i], head_df["rot.y"][i], head_df["rot.z"][i]])

        px = np.array([])
        py = np.array([])
        pz = np.array([])
        pitch = np.array([])
        yaw = np.array([])
        roll= np.array([])

        px = np.append(px, pos0[0])
        py = np.append(py, pos0[1])
        pz = np.append(pz, pos0[2])
        pitch = np.append(pitch, rot0[0])
        yaw = np.append(yaw, rot0[1])
        roll = np.append(roll, rot0[2])

        file_head = f"{save_folder}/head_centered/{file_name}_head_{spliceNr}.avi"
        file_gaze = f"{save_folder}/retina_centered/{file_name}_gaze_{spliceNr}.avi"
        out_head = cv.VideoWriter(file_head, cv.VideoWriter_fourcc('M','J','P','G'), num_frames, (FRAME_WIDTH,FRAME_HEIGHT))
        out_gaze = cv.VideoWriter(file_gaze, cv.VideoWriter_fourcc('M','J','P','G'), num_frames, (FRAME_WIDTH,FRAME_HEIGHT))

        out_head.write(frameHead)
        out_gaze.write(frameGaze)

    if i > start and i < end:
        out_head.write(frameHead)
        out_gaze.write(frameGaze)

        px = np.append(px, head_df["pos.x"][i])
        py = np.append(py, head_df["pos.y"][i])
        pz = np.append(pz, head_df["pos.z"][i])
        pitch = np.append(pitch, head_df["rot.x"][i])
        yaw = np.append(yaw, head_df["rot.x"][i])
        roll = np.append(roll, head_df["rot.x"][i])

    if i == end:
        time1 = head_df["time"][i]
        pos1 = np.array([head_df["pos.x"][i], head_df["pos.y"][i], head_df["pos.z"][i]])
        rot1 = np.array([head_df["rot.x"][i], head_df["rot.y"][i], head_df["rot.z"][i]])
        delta_t = time1 - time0
        vel = (pos1 - pos0) / delta_t
        rot_diff = rot1 - rot0

        px = np.append(px, pos1[0])
        py = np.append(py, pos1[1])
        pz = np.append(pz, pos1[2])
        pitch = np.append(pitch, rot1[0])
        yaw = np.append(yaw, rot1[1])
        roll = np.append(roll, rot1[2])

        new_row = pd.Series({"file_head": file_head, "file_gaze": file_gaze, "velX": vel[0], "velY": vel[1], "VelZ": vel[2], "pitch": pitch.mean(), "yaw": yaw.mean(), "roll": roll.mean()})
        headingData = pd.concat([headingData, new_row.to_frame().T], ignore_index=True)

        # print(f"posX: {vel[0]*delta_t:0.2f}, {px.mean():0.2f}, {np.median(px):0.2f}")

        out_head.release()
        out_gaze.release()
        spliceNr += 1
        if spliceNr == len(split_idx):
            break

headingData.to_csv(f"{save_folder}/{file_name}_ws.csv")