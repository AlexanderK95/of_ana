import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

head = "Data/20220921-115719_1_head_cam.mp4"
gaze = "Data/20220921-115719_1_gaze_cam.mp4"

def calc_of(video):
    cap = cv.VideoCapture(cv.samples.findFile(video))
    # flow_all = np.zeros([int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), 2, int(cap.get(cv.CAP_PROP_FRAME_COUNT))])

    cnt = 0

    mean_of = np.zeros([int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), 2])

    

    Y, X = np.mgrid[0:int(cap.get(cv.CAP_PROP_FRAME_WIDTH)):1, 0:int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)):1]

    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    # hsv = np.zeros_like(frame1)
    # hsv[..., 1] = 255
    # fig, ax = plt.subplots(1,1)
    # plt.ion()
    # plt.show()
    for i in tqdm(np.arange(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
        # fig.clf()
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow_all[:, :, :, cnt] = flow
        mean_of += flow / int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # U = flow[:,:,0]
        # V = flow[:,:,1]

        # speed = np.sqrt(U**2 + V**2)
        # lw = 5*speed / speed.max()
        # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang*180/np.pi/2
        # # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # hsv[..., 2] = mag
        # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # cv.imshow('frame2', frame2)
        # plt.imshow(frame2)
        # plt.streamplot(X, Y, U, V, density=[0.5, 1], linewidth=lw)
        # plt.pause(0.01)
        # k = cv.waitKey(5) & 0xff
        # if k == 27:
        #     break
        prvs = next
        cnt += 1
    # cv.destroyAllWindows()
    # return flow_all, mean_of
    return mean_of


body_df = pd.read_csv("Data/20220921-115719_1_body.csv", sep=",")
speed = np.sqrt(body_df['rot.x'] ** 2 + body_df['rot.z'] ** 2)
plt.plot(body_df['time'], speed)
plt.plot(body_df['time'], body_df['rot.x'])
plt.plot(body_df['time'], body_df['rot.z'])
plt.show()

pass

# gaze_of_mean = calc_of(gaze)
# head_of_mean = calc_of(head)


# Y, X = np.mgrid[0:512:1, 0:512:1]

# plt.figure()
# U = gaze_of_mean[:,:,0]
# V = gaze_of_mean[:,:,1]

# speed = np.sqrt(U**2 + V**2)
# lw = 5*speed / speed.max()

# plt.streamplot(X, Y, U, V, density=[0.5, 1], linewidth=lw)
# plt.ylim(512, 0)

# plt.figure()
# U = head_of_mean[:,:,0]
# V = head_of_mean[:,:,1]

# speed = np.sqrt(U**2 + V**2)
# lw = 5*speed / speed.max()

# plt.streamplot(X, Y, U, V, density=[0.5, 1], linewidth=lw)
# plt.ylim(512, 0)
# plt.show() 

# pass