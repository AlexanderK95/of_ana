import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from moviepy.video.io.bindings import mplfig_to_npimage

def make_video(video, sp):
    cap = cv.VideoCapture(cv.samples.findFile(video))
    cnt = 0

    video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter('output.mp4', -1, 45.0, (video_width,video_height))

    Y, X = np.mgrid[0:video_width:1, 0:video_height:1]

    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    for i in tqdm(np.arange(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
        # fig.clf()
        ret, frame2 = cap.read()
        if not ret:
            # print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        U = flow[:,:,0,]
        V = flow[:,:,1]
        speed = np.sqrt(U**2 + V**2)
        lw = 5*speed / speed.max()

        fig, ax = plt.subplots()
        ax.streamplot(X, Y, U, V, linewidth=lw, start_points=sp.T)
        ax.imshow(next)
        img = mplfig_to_npimage(fig)
        out.write(img)

        
        cv.imshow('frame',img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            out.release()
            break
        # flow_all[:, :, :, cnt] = flow
        # mean_of += flow / int(cap.get(cv.CAP_PROP_FRAME_COUNT))

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
    # return mean_of
    cap.release()
    out.release()




src_file = "C:/Users/AlexanderK/Documents/GitHub/of_ana/Data/20221004-135401_1_head_cam.mp4"

width = 512
height = 512

src_points = [5, 5]

#####
x = np.linspace(width/src_points[0], width-width/src_points[0], src_points[0], endpoint=True)
y = np.linspace(height/src_points[1], height-height/src_points[1], src_points[1], endpoint=True)

starting_points_x, starting_points_y = np.meshgrid(x,y)

sp = np.array([starting_points_x.flatten(), starting_points_y.flatten()])

make_video(src_file, sp)