import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats.stats import pearsonr 

def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def curl_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl

def calc_of(video, shouldPlot=False):
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
    for i in np.arange(int(cap.get(cv.CAP_PROP_FRAME_COUNT))):
        # fig.clf()
        ret, frame2 = cap.read()
        if not ret:
            # print('No frames grabbed!')
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
    cap.release()
    return mean_of

### Settings ###
data_file = "Stimuli/20221005-170121_2.csv"

width = 512
height = 512

shouldPlot = True

### Loading Data ###
df = pd.read_csv(data_file, sep=",")

data_cnt = df.shape[0]

mean_of_head = np.zeros([width, height, 2, data_cnt])
mean_of_retina = np.zeros([width, height, 2, data_cnt])
r = np.zeros(data_cnt)
curl_head = np.zeros([width, height, data_cnt])
curl_retina = np.zeros([width, height, data_cnt])
curl_mean_head = np.zeros(data_cnt)
curl_mean_retina = np.zeros(data_cnt)
rot = np.zeros(data_cnt)


div_head = np.zeros([width, height, data_cnt])
div_retina = np.zeros([width, height, data_cnt])
div_mean_head = np.zeros(data_cnt)
div_mean_retina = np.zeros(data_cnt)
vel = np.zeros(data_cnt)


for sm in tqdm(np.arange(data_cnt)):
    flow_head = calc_of(df['file_head'][sm])
    flow_retina = calc_of(df['file_gaze'][sm])
    mean_of_head[:,:,:,sm] = flow_head
    mean_of_retina[:,:,:,sm] = flow_retina

    flow1_centered = flow_head - np.mean(flow_head, axis=(0, 1))
    flow2_centered = flow_retina - np.mean(flow_retina, axis=(0, 1))
    inner_product = np.sum(flow1_centered*flow2_centered)
    r[sm] = inner_product/np.sqrt(np.sum(flow1_centered**2) * np.sum(flow2_centered**2))

    curl_head[:,:,sm] = curl_npgrad(flow_head)
    curl_retina[:,:,sm] = curl_npgrad(flow_retina)
    curl_mean_head[sm] = curl_head[:,:,sm].mean()
    curl_mean_retina[sm] = curl_retina[:,:,sm].mean()
    rot[sm] = np.sqrt(df['pitch'][sm]**2 + df['yaw'][sm]**2 + df['roll'][sm]**2)

    div_head[:,:,sm] = divergence_npgrad(flow_head)
    div_retina[:,:,sm] = divergence_npgrad(flow_retina)
    div_mean_head[sm] = div_head[:,:,sm].mean()
    div_mean_retina[sm] = div_retina[:,:,sm].mean()
    vel[sm] = np.sqrt(df['velX'][sm]**2 + df['velY'][sm]**2 + df['VelZ'][sm]**2)


curl_rot_corr_head = pearsonr(curl_mean_head, rot)
curl_rot_corr_retina = pearsonr(curl_mean_retina, rot)
div_vel_corr_head = pearsonr(div_mean_head, vel)
div_vel_corr_retina = pearsonr(div_mean_retina, vel)
print(f"Pearson R (p val) curl - rot | head: {curl_rot_corr_head[0]:0.4f} ({curl_rot_corr_head[1]:0.4f}) retina: {curl_rot_corr_retina[0]:0.4f} ({curl_rot_corr_retina[1]:0.4f})")
print(f"Pearson R (p val) div - vel | head: {div_vel_corr_head[0]:0.4f} ({div_vel_corr_head[1]:0.4f}) retina: {div_vel_corr_retina[0]:0.4f} ({div_vel_corr_retina[1]:0.4f})")
# Y, X = np.mgrid[0:512:1, 0:512:1]    

# u = np.sin(X/256 + Y/256)
# v = np.cos(X/256 - Y/256)

# field2 = np.stack((u, v), axis=-1)
# plt.streamplot(X, Y, field2[:, :, 0], field2[:, :, 1])

plt.figure()
plt.boxplot(r)
plt.show(block=False)
plt.pause(.001)

if shouldPlot:
    n = 2
    samples = np.array(df.sample(n=n).index)
    samples[0] = 18
    samples[1] = 11
    fig, axs = plt.subplots(n, 1, sharex=True, sharey=True)
    Y, X = np.mgrid[0:512:1, 0:512:1]
    plt.ion
    print("now Plotting...")
    for i in np.arange(n):
        Uh = mean_of_head[:,:,0,samples[i]]
        Vh = mean_of_head[:,:,1,samples[i]]
        Ur = mean_of_retina[:,:,0,samples[i]]
        Vr = mean_of_retina[:,:,1,samples[i]]

        flow1 = np.stack((Uh, Vh), axis=-1)
        flow2 = np.stack((Ur, Vr), axis=-1)
        flow1_centered = flow1 - np.mean(flow1, axis=(0, 1))
        flow2_centered = flow2 - np.mean(flow2, axis=(0, 1))
        inner_product = np.sum(flow1_centered*flow2_centered)
        r = inner_product/np.sqrt(np.sum(flow1_centered**2) * np.sum(flow2_centered**2))

        speedH = np.sqrt(Uh**2 + Vh**2)
        speedR = np.sqrt(Ur**2 + Vr**2)
        lwH = 5*speedH / speedH.max()
        lwR = 5*speedR / speedR.max()
        axs[i].streamplot(X, Y, Uh, Vh, density=[0.5, 1], linewidth=lwH)
        axs[i].set_title(f"H {samples[i]} (velX {df['velX'][samples[i]]:0.2f}, velY {df['velY'][samples[i]]:0.2f}, VelZ {df['VelZ'][samples[i]]:0.2f}, pitch {df['pitch'][samples[i]]:0.2f}, yaw {df['yaw'][samples[i]]:0.2f}, roll {df['roll'][samples[i]]:0.2f}, Corr. {r:0.2f}", fontsize=11)
        axs[i].streamplot(X+650, Y, Ur, Vr, density=[0.5, 1], linewidth=lwR)
    plt.ylim(512, 0)
    plt.xlim(0, 512+650)
    plt.show()


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