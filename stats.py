import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter
from scipy import signal
from scipy import stats as st
from detect_saccades import splice
from detectors import saccade_detection

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def cart2pol(x, y):
    if x > 0:
        return np.arctan2(y,x)
    if x < 0 and y >= 0:
        return np.arctan2(y,x) + np.pi
    if x < 0 and y < 0:
        return np.arctan2(y,x) - np.pi
    if x == 0 and y > 0:
        return np.pi/2
    if x == 0 and y < 0:
        return -np.pi/2

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/(sd+np.finfo(float).eps))

def autocorr1(x,lags):
    '''numpy.corrcoef, partial'''

    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)


### Settings ###
file_name = "20220926-142032_5"
save_folder = "Stimuli"
split_length = 0.3      # length of image sequences in seconds

thresh_vel = 200        # velocity threshold for saccade detection in degrees / s
thresh_acc = 8000        # acceleration threshold for saccade detection degrees / s²
max_isi = 1             # maximum time between saccades in seconds (isi above that will be excludes for the statistics)

shouldPlot = True



### Calculations ###

gaze_pos_df = pd.read_csv(f"Data/{file_name}_gaze_hr.csv", sep=",")
body_df = pd.read_csv(f"Data/{file_name}_body.csv", sep=",")
head_df = pd.read_csv(f"Data/{file_name}_head.csv", sep=",")
gaze_df = pd.read_csv(f"Data/{file_name}_gaze.csv", sep=",")

# length = gaze_df.shape[0]

# gaze_exc = np.zeros(length)
# phi = np.zeros(length)
# head_horizontal = np.zeros(length)
# head_vertical = np.zeros(length)

# for i in tqdm(np.arange(length)):
#     gaze_exc[i] = np.sqrt(gaze_df['rot.x'][i]**2 + gaze_df['rot.y'][i]**2)
#     phi[i] = cart2pol(gaze_df['rot.x'][i], gaze_df['rot.y'][i])
#     head_horizontal[i] = head_df['rot.y'][i]
#     head_vertical[i] = head_df['rot.x'][i]



body_pos_x = body_df['pos.x']
head_rot_x = head_df['rot.y']
gaze_rot_x = body_df['rot.y']

window_length = np.arange(2, 300, 1)
signal_length = body_df.shape[0]

snr_body = np.zeros_like(window_length, dtype=float)
snr_head = np.zeros_like(window_length, dtype=float)
snr_gaze = np.zeros_like(window_length, dtype=float)

cnt = 0
for w in tqdm(window_length):
    for n in np.arange(signal_length-w):
        # snr_body[cnt] += signaltonoise(body_pos_x[n:n+w]) / (signal_length-w)
        # snr_head[cnt] += signaltonoise(head_rot_x[n:n+w]) / (signal_length-w)
        # snr_gaze[cnt] += signaltonoise(gaze_rot_x[n:n+w]) / (signal_length-w)

        snr_body[cnt] += np.var(body_pos_x[n:n+w]) / (signal_length-w)
        snr_head[cnt] += np.var(head_rot_x[n:n+w]) / (signal_length-w)
        snr_gaze[cnt] += np.var(gaze_rot_x[n:n+w]) / (signal_length-w)

    cnt += 1


autocorr_body = autocorr1(body_pos_x, window_length)
autocorr_head = autocorr1(head_rot_x, window_length)
autocorr_gaze = autocorr1(gaze_rot_x, window_length)

plt.figure()
plt.plot(snr_body/window_length/snr_body.max())
plt.plot(snr_head/window_length/snr_head.max())
plt.plot(snr_gaze/window_length/snr_gaze.max())
plt.show()

plt.figure()
plt.plot(autocorr_body/autocorr_body.max())
plt.plot(autocorr_head/autocorr_head.max())
plt.plot(autocorr_gaze/autocorr_gaze.max())
plt.show()
# fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
# circular_hist(ax[0], phi)

plt.figure()
plt.title('gaze position (head centered)')
plt.xlabel('horizontal excentricity [°]')
plt.xlabel('vertical excentricity [°]')
plt.scatter(gaze_df['rot.x'][:], gaze_df['rot.y'][:], s=0.1)
plt.show(block=False)
plt.pause(.001)

plt.figure()
plt.title('gaze excentricity (head centered)')
plt.xlabel('excentricity [°]')
plt.hist(gaze_exc)
plt.show(block=False)
plt.pause(.001)


plt.figure()
plt.hist(head_horizontal)
plt.title('head rotation horizontal')
plt.xlabel('horizontal rotation [°]')
plt.show(block=False)
plt.pause(.001)


plt.figure()
plt.hist(head_vertical)
plt.title('head rotation vertical')
plt.xlabel('vertical rotation [°]')
plt.show(block=False)
plt.pause(.001)

pass


maze = "20220926-142032_5"
nature = "20221005-164755_4"
warehouse = "20221005-170121_2"

maze_df = pd.read_csv(f"Data/{maze}_gaze_hr.csv", sep=",")
nature_df = pd.read_csv(f"Data/{nature}_gaze_hr.csv", sep=",")
warehouse_df = pd.read_csv(f"Data/{warehouse}_gaze_hr.csv", sep=",")

maze_exc = np.zeros(maze_df.shape[0])
nature_exc = np.zeros(nature_df.shape[0])
warehouse_exc = np.zeros(warehouse_df.shape[0])

for i in tqdm(np.arange(maze_df.shape[0])):
    maze_exc[i] = np.sqrt(maze_df['rot.x'][i]**2 + maze_df['rot.y'][i]**2)

for i in tqdm(np.arange(nature_df.shape[0])):
    nature_exc[i] = np.sqrt(nature_df['rot.x'][i]**2 + nature_df['rot.y'][i]**2)
    
for i in tqdm(np.arange(warehouse_df.shape[0])):
    warehouse_exc[i] = np.sqrt(warehouse_df['rot.x'][i]**2 + warehouse_df['rot.y'][i]**2)


dict = {"maze": maze_exc, "nature": nature_exc, "warehouse": warehouse_exc}
fig, ax = plt.subplots()
ax.boxplot(dict.values(), sym="")
ax.set_xticklabels(dict.keys())
plt.show()
