# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: seis310
#     language: python
#     name: seis310
# ---

# %% [markdown]
# # Geolocalization of DAS channels using GPS-tracked vehicle
# @Author: Ettore Biondi - ebiondi@caltech.edu 
#
# This notebook shows how the latitude and longitude positions of a vehicle can be used to geolocate the channels of a distributed acoustic sensing (DAS) system that employs an onshore dark fiber.

# %%
# %matplotlib widget
import sys
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from tqdm import tqdm

time_format = "%Y-%m-%dT%H:%M:%S.%f+00:00"
time_format1 = "%Y-%m-%dT%H:%M:%S.%fZ"

import matplotlib
import matplotlib.pyplot as plt
import datetime, pytz
import dateutil.parser

# Adding pyDAS location
pyDAS_path = "/home/qzhai/opt/DAS-utilities/build/"
try:
    os.environ['LD_LIBRARY_PATH'] += ":" + pyDAS_path
except:
    os.environ['LD_LIBRARY_PATH'] = pyDAS_path
sys.path.insert(0, pyDAS_path)
sys.path.insert(0, '/home/qzhai/opt/DAS-utilities/python')
sys.path.insert(0, "../Python")
import pyDAS
import DASutils
import TapTestWdgts

import importlib

importlib.reload(pyDAS)
importlib.reload(DASutils)
importlib.reload(TapTestWdgts)
# from DASutils import get_fft, get_fft_and_psd_median_of_segs_of_one_file, detect_local_minima,bp_filter,narrow_filter,envelop, median_filter, velocity_shift, kmps2mph, normalize_by_maxabs, get_n_mad_threshold, n_mad_above_moving_median, find_picks
from DASutils import *

from inspect import getmembers, isfunction

functions_list = getmembers(DASutils, isfunction)
print([i[0] for i in functions_list])

from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {
    "figure.figsize": (8.5 * 0.7, 8.5 * 0.7 / 9 * 6),
    "image.interpolation": "nearest",
    "image.cmap": "gray",
    "figure.dpi": 150,  # to adjust notebook inline plot size
    "savefig.dpi": 300,  # to adjust notebook inline plot size
    "axes.labelsize": 14,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 12,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": False,
}
matplotlib.rcParams.update(params)

# %%
LAX_db = pd.read_csv(
    '/home/ebiondi/research/projects/LAX-DAS/Catalog/LA-DAS-Oct2022.txt',
    delim_whitespace=True,
)
startDate = "2022-02-01T17:26:00.00+00:00"
endDate = "2022-02-01T17:27:16.48+00:00"
eventTime = "2022-02-01T17:26:16.48"
eventDateTime = dateutil.parser.parse(eventTime)
selectedFiles, startTime = DASutils.selectFile(
    LAX_db, "LAX36-LAX28", startDate, endDate
)
print(selectedFiles)
DAS_data, info = DASutils.readFile_HDF(
    selectedFiles,
    0.05,
    40.0,
    verbose=1,
    preproc=False,
    desampling=False,
    nChbuffer=5000,
    system="OptaSense",
)
nt = DAS_data.shape[1]
fs = info['fs']
dChEq = info['dx']
dt = 1.0 / fs

dataEq = DASutils.bandpass2D_c(DAS_data[:, :], 0.5, 16, 1.0 / fs, zerophase=True)

# %% [markdown]
# Now we can start the interactive plots to easily identify the channels that we want to be removed from the data. The "bad channels" widget can be used to type the channel that are not recording useful seismic signal. The user has to provide a comma-separated list of channel numbers from 0 to the total number of channels (e.g., 0,1,2,3,...). It is also possible to provide an interval of channels by using two indices separated by a column (e.g., 0:3,10:32,...).
# For this specific system, the identified bad channels are the following: 
#
# "bad channels for channel loop": 198:222,268:271,279:283,324:327,371:374,402:406,423:427,452:454,467:470,495:499,525:528,552:555,600:603,648:652,691:694,727:730,746,768,810,818:820,852:855,881:883,902:906,922:924,968,990:993,996:999,1029:1032,1064:1066,1089:1092,1046:1051,1146:1151,1382:1384,1430:1432,1458:1461,1521,1568:1570,1594:1597,1646:1659,1696,1859,1897:1899,1925:1931,1954:1957,1972:1975,2023:2025,2042,2065:2069,2072,2098:2101,2144:2147,2192:2196,2231,2233,2285:2288,2293:2296,2342,2403:2405,2450,2482,2633:2635,2658,2688:2692,2705,2717:2721,2738:2741,2797:2800,2818:2821,2868:2870,2909:2912,2932:2935,2985:2989,3016:3020,3076:3079,3120,3188,3191,3222:3225,3267:3270,3297:3301,3321:3323,3343,3386:3389,3419:3422,3442:3445,3738:3742,3768:3770,3786,3797:3799,3832:3835,3857,3929:3932,3941:3945,3952,3969:3972,3991,4035:4038,4068,4096:4101,4204,4265:4269,4292:4296,4308:4312,4345:4348,4444,4481:4484,4496,4522:4524,4542,4609,4718:4721,4765:4779
#
# "additional channel loops":
# 2721,2737,2767,2906:2919,2971:2975,3119:3122,3176,3188:3191,3232
#
# "change bad back to good"
# 3191
#
# "bad channels for channle loop, no locaiton at two ends,  and high noise (aerial) sections":
# 0:300,324:327,371:374,402:406,423:427,452:454,467:470,495:499,525:528,552:555,600:603,648:652,691:694,727:730,746,768,769:809,810,818:820,852:855,881:883,902:906,922:924,968,990:993,996:999,1029:1032,1064:1066,1089:1092,1046:1051,1146:1151,1163:1356,1382:1384,1430:1432,1458:1461,1521,1568:1570,1594:1597,1646:1659,1696,1859,1897:1899,1925:1931,1940:1957,1972:1975,2023:2025,2042,2065:2069,2072,2098:2101,2144:2147,2192:2196,2231,2233,2285:2288,2293:2296,2342,2403:2405,2412:2450,2482,2506:2554,2574:2587,2605:2625,2633:2654,2658,2688:2692,2705,2717:2722,2737:2741,2767,2797:2800,2818:2821,2868:2870,2906:2919,2932:2935,2971:2975,2985:2989,3016:3020,3076:3079,3119:3122,3176,3188:3191,3222:3225,3232,3267:3270,3297:3301,3321:3323,3343,3386:3389,3419:3422,3442:3445,3738:3742,3768:3770,3786,3797:3799,3832:3835,3857,3929:3932,3941:3945,3952,3969:3972,3991,4035:4038,4068,4096:4101,4204,4265:4269,4292:4296,4308:4312,4345:4348,4444,4481:4484,4496,4522:4524,4542,4580:4780
#
#
# One can copy this list and paste it in the "bad channels" widget.
# The bottom widgets provide control on the visualized data. The interface should be intuitive. If non-valid values are set, an error message should appear below the widgets.

# %%
# Earthquake at 350 seconds
importlib.reload(TapTestWdgts)
badCh = TapTestWdgts.badchnnlwdgt(dataEq, dt)
badCh

# %%
# LAX_db = pd.read_csv('/home/qzhai/projects/lax/data/LAX_DAS_db.txt',delim_whitespace=True)
# print(LAX_db)
# startDate = "2022-02-27T:01:43:18.00+00:00"
# endDate = "2022-02-27T01:48:18.00+00:00"
# eventTime = "2022-02-27T01:44:18.00+00:00"

# eventDateTime = dateutil.parser.parse(eventTime).replace(tzinfo=None)
# selectedFiles, startTime = DASutils.selectFile(LAX_db, "LAX", startDate, endDate)
# print(selectedFiles)
# DAS_data, info = DASutils.readFile_HDF(
#     selectedFiles,
#     0.1,
#     40.0,
#     verbose=1,
#     preproc=False,
#     desampling=False,
#     nChbuffer=6000,
#     system="OptaSense",
# )
# dataEq = DASutils.bandpass2D_c(DAS_data[:,:], 0.5, 16, 1.0/fs, zerophase=True)
# nt = DAS_data.shape[1]
# fs = info['fs']
# dChEq = info['dx']
# dt = 1.0 / fs
# # Earthquake at 350 seconds
# badCh = TapTestWdgts.badchnnlwdgt(dataEq, dt)
# badCh

# %% [markdown]
# Let's now extract a mask to plot the data without the bad channels.

# %%
mask_bad = np.ones(dataEq.shape[0], dtype=bool)
mask_bad[badCh.bad_channels] = False
data_good = dataEq[mask_bad]

# %%
# Plotting w/o loops and trace normalization
data_norm = data_good.copy()
# Plotting the two systems' data together
fig, ax = plt.subplots(figsize=(12, 10))
# North system
min_t = 340
max_t = 380
it_min = int((min_t) / dt + 0.5)
it_max = int((max_t) / dt + 0.5)
std_data = np.std(data_norm[:, it_min:it_max], axis=1)
data_norm /= np.expand_dims(std_data, axis=1)
clipVal = np.percentile(np.absolute(data_norm[:, it_min:it_max]), 95)
ax.imshow(
    data_norm[:, it_min:it_max].T,
    extent=[0, data_norm.shape[0], max_t, min_t],
    aspect='auto',
    vmin=-clipVal,
    vmax=clipVal,
    cmap=plt.get_cmap('seismic'),
)
ax.set_ylabel("Time [s]")
ax.set_xlabel("Channel number")
ax.grid()

# %%
# with np.load("../Dat/Traffic.npz",allow_pickle=True) as dat:
#     dataTap = dat['data']
#     dt       = dat['dt'].item()
#     ot       = dat['ot'].item()
#     dCh      = dat['dCh'].item()
with np.load(
    "/home/ebiondi/research/projects/LAX-DAS/Catalog/CarTrackLAX2022.npz",
    allow_pickle=True,
) as dat:
    gps_lat = dat['gps_lat']
    gps_lon = dat['gps_lon']
    gps_time = dat['gps_time']

# %%
startDate = (gps_time[0] - timedelta(minutes=3)).strftime(time_format)
endDate = (gps_time[-1] + timedelta(minutes=3)).strftime(time_format)
selectedFiles, startTime = DASutils.selectFile(
    LAX_db, "LAX36-LAX28", startDate, endDate
)
DAS_data, info = DASutils.readFile_HDF(
    selectedFiles,
    0.05,
    2.5,
    verbose=1,
    preproc=False,
    desampling=True,
    nChbuffer=5000,
    system="OptaSense",
)
nt = DAS_data.shape[1]
fs = info['fs']
dCh = info['dx']
dt = 1.0 / fs
ot = info['begTime']
# Adding time zone to begTime
ot = dateutil.parser.parse(ot.strftime(time_format1))

# %%
dataTap = DASutils.bandpass2D_c(DAS_data[:, :], 0.25, 2, 1.0 / fs, zerophase=True)

# %%
# Loading fiber locations at 1 m spacing
with np.load(
    "/home/ebiondi/research/projects/LAX-DAS/Catalog/FiberLAXReg.npz", allow_pickle=True
) as dat:
    fiber_lat = dat['lat']
    fiber_lon = dat['lon']
    fiber_dist = dat['dist']

# Let's map the car locations onto the closest fiber points
close_ch = TapTestWdgts.find_close_ch(gps_lat, gps_lon, fiber_lat, fiber_lon)
fiber_dist_vehicle = fiber_dist[close_ch]

# %%
fig, ax = plt.subplots(figsize=(10, 10))
min_idx = 0
max_idx = 2000
ax.plot(fiber_lon, fiber_lat, "bo-")
ax.plot(gps_lon[min_idx:max_idx], gps_lat[min_idx:max_idx], "ro")
ax.grid()

# %% [markdown]
# Now we have all the necessary inputs for proceeding with the geolocalization of the DAS channels of this system. Once the interactive plot is started, the first step is to copy the list of bad channels that we have identified before. Obviously, in this step, additional bad channels can be found and removed if necessary. To correctly calibrate the channel positions, one must uncheck the "Show bad channel" toggle switch.
# After this visual check, we need to align the red dashed line with the car-tracked generated signal. To provide an example of this process, change the display options to visualize the data between the channel numbers 500 and 1800 and the first 600 seconds. Now, apply a "Tap-test shift" of -46.4 and turn off and on the "Show tap-test line" switch to see the alignment of a signal with the calibration line. Thus, we can map with confidence that the displayed channels have been excited by the tracked-car movement. Change the min and max "map ch" slidebars to 500 and 1800, and hit map channels. You should be able to see the mapped channels in the top panel of the interactive plot.
# We can now proceed to other sections of the cable. 
#
# The second section can be mapped with the following parameters: 
# min channel=1800
# max channel=2300
# min time=500
# max time=910
# min map ch=500
# max map ch=910
#
# The small gap between the two section can be easily filled by a linear interpolation, which is done in the cells below. One could use u-turns to identify the tracked car, but when mapping one should visualize only a single direction curve (i.e., towards or away from the interrogator unit) before hitting the map channel button.
#
# For the purpose of this tutorial, only the first 25 km of the fiber geometry are provided. Thus, one can only map approximately the first 2500 DAS channels.
#
# The variable **filename** below should be populated by the user in that previous box, or else the next box will throw an error. Finally, if bad channels have been identified, remember to uncheck the **Show bad channels** box before mapping any channel.
#
# LAX:errorGPS=-6.6

# %%
importlib.reload(TapTestWdgts)
errorGPS = -6.6
taptest = TapTestWdgts.taptestwdgt(
    dataTap,
    dt,
    dCh,
    gps_time,
    fiber_dist_vehicle,
    fiber_lat[close_ch],
    fiber_lon[close_ch],
    ot + timedelta(seconds=errorGPS),
    pclip=90,
)
# This is the CSV filename in which the channel positions will be saved, see cells below; it can be also
# be used to load temporary results while calibrating the channel locations
filename = "/home/qzhai/opt/DAS-taptest-widgets/Notebooks/Tap_test_LAX_QZ.csv"
# if statement is useful to reload temporary results in case of errors during the calibration
# Recopy the bad channel list and uncheck the bad channel toggle button
# The plot will show the temporary progress if it was saved on file
if os.path.isfile(filename):
    ch_db = pd.read_csv(filename)
    taptest.mapped_channels = (
        ch_db[ch_db["status"] == "good"]["channel"].astype(int).to_numpy()
    )
    taptest.mapped_lat = (
        ch_db[ch_db["status"] == "good"]["latitude"].astype(float).to_numpy()
    )
    taptest.mapped_lon = (
        ch_db[ch_db["status"] == "good"]["longitude"].astype(float).to_numpy()
    )
# To start the interactive tool, run the next cell.

# %%
taptest

# %%

# %%
bad_ch = np.copy(taptest.bad_channels)
mapped_ch = np.copy(taptest.mapped_channels)
mapped_lat = np.copy(taptest.mapped_lat)
mapped_lon = np.copy(taptest.mapped_lon)
df_ch = pd.DataFrame(
    columns=['channel', 'status', 'latitude', 'longitude', 'elevation']
)
nCh = dataTap.shape[0]
for ich in range(nCh):
    if ich in mapped_ch:
        idx = np.argwhere(mapped_ch == ich)[0]
        df_ch.loc[str(ich)] = [
            ich,
            "good",
            mapped_lat[idx][0],
            mapped_lon[idx][0],
            np.nan,
        ]
    if ich in bad_ch:
        idx = np.argwhere(bad_ch == ich)[0]
        df_ch.loc[str(ich)] = [ich, "bad", np.nan, np.nan, np.nan]
if filename != "":
    df_ch.to_csv(filename, index=None, sep=',', mode='w')

# %% [markdown]
# #### Interpolating and mapping to closest fiber locations

# %% [markdown]
# Once all the certain channels have been located, we can interpolate the uncertain ones (i.e., the ones with no clear vehicle signal). This step allows us to geolocate channels that could not be reached by the car signal. In our case, most of the channels could be excited by the car-related deformation, so the interpolated channels are within an accuracy of approximately 5 m.

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fiber_lon, fiber_lat, "bo-", label="Fiber geometry")
ax.plot(mapped_lon, mapped_lat, "ro", label="Mapped channels")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.legend()
ax.grid()

# %%
bad_ch_before_remove_bad = bad_ch
mapped_ch_before_remove_bad = mapped_ch
close_finech = TapTestWdgts.find_close_ch(mapped_lat, mapped_lon, fiber_lat, fiber_lon)
fiber_dist_taptest = fiber_dist[close_finech]

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(mapped_ch_before_remove_bad, fiber_dist_taptest, "ro", label="Mapped Channels")
ax.set_xlabel("mapped_ch_before_remove_bad (#)")
ax.set_ylabel("fiber_dist_taptest (m)")
ax.legend()
ax.grid()

# %%
id_mapped_selected = np.where(fiber_dist_taptest > 0.1)[0]
fiber_dist_taptest_selected = fiber_dist_taptest[id_mapped_selected]
mapped_ch_before_remove_bad_selected = mapped_ch_before_remove_bad[id_mapped_selected]
is_mapped_before_remove_bad = np.full((nCh,), False)
is_mapped_before_remove_bad[mapped_ch_before_remove_bad_selected] = True
print(is_mapped_before_remove_bad)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(
    mapped_ch_before_remove_bad_selected,
    fiber_dist_taptest_selected,
    "ro",
    label="Selected Mapped Channels",
)
ax.set_xlabel("mapped_ch_before_remove_bad (#)")
ax.set_ylabel("fiber_dist_taptest (m)")
ax.legend()
ax.grid()

# %%
# only plot the channel spacing for mapped chs
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

ax.plot(
    mapped_ch_before_remove_bad_selected,
    fiber_dist_taptest_selected,
    "ro",
    label="Selected Mapped Channels",
)
mapped_ch_before_remove_bad_selected_set = set(mapped_ch_before_remove_bad_selected)
ch_spaces_all = []
for i_ch, ch in enumerate(mapped_ch_before_remove_bad_selected):
    if int(ch + 1) in mapped_ch_before_remove_bad_selected_set:
        ch_spaces_all.append(
            fiber_dist_taptest_selected[int(i_ch + 1)]
            - fiber_dist_taptest_selected[int(i_ch)]
        )

ch_spaces_all = np.array(ch_spaces_all)
print(np.nanmean(ch_spaces_all))
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(
    ch_spaces_all,
    bins=np.arange(np.nanmin(ch_spaces_all - 1), np.nanmax(ch_spaces_all) + 1, 1),
    color='black',
)
ax.plot([np.nanmean(ch_spaces_all), np.nanmean(ch_spaces_all)], [0, 1600], color='red')
ax.text(
    np.nanmean(ch_spaces_all),
    1610,
    "Mean channel spacing (mapped) is "
    + '{:.2f}'.format(np.nanmean(ch_spaces_all))
    + "m",
    color='red',
    horizontalalignment='center',
    fontsize=16,
)
ax.set_xlabel("Channel spacing (m)")
ax.set_ylabel("Count (#)")
ax.set_xlim((0, np.nanmean(ch_spaces_all) * 2))
# ax.grid()

# %%
np.delete(np.arange(10), np.arange(3))

# %%
# if need to add more channel loops:

# additional_channel_loops_str=""
additional_channel_loops_str = (
    "2721,2737,2767,2906:2919,2971:2975,3119:3122,3176,3188:3191,3232"
)
additional_channel_loops = np.array([], dtype=int)
if len(additional_channel_loops_str) > 0:
    if "," in additional_channel_loops_str:
        for val in additional_channel_loops_str.split(","):
            if ":" in val:
                strt_ch = int(val.split(":")[0])
                lst_ch = int(val.split(":")[1])
                additional_channel_loops = np.append(
                    additional_channel_loops, np.arange(strt_ch, lst_ch)
                )
            else:
                additional_channel_loops = np.append(additional_channel_loops, int(val))
    else:
        if ":" in additional_channel_loops_str:
            strt_ch = int(additional_channel_loops_str.split(":")[0])
            lst_ch = int(additional_channel_loops_str.split(":")[1])
            additional_channel_loops = np.append(
                additional_channel_loops, np.arange(strt_ch, lst_ch)
            )
        else:
            additional_channel_loops = np.append(
                additional_channel_loops, int(additional_channel_loops_str)
            )


chs_back_to_good_str = "3191"
chs_back_to_good = np.array([], dtype=int)
if len(chs_back_to_good_str) > 0:
    if "," in chs_back_to_good_str:
        for val in chs_back_to_good_str.split(","):
            if ":" in val:
                strt_ch = int(val.split(":")[0])
                lst_ch = int(val.split(":")[1])
                chs_back_to_good = np.append(
                    chs_back_to_good, np.arange(strt_ch, lst_ch)
                )
            else:
                chs_back_to_good = np.append(chs_back_to_good, int(val))
    else:
        if ":" in chs_back_to_good_str:
            strt_ch = int(chs_back_to_good_str.split(":")[0])
            lst_ch = int(chs_back_to_good_str.split(":")[1])
            chs_back_to_good = np.append(chs_back_to_good, np.arange(strt_ch, lst_ch))
        else:
            chs_back_to_good = np.append(chs_back_to_good, int(chs_back_to_good_str))

bad_ch_before_remove_bad_tmp = np.int32(
    np.sort(np.unique(np.append(bad_ch_before_remove_bad, additional_channel_loops)))
)
bad_ch_before_remove_bad = np.int32(
    np.sort(list(set(bad_ch_before_remove_bad_tmp) - set(chs_back_to_good)))
)

# %%
# Interpolating "uncertain" channels using fiber distance
mask_before_remove_bad = np.ones(dataTap.shape[0], dtype=bool)
mask_before_remove_bad[bad_ch_before_remove_bad] = False
good_ch_before_remove_bad = np.where(mask_before_remove_bad)[
    0
]  # indices of good channels
i_ch_after_remove_bad = np.arange(good_ch_before_remove_bad.shape[0])
# Mask of the mapped good channels
mask_after_remove_bad = np.zeros(good_ch_before_remove_bad.shape[0], dtype=bool)
for i_ch_before_remove_bad in mapped_ch_before_remove_bad_selected:
    mask_after_remove_bad[
        np.argwhere(good_ch_before_remove_bad == i_ch_before_remove_bad)[0]
    ] = True

# Interpolating missing good channels
f_dist = interp1d(
    np.where(mask_after_remove_bad)[0],
    fiber_dist_taptest_selected,
    kind='linear',
    bounds_error=False,
)
mapped_dist_int = f_dist(i_ch_after_remove_bad)
f_lat = interp1d(fiber_dist, fiber_lat, kind='linear', bounds_error=False)
mapped_lat_int = f_lat(mapped_dist_int)
f_lon = interp1d(fiber_dist, fiber_lon, kind='linear', bounds_error=False)
mapped_lon_int = f_lon(mapped_dist_int)

# Projecting calibrated and interpolated positions onto fiber geometry
i_not_nan = np.argwhere(~np.isnan(mapped_lat_int))[:, 0]
# close_finech_not_nan = TapTestWdgts.find_close_ch(mapped_lat_int[i_not_nan],mapped_lon_int[i_not_nan],fiber_lat,fiber_lon)
# mapped_lat_int[i_not_nan] = fiber_lat[close_finech_not_nan]
# mapped_lon_int[i_not_nan] = fiber_lon[close_finech_not_nan]

# Interpolating bad and good channels before remove bad
f_dist_before_remove_bad = interp1d(
    mapped_ch_before_remove_bad_selected,
    fiber_dist_taptest_selected,
    kind='linear',
    bounds_error=False,
)
good_dist_int = f_dist_before_remove_bad(good_ch_before_remove_bad)
good_lat_int = f_lat(good_dist_int)
good_lon_int = f_lon(good_dist_int)
bad_dist_int = f_dist_before_remove_bad(bad_ch_before_remove_bad)
bad_lat_int = f_lat(bad_dist_int)
bad_lon_int = f_lon(bad_dist_int)

# %%
# only plot the channel spacing for mapped chs
ch_spaces_all = []
L_ch_spaces = np.diff(mapped_dist_int)
N_ch_spaces = np.diff(i_ch_after_remove_bad)
ch_spaces = L_ch_spaces / N_ch_spaces
for i in range(ch_spaces.shape[0]):
    if N_ch_spaces[i] > 0 and N_ch_spaces[i] <= len(mapped_dist_int):
        ch_spaces_all = ch_spaces_all + [ch_spaces[i]] * N_ch_spaces[i]
ch_spaces_all = np.array(ch_spaces_all)
print(np.nanmean(ch_spaces_all))
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(
    ch_spaces_all,
    bins=np.arange(np.nanmin(ch_spaces_all - 1), np.nanmax(ch_spaces_all) + 1, 1),
    color='black',
)
ax.plot([np.nanmean(ch_spaces_all), np.nanmean(ch_spaces_all)], [0, 1600], color='red')
ax.text(
    np.nanmean(ch_spaces_all),
    1610,
    "Mean channel spacing is " + '{:.2f}'.format(np.nanmean(ch_spaces_all)) + "m",
    color='red',
    horizontalalignment='center',
)
ax.set_xlabel("Channel spacing (m)")
ax.set_ylabel("Count (#)")
ax.set_xlim((0, np.nanmean(ch_spaces_all) * 2))
# ax.grid()

# %%
fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(fiber_lon, fiber_lat, "bo", label="Fiber geometry")
ax.plot(mapped_lon_int, mapped_lat_int, "go", label="Interpolated channels")
# ax.plot(good_lon_int,good_lat_int,"co",label="Good channels")
# ax.plot(bad_lon_int,bad_lat_int,"mo",label="Bad channels")
ax.plot(mapped_lon, mapped_lat, "ro", label="Mapped channels")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.legend(fontsize=22, markerscale=5)
ax.grid()

# %% [markdown]
# ### Getting channel elevations and writing to final output file

# %% [markdown]
# Finally, we can get the elevation of each channel, verify the tap-test results, and then write the final result onto a file.

# %%
# Getting elevation of each channel from USGS query form (this might take some time)
mapped_ele_int = np.zeros_like(mapped_lat_int)
# for idx in tqdm(maskNan):
#     mapped_ele_int[idx] = TapTestWdgts.elevation_function(mapped_lat_int[idx],mapped_lon_int[idx])

i_not_nan = np.argwhere(~np.isnan(mapped_lat_int))[:, 0]
n_jobs = 40
from joblib import Parallel, delayed
from tqdm import tqdm

output_list_with_None = Parallel(n_jobs=n_jobs)(
    delayed(TapTestWdgts.elevation_function)(mapped_lat_int[idx], mapped_lon_int[idx])
    for idx in tqdm(i_not_nan)
)

for i_idx, idx in enumerate(i_not_nan):
    mapped_ele_int[idx] = output_list_with_None[i_idx]

# %% [markdown]
# Now, let's verify if the calibration process provided accurate channel positions by mapping the car GPS points to the closest mapped channel.

# %%
close_ch = (
    TapTestWdgts.find_close_ch(
        gps_lat, gps_lon, mapped_lat_int[i_not_nan], mapped_lon_int[i_not_nan]
    )
    + i_not_nan[0]
)
close_gps_time = [tm.timestamp() - ot.timestamp() - errorGPS for tm in gps_time]

# %%
# fig, ax = plt.subplots(figsize=(12,100))
fig, ax = plt.subplots(figsize=(12, 10))
mask_tmp = np.ones(dataTap.shape[0], dtype=bool)
mask_tmp[bad_ch] = False
env = np.abs(hilbert(dataTap[mask_tmp, :]))
env /= env.max()
clipVal = np.percentile(np.absolute(env[:, :]), 85)
im = plt.imshow(
    env.T,
    extent=[
        i_ch_after_remove_bad[0],
        i_ch_after_remove_bad[-1],
        (env.shape[1] - 1) * dt,
        0.0,
    ],
    aspect='auto',
    vmin=0.0,
    vmax=clipVal,
    cmap=plt.get_cmap('jet'),
)
plt.plot(close_ch, close_gps_time, "r--", lw=3, alpha=0.5)
plt.ylabel("Time [s]")
plt.xlabel("Channel number")
plt.grid()
plt.ylim([(env.shape[1] - 1) * dt, 0.0])
plt.gca().invert_yaxis()
# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1%", pad=0.5)
cbar = plt.colorbar(im, orientation="vertical", cax=cax)
cbar.set_label('Amplitude @ 0.25-2.0 Hz')

# %% [markdown]
# And finally save the end result into a file.

# %%
filenameFinal = (
    "/home/qzhai/opt/DAS-taptest-widgets/Notebooks/Tap_test_LAX_Final_QZ.csv"
)
df_ch = pd.DataFrame(
    columns=['channel', 'status', 'latitude', 'longitude', 'elevation', 'mapped']
)
for ich in range(nCh):
    if ich in good_ch_before_remove_bad:
        idx = np.where(good_ch_before_remove_bad == ich)[0][0]
        if np.isnan(mapped_lat_int[idx]):
            df_ch.loc[str(ich)] = [ich, "bad", np.nan, np.nan, np.nan, False]
        else:
            df_ch.loc[str(ich)] = [
                ich,
                "good",
                mapped_lat_int[idx],
                mapped_lon_int[idx],
                mapped_ele_int[idx],
                is_mapped_before_remove_bad[idx],
            ]
    else:
        df_ch.loc[str(ich)] = [ich, "bad", np.nan, np.nan, np.nan, False]
df_ch.to_csv(filenameFinal, index=None, sep=',', mode='w')

# %%
df_ch = pd.read_csv(filenameFinal)
print(df_ch)
print(df_ch[df_ch['mapped'] == True])

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fiber_lon, fiber_lat, "bo", label="Fiber geometry")
ax.plot(mapped_lon_int, mapped_lat_int, "go", label="Interpolated channels")
# ax.plot(good_lon_int,good_lat_int,"co",label="Good channels")
# ax.plot(bad_lon_int,bad_lat_int,"mo",label="Bad channels")
ax.plot(mapped_lon, mapped_lat, "ro", label="Tap-test mapped channels")
ax.plot(
    df_ch[df_ch['status'] == 'good']['longitude'],
    df_ch[df_ch['status'] == 'good']['latitude'],
    "k.",
    markersize=1,
    label="Located channels",
)
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.legend(fontsize=22, markerscale=5)
ax.grid()

# %% [markdown]
# # refine the results

# %% tags=[]
df_ch = pd.read_csv(filenameFinal)
df_ch

# %%
# "bad channels for channle loop, no locaiton at two ends,  and high noise (aerial) sections"

not_work_chs_str = "0:300,324:327,371:374,402:406,423:427,452:454,467:470,495:499,525:528,552:555,600:603,648:652,691:694,727:730,746,768,769:809,810,818:820,852:855,881:883,902:906,922:924,968,990:993,996:999,1029:1032,1064:1066,1089:1092,1046:1051,1146:1151,1163:1356,1382:1384,1430:1432,1458:1461,1521,1568:1570,1594:1597,1646:1659,1696,1859,1897:1899,1925:1931,1940:1957,1972:1975,2023:2025,2042,2065:2069,2072,2098:2101,2144:2147,2192:2196,2231,2233,2285:2288,2293:2296,2342,2403:2405,2412:2450,2482,2506:2554,2574:2587,2605:2625,2633:2654,2658,2688:2692,2705,2717:2722,2737:2741,2767,2797:2800,2818:2821,2868:2870,2906:2919,2932:2935,2971:2975,2985:2989,3016:3020,3076:3079,3119:3122,3176,3188:3191,3222:3225,3232,3267:3270,3297:3301,3321:3323,3343,3386:3389,3419:3422,3442:3445,3738:3742,3768:3770,3786,3797:3799,3832:3835,3857,3929:3932,3941:3945,3952,3969:3972,3991,4035:4038,4068,4096:4101,4204,4265:4269,4292:4296,4308:4312,4345:4348,4444,4481:4484,4496,4522:4524,4542,4580:4780"
not_work_chs = np.array([], dtype=int)
for val in not_work_chs_str.split(","):
    if ":" in val:
        strt_ch = int(val.split(":")[0])
        lst_ch = int(val.split(":")[1])
        not_work_chs = np.append(not_work_chs, np.arange(strt_ch, lst_ch))
    else:
        not_work_chs = np.append(not_work_chs, int(val))

# %%
for i in range(df_ch.shape[0]):
    if i in not_work_chs:
        df_ch.loc[i, 'status'] = "bad"
df_ch.to_csv(filenameFinal + ".remove.aerial.csv", index=None, sep=',', mode='w')

# %%
1279 in not_work_chs

# %%
df_ch.loc[1279]

# %%
df_ch[df_ch['channel'] >= 2905]

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(fiber_lon, fiber_lat, "bo", label="Fiber geometry")
ax.plot(mapped_lon_int, mapped_lat_int, "go", label="Interpolated channels")
# ax.plot(good_lon_int,good_lat_int,"co",label="Good channels")
# ax.plot(bad_lon_int,bad_lat_int,"mo",label="Bad channels")
ax.plot(mapped_lon, mapped_lat, "ro", label="Tap-test mapped channels")
ax.plot(
    df_ch[df_ch['status'] == 'good']['longitude'],
    df_ch[df_ch['status'] == 'good']['latitude'],
    "k.",
    markersize=2,
    label="Located good channels",
)
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.legend(fontsize=22, markerscale=5)
ax.grid()
