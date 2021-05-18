import readTrodesExtractedDataFile3
from scipy.ndimage.morphology import binary_dilation
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


# mkfigs = np.zeros((100, ))
mkfigs = np.ones((100, ))
# mkfigs[1] = True
# mkfigs[2] = True
# mkfigs[3] = True
mkfigs[0] = False
mkfigs[1] = False
mkfigs[2] = False
mkfigs[3] = False
mkfigs[4] = False
mkfigs[5] = False
# mkfigs[6] = False
mkfigs[7] = False
# mkfigs[7] = True
# mkfigs[9] = True

DOWN_DEFLECT_NOISE_THRESH = -20000
NOISE_BWD_EXCLUDE_SECS = 1
NOISE_FWD_EXCLUDE_SECS = 1

UP_DEFLECT_STIM_THRESH = 10000
UP_DEFLECT_STIM_PROMINENCE = 5000
SPK_BIN_SZ_MS = 50
PSTH_MARGIN_MS = 500
PSTH_MARGIN_SECS = float(PSTH_MARGIN_MS) / 1000.0
SPK_BIN_SZ_SECS = float(SPK_BIN_SZ_MS) / 1000.0

LFP_HZ = 1500


# data_file = "/home/wcroughan/data/20210108_162804/20210108_162804.spikes/20210108_162804.spikes_nt7.dat"
data_file = "/media/WDC2/B8/20210517_180358/20210517_180358.spikes/20210517_180358.spikes_nt2.dat"
spike_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(data_file)
spike_data = spike_data_dict['data']
# features = np.zeros((spike_data.size, 3))
features = np.zeros((spike_data.size, 4))

if np.any(mkfigs[0:3]) or np.any(mkfigs[8:]):
    next_pct = spike_data.size / 100

    for si, spike in enumerate(spike_data):
        ts = spike[0]
        wf1 = spike[1]
        wf2 = spike[2]
        wf3 = spike[3]
        wf4 = spike[4]

        # wfs = np.array((wf1, wf2, wf3))
        wfs = np.array((wf1, wf2, wf3, wf4))

        ntp = wf1.size
        maxpt = np.argmax(wfs)
        maxtet = maxpt // ntp
        maxslice = maxpt % ntp
        features[si, :] = wfs[:, maxslice]

        if si > next_pct:
            print("{}/{}".format(si, spike_data.size))
            next_pct += spike_data.size / 100

            if mkfigs[0]:
                plt.plot(wf1)
                plt.plot(wf2)
                plt.plot(wf3)
                plt.plot(wf4)
                plt.show()

    if mkfigs[1]:
        plt.subplot(321)
        plt.scatter(features[:, 0], features[:, 1], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(322)
        plt.scatter(features[:, 0], features[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(323)
        plt.scatter(features[:, 0], features[:, 3], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(324)
        plt.scatter(features[:, 1], features[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(325)
        plt.scatter(features[:, 1], features[:, 3], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(326)
        plt.scatter(features[:, 2], features[:, 3], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.show()

    # c1spks_idxs = np.logical_and(features[:, 1] > 1100, features[:, 2] < 1000)
    c1spks_idxs = features[:, 3] < features[:, 1] - 500
    c1spks = features[c1spks_idxs, :]
    c1ts = spike_data['time'][c1spks_idxs]
    if mkfigs[2]:
        plt.subplot(321)
        plt.scatter(c1spks[:, 0], c1spks[:, 1], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(322)
        plt.scatter(c1spks[:, 0], c1spks[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(323)
        plt.scatter(c1spks[:, 0], c1spks[:, 3], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(324)
        plt.scatter(c1spks[:, 1], c1spks[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(325)
        plt.scatter(c1spks[:, 1], c1spks[:, 3], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(326)
        plt.scatter(c1spks[:, 2], c1spks[:, 3], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.show()

# lfp_data_file = "/home/wcroughan/data/20210108_162804/20210108_162804.LFP/20210108_162804.LFP_nt7ch1.dat"
lfp_data_file = "/media/WDC2/B8/20210517_180358/20210517_180358.LFP/20210517_180358.LFP_nt2ch2.dat"
lfp_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_data_file)
lfp_data = lfp_data_dict['data']
lfp_data = lfp_data.astype(float)

# lfp_ts_file = "/home/wcroughan/data/20210108_162804/20210108_162804.LFP/20210108_162804.timestamps.dat"
lfp_ts_file = "/media/WDC2/B8/20210517_180358/20210517_180358.LFP/20210517_180358.timestamps.dat"
lfp_ts_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_ts_file)
lfp_ts = lfp_ts_dict['data']

if mkfigs[3]:
    print(lfp_ts)
    plt.plot(lfp_ts.astype(float) / 30000, lfp_data)
    plt.show()

if mkfigs[4]:
    LFP_HZ = 4
    NOISE_BWD_EXCLUDE_SECS = 0.5
    NOISE_FWD_EXCLUDE_SECS = 2
    lfp_data = np.sin(np.linspace(0, 2*np.pi, num=100))
    DOWN_DEFLECT_NOISE_THRESH = -0.9

is_noise = lfp_data < DOWN_DEFLECT_NOISE_THRESH
NOISE_BWD_EXCLUDE_FRAMES = int(NOISE_BWD_EXCLUDE_SECS * LFP_HZ)
NOISE_FWD_EXCLUDE_FRAMES = int(NOISE_FWD_EXCLUDE_SECS * LFP_HZ)
STRSZHLF = max(NOISE_BWD_EXCLUDE_FRAMES, NOISE_FWD_EXCLUDE_FRAMES)
dilstr = np.zeros((STRSZHLF * 2))
dilstr[STRSZHLF-NOISE_BWD_EXCLUDE_FRAMES:STRSZHLF+NOISE_FWD_EXCLUDE_FRAMES] = 1
noise_mask = binary_dilation(is_noise, structure=dilstr).astype(bool)

if mkfigs[4]:
    plt.subplot(121)
    plt.plot(dilstr)
    plt.subplot(122)
    plt.plot(is_noise)
    plt.plot(noise_mask.astype(int))
    plt.show()

    exit()

if mkfigs[5]:
    ts = lfp_ts.astype(float) / 30000
    plt.plot(ts, lfp_data)
    plt.plot(ts, noise_mask.astype(float) * -np.max(lfp_data))
    plt.scatter(ts[is_noise], lfp_data[is_noise], c="#ff7f0e")
    plt.show()

peaks, props = signal.find_peaks(lfp_data, height=UP_DEFLECT_STIM_THRESH,
                                 prominence=UP_DEFLECT_STIM_PROMINENCE)

if mkfigs[6]:
    ts = lfp_ts.astype(float) / 30000
    plt.plot(ts, lfp_data)
    plt.scatter(ts[peaks], props['peak_heights'], c="#ff7f0e")

    t1 = ts[peaks[0]]
    t2 = ts[peaks[-1]]
    print("{} ({}-{}), {} total stims = {} Hz".format(
        t2-t1, t1, t2, len(peaks), float(len(peaks))/float(t2-t1)
    ))

    plt.show()

stim_artifact_bool = np.logical_not(noise_mask[peaks])
peak_heights = props['peak_heights']
stimpeaks = peaks[stim_artifact_bool]
stimpeak_heights = peak_heights[stim_artifact_bool]

if mkfigs[7]:
    ts = lfp_ts.astype(float) / 30000
    plt.plot(ts, lfp_data)
    plt.scatter(ts[stimpeaks], stimpeak_heights, c="#ff7f0e")
    plt.show()

if mkfigs[8]:
    ts = lfp_ts.astype(float) / 30000
    plt.plot(ts, lfp_data)
    plt.scatter(c1ts.astype(float) / 30000, np.ones_like(c1ts), c="#ff7f0e")
    plt.show()

NUM_PSTH_BINS = int(PSTH_MARGIN_MS / SPK_BIN_SZ_MS)
psth = np.zeros((stimpeaks.size, 2*NUM_PSTH_BINS))
spike_times = c1ts.astype(float) / 30000
stim_times = lfp_ts[stimpeaks].astype(float) / 30000

NUM_LFP_PSTH_BINS = int(PSTH_MARGIN_SECS * LFP_HZ)
lfp_psth = np.zeros((stimpeaks.size, 2*NUM_LFP_PSTH_BINS))
for si, (st, stidx) in enumerate(zip(stim_times, stimpeaks)):
    bins = np.linspace(st - PSTH_MARGIN_SECS, st + PSTH_MARGIN_SECS, num=2*NUM_PSTH_BINS+1)
    h, b = np.histogram(spike_times, bins=bins)
    psth[si, :] = h

    lfp_psth[si, :] = lfp_data[stidx-NUM_LFP_PSTH_BINS:stidx+NUM_LFP_PSTH_BINS]

avg_fr_psth = np.mean(psth, axis=0)
std_fr_psth = np.std(psth, axis=0)
avg_lfp_psth = np.mean(lfp_psth, axis=0)
std_lfp_psth = np.std(lfp_psth, axis=0)
if mkfigs[9]:
    x1 = np.linspace(-PSTH_MARGIN_SECS, PSTH_MARGIN_SECS, NUM_PSTH_BINS*2)
    x2 = np.linspace(-PSTH_MARGIN_SECS, PSTH_MARGIN_SECS, NUM_LFP_PSTH_BINS*2)
    y1 = avg_fr_psth - np.min(avg_fr_psth)
    y2 = avg_lfp_psth - np.min(avg_lfp_psth)
    y1 = y1 / np.max(y1)
    y2 = y2 / np.max(y2)

    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.show()
