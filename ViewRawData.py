import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import itertools

import readTrodesExtractedDataFile3
from MDAParse import parseMDAFile

mkfigs = np.zeros((100, ))
# mkfigs[0] = True
mkfigs[6] = True
OFFSET_FROM_REC_START = False

output_dir = '/home/wcroughan/data/B2/figs/'

# data_file = "/home/wcroughan/data/20210108_162804/20210108_162804.spikes/20210108_162804.spikes_nt7.dat"
# spike_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(data_file)
# spike_data = spike_data_dict['data']
# features = np.zeros((spike_data.size, 3))
# # features = np.zeros((spike_data.size, 4))

# lfp_data_file = "/media/WDC5/20210112_135933/20210112_135933.LFP/20210112_135933.LFP_nt7ch1.dat"
lfp_data_file = "/home/wcroughan/data/20210113_143855/20210113_143855.LFP_nt7ch1.dat"
lfp_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_data_file)
lfp_data = lfp_data_dict['data']

# lfp_ts_file = "/media/WDC5/20210112_135933/20210112_135933.LFP/20210112_135933.timestamps.dat"
lfp_ts_file = "/home/wcroughan/data/20210113_143855/20210113_143855.timestamps.dat"
lfp_ts_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_ts_file)
lfp_ts = lfp_ts_dict['data']
lfp_ts = np.array(list(itertools.chain(*lfp_ts)))
if OFFSET_FROM_REC_START:
    lfp_ts_mins = (lfp_ts - lfp_ts[0]).astype(float) / 30000 / 60
else:
    lfp_ts_mins = lfp_ts.astype(float) / 30000 / 60

if mkfigs[6]:
    plt.plot(lfp_ts_mins, lfp_data.astype(float))
    # fname = "raw.png"
    # plt.savefig(os.path.join(output_dir, fname), dpi=800)
    plt.show()
    exit()


# turned on
# tstart = 2600 / 60
# tend = 2700 / 60
# turned back off
tstart = 6
tend = 10
t1 = np.searchsorted(lfp_ts_mins, tstart)
t2 = np.searchsorted(lfp_ts_mins, tend)
lfpshort = lfp_data[t1:t2].astype(float)
lfp_tshort = lfp_ts_mins[t1:t2]

if mkfigs[0]:
    fname = "raw_short.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=800)
    plt.plot(lfp_tshort, lfpshort)
    plt.show()

if mkfigs[2]:
    fs = 1500
    f, t, Sxx = signal.spectrogram(lfpshort, fs)
    # print(f)
    # vmin = np.percentile(Sxx, 10)
    # vmax = np.percentile(Sxx, 98)
    vmin = 0
    vmax = 200
    plt.pcolormesh(t, f, Sxx, shading='gouraud', vmin=vmin, vmax=vmax)
    fname = "raw.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=800)
    plt.show()
    exit()

if mkfigs[3]:
    fs = 1500
    f, t, Sxx = signal.spectrogram(lfp_data.astype(float), fs)
    # print(f)
    vmin = np.percentile(Sxx, 10)
    vmax = np.percentile(Sxx, 98)
    plt.pcolormesh(t, f, Sxx, shading='gouraud', vmin=vmin, vmax=vmax)
    plt.show()
    exit()


# if mkfigs[0]:
#     print(lfp_ts)
#     plt.plot(lfp_ts.astype(float) / 30000, lfp_data)
#     plt.show()


if mkfigs[4] or mkfigs[5]:
    # PSTH around on and off times of 60Hz power and 0-40Hz power
    # on_times_mins = [27, 43, 74]
    # on_times_secs = [40, 45, 50]
    # off_times_mins = [28, 56, 89]
    # off_times_secs = [10, 20, 0]

    # on_times_mins = [19, 32]
    # on_times_secs = [15, 0]
    on_times_mins = [19, 32]
    on_times_secs = [15, 0]
    off_times_mins = [13, 24, 37]
    off_times_secs = [15, 45, 30]

    lfp_tsecs = lfp_ts_mins * 60

    PSTH_WINDOW = 120

    on_p1 = np.array([])
    on_p2 = np.array([])

    frange_1l = 48
    frange_1u = 72
    frange_2l = 0
    frange_2u = 48
    for ti in range(len(on_times_mins)):
        ot = on_times_mins[ti] * 60 + on_times_secs[ti]
        t1 = np.searchsorted(lfp_tsecs, ot - PSTH_WINDOW)
        t2 = np.searchsorted(lfp_tsecs, ot + PSTH_WINDOW)
        fs = 1500
        f, t, s = signal.spectrogram(lfp_data[t1:t2].astype(float), fs)

        r1i = np.where((f >= frange_1l) & (f <= frange_1u))[0]
        r2i = np.where((f >= frange_2l) & (f <= frange_2u))[0]

        if on_p1.size == 0:
            on_p1 = np.sum(s[r1i, :], axis=0)
            on_p2 = np.sum(s[r2i, :], axis=0)
        else:
            on_p1 = np.vstack((on_p1, np.sum(s[r1i, :], axis=0)))
            on_p2 = np.vstack((on_p2, np.sum(s[r2i, :], axis=0)))

        if mkfigs[5]:
            # vmin = np.percentile(s, 10)
            # vmax = np.percentile(s, 98)
            vmin = 0
            vmax = 200
            print("color range: {}-{}".format(vmin, vmax))
            plt.pcolormesh(t, f, s, shading='gouraud', vmin=vmin, vmax=vmax)
            plt.title("on #{}".format(ti))
            plt.plot([120, 120], [0, np.max(f)], c="#ff0000")
            fname = "spec_on_{}.png".format(ti)
            plt.savefig(os.path.join(output_dir, fname), dpi=800)
            plt.show()

    off_p1 = np.array([])
    off_p2 = np.array([])

    frange_1l = 48
    frange_1u = 72
    frange_2l = 0
    frange_2u = 48
    for ti in range(len(off_times_mins)):
        ot = off_times_mins[ti] * 60 + off_times_secs[ti]
        t1 = np.searchsorted(lfp_tsecs, ot - PSTH_WINDOW)
        t2 = np.searchsorted(lfp_tsecs, ot + PSTH_WINDOW)
        fs = 1500
        f, t, s = signal.spectrogram(lfp_data[t1:t2].astype(float), fs)

        r1i = np.where((f >= frange_1l) & (f <= frange_1u))[0]
        r2i = np.where((f >= frange_2l) & (f <= frange_2u))[0]

        if off_p1.size == 0:
            off_p1 = np.sum(s[r1i, :], axis=0)
            off_p2 = np.sum(s[r2i, :], axis=0)
        else:
            off_p1 = np.vstack((off_p1, np.sum(s[r1i, :], axis=0)))
            off_p2 = np.vstack((off_p2, np.sum(s[r2i, :], axis=0)))

        if mkfigs[5]:
            # vmin = np.percentile(s, 10)
            # vmax = np.percentile(s, 98)
            vmin = 0
            vmax = 200
            print("color range: {}-{}".format(vmin, vmax))
            plt.pcolormesh(t, f, s, shading='gouraud', vmin=vmin, vmax=vmax)
            plt.title("off #{}".format(ti))
            plt.plot([120, 120], [0, np.max(f)], c="#ff0000")
            fname = "spec_off_{}.png".format(ti)
            plt.savefig(os.path.join(output_dir, fname), dpi=800)
            plt.show()

    if mkfigs[4]:
        nt = on_p1.shape[1]
        px = np.linspace(-PSTH_WINDOW, PSTH_WINDOW, nt, endpoint=False)
        plt.subplot(221)
        plt.plot(px, on_p1.T)
        plt.title("stim on, {}-{}Hz".format(frange_1l, frange_1u))
        basemax = np.max(
            np.hstack((on_p1[:, (nt // 5):(2*nt // 5)], on_p1[:, (3*nt // 5):(4*nt // 5)])))
        plt.ylim((0, basemax))
        plt.subplot(222)
        plt.plot(px, off_p1.T)
        plt.title("stim off, {}-{}Hz".format(frange_1l, frange_1u))
        # basemax = np.percentile(
        # np.hstack((off_p1[:, (nt // 5):(2*nt // 5)], off_p1[:, (3*nt // 5):(4*nt // 5)])), 99)
        plt.ylim((0, basemax))
        plt.subplot(223)
        plt.plot(px, on_p2.T)
        plt.title("stim on, {}-{}Hz".format(frange_2l, frange_2u))
        basemax = np.max(
            np.hstack((on_p2[:, (nt // 5):(2*nt // 5)], on_p2[:, (3*nt // 5):(4*nt // 5)])))
        plt.ylim((0, basemax / 2.0))
        plt.subplot(224)
        plt.plot(px, off_p2.T)
        plt.title("stim off, {}-{}Hz".format(frange_2l, frange_2u))
        # basemax = np.percentile(
        # np.hstack((off_p2[:, (nt // 5):(2*nt // 5)], off_p2[:, (3*nt // 5):(4*nt // 5)])), 99)
        plt.ylim((0, basemax / 2.0))
        fname = "powers.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=800)
        plt.show()
        exit()


# raw_data_file = "/media/WDC5/20210112_135933/20210112_135933.mda/20210112_135933.nt7.mda"
raw_data_file = "/home/wcroughan/data/20210113_143855/20210113_143855.nt7.mda"
raw_data = parseMDAFile(raw_data_file)
print("Got raw data with shape {}".format(raw_data.shape))

# raw_ts_file = "/media/WDC5/20210112_135933/20210112_135933.mda/20210112_135933.timestamps.mda"
raw_ts_file = "/home/wcroughan/data/20210113_143855/20210113_143855.timestamps.mda"
raw_ts = parseMDAFile(raw_ts_file)
ts_mins = (raw_ts - raw_ts[0]).astype(float) / 30000 / 60

if mkfigs[1]:
    plt.plot(ts_mins, raw_data[0, :])
    plt.show()

tstart = 27
tend = 29
t1 = np.searchsorted(ts_mins, tstart)
t2 = np.searchsorted(ts_mins, tend)
ch1short = raw_data[0, t1:t2]
tshort = ts_mins[t1:t2]

if mkfigs[0]:
    plt.plot(tshort, ch1short)
    plt.show()

if mkfigs[2]:
    fs = 30000
    f, t, Sxx = signal.spectrogram(ch1short, fs)
    print(f)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.show()
