import numpy as np
import matplotlib.pyplot as plt

import readTrodesExtractedDataFile3

mkfigs = np.zeros((100, ))
# mkfigs[1] = True
# mkfigs[2] = True
mkfigs[3] = True

DOWN_DEFLECT_NOISE_THRESH = -5000
NOISE_BWD_EXCLUDE_SECS = 1
NOISE_FWD_EXCLUDE_SECS = 1

UP_DEFLECT_STIM_THRESH = 12000
SPK_BIN_SZ = 1


data_file = "/home/wcroughan/data/20210108_162804/20210108_162804.spikes/20210108_162804.spikes_nt7.dat"
spike_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(data_file)
spike_data = spike_data_dict['data']
features = np.zeros((spike_data.size, 3))
# features = np.zeros((spike_data.size, 4))

if np.any(mkfigs[0:3]):
    next_pct = spike_data.size / 100

    for si, spike in enumerate(spike_data):
        ts = spike[0]
        wf1 = spike[1]
        wf2 = spike[2]
        wf3 = spike[3]
        # wf4 = spike[4]

        wfs = np.array((wf1, wf2, wf3))
        # wfs = np.array((wf1, wf2, wf3, wf4))

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
                plt.show()

    if mkfigs[1]:
        plt.subplot(311)
        plt.scatter(features[:, 0], features[:, 1], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(312)
        plt.scatter(features[:, 0], features[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(313)
        plt.scatter(features[:, 1], features[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.show()

    c1spks = features[np.logical_and(features[:, 1] > 1100, features[:, 2] < 1000), :]
    if mkfigs[2]:
        plt.subplot(311)
        plt.scatter(c1spks[:, 0], c1spks[:, 1], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(312)
        plt.scatter(c1spks[:, 0], c1spks[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.subplot(313)
        plt.scatter(c1spks[:, 1], c1spks[:, 2], marker=".", s=0.5)
        plt.xlim(0, 2500)
        plt.ylim(0, 2500)
        plt.show()

lfp_data_file = "/home/wcroughan/data/20210108_162804/20210108_162804.LFP/20210108_162804.LFP_nt7ch1.dat"
lfp_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_data_file)
lfp_data = lfp_data_dict['data']

lfp_ts_file = "/home/wcroughan/data/20210108_162804/20210108_162804.LFP/20210108_162804.timestamps.dat"
lfp_ts_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_ts_file)
lfp_ts = lfp_ts_dict['data']

if mkfigs[3]:
    print(lfp_ts)
    plt.plot(lfp_ts.astype(float) / 30000, lfp_data)
    plt.show()
