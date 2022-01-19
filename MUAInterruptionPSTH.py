import sys
import numpy as np
import os
import matplotlib.pyplot as plt

import readTrodesExtractedDataFile3
from SpikeCalibration import MUAClusterFunc, runTheThing, makeClusterFuncFromFile, loadTrodesClusters
from BTData import BTData

REMAKE_PSTH = True
MANUAL_DIAG_POINT_INPUT = True
USE_MUA = False

if len(sys.argv) == 2:
    animal_name = sys.argv[1]
else:
    animal_name = 'B13'
print("Generating MUA PSTH for animal " + animal_name)


def makePSTH(animal_name):
    if animal_name == "B13":
        data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
        output_dir = "/media/WDC7/B13/figs/"
        if USE_MUA:
            diagPoint = np.array([3408, 682, 3053, 690], dtype=float)
            clusterPolygons = None

        else:
            clusterFileName = "/media/WDC7/B13/20211206_151659/20211206_151659.trodesClusters"
            spk_tet = 7
            clusterIndex = 0

            clfunc = makeClusterFuncFromFile(clusterFileName, spk_tet-1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName)
            clusterPolygons = clusters[spk_tet-1][clusterIndex]
    elif animal_name == "B14":
        data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
        output_dir = "/media/WDC7/B14/figs/"
        diagPoint = None
    elif animal_name == "Martin":
        data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
        output_dir = "/media/WDC7/Martin/figs/"
        diagPoint = None
    else:
        raise Exception("Unknown rat " + animal_name)

    print("Loading data for " + animal_name)
    ratData = BTData()
    ratData.loadFromFile(data_filename)
    allSessions = ratData.getSessions()

    ctrlPSTH = np.empty((0, 40))
    swrPSTH = np.empty((0, 40))

    for sesh in allSessions:
        lfp_file = sesh.bt_lfp_fnames[-1]
        lfp_timestamp_file = ".".join(lfp_file.split(".")[0:-2]) + ".timestamps.dat"
        output_fname = sesh.name + "_psth.png"
        spike_file = lfp_file.replace("LFP", "spikes")[0:-7] + ".dat"
        print(sesh.name, lfp_file, lfp_timestamp_file, output_fname, spike_file, sep="\n")

        # If necessary, generate spike file
        if not os.path.exists(spike_file):
            file_str = spike_file.split("spikes")[0] + "rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportspikes"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportspikes -rec " + file_str
            else:
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportspikes -rec " + file_str
            print(syscmd)
            os.system(syscmd)

        if MANUAL_DIAG_POINT_INPUT and USE_MUA:
            spike_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(spike_file)
            spike_data = spike_data_dict['data']

            features = np.zeros((spike_data.size, 4))
            chmaxes = np.zeros((spike_data.size, 4))
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

            CHLIM = 5000
            step = 20
            plt.subplot(321)
            plt.scatter(features[::step, 0], features[::step, 1], marker=".", s=0.5)
            if diagPoint is not None:
                plt.scatter(diagPoint[0], diagPoint[1], c="red")
            plt.xlim(0, CHLIM)
            plt.ylim(0, CHLIM)
            plt.subplot(322)
            plt.scatter(features[::step, 0], features[::step, 2], marker=".", s=0.5)
            if diagPoint is not None:
                plt.scatter(diagPoint[0], diagPoint[2], c="red")
            plt.xlim(0, CHLIM)
            plt.ylim(0, CHLIM)
            plt.subplot(323)
            plt.scatter(features[::step, 0], features[::step, 3], marker=".", s=0.5)
            if diagPoint is not None:
                plt.scatter(diagPoint[0], diagPoint[3], c="red")
            plt.xlim(0, CHLIM)
            plt.ylim(0, CHLIM)
            plt.subplot(324)
            plt.scatter(features[::step, 1], features[::step, 2], marker=".", s=0.5)
            if diagPoint is not None:
                plt.scatter(diagPoint[1], diagPoint[2], c="red")
            plt.xlim(0, CHLIM)
            plt.ylim(0, CHLIM)
            plt.subplot(325)
            plt.scatter(features[::step, 1], features[::step, 3], marker=".", s=0.5)
            if diagPoint is not None:
                plt.scatter(diagPoint[1], diagPoint[3], c="red")
            plt.xlim(0, CHLIM)
            plt.ylim(0, CHLIM)
            plt.subplot(326)
            plt.scatter(features[::step, 2], features[::step, 3], marker=".", s=0.5)
            if diagPoint is not None:
                plt.scatter(diagPoint[2], diagPoint[3], c="red")
            plt.xlim(0, CHLIM)
            plt.ylim(0, CHLIM)
            plt.show()

            txtDiagPoints = input("diag point sep by spaces:")
            if len(txtDiagPoints) > 0:
                diagPoint = np.array([float(v) for v in txtDiagPoints.split(" ")])
            print(diagPoint)

        assert not USE_MUA or diagPoint is not None

        if USE_MUA:
            def clfunc(f, _, __): return MUAClusterFunc(f, diagPoint)

        fr_psth = runTheThing(spike_file, lfp_file, lfp_timestamp_file,
                              output_fname, clfunc, makeFigs=MANUAL_DIAG_POINT_INPUT or not USE_MUA, clusterPolygons=clusterPolygons)

        if sesh.isRippleInterruption:
            swrPSTH = np.vstack((swrPSTH, fr_psth))
        else:
            ctrlPSTH = np.vstack((ctrlPSTH, fr_psth))

    np.savetxt(os.path.join(output_dir, "swrPSTH"), swrPSTH)
    np.savetxt(os.path.join(output_dir, "ctrlPSTH"), ctrlPSTH)


if __name__ == "__main__":
    if animal_name == "B13":
        data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
        output_dir = "/media/WDC7/B13/figs/"
    elif animal_name == "B14":
        data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
        output_dir = "/media/WDC7/B14/figs/"
    elif animal_name == "Martin":
        data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
        output_dir = "/media/WDC7/Martin/figs/"
    else:
        raise Exception("Unknown rat " + animal_name)
    swrPSTHFile = os.path.join(output_dir, "swrPSTH")
    if REMAKE_PSTH or not os.path.exists(swrPSTHFile):
        makePSTH(animal_name)
    ctrlPSTHFile = os.path.join(output_dir, "ctrlPSTH")

    swrPSTH = np.loadtxt(swrPSTHFile)
    ctrlPSTH = np.loadtxt(ctrlPSTHFile)

    plt.clf()
    xs = np.linspace(-1, 1, 40)
    plt.plot(xs, np.nanmean(swrPSTH, axis=0))
    plt.plot(xs + 0.2, np.nanmean(ctrlPSTH, axis=0))
    plt.plot([0, 0], (plt.ylim()))
    output_fname = os.path.join(output_dir, animal_name + "_psth.png")
    plt.savefig(output_fname, dpi=800)
    # plt.show()
