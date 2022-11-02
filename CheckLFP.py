import os
import MountainViewIO
import glob
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from UtilFunctions import getRipplePower, detectRipples
from consts import LFP_SAMPLING_RATE

def loadLFPData(recFileName, detTet, baseTet):
    fileStartString = ".".join(recFileName.split(".")[:-1])
    runName = fileStartString.split("/")[-1]
    lfpdir = fileStartString + ".LFP"
    if not os.path.exists(lfpdir):
        print(lfpdir, "doesn't exists, gonna try and extract the LFP")
        if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
            syscmd = f"/home/wcroughan/Software/Trodes21/exportLFP -rec {recFileName}"
        elif os.path.exists("/home/wcroughan/Software/Trodes21/linux/exportLFP"):
            syscmd = f"/home/wcroughan/Software/Trodes21/linux/exportLFP -rec {recFileName}"
        else:
            syscmd = f"/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804/exportLFP -rec {recFileName}"
        print(syscmd)
        os.system(syscmd)

    gl = lfpdir + "/" + runName + ".LFP_nt" + str(detTet) + "ch*.dat"
    lfpfilelist = glob.glob(gl)
    lfpfilename = lfpfilelist[0]
    lfpData = MountainViewIO.loadLFP(data_file=lfpfilename)

    gl = lfpdir + "/" + runName + ".LFP_nt" + str(baseTet) + "ch*.dat"
    lfpfilelist = glob.glob(gl)
    lfpfilename = lfpfilelist[0]
    baselineLfpData = MountainViewIO.loadLFP(data_file=lfpfilename)

    return lfpData, baselineLfpData

def checkLFP(lfpData):
    lfpV = lfpData[0][1]['voltage']
    lfpTimestamps = lfpData[0][0]['time']

    lfp_deflections = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=6000.0, distance=int(0.05 * LFP_SAMPLING_RATE))
    interruptionIdxs = lfp_deflections[0]

    _, ripplePower, _, _ = getRipplePower(lfpV, lfp_deflections=interruptionIdxs)
    ripStarts, ripLens, ripPeakIdxs, ripPeakAmps, _ = detectRipples(ripplePower)
    
    plt.plot(lfpTimestamps, lfpV)
    plt.plot(lfpTimestamps, ripplePower)
    plt.scatter(lfpTimestamps[interruptionIdxs], lfpV[interruptionIdxs])
    for rs, rl in zip(ripStarts, ripLens):
        plt.plot([lfpTimestamps[rs], lfpTimestamps[rs+rl]], [0, 0], "k")

    plt.show()

if __name__ == "__main__":
    recName = "/media/fosterlab/WDC8/B16/bradtasksessions/20220919_122046/20220919_122046.rec"
    detTet = 4
    baseTet = 7
    lfpData, baselineLfpData = loadLFPData(recName, detTet, baseTet)
    checkLFP(lfpData)