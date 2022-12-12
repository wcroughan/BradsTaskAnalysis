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
        elif os.path.exists("/home/fosterlab/Software/Trodes223/exportLFP"):
            syscmd = f"/home/fosterlab/Software/Trodes223/exportLFP -rec {recFileName}"
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
    # print(lfpData[1])
    # lfpV = lfpData[0][1]['voltage']
    # lfpTimestamps = lfpData[0][0]['time']
    lfpV = lfpData[1].astype(float)
    lfpTimestamps = lfpData[0].astype(float)

    lfp_deflections = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=6000.0, distance=int(0.05 * LFP_SAMPLING_RATE))
    interruptionIdxs = lfp_deflections[0]

    _, ripplePower, _, _ = getRipplePower(lfpV, lfp_deflections=interruptionIdxs)
    ripStarts, ripLens, ripPeakIdxs, ripPeakAmps, _ = detectRipples(ripplePower)
    
    plt.plot(lfpTimestamps, lfpV)
    plt.plot(lfpTimestamps, ripplePower)
    # plt.scatter(lfpTimestamps[interruptionIdxs], lfpV[interruptionIdxs], color="orange")
    plt.scatter(lfpTimestamps[interruptionIdxs], np.zeros_like(interruptionIdxs) + 10000, color="orange")
    for ri in range(len(ripStarts)):
        rs = ripStarts[ri]
        rl = ripLens[ri]
        ra = ripPeakAmps[ri]

        plt.plot([lfpTimestamps[rs], lfpTimestamps[rs+rl]], [ra, ra], "k")

    print(f"{len(interruptionIdxs)} interruptions found")
    plt.show()

if __name__ == "__main__":
    # recName = "/media/fosterlab/WDC8/B16/bradtasksessions/20220919_122046/20220919_122046.rec"
    # detTet = 4
    # baseTet = 7

    # recName = "/media/fosterlab/WDC8/B18/20221102_100012/20221102_100012.rec"
    # detTet = 8
    # baseTet = 7

    # recName = "/media/fosterlab/WDC8/B16/20221102_092531/20221102_092531.rec"
    # detTet = 8
    # baseTet = 7

    # recName = "/media/fosterlab/WDC8/B17/20221118_114509/20221118_114509.rec"
    # recName = "/media/fosterlab/WDC8/B17/20221117_154325/20221117_154325.rec"
    # recName = "/media/fosterlab/WDC8/B17/20221116_094757/20221116_094757.rec"
    recName = "/media/fosterlab/WDC8/B17/20221115_095659/20221115_095659.rec"

    detTet = 6
    baseTet = 5

    lfpData, baselineLfpData = loadLFPData(recName, detTet, baseTet)
    checkLFP(lfpData)