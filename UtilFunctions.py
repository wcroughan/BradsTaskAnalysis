import numpy as np
from consts import allWellNames, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE
import csv
import glob
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from itertools import groupby
from datetime import date
import sys
import os
from typing import List, Dict, Tuple, Optional
import functools
import time


def findDataDir(possibleDataDirs=["/media/WDC8/",
                                  "/media/WDC6/",
                                  "/media/fosterlab/WDC6/",
                                  "/home/wcroughan/data/"]) -> str:
    for dd in possibleDataDirs:
        if os.path.exists(dd):
            return dd

    return None


def parseCmdLineAnimalNames(default: Optional[List[str]] = None) -> List[str]:
    if len(sys.argv) >= 2:
        if len(sys.argv) == 2 and sys.argv[1] == "new":
            return ["B16", "B17", "B18"]
        elif len(sys.argv) == 2 and sys.argv[1] == "old":
            return ["Martin", "B13", "B14"]
        elif len(sys.argv) == 2 and sys.argv[1] == "all":
            return ["Martin", "B13", "B14", "B16", "B17", "B18"]

        return sys.argv[1:]
    else:
        return default


def readWellCoordsFile(wellCoordsFile: str) -> Dict[str, Tuple[int, int]]:
    # For some reason json saving and loading turns the keys into strings, just going to change that here so
    # it's consistent
    with open(wellCoordsFile, 'r') as wcf:
        wellCoordsMap = {}
        csvReader = csv.reader(wcf)
        for dataRow in csvReader:
            try:
                wellCoordsMap[str(int(dataRow[0]))] = (
                    int(dataRow[1]), int(dataRow[2]))
            except Exception as err:
                if dataRow[1] != '':
                    print(err)

        return wellCoordsMap


def readRawPositionData(data_filename):
    try:
        with open(data_filename, 'rb') as datafile:
            dt = np.dtype([('timestamp', np.uint32), ('x1', np.uint16),
                           ('y1', np.uint16), ('x2', np.uint16), ('y2', np.uint16)])
            lineText = ""
            max_iter = 8
            iter = 0
            settings = {}
            while lineText != b'<end settings>\n':
                lineText = datafile.readline().lower()
                if lineText != b'<end settings>\n' and lineText != b'<start settings>\n':
                    ss = str(lineText).split(":")
                    if len(ss) == 2:
                        settings[str(ss[0])[2:].strip()] = str(ss[1])[0:-3].strip()
                # print(lineText)
                iter += 1
                if iter > max_iter:
                    raise Exception
            return settings, np.fromfile(datafile, dtype=dt)
    except Exception as err:
        print(err)
        return None, None


def quickPosPlot(tt, xx, yy, title, irange=None):
    if True:
        if irange is not None:
            xx = xx[irange[0]:irange[1]]
            yy = yy[irange[0]:irange[1]]
            tt = tt[irange[0]:irange[1]]
        plt.plot(xx, yy)
        plt.title(title)
        plt.show()
        plt.plot(tt, xx)
        plt.plot(tt, yy)
        plt.show()
        plt.scatter(tt, xx)
        plt.scatter(tt, yy)
        plt.show()


def timeStrForTrodesTimestamp(ts):
    if ts is None:
        return "[None]"
    s = ts // TRODES_SAMPLING_RATE
    ssecs = str(s % 60)
    if len(ssecs) == 1:
        ssecs = "0" + ssecs
    s = s // 60
    smins = str(s % 60)
    if len(smins) == 1:
        smins = "0" + smins
    shrs = s // 60
    return f"{shrs}:{smins}:{ssecs}"


def getWellPosCoordinates(wellName: int) -> Tuple[float, float]:
    assert wellName in allWellNames
    y = float(wellName // 8) + 0.5
    x = float(wellName % 8 - 2) + 0.5
    return x, y


# def detectStimArtifacts(lfp_data):
#     deflection_metrics = signal.find_peaks(np.abs(np.diff(lfp_data,
#                                                           prepend=lfp_data[0])), height=DEFLECTION_THRESHOLD_LO,
#                                            distance=MIN_ARTIFACT_DISTANCE)

#     return deflection_metrics[0]


def getRipplePower(lfpData, omitArtifacts=True, causalSmoothing=False,
                   lfpDeflections=None, meanPower=None, stdPower=None,
                   showPlot=False, rippleFilterBand=(150, 250), rippleFilterOrder=4,
                   skipTimePointsFoward=int(0.075 * LFP_SAMPLING_RATE),
                   skipTimePointsBackward=int(0.02 * LFP_SAMPLING_RATE)):
    """
    Get ripple power in LFP
    """
    lfpDataCopy = lfpData.copy().astype(float)

    if (meanPower is None) != (stdPower is None):
        raise Exception("meanPower and stdPower must both be provided or None")

    if lfpDeflections is None:
        if omitArtifacts:
            raise Exception("this hasn't been updated")
            # # Remove all the artifacts in the raw ripple amplitude data
            # deflectionMetrics = signal.find_peaks(np.abs(np.diff(lfpData,
            #                                                       prepend=lfpData[0])),
            # height=DEFLECTION_THRESHOLD_LO,
            #                                        distance=MIN_ARTIFACT_DISTANCE)
            # lfpDeflections = deflectionMetrics[0]

    # After this preprocessing, clean up the data if needed.
    lfpMask = np.zeros_like(lfpDataCopy)
    if lfpDeflections is not None:
        for artifactIdx in lfpDeflections:
            if artifactIdx < 0 or artifactIdx > len(lfpData):
                continue
            cleanupStart = max(0, artifactIdx - skipTimePointsBackward)
            cleanupFinish = min(len(lfpData), artifactIdx +
                                skipTimePointsFoward)
            # lfpDataCopy[cleanupStart:cleanupFinish] = np.nan
            lfpMask[cleanupStart:cleanupFinish] = 1

        # print("LFP mask letting {} of signal through".format(
        #     1 - (np.count_nonzero(lfpMask) / len(lfpMask))))

    nyqFreq = LFP_SAMPLING_RATE * 0.5
    loCutoff = rippleFilterBand[0] / nyqFreq
    hiCutoff = rippleFilterBand[1] / nyqFreq
    pl, ph = signal.butter(rippleFilterOrder, [loCutoff, hiCutoff], btype='band')
    if causalSmoothing:
        rippleAmplitude = signal.lfilter(pl, ph, lfpDataCopy)
    else:
        rippleAmplitude = signal.filtfilt(pl, ph, lfpDataCopy)

    rippleAmplitude[lfpMask == 1] = np.nan

    # Smooth this data and get ripple power
    # smoothingWindowLength = RIPPLE_POWER_SMOOTHING_WINDOW * LFP_SAMPLING_RATE
    # smoothingWeights = np.ones(int(smoothingWindowLength))/smoothingWindowLength
    # ripplePower = np.convolve(np.abs(rippleAmplitude), smoothingWeights, mode='same')

    # Use a Gaussian kernel for filtering - Make the Kernel Causal bu keeping only one half of the values
    smoothingWindowLength = 10
    if causalSmoothing:
        # In order to no have NaN values affect the filter output, create a copy with the artifacts
        rippleAmplitudeCopy = rippleAmplitude.copy()

        halfSmoothingSignal = \
            np.exp(-np.square(np.linspace(0, -4 * smoothingWindowLength, 4 *
                                          smoothingWindowLength)) / (
                2 * smoothingWindowLength * smoothingWindowLength))
        smoothingSignal = np.concatenate(
            (np.zeros_like(halfSmoothingSignal), halfSmoothingSignal), axis=0)
        ripplePower = signal.convolve(np.abs(rippleAmplitudeCopy),
                                      smoothingSignal, mode='same') / np.sum(smoothingSignal)
        ripplePower[np.isnan(rippleAmplitude)] = np.nan
    else:
        ripplePower = gaussian_filter(np.abs(rippleAmplitude), smoothingWindowLength)

    # Get the mean/standard deviation for ripple power and adjust for those
    if meanPower is None:
        meanPower = np.nanmean(ripplePower)
        stdPower = np.nanstd(ripplePower)
    zpower = (ripplePower - meanPower) / stdPower

    if showPlot:
        lc = lfpData.copy()
        lc = lc / np.nanmax(np.abs(lc)) * 10
        rc = np.array([min(10, p) for p in zpower])
        ts = np.linspace(0, len(lc) / 1500, len(lc))
        plt.plot(ts, rc, c="orange", zorder=0)
        plt.plot(ts, lc, c="blue", zorder=1)
        # plt.plot(np.diff(lc), c="red")
        # plt.plot([0, len(lc)], [3, 3], color="red", zorder=-1)
        if lfpDeflections is not None:
            plt.scatter(lfpDeflections / 1500, [0] * len(lfpDeflections), zorder=2, c="red")

        plt.show()

    return ripplePower, zpower, meanPower, stdPower


def detectRipples(ripplePower, minHeight=3.0, minLen=0.05, maxLen=0.3, edgeThresh=0.0):
    pks, peakInfo = signal.find_peaks(ripplePower, height=minHeight)

    ripStartIdxs = []
    ripLens = []
    ripPeakIdxs = []
    ripPeakAmps = []
    ripCrossThreshIdx = []

    i = 0
    while i < len(pks):
        pkidx = pks[i]
        ii = pkidx
        while ii >= 0 and ripplePower[ii] > edgeThresh:
            ii -= 1
        ii += 1
        ripStart = ii

        length = 0
        pkAmp = 0
        pkAmpI = 0
        crossI = 0
        crossed = False
        while ii < len(ripplePower) and ripplePower[ii] > edgeThresh:
            if ripplePower[ii] > pkAmp:
                pkAmp = ripplePower[ii]
                pkAmpI = ii

            if not crossed and ripplePower[ii] > minHeight:
                crossed = True
                crossI = ii

            ii += 1
            length += 1

        assert crossed

        lensec = float(length) / LFP_SAMPLING_RATE
        if lensec >= minLen and lensec <= maxLen:
            ripStartIdxs.append(ripStart)
            ripLens.append(length)
            ripPeakAmps.append(pkAmp)
            ripPeakIdxs.append(pkAmpI)
            ripCrossThreshIdx.append(crossI)

        while i < len(pks) and pks[i] < ii:
            i += 1

    return ripStartIdxs, ripLens, ripPeakIdxs, ripPeakAmps, ripCrossThreshIdx


# 2 3
# 0 1
def quadrantOfWell(well_idx):
    if well_idx > 24:
        res = 2
    else:
        res = 0

    if (well_idx - 1) % 8 >= 4:
        res += 1

    return res


def getListOfVisitedWells(nearestWells, countFirstVisitOnly):
    if countFirstVisitOnly:
        return list(set(nearestWells))
    else:
        return [k for k, g in groupby(nearestWells)]


def onWall(well):
    return well < 9 or well > 40 or well % 8 in [2, 7]


def offWall(well):
    return not onWall(well)


class AnimalInfo:
    def __init__(self):
        self.X_START = None
        self.X_FINISH = None
        self.Y_START = None
        self.Y_FINISH = None
        self.excludeBoxes = None
        self.data_dir = ""
        self.output_dir = ""
        self.fig_output_dir = ""
        self.out_filename = ""

        self.excluded_dates = []
        self.excluded_sessions = []
        self.minimum_date = None

        self.DEFAULT_RIP_DET_TET = None
        self.DEFAULT_RIP_BAS_TET = None

        self.DLC_dir = None


def getInfoForAnimal(animalName: str) -> AnimalInfo:
    ret = AnimalInfo()
    if animalName == "Martin":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = '/media/WDC1/martindata/bradtask/'
        ret.output_dir = '/media/WDC6/Martin/processed_data/'
        ret.fig_output_dir = ret.output_dir
        ret.out_filename = "martin_bradtask.dat"

        ret.excluded_dates = ["20200528", "20200630", "20200702", "20200703"]
        ret.excluded_dates += ["20200531", "20200603", "20200602",
                               "20200606", "20200605", "20200601"]
        ret.excluded_dates += ["20200526"]
        ret.excluded_sessions = ["20200624_1", "20200624_2", "20200628_2"]
        ret.minimum_date = None

        ret.excluded_dates += ["20200527"]
        ret.excluded_dates += ["20200604"]
        ret.excluded_dates += ["20200608"]
        ret.excluded_dates += ["20200609"]

        # Too few interruptions to really call this an interruption session probably:
        ret.excluded_dates += ["20200607"]

        ret.DEFAULT_RIP_DET_TET = 37
        ret.DEFAULT_RIP_BAS_TET = None

    elif animalName == "B12":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_bradtask.dat"

        ret.excluded_dates = []
        ret.minimum_date = None
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 7

        ret.DLC_dir = "/media/WDC6/DLC/trainingVideos/"

    elif animalName == "B12_goodpos":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_goodpos_bradtask.dat"

        ret.excluded_dates = ["20210816", "20210817", "20210818", "20210819"]
        ret.minimum_date = None
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 7

    elif animalName == "B12_no19":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_no19_bradtask.dat"

        ret.excluded_dates = ["20210819"]
        ret.minimum_date = None
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 7

    elif animalName == "B12_highthresh":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_highthresh_bradtask.dat"

        ret.excluded_dates = []
        ret.minimum_date = "20210916"
        ret.excluded_sessions = ["20210917_1", "20210923_1",
                                 "20211004_1", "20211005_2", "20211006_1"]
        ret.DEFAULT_RIP_DET_TET = 8

    elif animalName == "B13":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        ret.data_dir = "/media/WDC6/B13/bradtasksessions/"
        ret.output_dir = "/media/WDC6/B13/processed_data/"
        ret.fig_output_dir = "/media/WDC6/B13/processed_data/"
        ret.out_filename = "B13_bradtask.dat"

        ret.excluded_dates = ["20220209"]
        # minimum_date = "20211209"  # had one run on the 8th with probe but used high ripple threshold and a
        # different reference tetrode
        ret.minimum_date = None
        # high ripple thresh on 12/08-1, forgot to turn stim on til after first home on 12/16-2
        ret.excluded_sessions = ["20211208_1", "20211216_2"]
        ret.DEFAULT_RIP_DET_TET = 7
        ret.DEFAULT_RIP_BAS_TET = 2

        # Messed up away well order, marked down 20 when he got 12. Ended up giving him reward at 12 twice
        ret.excluded_sessions += ["20220131_2"]
        # Made a custom foundwells field in the behaviornotes for this guy, but would need to update the rest of the
        # import code
        # (i.e. clips loading assumes alternation with home)
        ret.excluded_sessions += ["20220222_2"]
        # video skips
        ret.excluded_sessions += ["20220304_2"]
        ret.excluded_sessions += ["20220307_1"]

        # Cable got messed up during the task, pulley wasn't rolling horizontally
        ret.excluded_sessions += ["20211217_1"]

        ret.rerun_trodes_videos = []
        ret.rerun_usb_videos = []

    elif animalName == "B14":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        ret.data_dir = "/media/WDC6/B14/bradtasksessions/"
        ret.output_dir = "/media/WDC6/B14/processed_data/"
        ret.fig_output_dir = "/media/WDC6/B14/processed_data/"
        ret.out_filename = "B14_bradtask.dat"

        ret.excluded_dates = []
        # minimum_date = "20211209"  # one run with high thresh and one 15 min run on the 8th
        ret.minimum_date = "20220220"  # Only after adjusting stim electrode to correct place!
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 3
        ret.DEFAULT_RIP_BAS_TET = 2
        # video skips
        ret.excluded_sessions += ["20220307_2"]
        ret.excluded_sessions += ["20220310_2"]
        # forgot to turn stim on until after first home well
        ret.excluded_sessions += ["20220311_1"]

        ret.rerun_usb_videos = []
        ret.rerun_trodes_videos = []
    elif animalName == "B18":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        ret.excludeBoxes = [(0, 0, 130, 60)]
        # ret.data_dir = "/home/wcroughan/data/B18/bradtasksessions/"
        # ret.output_dir = "/home/wcroughan/data/B18/processed_data/"
        # ret.fig_output_dir = "/home/wcroughan/data/B18/processed_data/"
        ret.data_dir = "/media/WDC8/B18/bradtasksessions/"
        ret.output_dir = "/media/WDC8/B18/processed_data/"
        ret.fig_output_dir = "/media/WDC8/B18/processed_data/"
        ret.out_filename = "B18_bradtask.dat"

        ret.excluded_dates = []
        ret.minimum_date = None
        # ret.minimum_date = "20221109"
        ret.excluded_sessions = []

        # Add back in later, just issue with info file
        # ret.excluded_sessions += ["20221114_1"]
        # ret.excluded_sessions += ["20221115_1"]
        # ret.excluded_sessions += ["20221116_1"]
        # ret.excluded_sessions += ["20221118_1"]

        # Found wells doesn't seem like it lines up perfectly with usb video, check it out and fix
        ret.excluded_sessions += ["20221113_2"]

        # Need to improve trodes tracking
        ret.excluded_sessions += ["20221108_1"]

        # usb video skips (add back later, just annoying for now)
        ret.excluded_sessions += ["20220617_1"]

        # Trodes video skips
        ret.excluded_sessions += ["20220620_1"]
        ret.excluded_sessions += ["20220622_2"]

        # Trodes camera partially blocked during probe by pulley system. It's just wells 7-4 ish, might be able to
        # deal with it in analysis
        # At least it shouldn't effect measures at off wall wells
        # ret.excluded_sessions += ["20220621_2"]

        # Amplitude at 100uA, probably don't need to exclude but should note
        # ret.excluded_sessions += ["20220620_2"]

        # stim wire was broken, shoulda been iterruption but was no-stim control
        # (both leads were disconnected, so no stim at all made it to brain)
        ret.excluded_sessions += ["20220624_2"]

        # Messed up the well order, so current pipeline can't handle clips. SHould still be able to analyze probe
        # behavior though if I wanted to
        ret.excluded_sessions += ["20220621_2"]

        # Wasn't actually detecting ripples during these
        ret.excluded_sessions += ["20221102_1"]
        ret.excluded_sessions += ["20221102_2"]
        ret.excluded_sessions += ["20221103_1"]
        ret.excluded_sessions += ["20221103_2"]
        ret.excluded_sessions += ["20221105_1"]

        ret.DEFAULT_RIP_DET_TET = 5
        ret.DEFAULT_RIP_BAS_TET = 3
        ret.rerun_usb_videos = []
        ret.rerun_trodes_videos = []

        # ======================
        # Temporary
        # Just annoying behavior notes thing, fix and add back in
        # ret.excluded_sessions += ["20220618_2"]
        # ret.excluded_sessions += ["20220625_2"]
        # ret.excluded_sessions += ["20220626_2"]
        # ret.minimum_date = "20220621"
        # ret.excluded_sessions = ["20220621_2", "20220622_2", "20220623_2"]
        # ret.excluded_sessions += ["2022062{}_1".format(v) for v in range(24, 29)]
        # ret.excluded_sessions += ["2022062{}_2".format(v) for v in range(24, 29)]

    elif animalName == "B16":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        # ret.data_dir = "/media/WDC8/B16/bradtasksessions/"
        # ret.output_dir = "/media/WDC8/B16/processed_data/"
        # ret.fig_output_dir = "/media/WDC8/B16/processed_data/"
        # TODO make this dynamic like findDataDir
        ret.data_dir = "/home/wcroughan/data/B16/bradtasksessions/"
        ret.output_dir = "/home/wcroughan/data/B16/processed_data/"
        ret.fig_output_dir = "/home/wcroughan/data/B16/processed_data/"
        ret.out_filename = "B16_bradtask.dat"

        ret.excluded_dates = []
        # Seems like the older usb videos are all skipping for some reason
        ret.minimum_date = "20221101"
        # ret.minimum_date = None
        ret.excluded_sessions = []

        # Trodes video skips
        ret.excluded_sessions += ["20221107_2"]
        ret.excluded_sessions += ["20221116_1"]

        # USB video skips, can add back in later
        ret.excluded_sessions += ["20220919_1"]
        ret.excluded_sessions += ["20220919_2"]

        # USB and trodes misaligned
        # ret.excluded_sessions += ["20221113_1"]

        # Just stayed in the corner for ten minutes then I took him out
        ret.excluded_sessions += ["20221118_1"]

        # Just annoying info file stuff, fix and add back in
        # ret.excluded_sessions += ["20221111_2"]

        # ret.excluded_sessions += ["20220919_122046"]

        # Used the wrong detection tetrode
        ret.excluded_sessions += ["20221103_1"]

        # USB video starts right after I put him in
        # ret.excluded_sessions += ["20221109_1"]

        ret.DEFAULT_RIP_DET_TET = 4
        ret.DEFAULT_RIP_BAS_TET = 7
        ret.rerun_usb_videos = []
        ret.rerun_trodes_videos = []

    elif animalName == "B17":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        ret.data_dir = "/media/WDC8/B17/bradtasksessions/"
        ret.output_dir = "/media/WDC8/B17/processed_data/"
        ret.fig_output_dir = "/media/WDC8/B17/processed_data/"
        ret.out_filename = "B17_bradtask.dat"

        ret.excluded_dates = []
        ret.minimum_date = None
        ret.excluded_sessions = []

        # Trodes video skips
        ret.excluded_sessions += ["20221104_1"]

        ret.DEFAULT_RIP_DET_TET = 6
        ret.DEFAULT_RIP_BAS_TET = 5
        ret.rerun_usb_videos = []
        ret.rerun_trodes_videos = []

    else:
        raise Exception("Unknown animal name")

    return ret


def getUSBVideoFile(seshName, possibleDirectories, seshIdx=None, useSeshIdxDirectly=False):
    # print(seshName)
    seshDate, seshTime = seshName.split("_")
    if len(seshTime) == 1 or seshIdx is not None:
        # seshTime is actually session idx, or it's been provided directly
        if seshIdx is None:
            seshIdx = int(seshTime) - 1
        else:
            seshIdx -= 1

        if seshDate > "20221100" and seshDate < "20221122":
            # In november, trimmed videos are labeled by session name
            print("\tusing date string directly with no dashes")
            usbDateStr = seshDate
        else:
            usbDateStr = "-".join([seshDate[0:4], seshDate[4:6], seshDate[6:8]])
        possibleUSBVids = []
        for pd in possibleDirectories:
            directName = os.path.join(pd, usbDateStr + "_" + str(seshIdx + 1) + ".mkv")
            if os.path.exists(directName):
                return directName

            if not useSeshIdxDirectly:
                gl = pd + "/" + usbDateStr + "*.mkv"
                possibleUSBVids += glob.glob(gl)

        if len(possibleUSBVids) == 0:
            print("\tCouldn't find any matching usb video files")
            return None

        possibleUSBVids = sorted(possibleUSBVids)
        return possibleUSBVids[seshIdx]
    else:
        seshTimeVal = float(seshTime[0:2]) * 3600 + \
            float(seshTime[2:4]) * 60 + float(seshTime[4:6])

        usbDateStr = "-".join([seshDate[0:4], seshDate[4:6], seshDate[6:8]])
        possibleUSBVids = []
        for pd in possibleDirectories:
            gl = pd + "/" + usbDateStr + "*.mkv"
            possibleUSBVids += glob.glob(gl)

        if len(possibleUSBVids) == 0:
            return None

        minDiff = 24 * 3600
        usbVidFile = None
        for uvi, uv in enumerate(sorted(possibleUSBVids)):
            fname = uv.split("/")[-1]
            print(fname)
            if " " in fname:
                timeStr = fname.split(" ")[1].split(".")[0]
            else:
                timeStr = fname.split("_")[1].split(".")[0]
            timeVals = [float(v) for v in timeStr.split("-")]
            print(timeVals)
            usbTime = timeVals[0] * 3600 + timeVals[1] * 60 + timeVals[0]

            diff = abs(usbTime - seshTimeVal)
            if diff < minDiff:
                minDiff = diff
                usbVidFile = uv
                # seshWithinDay = uvi

        return usbVidFile


def getTrodesVideoFile(seshInfoFileName, data_dir):
    seshDate, seshIdx = seshInfoFileName.split("_")

    gl = data_dir + "/" + seshDate + "_*/" + seshDate + "_*.1.h264"
    possibleTrodesVids = glob.glob(gl)

    if len(possibleTrodesVids) == 0:
        return None

    seshIdx = int(seshIdx) - 1
    possibleTrodesVids = sorted(possibleTrodesVids)
    return possibleTrodesVids[seshIdx]


def numWellsVisited(nearestWells, countReturns=False, wellSubset=None):
    g = groupby(nearestWells)
    if wellSubset is None:
        wellSubset = allWellNames
    if countReturns:
        return len([k for k, _ in g if k in wellSubset])
    else:
        return len(set([k for k, _ in g if k in wellSubset]))


def weekIdxForDateStr(datestr, d0=date(2016, 1, 4)):
    d = date(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]))
    return (d - d0).days // 7


class TimeThisFunction:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.totalTime = 0

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = self.func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        self.totalTime += run_time
        return value
