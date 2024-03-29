import sys
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from SpikeCalibration import MUAClusterFunc, runTheThing, makeClusterFuncFromFile, loadTrodesClusters, ConvertTimeToTrodesTS


def makePSTH(animal_name, ALWAYS_REMAKE_SG_EXPORTS=False):
    possible_drive_dirs = ["/media/WDC7/", "/media/fosterlab/WDC7/",
                           "/media/WDC6/", "/media/fosterlab/WDC6/",
                           "/media/WDC8/", "/media/fosterlab/WDC8/",
                           "/home/fosterlab/Desktop/"]
    # print(possible_drive_dirs)

    drive_dir = None
    for dd in possible_drive_dirs:
        if os.path.exists(dd):
            drive_dir = dd
            print("drive_dir = ", drive_dir)
            break

    if drive_dir is None:
        print("Couldnt' find data directory among any of these: {}".format(possible_drive_dirs))
        return

    UP_DEFLECT_STIM_THRESH = None

    clfuncCtrl = None
    clusterPolygonsCtrl = None
    twoFiles = False

    if animal_name == "B13":
        runName = "20220125_094035"
        lfpTet = 7
        spikeTet = 7
        clusterIndex = -1

        recFileName = os.path.join(drive_dir, "B13/{}/{}.rec".format(runName, runName))
        gl = os.path.join(drive_dir, "B13/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(
            runName, runName, runName, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = os.path.join(drive_dir, "B13/{}/{}.spikes/{}.spikes_nt{}.dat".format(
            runName, runName, runName, spikeTet))

        clusterFileName = os.path.join(
            drive_dir, "B13/{}/{}.trodesClusters".format(runName, runName))

        clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)
        clusters = loadTrodesClusters(clusterFileName)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygons = clusters[spikeTet - 1]

        swrt0 = ConvertTimeToTrodesTS(0, 5, 0)
        swrt1 = ConvertTimeToTrodesTS(0, 38, 0)
        ctrlt0 = ConvertTimeToTrodesTS(1, 7, 30)
        ctrlt1 = ConvertTimeToTrodesTS(1, 38, 0)

        outputDir = os.path.join(drive_dir, "B13/figs/")
    elif animal_name == "B14":
        lfpTet = 2
        spikeTet = 2
        clusterIndex = -1
        clfunc = None

        if False:
            runName = "20220124_100811"
            swrt0 = ConvertTimeToTrodesTS(0, 5, 0)
            swrt1 = ConvertTimeToTrodesTS(0, 35, 0)
            ctrlt0 = ConvertTimeToTrodesTS(1, 5, 0)
            ctrlt1 = ConvertTimeToTrodesTS(1, 35, 0)
        elif False:
            runName = "20220126_110046"
            swrt0 = ConvertTimeToTrodesTS(0, 1, 0)
            swrt1 = ConvertTimeToTrodesTS(0, 34, 0)
            ctrlt0 = ConvertTimeToTrodesTS(1, 6, 0)
            ctrlt1 = ConvertTimeToTrodesTS(1, 36, 0)
        elif False:
            runName = "20220127_090603"
            swrt0 = ConvertTimeToTrodesTS(1, 6, 15)
            swrt1 = ConvertTimeToTrodesTS(1, 25, 0)
            ctrlt0 = ConvertTimeToTrodesTS(1, 30, 25)
            ctrlt1 = ConvertTimeToTrodesTS(1, 50, 0)
            spikeTet = 5
        elif False:
            runName = "20220128_093620"
            swrt0 = ConvertTimeToTrodesTS(0, 3, 15)
            swrt1 = ConvertTimeToTrodesTS(0, 36, 30)
            ctrlt0 = ConvertTimeToTrodesTS(0, 53, 41)
            ctrlt1 = ConvertTimeToTrodesTS(1, 27, 0)
            spikeTet = 5
        elif False:
            runName = "20220202_085258"
            swrt0 = ConvertTimeToTrodesTS(1, 8, 0)
            swrt1 = ConvertTimeToTrodesTS(1, 42, 15)
            ctrlt0 = ConvertTimeToTrodesTS(2, 19, 0)
            ctrlt1 = ConvertTimeToTrodesTS(2, 51, 0)
            spikeTet = 2

            clusterFileName = os.path.join(
                drive_dir, "B14/{}/{}.trodesClusters".format(runName, runName))
            f = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)

            def clfunc(features, chmaxes, maxfeature, endFeatures, chmins):
                ret = f(features, chmaxes, maxfeature)
                return np.logical_and(ret, endFeatures[:, 2] < 0)
            clusters = loadTrodesClusters(clusterFileName)
            clusterPolygons = clusters[spikeTet - 1]
        elif False:
            runName = "20220202_125441"
            swrt0 = ConvertTimeToTrodesTS(4, 2, 38)
            swrt1 = ConvertTimeToTrodesTS(4, 33, 0)
            ctrlt0 = ConvertTimeToTrodesTS(5, 13, 1)
            ctrlt1 = ConvertTimeToTrodesTS(5, 44, 0)
            spikeTet = 2
        elif False:
            runName = "20220217_142910"
            swrt0 = ConvertTimeToTrodesTS(0, 21, 2)
            swrt1 = ConvertTimeToTrodesTS(0, 59, 11)
            ctrlt0 = ConvertTimeToTrodesTS(1, 38, 22)
            ctrlt1 = ConvertTimeToTrodesTS(2, 10, 45)
            spikeTet = 8
        else:
            runName = "20220218_145108"
            swrt0 = ConvertTimeToTrodesTS(0, 39, 10)
            swrt1 = ConvertTimeToTrodesTS(1, 12, 50)
            ctrlt0 = ConvertTimeToTrodesTS(1, 39, 51)
            ctrlt1 = ConvertTimeToTrodesTS(2, 9, 45)
            spikeTet = 8

        recFileName = os.path.join(drive_dir, "B14/{}/{}.rec".format(runName, runName))
        gl = os.path.join(drive_dir, "B14/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(
            runName, runName, runName, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = os.path.join(drive_dir, "B14/{}/{}.spikes/{}.spikes_nt{}.dat".format(
            runName, runName, runName, spikeTet))

        if clfunc is None:
            clusterFileName = os.path.join(
                drive_dir, "B14/{}/{}.trodesClusters".format(runName, runName))
            clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygons = clusters[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "B14/figs/")

    # elif animal_name == "Martin":
    #     data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
    #     outputDir = "/media/WDC7/Martin/figs/"
    #     diagPoint = None
    elif animal_name == "B16":
        lfpTet = 2
        spikeTet = 2
        clusterIndex = -1
        clfunc = None
        clFileSuffix = ""

        if False:
            runName = "20220523_092912"
            swrt0 = ConvertTimeToTrodesTS(3, 47, 30)
            swrt1 = ConvertTimeToTrodesTS(4, 17, 15)
            ctrlt0 = ConvertTimeToTrodesTS(4, 48, 45)
            ctrlt1 = ConvertTimeToTrodesTS(5, 18, 38)
            spikeTet = 2
        elif False:
            runName = "20220531_201756"
            swrt0 = ConvertTimeToTrodesTS(0, 4, 12)
            swrt1 = ConvertTimeToTrodesTS(0, 37, 58)
            ctrlt0 = ConvertTimeToTrodesTS(1, 9, 35)
            ctrlt1 = ConvertTimeToTrodesTS(1, 39, 50)
            lfpTet = 4
            spikeTet = 4
        elif False:
            runName = "20220601_114315"
            swrt0 = ConvertTimeToTrodesTS(0, 14, 16)
            swrt1 = ConvertTimeToTrodesTS(0, 51, 15)
            ctrlt0 = ConvertTimeToTrodesTS(1, 23, 3)
            ctrlt1 = ConvertTimeToTrodesTS(1, 58, 34)
            lfpTet = 4
            spikeTet = 4
            UP_DEFLECT_STIM_THRESH = 5000
            clFileSuffix = "_noisy"
        else:
            runName = "20220819_111500"
            swrt0 = ConvertTimeToTrodesTS(0, 39, 53)
            swrt1 = ConvertTimeToTrodesTS(1, 13, 52)
            ctrlt0 = ConvertTimeToTrodesTS(1, 42, 42)
            ctrlt1 = ConvertTimeToTrodesTS(2, 21, 42)
            lfpTet = 4
            spikeTet = 4

        recFileName = os.path.join(drive_dir, "B16/{}/{}.rec".format(runName, runName))
        gl = os.path.join(drive_dir, "B16/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(
            runName, runName, runName, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = os.path.join(drive_dir, "B16/{}/{}.spikes/{}.spikes_nt{}.dat".format(
            runName, runName, runName, spikeTet))

        if clfunc is None:
            clusterFileName = os.path.join(
                drive_dir, "B16/{}/{}{}.trodesClusters".format(runName, runName, clFileSuffix))
            clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygons = clusters[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "B16/figs/")
    elif animal_name == "B17":
        lfpTet = 2
        spikeTet = 3
        clusterIndex = -1
        clfunc = None
        clFileSuffix = ""
        clFileSuffix_delay = None
        harsherClFunc = False

        if False:
            runName = "20220526_135145"
            swrt0 = ConvertTimeToTrodesTS(1, 17, 45)
            swrt1 = ConvertTimeToTrodesTS(1, 48, 19)
            ctrlt0 = ConvertTimeToTrodesTS(2, 25, 40)
            ctrlt1 = ConvertTimeToTrodesTS(2, 55, 45)
        elif False:
            runName = "20220601_150729"
            swrt0 = ConvertTimeToTrodesTS(0, 16, 7)
            swrt1 = ConvertTimeToTrodesTS(0, 48, 39)
            ctrlt0 = ConvertTimeToTrodesTS(1, 16, 39)
            ctrlt1 = ConvertTimeToTrodesTS(1, 50, 54)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220612_182252"
            swrt0 = ConvertTimeToTrodesTS(2, 15, 36)
            swrt1 = ConvertTimeToTrodesTS(2, 47, 20)
            ctrlt0 = ConvertTimeToTrodesTS(3, 15, 20)
            ctrlt1 = ConvertTimeToTrodesTS(3, 46, 8)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220612_220046"
            swrt0 = ConvertTimeToTrodesTS(0, 10, 4)
            swrt1 = ConvertTimeToTrodesTS(0, 30, 33)
            ctrlt0 = ConvertTimeToTrodesTS(0, 49, 30)
            ctrlt1 = ConvertTimeToTrodesTS(1, 13, 10)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220614_113602"
            swrt0 = ConvertTimeToTrodesTS(3, 9, 45)
            swrt1 = ConvertTimeToTrodesTS(3, 38, 50)
            ctrlt0 = ConvertTimeToTrodesTS(4, 13, 2)
            ctrlt1 = ConvertTimeToTrodesTS(4, 50, 0)
            lfpTet = 3
            spikeTet = 3
            # clFileSuffix_delay = "_2"
            clFileSuffix = "_again"
        elif False:
            runName = "20220616_100325"
            swrt0 = ConvertTimeToTrodesTS(0, 39, 0)
            swrt1 = ConvertTimeToTrodesTS(1, 9, 15)
            ctrlt0 = ConvertTimeToTrodesTS(1, 42, 8)
            ctrlt1 = ConvertTimeToTrodesTS(2, 12, 13)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220617_121743"
            swrt0 = ConvertTimeToTrodesTS(0, 25, 15)
            swrt1 = ConvertTimeToTrodesTS(0, 55, 12)
            ctrlt0 = ConvertTimeToTrodesTS(1, 32, 2)
            ctrlt1 = ConvertTimeToTrodesTS(2, 5, 50)
            lfpTet = 3
            spikeTet = 3
            clFileSuffix = "_2"
        elif False:
            runName = "20220620_123733"
            swrt0 = ConvertTimeToTrodesTS(0, 42, 43)
            swrt1 = ConvertTimeToTrodesTS(1, 12, 33)
            ctrlt0 = ConvertTimeToTrodesTS(1, 43, 17)
            ctrlt1 = ConvertTimeToTrodesTS(2, 18, 3)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220620_173330"
            swrt0 = ConvertTimeToTrodesTS(1, 22, 37)
            swrt1 = ConvertTimeToTrodesTS(1, 52, 40)
            ctrlt0 = ConvertTimeToTrodesTS(2, 51, 1)
            ctrlt1 = ConvertTimeToTrodesTS(3, 24, 45)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220623_114549"
            swrt0 = ConvertTimeToTrodesTS(3, 2, 24)
            swrt1 = ConvertTimeToTrodesTS(3, 32, 30)
            ctrlt0 = ConvertTimeToTrodesTS(4, 5, 25)
            ctrlt1 = ConvertTimeToTrodesTS(4, 44, 0)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220627_134658"
            swrt0 = ConvertTimeToTrodesTS(0, 30, 2)
            swrt1 = ConvertTimeToTrodesTS(1, 0, 37)
            ctrlt0 = ConvertTimeToTrodesTS(1, 33, 6)
            ctrlt1 = ConvertTimeToTrodesTS(2, 18, 0)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220627_171258"
            swrt0 = ConvertTimeToTrodesTS(0, 39, 55)
            swrt1 = ConvertTimeToTrodesTS(1, 10, 55)
            ctrlt0 = ConvertTimeToTrodesTS(1, 39, 15)
            ctrlt1 = ConvertTimeToTrodesTS(2, 25, 10)
            lfpTet = 3
            spikeTet = 3
        elif False:
            runName = "20220928_144436"
            swrt0 = ConvertTimeToTrodesTS(1, 53, 39)
            swrt1 = ConvertTimeToTrodesTS(2, 29, 29)
            ctrlt0 = ConvertTimeToTrodesTS(2, 45, 53)
            ctrlt1 = ConvertTimeToTrodesTS(3, 22, 26)
            lfpTet = 6
            spikeTet = 3
        elif False:
            runName = "20220929_140856"
            swrt0 = ConvertTimeToTrodesTS(0, 19, 29)
            swrt1 = ConvertTimeToTrodesTS(0, 59, 40)
            ctrlt0 = ConvertTimeToTrodesTS(1, 37, 19)
            ctrlt1 = ConvertTimeToTrodesTS(2, 14, 28)
            lfpTet = 6
            spikeTet = 3
        elif False:
            runName = "20221012_161840"
            swrt0 = ConvertTimeToTrodesTS(0, 18, 1)
            swrt1 = ConvertTimeToTrodesTS(0, 48, 1)
            ctrlt0 = ConvertTimeToTrodesTS(1, 24, 54)
            ctrlt1 = ConvertTimeToTrodesTS(1, 55, 5)
            lfpTet = 6
            spikeTet = 3
        else:
            runName = "20221013_115514"
            swrt0 = ConvertTimeToTrodesTS(0, 9, 6)
            swrt1 = ConvertTimeToTrodesTS(0, 40, 8)
            ctrlt0 = ConvertTimeToTrodesTS(1, 21, 10)
            ctrlt1 = ConvertTimeToTrodesTS(1, 54, 24)
            lfpTet = 6
            spikeTet = 3
            harsherClFunc = True

        if clFileSuffix_delay is None:
            clFileSuffix_delay = clFileSuffix

        recFileName = os.path.join(drive_dir, "{}/{}/{}.rec".format(animal_name, runName, runName))
        gl = os.path.join(drive_dir, "{}/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name,
                                                                              runName, runName, runName, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = os.path.join(drive_dir, "{}/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name,
                                                                                            runName, runName, runName, spikeTet))

        if clfunc is None:
            clusterFileName = os.path.join(
                drive_dir, "{}/{}/{}{}.trodesClusters".format(animal_name, runName, runName, clFileSuffix))
            tmpfunc = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)
            if harsherClFunc:
                def clfunc(features, chmaxes, maxfeature, endFeatures, chmins):
                    return np.logical_and(
                        tmpfunc(features, chmaxes, maxfeature, endFeatures, chmins),
                        np.max(endFeatures, axis=1) < 300
                    )
            else:
                clfunc = tmpfunc
            clusters = loadTrodesClusters(clusterFileName)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygons = clusters[spikeTet - 1]

        if clFileSuffix_delay != clFileSuffix:
            clusterFileName_delay = os.path.join(
                drive_dir, "{}/{}/{}{}.trodesClusters".format(animal_name, runName, runName, clFileSuffix_delay))
            clfuncCtrl = makeClusterFuncFromFile(clusterFileName_delay, spikeTet - 1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName_delay)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygonsCtrl = clusters[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "{}/figs/".format(animal_name))
    elif animal_name == "B18":
        lfpTet = 5
        spikeTet = 5
        clusterIndex = -1
        clfunc = None

        if False:
            pass
        else:
            runName = "20220521_094602"
            swrt0 = ConvertTimeToTrodesTS(3, 12, 45)
            swrt1 = ConvertTimeToTrodesTS(3, 44, 10)
            ctrlt0 = ConvertTimeToTrodesTS(4, 15, 30)
            ctrlt1 = ConvertTimeToTrodesTS(4, 45, 45)

        recFileName = os.path.join(drive_dir, "{}/{}/{}.rec".format(animal_name, runName, runName))
        gl = os.path.join(drive_dir, "{}/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name,
                                                                              runName, runName, runName, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = os.path.join(drive_dir, "{}/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name,
                                                                                            runName, runName, runName, spikeTet))

        if clfunc is None:
            clusterFileName = os.path.join(
                drive_dir, "{}/{}/{}.trodesClusters".format(animal_name, runName, runName))
            clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygons = clusters[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "{}/figs/".format(animal_name))
    elif animal_name == "B19":
        lfpTet = 2
        spikeTet = 6
        clusterIndex = -1
        clfunc = None
        clFileSuffix = ""

        if False:
            runName = "20220504_150715"
            swrt0 = ConvertTimeToTrodesTS(0, 1, 45)
            swrt1 = ConvertTimeToTrodesTS(0, 31, 20)
            ctrlt0 = ConvertTimeToTrodesTS(0, 48, 30)
            ctrlt1 = ConvertTimeToTrodesTS(1, 18, 30)
        elif False:
            runName = "20220525_140759"
            swrt0 = ConvertTimeToTrodesTS(0, 36, 30)
            swrt1 = ConvertTimeToTrodesTS(1, 10, 0)
            ctrlt0 = ConvertTimeToTrodesTS(1, 41, 0)
            ctrlt1 = ConvertTimeToTrodesTS(2, 41, 30)
            lfpTet = 8
            spikeTet = 8
        elif False:
            runName = "20220531_114750"
            swrt0 = ConvertTimeToTrodesTS(0, 56, 16)
            swrt1 = ConvertTimeToTrodesTS(1, 32, 6)
            ctrlt0 = ConvertTimeToTrodesTS(2, 7, 4)
            ctrlt1 = ConvertTimeToTrodesTS(2, 38, 1)
            lfpTet = 8
            spikeTet = 8
        elif False:
            runName = "20220605_162549"
            swrt0 = ConvertTimeToTrodesTS(0, 15, 26)
            swrt1 = ConvertTimeToTrodesTS(0, 56, 30)
            ctrlt0 = ConvertTimeToTrodesTS(1, 29, 2)
            ctrlt1 = ConvertTimeToTrodesTS(1, 59, 40)
            lfpTet = 8
            spikeTet = 8
        elif False:
            runName = "20220609_131504"
            swrt0 = ConvertTimeToTrodesTS(0, 25, 35)
            swrt1 = ConvertTimeToTrodesTS(0, 59, 25)
            ctrlt0 = ConvertTimeToTrodesTS(1, 30, 45)
            ctrlt1 = ConvertTimeToTrodesTS(2, 10, 0)
            lfpTet = 8
            spikeTet = 8
        elif False:
            runName = "20220610_194356"
            swrt0 = ConvertTimeToTrodesTS(0, 13, 34)
            swrt1 = ConvertTimeToTrodesTS(0, 47, 58)
            ctrlt0 = ConvertTimeToTrodesTS(1, 15, 0)
            ctrlt1 = ConvertTimeToTrodesTS(1, 47, 4)
            lfpTet = 8
            spikeTet = 8
            clFileSuffix = "_2"
        else:
            runName = "20220816_144045"
            swrt0 = ConvertTimeToTrodesTS(0, 48, 14)
            swrt1 = ConvertTimeToTrodesTS(1, 17, 45)
            ctrlt0 = ConvertTimeToTrodesTS(2, 4, 47)
            ctrlt1 = ConvertTimeToTrodesTS(2, 30, 0)
            lfpTet = 8
            spikeTet = 8

        recFileName = os.path.join(drive_dir, "{}/{}/{}.rec".format(animal_name, runName, runName))
        gl = os.path.join(drive_dir, "{}/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name,
                                                                              runName, runName, runName, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = os.path.join(drive_dir, "{}/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name,
                                                                                            runName, runName, runName, spikeTet))

        if clfunc is None:
            clusterFileName = os.path.join(
                drive_dir, "{}/{}/{}{}.trodesClusters".format(animal_name, runName, runName, clFileSuffix))
            clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygons = clusters[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "{}/figs/".format(animal_name))
    elif animal_name == "B20":
        lfpTet = 2
        spikeTet = 2
        clusterIndex = -1
        clfunc = None

        if False:
            runName = "20220602_144759"
            swrt0 = ConvertTimeToTrodesTS(3, 44, 15)
            swrt1 = ConvertTimeToTrodesTS(4, 6, 55)
            ctrlt0 = ConvertTimeToTrodesTS(4, 25, 7)
            ctrlt1 = ConvertTimeToTrodesTS(4, 47, 50)
        elif False:
            runName = "20220606_132221"
            swrt0 = ConvertTimeToTrodesTS(0, 45, 36)
            swrt1 = ConvertTimeToTrodesTS(1, 16, 7)
            ctrlt0 = ConvertTimeToTrodesTS(1, 46, 13)
            ctrlt1 = ConvertTimeToTrodesTS(2, 16, 1)
            spikeTet = 1
        elif False:
            runName = "20220609_181141"
            swrt0 = ConvertTimeToTrodesTS(1, 6, 37)
            swrt1 = ConvertTimeToTrodesTS(1, 26, 28)
            ctrlt0 = ConvertTimeToTrodesTS(1, 39, 31)
            ctrlt1 = ConvertTimeToTrodesTS(2, 3, 20)
            spikeTet = 8
        elif False:
            runName = "20220610_121139"
            swrt0 = ConvertTimeToTrodesTS(1, 53, 40)
            swrt1 = ConvertTimeToTrodesTS(2, 22, 22)
            ctrlt0 = ConvertTimeToTrodesTS(3, 6, 23)
            ctrlt1 = ConvertTimeToTrodesTS(3, 38, 23)
            spikeTet = 8
        else:
            runName = "20220614_230037"
            swrt0 = ConvertTimeToTrodesTS(5, 17, 50)
            swrt1 = ConvertTimeToTrodesTS(5, 56, 30)
            ctrlt0 = ConvertTimeToTrodesTS(6, 59, 30)
            ctrlt1 = ConvertTimeToTrodesTS(7, 30, 0)
            spikeTet = 2
            lfpTet = 4

        recFileName = os.path.join(drive_dir, "{}/{}/{}.rec".format(animal_name, runName, runName))
        gl = os.path.join(drive_dir, "{}/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name,
                                                                              runName, runName, runName, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = os.path.join(drive_dir, "{}/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name,
                                                                                            runName, runName, runName, spikeTet))

        if clfunc is None:
            clusterFileName = os.path.join(
                drive_dir, "{}/{}/{}.trodesClusters".format(animal_name, runName, runName))
            clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet - 1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygons = clusters[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "{}/figs/".format(animal_name))
    elif animal_name == "B13_run":
        lfpTet = 7
        spikeTet = 7
        clusterIndex = -1
        twoFiles = True

        if True:
            runNameSWR = "20220307_140442"
            runNameCtrl = "20220307_174753"
            swrt0 = None
            swrt1 = None
            ctrlt0 = None
            ctrlt1 = None
            # spikeTet = 2
            # lfpTet = 4
            recFileNameSWR = os.path.join(
                drive_dir, "{}/bradtasksessions/{}/{}.rec".format(animal_name.split("_")[0], runNameSWR, runNameSWR))
            recFileNameCtrl = os.path.join(
                drive_dir, "{}/bradtasksessions/{}/{}.rec".format(animal_name.split("_")[0], runNameCtrl, runNameCtrl))

        gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                               runNameSWR, runNameSWR, runNameSWR, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileNameSWR = lfpfilelist[0]
        else:
            lfpFileNameSWR = "nofile"

        spikeFileNameSWR = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name.split("_")[0],
                                                                                                                runNameSWR, runNameSWR, runNameSWR, spikeTet))

        gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                               runNameCtrl, runNameCtrl, runNameCtrl, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileNameCtrl = lfpfilelist[0]
        else:
            lfpFileNameCtrl = "nofile"

        spikeFileNameCtrl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name.split("_")[0],
                                                                                                                 runNameCtrl, runNameCtrl, runNameCtrl, spikeTet))

        clusterFileNameSWR = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.trodesClusters".format(animal_name.split("_")[0], runNameSWR, runNameSWR))
        clfuncSWR = makeClusterFuncFromFile(clusterFileNameSWR, spikeTet - 1, clusterIndex)
        clustersSWR = loadTrodesClusters(clusterFileNameSWR)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygonsSWR = clustersSWR[spikeTet - 1]

        clusterFileNameCtrl = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.trodesClusters".format(animal_name.split("_")[0], runNameCtrl, runNameCtrl))
        clfuncCtrl = makeClusterFuncFromFile(clusterFileNameCtrl, spikeTet - 1, clusterIndex)
        clustersCtrl = loadTrodesClusters(clusterFileNameCtrl)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygonsCtrl = clustersCtrl[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "{}/figs/".format(animal_name.split("_")[0]))
    elif animal_name == "B14_run":
        lfpTet = 6
        spikeTet = 3
        clusterIndex = -1
        twoFiles = True

        if True:
            runNameSWR = "20220304_175709"
            runNameCtrl = "20220304_144221"
            swrt0 = None
            swrt1 = None
            ctrlt0 = None
            ctrlt1 = None
            # spikeTet = 2
            # lfpTet = 4
            recFileNameSWR = os.path.join(
                drive_dir, "{}/bradtasksessions/{}/{}.rec".format(animal_name.split("_")[0], runNameSWR, runNameSWR))
            recFileNameCtrl = os.path.join(
                drive_dir, "{}/bradtasksessions/{}/{}.rec".format(animal_name.split("_")[0], runNameCtrl, runNameCtrl))

        gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                               runNameSWR, runNameSWR, runNameSWR, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileNameSWR = lfpfilelist[0]
        else:
            lfpFileNameSWR = "nofile"

        spikeFileNameSWR = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name.split("_")[0],
                                                                                                                runNameSWR, runNameSWR, runNameSWR, spikeTet))

        gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                               runNameCtrl, runNameCtrl, runNameCtrl, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileNameCtrl = lfpfilelist[0]
        else:
            lfpFileNameCtrl = "nofile"

        spikeFileNameCtrl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name.split("_")[0],
                                                                                                                 runNameCtrl, runNameCtrl, runNameCtrl, spikeTet))

        clusterFileNameSWR = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.trodesClusters".format(animal_name.split("_")[0], runNameSWR, runNameSWR))
        clfuncSWR = makeClusterFuncFromFile(clusterFileNameSWR, spikeTet - 1, clusterIndex)
        clustersSWR = loadTrodesClusters(clusterFileNameSWR)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygonsSWR = clustersSWR[spikeTet - 1]

        clusterFileNameCtrl = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.trodesClusters".format(animal_name.split("_")[0], runNameCtrl, runNameCtrl))
        clfuncCtrl = makeClusterFuncFromFile(clusterFileNameCtrl, spikeTet - 1, clusterIndex)
        clustersCtrl = loadTrodesClusters(clusterFileNameCtrl)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygonsCtrl = clustersCtrl[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "{}/figs/".format(animal_name.split("_")[0]))
    elif animal_name == "B18_run":
        lfpTet = 3
        spikeTet = 5
        clusterIndex = -1
        twoFiles = True
        drive_dir = "/media/WDC8/"

        if False:
            runNameSWR = "20220626_181119"
            runNameCtrl = "20220626_145441"
            swrt0 = None
            swrt1 = None
            ctrlt0 = None
            ctrlt1 = None
            spikeTet = 6
        else:
            runNameSWR = "20220625_104058"
            runNameCtrl = "20220625_133942"
            swrt0 = None
            swrt1 = None
            ctrlt0 = None
            ctrlt1 = None
            spikeTet = 5

        recFileNameSWR = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.rec".format(animal_name.split("_")[0], runNameSWR, runNameSWR))
        recFileNameCtrl = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.rec".format(animal_name.split("_")[0], runNameCtrl, runNameCtrl))

        gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                               runNameSWR, runNameSWR, runNameSWR, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileNameSWR = lfpfilelist[0]
        else:
            lfpFileNameSWR = "nofile"

        spikeFileNameSWR = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name.split("_")[0],
                                                                                                                runNameSWR, runNameSWR, runNameSWR, spikeTet))

        gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                               runNameCtrl, runNameCtrl, runNameCtrl, lfpTet))
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileNameCtrl = lfpfilelist[0]
        else:
            lfpFileNameCtrl = "nofile"

        spikeFileNameCtrl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.spikes/{}.spikes_nt{}.dat".format(animal_name.split("_")[0],
                                                                                                                 runNameCtrl, runNameCtrl, runNameCtrl, spikeTet))

        clusterFileNameSWR = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.trodesClusters".format(animal_name.split("_")[0], runNameSWR, runNameSWR))
        clfuncSWR = makeClusterFuncFromFile(clusterFileNameSWR, spikeTet - 1, clusterIndex)
        clustersSWR = loadTrodesClusters(clusterFileNameSWR)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygonsSWR = clustersSWR[spikeTet - 1]

        clusterFileNameCtrl = os.path.join(
            drive_dir, "{}/bradtasksessions/{}/{}.trodesClusters".format(animal_name.split("_")[0], runNameCtrl, runNameCtrl))
        clfuncCtrl = makeClusterFuncFromFile(clusterFileNameCtrl, spikeTet - 1, clusterIndex)
        clustersCtrl = loadTrodesClusters(clusterFileNameCtrl)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygonsCtrl = clustersCtrl[spikeTet - 1]

        outputDir = os.path.join(drive_dir, "{}/figs/".format(animal_name.split("_")[0]))
    else:
        raise Exception("Unknown rat " + animal_name)

    # If necessary, generate lfp file
    possible_sg_dirs = ["/home/wcroughan/Software/Trodes223/", "/home/wcroughan/Software/Trodes21/",
                        "/home/wcroughan/Software/Trodes21/linux/", "/home/fosterlab/Software/Trodes223/",
                        "/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804"]
    sg_dir = None
    for dd in possible_sg_dirs:
        if os.path.exists(dd):
            sg_dir = dd
            print("sg_dir = ", sg_dir)
            break
    if sg_dir is None:
        print("Couldnt' find spike gadgets directory among any of these: {}".format(possible_sg_dirs))
        return

    if twoFiles:
        if not os.path.exists(lfpFileNameSWR) or ALWAYS_REMAKE_SG_EXPORTS:
            syscmd = os.path.join(sg_dir, "exportLFP") + " -rec " + recFileNameSWR
            print(syscmd)
            os.system(syscmd)
            gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                                   runNameSWR, runNameSWR, runNameSWR, lfpTet))
            lfpfilelist = glob.glob(gl)
            lfpFileNameSWR = lfpfilelist[0]

        # If necessary, generate spike file
        if not os.path.exists(spikeFileNameSWR) or ALWAYS_REMAKE_SG_EXPORTS:
            syscmd = os.path.join(sg_dir, "exportspikes") + " -rec " + recFileNameSWR
            print(syscmd)
            os.system(syscmd)

        if not os.path.exists(lfpFileNameCtrl) or ALWAYS_REMAKE_SG_EXPORTS:
            syscmd = os.path.join(sg_dir, "exportLFP") + " -rec " + recFileNameCtrl
            print(syscmd)
            os.system(syscmd)
            gl = os.path.join(drive_dir, "{}/bradtasksessions/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(animal_name.split("_")[0],
                                                                                                   runNameCtrl, runNameCtrl, runNameCtrl, lfpTet))
            lfpfilelist = glob.glob(gl)
            lfpFileNameCtrl = lfpfilelist[0]

        # If necessary, generate spike file
        if not os.path.exists(spikeFileNameCtrl) or ALWAYS_REMAKE_SG_EXPORTS:
            syscmd = os.path.join(sg_dir, "exportspikes") + " -rec " + recFileNameCtrl
            print(syscmd)
            os.system(syscmd)
    else:
        if not os.path.exists(lfpFileName) or ALWAYS_REMAKE_SG_EXPORTS:
            syscmd = os.path.join(sg_dir, "exportLFP") + " -rec " + recFileName
            print(syscmd)
            os.system(syscmd)
            lfpfilelist = glob.glob(gl)
            lfpFileName = lfpfilelist[0]

        # If necessary, generate spike file
        if not os.path.exists(spikeFileName) or ALWAYS_REMAKE_SG_EXPORTS:
            syscmd = os.path.join(sg_dir, "exportspikes") + " -rec " + recFileName
            print(syscmd)
            os.system(syscmd)

    if twoFiles:
        lfpTimestampFileNameSWR = ".".join(lfpFileNameSWR.split(".")[0:-2]) + ".timestamps.dat"
        lfpTimestampFileNameCtrl = ".".join(lfpFileNameCtrl.split(".")[0:-2]) + ".timestamps.dat"
        swrOutputFileName = os.path.join(outputDir, animal_name +
                                         "_" + runNameSWR + "_swr_psth.png")
        ctrlOutputFileName = os.path.join(
            outputDir, animal_name + "_" + runNameCtrl + "_ctrl_psth.png")
        comboOutputFileName = os.path.join(
            outputDir, animal_name + "_" + runNameSWR + "_" + runNameCtrl + "_combo_psth.png")
    else:
        lfpTimestampFileName = ".".join(lfpFileName.split(".")[0:-2]) + ".timestamps.dat"
        swrOutputFileName = os.path.join(outputDir, animal_name + "_" + runName + "_swr_psth.png")
        ctrlOutputFileName = os.path.join(outputDir, animal_name + "_" + runName + "_ctrl_psth.png")
        comboOutputFileName = os.path.join(
            outputDir, animal_name + "_" + runName + "_" + runName + "_combo_psth.png")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    if twoFiles:
        print(runNameSWR, runNameCtrl, "LFP tet " + str(lfpTet), "spike tet " + str(spikeTet), "cluster index " + str(clusterIndex),
              recFileNameSWR, recFileNameCtrl, lfpFileNameSWR, spikeFileNameSWR, lfpFileNameCtrl, spikeFileNameCtrl, clusterFileNameSWR, clusterFileNameCtrl)
    else:
        print(runName, "LFP tet " + str(lfpTet), "spike tet " + str(spikeTet), "cluster index " + str(clusterIndex),
              recFileName, lfpFileName, spikeFileName, clusterFileName)

    if clfuncCtrl is None:
        clfuncCtrl = clfunc
    if clusterPolygonsCtrl is None:
        clusterPolygonsCtrl = clusterPolygons

    if twoFiles:
        swrTvals, swrMeanPSTH, swrStdPSTH, swrN = runTheThing(spikeFileNameSWR, lfpFileNameSWR, lfpTimestampFileNameSWR,
                                                              swrOutputFileName, clfuncSWR, makeFigs=True, clusterPolygons=clusterPolygonsSWR,
                                                              tStart=swrt0, tEnd=swrt1, UP_DEFLECT_STIM_THRESH=UP_DEFLECT_STIM_THRESH)
        ctrlTvals, ctrlMeanPSTH, ctrlStdPSTH, ctrlN = runTheThing(spikeFileNameCtrl, lfpFileNameCtrl, lfpTimestampFileNameCtrl,
                                                                  ctrlOutputFileName, clfuncCtrl, makeFigs=False, clusterPolygons=clusterPolygonsCtrl,
                                                                  tStart=ctrlt0, tEnd=ctrlt1, UP_DEFLECT_STIM_THRESH=UP_DEFLECT_STIM_THRESH)
    else:
        swrTvals, swrMeanPSTH, swrStdPSTH, swrN = runTheThing(spikeFileName, lfpFileName, lfpTimestampFileName,
                                                              swrOutputFileName, clfunc, makeFigs=True, clusterPolygons=clusterPolygons,
                                                              tStart=swrt0, tEnd=swrt1, UP_DEFLECT_STIM_THRESH=UP_DEFLECT_STIM_THRESH)
        ctrlTvals, ctrlMeanPSTH, ctrlStdPSTH, ctrlN = runTheThing(spikeFileName, lfpFileName, lfpTimestampFileName,
                                                                  ctrlOutputFileName, clfuncCtrl, makeFigs=False, clusterPolygons=clusterPolygonsCtrl,
                                                                  tStart=ctrlt0, tEnd=ctrlt1, UP_DEFLECT_STIM_THRESH=UP_DEFLECT_STIM_THRESH)

    # print(swrMeanPSTH)
    # print(swrStdPSTH)
    # print(ctrlMeanPSTH)
    # print(ctrlStdPSTH)

    ctrlTvals += 0.2

    swrSEM = swrStdPSTH / np.sqrt(swrN)
    ctrlSEM = ctrlStdPSTH / np.sqrt(ctrlN)

    normFrac = 0.125
    normMaxIdx = math.floor(np.size(swrMeanPSTH) * normFrac)
    print(f"Normalizing to the first {normMaxIdx} datapoints, (out of {np.size(swrMeanPSTH)})")
    swrBaseline = np.mean(swrMeanPSTH[0:normMaxIdx])
    ctrlBaseline = np.mean(ctrlMeanPSTH[0:normMaxIdx])

    swrMeanPSTH /= swrBaseline
    swrSEM /= swrBaseline
    ctrlMeanPSTH /= ctrlBaseline
    ctrlSEM /= ctrlBaseline

    plt.plot(swrTvals, swrMeanPSTH, color="orange")
    plt.fill_between(swrTvals, swrMeanPSTH - swrSEM, swrMeanPSTH +
                     swrSEM, facecolor="orange", alpha=0.2)
    plt.plot(ctrlTvals, ctrlMeanPSTH, color="cyan")
    plt.fill_between(ctrlTvals, ctrlMeanPSTH - ctrlSEM, ctrlMeanPSTH +
                     ctrlSEM, facecolor="cyan", alpha=0.2)

    plt.savefig(comboOutputFileName, dpi=800)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        animal_name = sys.argv[1]
    else:
        animal_name = 'B13'
    print("Generating MUA PSTH for animal " + animal_name)

    makePSTH(animal_name, ALWAYS_REMAKE_SG_EXPORTS=False)
