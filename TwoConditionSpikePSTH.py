import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from SpikeCalibration import MUAClusterFunc, runTheThing, makeClusterFuncFromFile, loadTrodesClusters, ConvertTimeToTrodesTS


def makePSTH(animal_name):
    if animal_name == "B13":
        runName = "20220125_094035"
        lfpTet = 7
        spikeTet = 7
        clusterIndex = -1

        recFileName = "/media/WDC7/B13/{}/{}.rec".format(runName, runName)
        gl = "/media/WDC7/B13/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(
            runName, runName, runName, lfpTet)
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = "/media/WDC7/B13/{}/{}.spikes/{}.spikes_nt{}.dat".format(
            runName, runName, runName, spikeTet)

        clusterFileName = "/media/WDC7/B13/{}/{}.trodesClusters".format(runName, runName)

        clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet-1, clusterIndex)
        clusters = loadTrodesClusters(clusterFileName)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygons = clusters[spikeTet-1]

        swrt0 = ConvertTimeToTrodesTS(0, 5, 0)
        swrt1 = ConvertTimeToTrodesTS(0, 38, 0)
        ctrlt0 = ConvertTimeToTrodesTS(1, 7, 30)
        ctrlt1 = ConvertTimeToTrodesTS(1, 38, 0)

        outputDir = "/media/WDC7/B13/figs/"
    elif animal_name == "B14":
        if False:
            runName = "20220124_100811"
            swrt0 = ConvertTimeToTrodesTS(0, 5, 0)
            swrt1 = ConvertTimeToTrodesTS(0, 35, 0)
            ctrlt0 = ConvertTimeToTrodesTS(1, 5, 0)
            ctrlt1 = ConvertTimeToTrodesTS(1, 35, 0)
        else:
            runName = "20220126_110046"
            swrt0 = ConvertTimeToTrodesTS(0, 1, 0)
            swrt1 = ConvertTimeToTrodesTS(0, 34, 0)
            ctrlt0 = ConvertTimeToTrodesTS(1, 6, 0)
            ctrlt1 = ConvertTimeToTrodesTS(1, 36, 0)

        lfpTet = 2
        spikeTet = 2
        clusterIndex = -1

        recFileName = "/media/WDC7/B14/{}/{}.rec".format(runName, runName)
        gl = "/media/WDC7/B14/{}/{}.LFP/{}.LFP_nt{}ch*.dat".format(
            runName, runName, runName, lfpTet)
        lfpfilelist = glob.glob(gl)
        if len(lfpfilelist) > 0:
            lfpFileName = lfpfilelist[0]
        else:
            lfpFileName = "nofile"

        spikeFileName = "/media/WDC7/B14/{}/{}.spikes/{}.spikes_nt{}.dat".format(
            runName, runName, runName, spikeTet)

        clusterFileName = "/media/WDC7/B14/{}/{}.trodesClusters".format(runName, runName)

        clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet-1, clusterIndex)
        clusters = loadTrodesClusters(clusterFileName)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygons = clusters[spikeTet-1]

        outputDir = "/media/WDC7/B14/figs/"

    # elif animal_name == "Martin":
    #     data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
    #     outputDir = "/media/WDC7/Martin/figs/"
    #     diagPoint = None
    else:
        raise Exception("Unknown rat " + animal_name)

    # If necessary, generate lfp file
    if not os.path.exists(lfpFileName):
        if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
            syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + recFileName
        else:
            syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + recFileName
        print(syscmd)
        os.system(syscmd)

        lfpfilelist = glob.glob(gl)
        lfpFileName = lfpfilelist[0]

    # If necessary, generate spike file
    if not os.path.exists(spikeFileName):
        if os.path.exists("/home/wcroughan/Software/Trodes21/exportspikes"):
            syscmd = "/home/wcroughan/Software/Trodes21/exportspikes -rec " + recFileName
        else:
            syscmd = "/home/wcroughan/Software/Trodes21/linux/exportspikes -rec " + recFileName
        print(syscmd)
        os.system(syscmd)

    lfpTimestampFileName = ".".join(lfpFileName.split(".")[0:-2]) + ".timestamps.dat"
    swrOutputFileName = os.path.join(outputDir, animal_name + "_" + runName + "_swr_psth.png")
    ctrlOutputFileName = os.path.join(outputDir, animal_name + "_" + runName + "_ctrl_psth.png")
    print(runName, "LFP tet " + str(lfpTet), "spike tet " + str(spikeTet), "cluster index " + str(clusterIndex),
          recFileName, lfpFileName, spikeFileName, clusterFileName)

    swrTvals, swrMeanPSTH, swrStdPSTH, swrN = runTheThing(spikeFileName, lfpFileName, lfpTimestampFileName,
                                                          swrOutputFileName, clfunc, makeFigs=False, clusterPolygons=clusterPolygons,
                                                          tStart=swrt0, tEnd=swrt1)
    ctrlTvals, ctrlMeanPSTH, ctrlStdPSTH, ctrlN = runTheThing(spikeFileName, lfpFileName, lfpTimestampFileName,
                                                              ctrlOutputFileName, clfunc, makeFigs=False, clusterPolygons=clusterPolygons,
                                                              tStart=ctrlt0, tEnd=ctrlt1)

    # print(swrMeanPSTH)
    # print(swrStdPSTH)
    # print(ctrlMeanPSTH)
    # print(ctrlStdPSTH)

    ctrlTvals += 0.2

    swrSEM = swrStdPSTH / np.sqrt(swrN)
    ctrlSEM = ctrlStdPSTH / np.sqrt(ctrlN)

    plt.plot(swrTvals, swrMeanPSTH, color="orange")
    plt.fill_between(swrTvals, swrMeanPSTH - swrSEM, swrMeanPSTH +
                     swrSEM, facecolor="orange", alpha=0.2)
    plt.plot(ctrlTvals, ctrlMeanPSTH, color="cyan")
    plt.fill_between(ctrlTvals, ctrlMeanPSTH - ctrlSEM, ctrlMeanPSTH +
                     ctrlSEM, facecolor="cyan", alpha=0.2)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        animal_name = sys.argv[1]
    else:
        animal_name = 'B14'
    print("Generating MUA PSTH for animal " + animal_name)

    makePSTH(animal_name)
