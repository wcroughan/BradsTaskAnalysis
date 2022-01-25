import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from SpikeCalibration import MUAClusterFunc, runTheThing, makeClusterFuncFromFile, loadTrodesClusters, ConvertTimeToTrodesTS


def makePSTH(animal_name):
    if animal_name == "B14":
        runName = "20220124_100811"
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
        clusterPolygons = clusters[spikeTet-1][clusterIndex]

        swrt0 = ConvertTimeToTrodesTS(0, 5, 0)
        swrt1 = ConvertTimeToTrodesTS(0, 35, 0)
        ctrlt0 = ConvertTimeToTrodesTS(1, 5, 0)
        ctrlt1 = ConvertTimeToTrodesTS(1, 35, 0)
    # elif animal_name == "B13":
    #     data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
    #     output_dir = "/media/WDC7/B14/figs/"
    #     diagPoint = None
    # elif animal_name == "Martin":
    #     data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
    #     output_dir = "/media/WDC7/Martin/figs/"
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
    swrOutputFileName = animal_name + "_" + runName + "_swr_psth.png"
    ctrlOutputFileName = animal_name + "_" + runName + "_ctrl_psth.png"
    print(runName, "LFP tet " + str(lfpTet), "spike tet " + str(spikeTet), "cluster index " + str(clusterIndex),
          recFileName, lfpFileName, spikeFileName, clusterFileName)

    swrTvals, swrMeanPSTH, swrStdPSTH = runTheThing(spikeFileName, lfpFileName, lfpTimestampFileName,
                                                    swrOutputFileName, clfunc, makeFigs=True, clusterPolygons=clusterPolygons,
                                                    tStart=swrt0, tEnd=swrt1)
    ctrlTvals, ctrlMeanPSTH, ctrlStdPSTH = runTheThing(spikeFileName, lfpFileName, lfpTimestampFileName,
                                                       ctrlOutputFileName, clfunc, makeFigs=True, clusterPolygons=clusterPolygons,
                                                       tStart=ctrlt0, tEnd=ctrlt1)

    plt.plot(swrTvals, swrMeanPSTH, color="orange")
    plt.fill_between(swrTvals, swrMeanPSTH - swrStdPSTH, swrMeanPSTH +
                     swrStdPSTH, facecolor="orange", alpha=0.2)
    plt.plot(ctrlTvals, ctrlMeanPSTH, color="cyan")
    plt.fill_between(ctrlTvals, ctrlMeanPSTH - ctrlStdPSTH, ctrlMeanPSTH +
                     ctrlStdPSTH, facecolor="cyan", alpha=0.2)

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        animal_name = sys.argv[1]
    else:
        animal_name = 'B14'
    print("Generating MUA PSTH for animal " + animal_name)

    makePSTH(animal_name)
