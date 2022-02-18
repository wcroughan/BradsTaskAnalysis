import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from SpikeCalibration import MUAClusterFunc, runTheThing, makeClusterFuncFromFile, loadTrodesClusters, ConvertTimeToTrodesTS


def makePSTH(animal_name):
    possible_drive_dirs = ["/media/WDC7/", "/media/fosterlab/WDC7/"]
    drive_dir = None
    for dd in possible_drive_dirs:
        if os.path.exists(dd):
            drive_dir = dd
            break

    if drive_dir == None:
        print("Couldnt' find data directory among any of these: {}".format(possible_drive_dirs))
        return

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

        clusterFileName = os.path.join(drive_dir, "B13/{}/{}.trodesClusters".format(runName, runName))

        clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet-1, clusterIndex)
        clusters = loadTrodesClusters(clusterFileName)
        # clusterPolygons = clusters[spikeTet-1][clusterIndex]
        clusterPolygons = clusters[spikeTet-1]

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

            clusterFileName = os.path.join(drive_dir, "B14/{}/{}.trodesClusters".format(runName, runName))
            f = makeClusterFuncFromFile(clusterFileName, spikeTet-1, clusterIndex)
            def clfunc(features, chmaxes, maxfeature, endFeatures, chmins):
                ret = f(features, chmaxes, maxfeature)
                return np.logical_and(ret, endFeatures[:,2] < 0)
            clusters = loadTrodesClusters(clusterFileName)
            clusterPolygons = clusters[spikeTet-1]
        elif False:
            runName = "20220202_125441"
            swrt0 = ConvertTimeToTrodesTS(4, 2, 38)
            swrt1 = ConvertTimeToTrodesTS(4, 33, 0)
            ctrlt0 = ConvertTimeToTrodesTS(5, 13, 1)
            ctrlt1 = ConvertTimeToTrodesTS(5, 44, 0)
            spikeTet = 2
        else:
            runName = "20220217_142910"
            swrt0 = ConvertTimeToTrodesTS(0, 21, 2)
            swrt1 = ConvertTimeToTrodesTS(0, 59, 11)
            ctrlt0 = ConvertTimeToTrodesTS(1, 38, 22)
            ctrlt1 = ConvertTimeToTrodesTS(2, 10, 45)
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
            clusterFileName = os.path.join(drive_dir, "B14/{}/{}.trodesClusters".format(runName, runName))
            clfunc = makeClusterFuncFromFile(clusterFileName, spikeTet-1, clusterIndex)
            clusters = loadTrodesClusters(clusterFileName)
            # clusterPolygons = clusters[spikeTet-1][clusterIndex]
            clusterPolygons = clusters[spikeTet-1]

        outputDir = os.path.join(drive_dir, "B14/figs/")

    # elif animal_name == "Martin":
    #     data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
    #     outputDir = "/media/WDC7/Martin/figs/"
    #     diagPoint = None
    else:
        raise Exception("Unknown rat " + animal_name)

    # If necessary, generate lfp file
    possible_sg_dirs = ["/home/wcroughan/Software/Trodes21/", "/home/wcroughan/Software/Trodes21/linux/", "/home/fosterlab/Software/Trodes223/"]
    sg_dir = None
    for dd in possible_sg_dirs:
        if os.path.exists(dd):
            sg_dir = dd
            break
    if sg_dir == None:
        print("Couldnt' find spike gadgets directory among any of these: {}".format(possible_sg_dirs))
        return

    if not os.path.exists(lfpFileName):
        syscmd = os.path.join(sg_dir, "exportLFP") + " -rec " + recFileName
        print(syscmd)
        os.system(syscmd)
        lfpfilelist = glob.glob(gl)
        lfpFileName = lfpfilelist[0]

    # If necessary, generate spike file
    if not os.path.exists(spikeFileName):
        syscmd = os.path.join(sg_dir, "exportspikes") + " -rec " + recFileName
        print(syscmd)
        os.system(syscmd)

    lfpTimestampFileName = ".".join(lfpFileName.split(".")[0:-2]) + ".timestamps.dat"
    swrOutputFileName = os.path.join(outputDir, animal_name + "_" + runName + "_swr_psth.png")
    ctrlOutputFileName = os.path.join(outputDir, animal_name + "_" + runName + "_ctrl_psth.png")
    comboOutputFileName = os.path.join(outputDir, animal_name + "_" + runName + "_combo_psth.png")
    print(runName, "LFP tet " + str(lfpTet), "spike tet " + str(spikeTet), "cluster index " + str(clusterIndex),
          recFileName, lfpFileName, spikeFileName, clusterFileName)

    swrTvals, swrMeanPSTH, swrStdPSTH, swrN = runTheThing(spikeFileName, lfpFileName, lfpTimestampFileName,
                                                          swrOutputFileName, clfunc, makeFigs=True, clusterPolygons=clusterPolygons,
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

    plt.savefig(comboOutputFileName, dpi=800)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        animal_name = sys.argv[1]
    else:
        animal_name = 'B14'
    print("Generating MUA PSTH for animal " + animal_name)

    makePSTH(animal_name)
