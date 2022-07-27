import os
import sys
import readTrodesExtractedDataFile3
from scipy.ndimage.morphology import binary_dilation
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')
from xml.etree import ElementTree


def loadTrodesClusters(fname):
    TRODES_CLUSTER_MULT_FACTOR = 5.25
    doc = ElementTree.parse(fname)
    ntrodes = []
    for nt in doc.findall("PolygonClusters/nTrode"):
        clusters = []
        for cl in nt:
            polys = []
            for pg in cl:
                ax1 = int(pg.attrib['xAxis'])
                ax2 = int(pg.attrib['yAxis'])
                pts = []
                for pt in pg:
                    # x = int(float(pt.attrib['x']) / TRODES_CLUSTER_MULT_FACTOR)
                    # y = int(float(pt.attrib['y']) / TRODES_CLUSTER_MULT_FACTOR)
                    x = int(float(pt.attrib['x']))
                    y = int(float(pt.attrib['y']))
                    pts.append((x, y))
                polys.append((ax1, ax2, pts))
            clusters.append(polys)
        ntrodes.append(clusters)
    return ntrodes


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Return true if line segments AB and CD intersect


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def isInPolygons(polygons, points):
    ret = np.ones((points.shape[0],))
    for poly in polygons:
        ax1 = poly[0]
        ax2 = poly[1]
        pa = points[:, [ax1, ax2]]
        ppts = poly[2]
        numIntersections = np.zeros((points.shape[0],))
        for i in range(len(ppts)):
            p1 = ppts[i]
            p2 = ppts[(i + 1) % len(ppts)]
            for j in range(len(numIntersections)):
                if intersect(p1, p2, [0, 0], pa[j]):
                    numIntersections[j] += 1
        ret = np.logical_and(ret, numIntersections % 2 == 1)
    return ret


def testPolygonLogic():
    x = np.reshape(np.linspace(0, 1, 50), (50, 1))
    y = np.reshape(np.linspace(1, 0, 50), (50, 1))
    z = np.reshape(np.linspace(10, 11, 50), (50, 1))
    w = np.reshape(np.linspace(10, 11, 50), (50, 1))
    pts = np.hstack((x, y, z, w))
    print(pts.shape)

    poly = [(0, 1, [(0, 0.5), (1, 1), (0.5, 0)])]

    iip = isInPolygons(poly, pts)
    plt.scatter(pts[iip, 0], pts[iip, 1])
    plt.scatter(pts[np.logical_not(iip), 0], pts[np.logical_not(iip), 1])
    plt.show()


def ConvertTimeToTrodesTS(hours, mins, secs):
    return 30000 * (secs + 60 * (mins + 60 * (hours)))


def runTheThingWithFilenames(args, makeFigs=False):
    print(args)
    recfname = args[1]
    if not os.path.exists(recfname):
        print("rec file {} doesn't exist".format(recfname))
        return -1
    spk_tet = int(args[2])
    lfp_tet = int(args[3])
    if args[4] == "None":
        tStart = None
    else:
        ts = args[4].split(".")
        tStart = ConvertTimeToTrodesTS(float(ts[0]), float(ts[1]), float(ts[2]))
    if args[5] == "None":
        tEnd = None
    else:
        ts = args[5].split(".")
        tEnd = ConvertTimeToTrodesTS(float(ts[0]), float(ts[1]), float(ts[2]))
    clusterFileName = args[6]
    clusterIndex = int(args[7])

    fpfx = recfname[0:-4]
    fileParts = fpfx.split('/')
    runName = fileParts[-1]
    ratName = fileParts[-3]

    spike_file = fpfx + ".spikes/" + runName + ".spikes_nt" + str(spk_tet) + ".dat"
    lfp_ts_file = fpfx + ".LFP/" + runName + ".timestamps.dat"

    output_dir = os.path.join(drive_dir, ratName, "figs")
    output_fname = os.path.join(output_dir, runName + "_{}_{}_fr_psth.png".format(tStart, tEnd))

    if not os.path.exists(spike_file):
        execFileName = "/home/wcroughan/Software/Trodes210/exportspikes"
        if not os.path.exists(execFileName):
            execFileName = "/home/fosterlab/Software/Trodes210/linux/exportspikes"
        if not os.path.exists(execFileName):
            print("Couldn't find trodes executables")
            exit(0)

        syscmd = execFileName + " -rec " + recfname
        print(syscmd)
        os.system(syscmd)

    for lfp_ch in [1, 2, 3, 4]:
        lfp_file = fpfx + ".LFP/" + runName + ".LFP_nt" + str(lfp_tet) + "ch" + str(lfp_ch) + ".dat"
        if os.path.exists(lfp_file):
            break

    if not os.path.exists(lfp_file):
        execFileName = "/home/wcroughan/Software/Trodes210/exportLFP"
        if not os.path.exists(execFileName):
            execFileName = "/home/fosterlab/Software/Trodes210/linux/exportLFP"
        if not os.path.exists(execFileName):
            print("Couldn't find trodes executables")
            exit(0)

        syscmd = execFileName + " -rec " + recfname
        print(syscmd)
        os.system(syscmd)

        for lfp_ch in [1, 2, 3, 4]:
            lfp_file = fpfx + ".LFP/" + runName + ".LFP_nt" + \
                str(lfp_tet) + "ch" + str(lfp_ch) + ".dat"
            if os.path.exists(lfp_file):
                break

    print("spike_file={}\nlfp_file={}\nlfp_ts_file={}\noutput_fname={}\ntStart={}\ntEnd={}\nclusterFileName={}\nclusterIndex={}\n".format(
        spike_file, lfp_file, lfp_ts_file, output_fname, tStart, tEnd, clusterFileName, clusterIndex))

    if clusterFileName != "None":
        clusters = loadTrodesClusters(clusterFileName)
        clusterPolygons = clusters[spk_tet - 1][clusterIndex]
        clfunc = makeClusterFuncFromFile(clusterFileName, spk_tet - 1, clusterIndex)
    else:
        clusterPolygons = None
        diagPoint = np.array([3408, 682, 3053, 690], dtype=float)
        def clfunc(f, c, m): return MUAClusterFunc(f, diagPoint)

    runTheThing(spike_file, lfp_file, lfp_ts_file, output_fname, clfunc, tStart,
                tEnd, makeFigs=makeFigs, clusterPolygons=clusterPolygons)

    return 0


def MUAClusterFunc(features, channelRatios):
    # For channel ratios, pick a point along the actual noise diagonal and input that here. Less sensitive channels will be scaled up
    # Want a high-ish threshold and exclude the diagonal
    OFF_DIAG_MIN_DIST = 1000
    AMP_THRESH_MIN = 1000
    AMP_THRESH_MAX = 10000
    channelRatios = np.array(channelRatios)
    channelRatios /= np.max(channelRatios)
    scaledFeatures = features / channelRatios

    minfeature = np.min(scaledFeatures, axis=1)
    maxfeature = np.max(scaledFeatures, axis=1)
    return (maxfeature < AMP_THRESH_MAX) & (maxfeature > AMP_THRESH_MIN) & (maxfeature - minfeature > OFF_DIAG_MIN_DIST)


def makeClusterFuncFromFile(clusterFileName, trodeIndex, clusterIndex):
    clusters = loadTrodesClusters(clusterFileName)
    # print(clusters)
    for cl in clusters:
        while [] in cl:
            cl.remove([])
    print(clusters)

    def retF(features, chmaxes, maxfeature, endFeatures, chmins):
        if isinstance(clusterIndex, list):
            return any([isInPolygons(clusters[trodeIndex][i], features) for i in clusterIndex])
        elif clusterIndex == -1:
            # print(clusters)
            # print(trodeIndex)
            # print(clusters[trodeIndex])
            # print([isInPolygons(clusters[trodeIndex][i], features)
            #   for i in range(len(clusters[trodeIndex]))])
            # print(clusters[trodeIndex])
            ret1 = np.vstack([isInPolygons(clusters[trodeIndex][i], features)
                             for i in range(len(clusters[trodeIndex]))]).T
            # print(ret1.shape)
            # print(ret1)
            # print(np.sum(ret1, axis=0))
            ret = np.any(ret1, axis=1)
            # print(ret)
            return ret
        else:
            clusterPolygons = clusters[trodeIndex][clusterIndex]
            return isInPolygons(clusterPolygons, features)

    return retF


def runTheThingWithAnimalInfo(animal_name, condition, amplitude=40):
    clusterFileName = None
    tStart = None
    tEnd = None

    if animal_name == "B12":
        output_dir = "/media/WDC7/B12/figs/"
        if condition == "delay":
            if tet == 5:
                data_file = "/media/WDC7/B12/20210903_145837/20210903_145837.spikes/20210903_145837.spikes_nt5.dat"
                lfp_data_file = "/media/WDC7/B12/20210903_145837/20210903_145837.LFP/20210903_145837.LFP_nt5ch2.dat"

                def clfunc(features, chmaxes, maxfeature): return np.logical_and(
                    np.logical_and(features[:, 0] < 300, features[:, 3] > 2250), features[:, 3] < 5000)
            elif tet == 7:
                data_file = "/media/WDC7/B12/20210903_145837/20210903_145837.spikes/20210903_145837.spikes_nt7.dat"
                lfp_data_file = "/media/WDC7/B12/20210903_145837/20210903_145837.LFP/20210903_145837.LFP_nt7ch2.dat"
                def clfunc(features, chmaxes, maxfeature): return np.logical_and(np.logical_and(np.logical_and(
                    features[:, 1] > 1000, features[:, 2] < features[:, 1] - 250), features[:, 3] < features[:, 1]), chmaxes[:, 1] < 3000)
            lfp_ts_file = "/media/WDC7/B12/20210903_145837/20210903_145837.LFP/20210903_145837.timestamps.dat"
            raise Exception("Haven'te implemented changing these 'globals' yet")
            # UP_DEFLECT_STIM_THRESH = 8000
        elif condition == "interruption" or condition == "swr":
            data_file = "/media/WDC7/B12/20210901_152223/20210901_152223.spikes/20210901_152223.spikes_nt5.dat"
            lfp_data_file = "/media/WDC7/B12/20210901_152223/20210901_152223.LFP/20210901_152223.LFP_nt5ch2.dat"
            lfp_ts_file = "/media/WDC7/B12/20210901_152223/20210901_152223.LFP/20210901_152223.timestamps.dat"

            def clfunc(features, chmaxes, maxfeature): return np.logical_and(
                np.logical_and(features[:, 0] < 250, features[:, 3] > 2500), chmaxes[:, 3] < 5200)
        else:
            print("Condition uknonw")
    elif animal_name == "B13":
        output_dir = os.path.join(drive_dir, "B13/figs/")
        if amplitude == 40:
            data_file = os.path.join(
                drive_dir, "B13/20211129_162149/20211129_162149.spikes/20211129_162149.spikes_nt7.dat")
            lfp_data_file = os.path.join(
                drive_dir, "B13/20211129_162149/20211129_162149.LFP/20211129_162149.LFP_nt7ch3.dat")
            lfp_ts_file = os.path.join(
                drive_dir, "B13/20211129_162149/20211129_162149.LFP/20211129_162149.timestamps.dat")
            if condition == "delay":
                tStart = ConvertTimeToTrodesTS(2, 4, 44)
                tEnd = ConvertTimeToTrodesTS(2, 31, 53)
            elif condition == "swr" or condition == "interruption":
                tStart = ConvertTimeToTrodesTS(1, 29, 7)
                tEnd = ConvertTimeToTrodesTS(2, 2, 0)
            else:
                print("condition unkonwn")

            def clfunc(features, chmaxes, maxfeature):
                c1spks_idxs = np.logical_and(np.logical_and(np.logical_and(
                    features[:, 1] < 7500, features[:, 0] < 1000), features[:, 3] > 400), maxfeature < 1000)
                for si in range(len(c1spks_idxs)):
                    if not c1spks_idxs[si]:
                        continue
                    wf = spike_data[si][4]
                    if np.any(wf[25:] > 200):
                        c1spks_idxs[si] = False
                return c1spks_idxs
        elif amplitude in [60, 100, 140]:
            data_file = os.path.join(
                drive_dir, "B13/20211202_192250/20211202_192250.spikes/20211202_192250.spikes_nt7.dat")
            lfp_data_file = os.path.join(
                drive_dir, "B13/20211202_192250/20211202_192250.LFP/20211202_192250.LFP_nt7ch3.dat")
            lfp_ts_file = os.path.join(
                drive_dir, "B13/20211202_192250/20211202_192250.LFP/20211202_192250.timestamps.dat")

            def clfunc(features, chmaxes, maxfeature): return np.logical_and(np.logical_and(
                features[:, 3] > features[:, 2] * 0.919 + 606, features[:, 2] > 800), maxfeature < 2000)
            if amplitude == 60:
                if condition == "swr" or condition == "interruption":
                    tStart = ConvertTimeToTrodesTS(0, 14, 30)
                    tEnd = ConvertTimeToTrodesTS(0, 46, 44)
                elif condition == "delay":
                    tStart = ConvertTimeToTrodesTS(0, 47, 14)
                    tEnd = ConvertTimeToTrodesTS(1, 22, 50)
            elif amplitude == 100:
                if condition == "swr" or condition == "interruption":
                    tStart = ConvertTimeToTrodesTS(2, 3, 16)
                    tEnd = ConvertTimeToTrodesTS(2, 38, 50)
                elif condition == "delay":
                    tStart = ConvertTimeToTrodesTS(1, 22, 50)
                    tEnd = ConvertTimeToTrodesTS(2, 2, 48)
            elif amplitude == 140:
                if condition == "swr" or condition == "interruption":
                    tStart = ConvertTimeToTrodesTS(2, 39, 20)
                    tEnd = ConvertTimeToTrodesTS(3, 13, 9)
                elif condition == "delay":
                    tStart = ConvertTimeToTrodesTS(3, 13, 40)
                    tEnd = ConvertTimeToTrodesTS(3, 44, 50)
        print(tStart / 30000, tEnd / 30000)
    elif animal_name == "B14":
        output_dir = os.path.join(drive_dir, "B14/figs/")
        if amplitude == 40:
            data_file = os.path.join(
                drive_dir, "B14/20211130_152749/20211130_152749.spikes/20211130_152749.spikes_nt1.dat")
            lfp_data_file = os.path.join(
                drive_dir, "B14/20211130_152749/20211130_152749.LFP/20211130_152749.LFP_nt1ch2.dat")
            lfp_ts_file = os.path.join(
                drive_dir, "B14/20211130_152749/20211130_152749.LFP/20211130_152749.timestamps.dat")

            def clfunc(features, chmaxes, maxfeature): return np.logical_and(np.logical_and(np.logical_and(
                features[:, 1] > 1350, features[:, 1] < 2000), features[:, 2] < -955 + 1.26 * features[:, 1]), maxfeature < 2500)
            if condition == "delay":
                tStart = ConvertTimeToTrodesTS(1, 0, 51)
                tEnd = ConvertTimeToTrodesTS(1, 30, 43)
            elif condition == "swr" or condition == "interruption":
                tStart = ConvertTimeToTrodesTS(0, 32, 40)
                tEnd = ConvertTimeToTrodesTS(1, 0, 0)
            else:
                print("condition unkonwn")
        elif amplitude in [60, 100, 140]:
            data_file = os.path.join(
                drive_dir, "B14/20211202_123504/20211202_123504.spikes/20211202_123504.spikes_nt6.dat")
            lfp_data_file = os.path.join(
                drive_dir, "B14/20211202_123504/20211202_123504.LFP/20211202_123504.LFP_nt3ch1.dat")
            lfp_ts_file = os.path.join(
                drive_dir, "B14/20211202_123504/20211202_123504.LFP/20211202_123504.timestamps.dat")
            clusterFileName = os.path.join(
                drive_dir, "B14/20211202_123504/20211202_123504_time_11773455.trodesClusters")
            clfunc = makeClusterFuncFromFile(clusterFileName, 5, 0)
            if amplitude == 60:
                if condition == "swr" or condition == "interruption":
                    tStart = ConvertTimeToTrodesTS(0, 42, 4)
                    tEnd = ConvertTimeToTrodesTS(1, 25, 40)
                elif condition == "delay":
                    tStart = ConvertTimeToTrodesTS(1, 26, 17)
                    tEnd = ConvertTimeToTrodesTS(1, 55, 10)
            elif amplitude == 100:
                if condition == "swr" or condition == "interruption":
                    tStart = ConvertTimeToTrodesTS(2, 24, 35)
                    tEnd = ConvertTimeToTrodesTS(3, 11, 15)
                elif condition == "delay":
                    tStart = ConvertTimeToTrodesTS(1, 55, 10)
                    tEnd = ConvertTimeToTrodesTS(2, 24, 10)
            elif amplitude == 140:
                if condition == "swr" or condition == "interruption":
                    tStart = ConvertTimeToTrodesTS(3, 11, 45)
                    tEnd = ConvertTimeToTrodesTS(3, 33, 47)
                elif condition == "delay":
                    tStart = ConvertTimeToTrodesTS(3, 34, 21)
                    tEnd = ConvertTimeToTrodesTS(4, 5, 35)

        print(tStart / 30000, tEnd / 30000)

    else:
        print("animal uknonw")

    fname = "{0}_{1}_{2}_fr_psth.png".format(animal_name, condition, amplitude)
    output_fname = os.path.join(output_dir, fname)

    runTheThing(data_file, lfp_data_file, lfp_ts_file, output_fname, clfunc, tStart, tEnd)


def runTheThing(spike_file, lfp_file, lfp_timestamp_file, output_fname, clfunc, tStart=None, tEnd=None, makeFigs=False, clusterPolygons=None,
                UP_DEFLECT_STIM_THRESH=10000):
    DOWN_DEFLECT_NOISE_THRESH = -40000
    NOISE_BWD_EXCLUDE_SECS = 1
    NOISE_FWD_EXCLUDE_SECS = 1

    if UP_DEFLECT_STIM_THRESH is None:
        UP_DEFLECT_STIM_THRESH = 10000
    UP_DEFLECT_STIM_PROMINENCE = 5000
    SPK_BIN_SZ_MS = 10
    PSTH_MARGIN_MS = 500
    PSTH_MARGIN_SECS = float(PSTH_MARGIN_MS) / 1000.0
    SPK_BIN_SZ_SECS = float(SPK_BIN_SZ_MS) / 1000.0

    LFP_HZ = 1500

    mkfigs = np.zeros((100, ))
    if makeFigs:
        # mkfigs[0] = True  # waveforms
        # mkfigs[10] = True  # waveform just cluster, individuals
        # mkfigs[4] = True  # Old test, don't use
        # mkfigs[1] = True  # spike peak amp scatter
        # mkfigs[2] = True  # spike feats just cluster
        mkfigs[11] = True  # waveform just cluster, all waveforms
        # mkfigs[3] = True  # all LFP
        # mkfigs[5] = True  # LFP with noise marked
        mkfigs[6] = True  # LFP with peaks marked
        # mkfigs[7] = True  # peaks marked with noise peaks excluded
        mkfigs[8] = True  # LFP with clustered spikes marked
        # mkfigs[9] = True  # final PSTH fig

    spike_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(spike_file)
    spike_data = spike_data_dict['data']
    # features = np.zeros((spike_data.size, 3))
    features = np.zeros((spike_data.size, 4))
    endFeatures = np.zeros((spike_data.size, 4))
    chmaxes = np.zeros((spike_data.size, 4))
    chmins = np.zeros((spike_data.size, 4))

    next_pct = spike_data.size / 100

    print("making features for {} spikes".format(spike_data.size))
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

        endSize = 10
        endWf = wfs[:, -endSize:]
        ntp = endSize
        maxpt = np.argmax(endWf)
        maxtet = maxpt // ntp
        maxslice = maxpt % ntp
        endFeatures[si, :] = endWf[:, maxslice]

        chmaxes[si, :] = np.max(wfs, axis=1)
        chmins[si, :] = np.min(wfs, axis=1)

        if si > next_pct:
            # print("{}/{}".format(si, spike_data.size))
            next_pct += spike_data.size / 100

            if mkfigs[0]:
                plt.plot(wf1)
                plt.plot(wf2)
                plt.plot(wf3)
                plt.plot(wf4)
                plt.show()

    CHLIM = 30000
    step = 20
    if mkfigs[1]:
        plt.subplot(321)
        plt.scatter(features[::step, 0], features[::step, 1], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(322)
        plt.scatter(features[::step, 0], features[::step, 2], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(323)
        plt.scatter(features[::step, 0], features[::step, 3], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(324)
        plt.scatter(features[::step, 1], features[::step, 2], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(325)
        plt.scatter(features[::step, 1], features[::step, 3], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(326)
        plt.scatter(features[::step, 2], features[::step, 3], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.show()

    # clusterPolygons = None
    # c1spks_idxs = np.logical_and(features[:, 1] > 1100, features[:, 2] < 1000)
    # c1spks_idxs = features[:, 3] < features[:, 1] - 500

    maxfeature = np.max(features, axis=1)
    c1spks_idxs = clfunc(features, chmaxes, maxfeature, endFeatures, chmins)

    c1spks = features[c1spks_idxs, :]
    c1ts = spike_data['time'][c1spks_idxs]
    print("{}/{} spikes included in psth".format(len(c1ts), features.shape[0]))
    if mkfigs[2]:
        plt.subplot(321)
        plt.scatter(features[::step, 0], features[::step, 1], marker=".", s=0.5)
        plt.scatter(c1spks[:, 0], c1spks[:, 1], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(322)
        plt.scatter(features[::step, 0], features[::step, 2], marker=".", s=0.5)
        plt.scatter(c1spks[:, 0], c1spks[:, 2], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(323)
        plt.scatter(features[::step, 0], features[::step, 3], marker=".", s=0.5)
        plt.scatter(c1spks[:, 0], c1spks[:, 3], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(324)
        plt.scatter(features[::step, 1], features[::step, 2], marker=".", s=0.5)
        plt.scatter(c1spks[:, 1], c1spks[:, 2], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(325)
        plt.scatter(features[::step, 1], features[::step, 3], marker=".", s=0.5)
        plt.scatter(c1spks[:, 1], c1spks[:, 3], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)
        plt.subplot(326)
        plt.scatter(features[::step, 2], features[::step, 3], marker=".", s=0.5)
        plt.scatter(c1spks[:, 2], c1spks[:, 3], marker=".", s=0.5)
        plt.xlim(0, CHLIM)
        plt.ylim(0, CHLIM)

        if clusterPolygons is not None:
            for poly in clusterPolygons:
                if not isinstance(poly, list):
                    poly = [poly]
                for p in poly:
                    print(p)
                    ax1 = p[0]
                    ax2 = p[1]
                    if ax1 == 0:
                        sp = ax2
                    elif ax1 == 1:
                        sp = ax2 + 2
                    else:
                        sp = 6
                    plt.subplot(3, 2, sp)
                    vtxs = p[2]
                    for i in range(len(vtxs)):
                        p1 = vtxs[i]
                        p2 = vtxs[(i + 1) % len(vtxs)]
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red")

        plt.show()

    if mkfigs[10]:
        next_pct = np.sum(c1spks_idxs) / 100
        print(next_pct)
        for si, csi in enumerate(np.argwhere(c1spks_idxs)):
            wf1 = spike_data[csi][0][1]
            wf2 = spike_data[csi][0][2]
            wf3 = spike_data[csi][0][3]
            wf4 = spike_data[csi][0][4]

            if si > next_pct:
                print("{}/{}".format(si, np.sum(c1spks_idxs)))
                next_pct += np.sum(c1spks_idxs) / 100

                plt.plot(wf1)
                plt.plot(wf2)
                plt.plot(wf3)
                plt.plot(wf4)
                plt.legend(["1", "2", "3", "4"])
                plt.show()

    if mkfigs[11]:
        a = spike_data[c1spks_idxs]
        wf1 = np.array([v[1] for v in a])
        wf2 = np.array([v[2] for v in a])
        wf3 = np.array([v[3] for v in a])
        wf4 = np.array([v[4] for v in a])

        # step = 100
        pwf1 = wf1[::step, :]
        pwf2 = wf2[::step, :]
        pwf3 = wf3[::step, :]
        pwf4 = wf4[::step, :]

        plt.subplot(141)
        plt.plot(pwf1.T, linewidth=0.1)
        plt.subplot(142)
        plt.plot(pwf2.T, linewidth=0.1)
        plt.subplot(143)
        plt.plot(pwf3.T, linewidth=0.1)
        plt.subplot(144)
        plt.plot(pwf4.T, linewidth=0.1)

        # plt.subplot(141)
        # plt.plot(wf1.T)
        # plt.subplot(142)
        # plt.plot(wf2.T)
        # plt.subplot(143)
        # plt.plot(wf3.T)
        # plt.subplot(144)
        # plt.plot(wf4.T)

        plt.show()

    # lfp_data_file = "/home/wcroughan/data/20210108_162804/20210108_162804.LFP/20210108_162804.LFP_nt7ch1.dat"
    # lfp_data_file = "/media/WDC2/B8/20210517_180358/20210517_180358.LFP/20210517_180358.LFP_nt2ch2.dat"
    lfp_data_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_file)
    lfp_data = lfp_data_dict['data']
    lfp_data = lfp_data.astype(float)

    # lfp_ts_file = "/home/wcroughan/data/20210108_162804/20210108_162804.LFP/20210108_162804.timestamps.dat"
    # lfp_ts_file = "/media/WDC2/B8/20210517_180358/20210517_180358.LFP/20210517_180358.timestamps.dat"
    lfp_ts_dict = readTrodesExtractedDataFile3.readTrodesExtractedDataFile(lfp_timestamp_file)
    lfp_ts = lfp_ts_dict['data']

    if mkfigs[3]:
        # print(lfp_ts.dtype)
        if tStart is not None:
            inRange = np.logical_and(lfp_ts.astype(int) > tStart, lfp_ts.astype(int) < tEnd)
            ts = lfp_ts[inRange].astype(float) / 30000
            ld = lfp_data[inRange]
        else:
            ts = lfp_ts.astype(float) / 30000
            ld = lfp_data
        plt.plot(ts, ld)
        plt.show()

    if mkfigs[4]:
        LFP_HZ = 4
        NOISE_BWD_EXCLUDE_SECS = 0.5
        NOISE_FWD_EXCLUDE_SECS = 2
        lfp_data = np.sin(np.linspace(0, 2 * np.pi, num=100))
        DOWN_DEFLECT_NOISE_THRESH = -0.9

    is_noise = lfp_data < DOWN_DEFLECT_NOISE_THRESH
    NOISE_BWD_EXCLUDE_FRAMES = int(NOISE_BWD_EXCLUDE_SECS * LFP_HZ)
    NOISE_FWD_EXCLUDE_FRAMES = int(NOISE_FWD_EXCLUDE_SECS * LFP_HZ)
    STRSZHLF = max(NOISE_BWD_EXCLUDE_FRAMES, NOISE_FWD_EXCLUDE_FRAMES)
    dilstr = np.zeros((STRSZHLF * 2))
    dilstr[STRSZHLF - NOISE_BWD_EXCLUDE_FRAMES:STRSZHLF + NOISE_FWD_EXCLUDE_FRAMES] = 1

    # Ugh this isn't working after arch linux upgrade (libffi newer version)
    # noise_mask = binary_dilation(is_noise, structure=dilstr).astype(bool)
    noise_mask = np.zeros_like(is_noise)
    i = 0
    while i < len(noise_mask):
        if is_noise[i]:
            for j in range(max(0, i - NOISE_BWD_EXCLUDE_FRAMES), min(len(is_noise), NOISE_FWD_EXCLUDE_FRAMES)):
                noise_mask[j] = 1
                i = j
        i += 1

    if mkfigs[4]:
        plt.subplot(121)
        plt.plot(dilstr)
        plt.subplot(122)
        plt.plot(is_noise)
        plt.plot(noise_mask.astype(int))
        plt.show()

        exit()

    if mkfigs[5]:
        ts = lfp_ts.astype(float) / 30000
        plt.plot(ts, lfp_data)
        plt.plot(ts, noise_mask.astype(float) * -np.max(lfp_data))
        plt.scatter(ts[is_noise], lfp_data[is_noise], c="#ff7f0e")
        plt.show()

    peaks, props = signal.find_peaks(np.abs(lfp_data), height=UP_DEFLECT_STIM_THRESH,
                                     prominence=UP_DEFLECT_STIM_PROMINENCE)

    if mkfigs[6]:
        ts = lfp_ts.astype(float) / 30000
        plt.plot(ts, lfp_data)
        plt.scatter(ts[peaks], props['peak_heights'], c="#ff7f0e")

        t1 = ts[peaks[0]]
        t2 = ts[peaks[-1]]
        print("{} ({}-{}), {} total stims = {} Hz".format(
            t2 - t1, t1, t2, len(peaks), float(len(peaks)) / float(t2 - t1)
        ))
        plt.title("lfp peaks with noise included")

        plt.show()

    stim_artifact_bool = np.logical_not(noise_mask[peaks])
    peak_heights = props['peak_heights']
    stimpeaks = peaks[stim_artifact_bool]
    stimpeak_heights = peak_heights[stim_artifact_bool]

    if mkfigs[7]:
        ts = lfp_ts.astype(float) / 30000
        plt.plot(ts, lfp_data)
        plt.scatter(ts[stimpeaks], stimpeak_heights, c="#ff7f0e")
        plt.title("lfp peaks with noise excluded")
        plt.show()

    if mkfigs[8]:
        ts = lfp_ts.astype(float) / 30000
        plt.plot(ts, lfp_data, zorder=1)
        plt.scatter(c1ts.astype(float) / 30000, np.ones_like(c1ts), c="#ff7f0e", zorder=2)
        plt.show()

    NUM_PSTH_BINS = int(PSTH_MARGIN_MS / SPK_BIN_SZ_MS)
    psth = np.zeros((stimpeaks.size, 2 * NUM_PSTH_BINS))
    spike_times = c1ts.astype(float) / 30000
    stim_times = lfp_ts[stimpeaks].astype(float) / 30000

    NUM_LFP_PSTH_BINS = int(PSTH_MARGIN_SECS * LFP_HZ)
    lfp_psth = np.zeros((stimpeaks.size, 2 * NUM_LFP_PSTH_BINS))
    for si, (st, stidx) in enumerate(zip(stim_times, stimpeaks)):
        if tStart is not None and st < tStart / 30000:
            psth[si, :] = np.nan
            continue
        if tEnd is not None and st > tEnd / 30000:
            print("Reached tEnd of {} with time {}, breaking".format(tEnd / 30000, st))
            psth[si:, :] = np.nan
            break
        bins = np.linspace(st - PSTH_MARGIN_SECS, st + PSTH_MARGIN_SECS, num=2 * NUM_PSTH_BINS + 1)
        h, b = np.histogram(spike_times, bins=bins)
        psth[si, :] = h / (SPK_BIN_SZ_MS / 1000)

        lfp_psth[si, :] = lfp_data[stidx - NUM_LFP_PSTH_BINS:stidx + NUM_LFP_PSTH_BINS]

    numStimsCounted = np.count_nonzero(np.logical_not(np.isnan(psth[:, 0])))
    print("{}/{} stims included".format(numStimsCounted, psth.shape[0]))
    avg_fr_psth = np.nanmean(psth, axis=0)
    std_fr_psth = np.nanstd(psth, axis=0)
    avg_lfp_psth = np.mean(lfp_psth, axis=0)
    std_lfp_psth = np.std(lfp_psth, axis=0)
    x1 = np.linspace(-PSTH_MARGIN_SECS, PSTH_MARGIN_SECS, NUM_PSTH_BINS * 2)
    x2 = np.linspace(-PSTH_MARGIN_SECS, PSTH_MARGIN_SECS, NUM_LFP_PSTH_BINS * 2)
    # y1 = avg_fr_psth - np.min(avg_fr_psth)
    y2 = avg_lfp_psth - np.min(avg_lfp_psth)
    # y1 = y1 / np.max(y1)
    y2 = y2 / np.max(y2)
    y1 = avg_fr_psth
    y2 = y2 * np.max(y1)

    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.savefig(output_fname, dpi=800)
    if mkfigs[9]:
        plt.show()
    plt.cla()

    # z_psth = avg_fr_psth
    # z_psth -= np.nanmean(z_psth[0:10])
    # z_psth /= np.nanstd(z_psth[0:10])
    # return z_psth
    return x1, avg_fr_psth, std_fr_psth, numStimsCounted


# TODO this shouldn't run here do I need this to be global for some reason??
possible_drive_dirs = ["/media/WDC7/", "/media/fosterlab/WDC7/",
                       "/media/WDC6/", "/media/fosterlab/WDC6/"]
drive_dir = None
for dd in possible_drive_dirs:
    if os.path.exists(dd):
        drive_dir = dd
        break

if drive_dir == None:
    print("Couldnt' find data directory among any of these: {}".format(possible_drive_dirs))
    exit()


def printUsage():
    print("python SpikeCalibration.py /path/to/recfile.rec spk_tet# lfp_tet# startTime(hrs.mins.secs) endTime(hrs.mins.secs) /path/to/clusterfile.trodesClusters clusterIndex#")


if __name__ == "__main__":

    # testPolygonLogic()
    # exit()

    # print("python SpikeCalibration.py /path/to/recfile.rec spk_tet# lfp_tet# startTime(hrs.mins.secs) endTime(hrs.mins.secs) /path/to/clusterfile.trodesClusters clusterIndex#")
    # B13 after one turn:
    # args = ["", "/media/fosterlab/WDC7/B13/20211206_130837/20211206_130837.rec", "7", "7", "None", "None", "/media/fosterlab/WDC7/B13/20211206_130837/20211206_130837.trodesClusters", "0"]

    # B14 after one turn
    # args = ["", "/media/fosterlab/WDC7/B14/20211206_133702/20211206_133702.rec", "6", "3", "None", "None", "/media/fosterlab/WDC7/B14/20211206_133702/20211206_133702_time_49343784.trodesClusters", "0"]

    # B13 after two turns
    args = ["", "/media/WDC7/B13/20211206_151659/20211206_151659.rec", "7", "7", "None",
            "None", "/media/WDC7/B13/20211206_151659/20211206_151659.trodesClusters", "0"]
    # MUA
    # args = ["", "/media/WDC7/B13/20211206_151659/20211206_151659.rec",
    # "7", "7", "None", "None", "None", "0"]

    # B14 after two turns
    # args = ["", "/media/fosterlab/WDC7/B14/20211206_155328/20211206_155328.rec", "6", "3", "None", "None", "/media/fosterlab/WDC7/B14/20211206_155328/20211206_155328.trodesClusters", "3"]

    # B14 after three turns
    # args = ["", "/media/fosterlab/WDC7/B14/20211206_182602/20211206_182602.rec", "1", "3", "None", "None", "/media/fosterlab/WDC7/B14/20211206_182602/20211206_182602_time_34936241.trodesClusters", "0"]
    # again!
    # args = ["", "/media/fosterlab/WDC7/B14/20211206_190308/20211206_190308.rec", "1", "3", "None", "None",
    # "/media/fosterlab/WDC7/B14/20211206_182602/20211206_182602_time_34936241.trodesClusters", "0"]

    # B14 MUA
    # args = ["", "/media/WDC7/B14/20211206_190308/20211206_190308.rec",
    # "3", "3", "None", "None", "None", "0"]

    if args is not None:
        if runTheThingWithFilenames(args, makeFigs=True):
            print("Couldn't run it!")
    elif len(sys.argv) > 1:
        if runTheThingWithFilenames(sys.argv):
            printUsage()
    else:
        runTheThingWithAnimalInfo("B14", "swr", 140, makeFigs=True)
        runTheThingWithAnimalInfo("B14", "delay", 140)
        runTheThingWithAnimalInfo("B14", "swr", 60)
        runTheThingWithAnimalInfo("B14", "delay", 60)
        runTheThingWithAnimalInfo("B14", "swr", 100)
        runTheThingWithAnimalInfo("B14", "delay", 100)
        runTheThingWithAnimalInfo("B14", "swr", 40)
        runTheThingWithAnimalInfo("B14", "delay", 40)
