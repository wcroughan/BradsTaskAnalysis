import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
import sys

MAKE_RAW_LFP_FIGS = False
RUN_RIPPLE_DETECTION_COMPARISON = False
RUN_RIPPLE_REFRAC_PERIOD_CHECK = True

possibleDataDirs = ["/media/WDC7/", "/media/fosterlab/WDC7/", "/home/wcroughan/data/"]
dataDir = None
for dd in possibleDataDirs:
    if os.path.exists(dd):
        dataDir = dd
        break

if dataDir == None:
    print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
    exit()

globalOutputDir = os.path.join(dataDir, "figures", "20220315_labmeeting")


if len(sys.argv) >= 2:
    animalNames = sys.argv[1:]
else:
    animalNames = ['B13']
print("Plotting data for animals ", animalNames)

allSessionsByRat = {}
allSessionsWithProbeByRat = {}

for an in animalNames:
    if an == "B13":
        dataFilename = os.path.join(dataDir, "B13/processed_data/B13_bradtask.dat")
    elif an == "B14":
        dataFilename = os.path.join(dataDir, "B14/processed_data/B14_bradtask.dat")
    elif an == "Martin":
        dataFilename = os.path.join(dataDir, "Martin/processed_data/martin_bradtask.dat")
    else:
        raise Exception("Unknown rat " + an)
    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    allSessionsByRat[an] = ratData.getSessions()
    allSessionsWithProbeByRat[an] = ratData.getSessions(lambda s: s.probe_performed)


pp = PlotCtx(globalOutputDir)
for ratName in animalNames:
    print("Running rat", ratName)
    pp.setOutputDir(os.path.join(globalOutputDir, ratName))
    sessions = allSessionsByRat[ratName]
    sessionsWithProbe = [sesh for sesh in sessions if sesh.probe_performed]

    if RUN_RIPPLE_DETECTION_COMPARISON:
        preBTThreshs = np.array([sesh.prebtMeanRipplePower + 3.0 *
                                sesh.prebtStdRipplePower for sesh in sessionsWithProbe])
        ITIThreshs = np.array([sesh.ITIMeanRipplePower + 3.0 *
                               sesh.ITIStdRipplePower for sesh in sessionsWithProbe])
        probeThreshs = np.array([sesh.probeMeanRipplePower + 3.0 *
                                sesh.probeStdRipplePower for sesh in sessionsWithProbe])
        preBTNAThreshs = np.array([sesh.prebtMeanRipplePowerArtifactsRemoved + 3.0 *
                                   sesh.prebtStdRipplePowerArtifactsRemoved for sesh in sessionsWithProbe])
        preBTNAFHThreshs = np.array([sesh.prebtMeanRipplePowerArtifactsRemovedFirstHalf + 3.0 *
                                     sesh.prebtStdRipplePowerArtifactsRemovedFirstHalf for sesh in sessionsWithProbe])
        # maxval = max(np.max(preBTThreshs), np.max(ITIThreshs),
        #              np.max(probeThreshs), np.max(preBTNAThreshs),
        #              np.max(preBTNAFHThreshs))

        threshArrs = [preBTThreshs, ITIThreshs, probeThreshs, preBTNAThreshs, preBTNAFHThreshs]
        maxval = max(*[np.max(arr) for arr in threshArrs])
        threshArrLabels = ["Pre task", "ITI", "Probe",
                           "Pre task, no artifacts", "Pre task first half, no artifacts"]

        with pp.newFig("rippleThreshComparison", subPlots=(1, 3)) as axs:
            axs[0].scatter(preBTThreshs, ITIThreshs)
            axs[0].plot([0, maxval], [0, maxval])
            axs[0].set_xlabel("preBT")
            axs[0].set_ylabel("ITI")
            axs[1].scatter(preBTThreshs, probeThreshs)
            axs[1].plot([0, maxval], [0, maxval])
            axs[1].set_xlabel("preBT")
            axs[1].set_ylabel("probe")
            axs[2].scatter(ITIThreshs, probeThreshs)
            axs[2].plot([0, maxval], [0, maxval])
            axs[2].set_xlabel("ITI")
            axs[2].set_ylabel("probe")

        with pp.newFig("rippleThreshComparison_preBT", subPlots=(1, 3)) as axs:
            axs[0].scatter(preBTThreshs, preBTNAThreshs)
            axs[0].plot([0, maxval], [0, maxval])
            axs[0].set_xlabel("preBT")
            axs[0].set_ylabel("preBT Artifacts Removed")
            axs[1].scatter(preBTThreshs, preBTNAFHThreshs)
            axs[1].plot([0, maxval], [0, maxval])
            axs[1].set_xlabel("preBT")
            axs[1].set_ylabel("preBT first half Artifacts Removed")
            axs[2].scatter(preBTNAThreshs, preBTNAFHThreshs)
            axs[2].plot([0, maxval], [0, maxval])
            axs[2].set_xlabel("preBT Artifacts Removed")
            axs[2].set_ylabel("preBT first half Artifacts Removed")

        with pp.newFig("rippleThreshComparison_all", subPlots=(5, 5)) as axs:
            for row in range(5):
                if col == 0:
                    axs[row, 0].set_ylabel(threshArrLabels[row])

                yvals = threshArrs[row]
                for col in range(row, 5):
                    xvals = threshArrs[col]
                    ax = axs[row, col]
                    # axi = 5*row + col
                    ax.scatter(xvals, yvals)
                    ax.plot([0, maxval], [0, maxval])

                    if row == 0:
                        ax.set_title(threshArrLabels[col])

    for sesh in sessionsWithProbe:
        print("Running session", sesh.name)
        pp.setOutputDir(os.path.join(globalOutputDir, ratName, sesh.name))

        if RUN_RIPPLE_REFRAC_PERIOD_CHECK:
            # interruptionI = 0
            # interruptionIdx = sesh.interruptionIdxs[interruptionI]
            LFP_HZ = 1500.0
            ripCrossIdxs = np.array(sesh.btRipCrossThreshIdxsProbeStats)
            mostRecentInterruption = np.searchsorted(
                sesh.interruptionIdxs, ripCrossIdxs)
            notTooEarly = mostRecentInterruption > 0
            mostRecentInterruption = mostRecentInterruption[notTooEarly]
            ripCrossIdxs = ripCrossIdxs[notTooEarly]
            interruptionToRipInterval = (
                ripCrossIdxs - sesh.interruptionIdxs[mostRecentInterruption-1]).astype(float) / LFP_HZ

            with pp.newFig("interruption_to_ripple_delay") as ax:
                ax.hist(interruptionToRipInterval, bins=np.linspace(0, 1, 40))

        if MAKE_RAW_LFP_FIGS:
            lfpFName = sesh.bt_lfp_fnames[-1]
            print("LFP data from file {}".format(lfpFName))
            lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
            lfpV = lfpData[1]['voltage']
            lfpT = np.array(lfpData[0]['time']) / BTSession.TRODES_SAMPLING_RATE

            amps = sesh.btRipPeakAmpsProbeStats

            for i in range(100):
                maxRipIdx = np.argmax(amps)

                ripStartIdx = sesh.btRipStartIdxsProbeStats[maxRipIdx]
                ripLen = sesh.btRipLensProbeStats[maxRipIdx]
                ripPk = sesh.btRipPeakIdxsProbeStats[maxRipIdx]
                margin = int(0.15 * 1500)
                i1 = max(0, ripStartIdx - margin)
                i2 = min(ripStartIdx + ripLen + margin, len(lfpV))
                x = lfpT[i1:i2] - lfpT[ripPk]
                y = lfpV[i1:i2]

                xStart = lfpT[ripStartIdx] - lfpT[ripPk]
                xEnd = lfpT[ripStartIdx + ripLen] - lfpT[ripPk]
                ymin = np.min(y)
                ymax = np.max(y)

                with pp.newFig("rawRip_task_{}".format(i)) as ax:
                    ax.plot(x, y, zorder=1)
                    ax.plot([xStart, xStart], [ymin, ymax], c="red", zorder=0)
                    ax.plot([0, 0], [ymin, ymax], c="red", zorder=0)
                    ax.plot([xEnd, xEnd], [ymin, ymax], c="red", zorder=0)

                amps[maxRipIdx] = 0

        if RUN_RIPPLE_DETECTION_COMPARISON:
            pp.writeToInfoFile("preBTmean={}\npreBTStd={}".format(
                sesh.prebtMeanRipplePower, sesh.prebtStdRipplePower))
            pp.writeToInfoFile("ITImean={}\nITIStd={}".format(
                sesh.ITIMeanRipplePower, sesh.ITIStdRipplePower))
            pp.writeToInfoFile("probemean={}\nprobeStd={}".format(
                sesh.probeMeanRipplePower, sesh.probeStdRipplePower))
