import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
import sys
from UtilFunctions import getRipplePower, numWellsVisited, onWall, weekIdxForDateStr
from consts import TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE, allWellNames
import math
from matplotlib import cm

MAKE_RAW_LFP_FIGS = False
RUN_RIPPLE_DETECTION_COMPARISON = False
RUN_RIPPLE_REFRAC_PERIOD_CHECK = False
RUN_RIPPLE_BASELINE_TET_ANALYSIS = False
MAKE_PROBE_EXPLORATION_OVER_TIME_FIGS = True

possibleDataDirs = ["/media/WDC6/", "/media/fosterlab/WDC6/", "/home/wcroughan/data/"]
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
    animalNames = ['B13', 'B14']
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


pp = PlotCtx(globalOutputDir, priorityLevel=8)
for ratName in animalNames:
    print("Running rat", ratName)
    pp.setOutputDir(os.path.join(globalOutputDir, ratName))
    sessions = allSessionsByRat[ratName]
    sessionsWithProbe = [sesh for sesh in sessions if sesh.probe_performed]
    numSessionsWithProbe = len(sessionsWithProbe)

    if MAKE_PROBE_EXPLORATION_OVER_TIME_FIGS:
        numCols = math.ceil(math.sqrt(numSessionsWithProbe))
        with pp.newFig("allProbeTraces", subPlots=(numCols, numCols), figScale=0.3, priority=8) as axs:
            for si, sesh in enumerate(sessionsWithProbe):
                ax = axs[si // numCols, si % numCols]
                ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                c = "orange" if sesh.isRippleInterruption else "cyan"
                setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                ax.set_title(str(si))
            for i in range(((numSessionsWithProbe-1) % numCols)+1, numCols):
                axs[-1, i].cla()
                axs[-1, i].tick_params(axis="both", which="both", label1On=False,
                                       label2On=False, tick1On=False, tick2On=False)

        with pp.newFig("probe_numVisitedOverTime_bysidx", priority=10) as ax:
            windowSlide = 5
            t1Array = np.arange(windowSlide, 60*5, windowSlide)
            numVisited = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisited[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisited[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False)
            # plotIndividualAndAverage(axs, numVisited, t1Array)
            cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
            for si in range(numSessionsWithProbe):
                jitter = np.random.uniform(size=(numVisited.shape[1],)) * 0.5
                ax.plot(t1Array, numVisited[si, :] + jitter, color=cmap(si))
            ax.set_xticks(np.arange(0, 60*5+1, 60))

        with pp.newFig("probe_numVisitedOverTime_bysidx_offwall", priority=10) as ax:
            windowSlide = 5
            t1Array = np.arange(windowSlide, 60*5, windowSlide)
            numVisited = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisited[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisited[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False,
                        wellSubset=[w for w in allWellNames if not onWall(w)])
            # plotIndividualAndAverage(axs, numVisited, t1Array)
            cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
            for si in range(numSessionsWithProbe):
                jitter = np.random.uniform(size=(numVisited.shape[1],)) * 0.5
                ax.plot(t1Array, numVisited[si, :] + jitter, color=cmap(si))
            ax.set_xticks(np.arange(0, 60*5+1, 60))

        with pp.newFig("probe_numVisitedOverTime_byweek_offwall", priority=10) as ax:
            windowSlide = 5
            t1Array = np.arange(windowSlide, 60*5, windowSlide)
            numVisited = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisited[:] = np.nan
            weeks = np.empty((numSessionsWithProbe,), dtype=int)
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisited[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False,
                        wellSubset=[w for w in allWellNames if not onWall(w)])
                weeks[si] = weekIdxForDateStr(sesh.name.split("_")[0])
            weeks = weeks - np.min(weeks)
            cmap = cm.get_cmap("coolwarm", np.max(weeks))
            for si in range(numSessionsWithProbe):
                jitter = np.random.uniform(size=(numVisited.shape[1],)) * 0.5
                ax.plot(t1Array, numVisited[si, :] + jitter, color=cmap(weeks[si]))
            ax.set_xticks(np.arange(0, 60*5+1, 60))

        weeks = np.empty((numSessionsWithProbe,), dtype=int)
        for si, sesh in enumerate(sessionsWithProbe):
            weeks[si] = weekIdxForDateStr(sesh.name.split("_")[0])
        weeks = weeks - np.min(weeks)
        numWeekGroups = np.count_nonzero(np.diff(weeks) >= 2) + 1
        if numWeekGroups > 1:
            with pp.newFig("probe_numVisitedOverTime_byweek_offwall_separate", priority=8, subPlots=(1, numWeekGroups)) as axs:
                windowSlide = 5
                t1Array = np.arange(windowSlide, 60*5, windowSlide)
                numVisited = np.empty((numSessionsWithProbe, len(t1Array)))
                numVisited[:] = np.nan
                for si, sesh in enumerate(sessionsWithProbe):
                    t1s = t1Array * TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                    i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                    for ii, i1 in enumerate(i1Array):
                        numVisited[si, ii] = numWellsVisited(
                            sesh.probe_nearest_wells[0:i1], countReturns=False,
                            wellSubset=[w for w in allWellNames if not onWall(w)])
                cmap = cm.get_cmap("coolwarm", np.max(weeks))
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisited.shape[1],)) * 0.5
                    wgi = np.count_nonzero(np.diff(weeks[:si]) >= 2)
                    ax = axs[wgi]
                    ax.plot(t1Array, numVisited[si, :] + jitter, color=cmap(weeks[si]))
                for wgi in range(numWeekGroups):
                    ax = axs[wgi]
                    ax.set_xticks(np.arange(0, 60*5+1, 60))

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
            ripCrossIdxs = np.array(sesh.btRipCrossThreshIdxsProbeStats)
            mostRecentInterruption = np.searchsorted(
                sesh.interruptionIdxs, ripCrossIdxs)
            notTooEarly = mostRecentInterruption > 0
            mostRecentInterruption = mostRecentInterruption[notTooEarly] - 1
            ripCrossIdxs = ripCrossIdxs[notTooEarly]
            interruptionToRipInterval = (
                ripCrossIdxs - sesh.interruptionIdxs[mostRecentInterruption]).astype(float) / float(LFP_SAMPLING_RATE)

            with pp.newFig("interruption_to_ripple_delay") as ax:
                ax.hist(interruptionToRipInterval, bins=np.linspace(0, 1, 40))

        if RUN_RIPPLE_BASELINE_TET_ANALYSIS:
            lfpFName = sesh.bt_lfp_fnames[-1]
            print("LFP data from file {}".format(lfpFName))
            lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
            lfpV = lfpData[1]['voltage']
            lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

            print("LFP data from file {}".format(sesh.bt_lfp_baseline_fname))
            lfpData = MountainViewIO.loadLFP(data_file=sesh.bt_lfp_baseline_fname)
            baselfpV = lfpData[1]['voltage']
            baselfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

            # TODO in when getting power: grab probe stats (need to remake foor baseline tetrode), use those here
            # rawRipplePower, zRipplePower, _, _ = getRipplePower(
            #     btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=session.prebtMeanRipplePower, stdPower=session.prebtStdRipplePower, showPlot=showPlot)
            # session.btRipStartIdxsPreStats, session.btRipLensPreStats, session.btRipPeakIdxsPreStats, session.btRipPeakAmpsPreStats, session.btRipCrossThreshIdxsPreStats = \
            #     detectRipples(ripple_power)

            # Run detection with and without subtracting baseline power
            # Make simple 2x2 confusion matrix
            #   col 1: detected without baseline, how many also detected vs not with baseline
            #   col 2: detected with baseline, how many also detected vs not
            pass

        if MAKE_RAW_LFP_FIGS:
            lfpFName = sesh.bt_lfp_fnames[-1]
            print("LFP data from file {}".format(lfpFName))
            lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
            lfpV = lfpData[1]['voltage']
            lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

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
