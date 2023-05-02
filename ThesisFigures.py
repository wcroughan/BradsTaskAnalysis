from typing import List
import time
from datetime import datetime
import os
import warnings
import pandas as pd
from matplotlib.axes import Axes
import sys
import numpy as np
from matplotlib.ticker import MaxNLocator

from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getLoadInfo, flagCheck
from PlotUtil import PlotManager
from BTData import BTData
from BTSession import BTSession
from MeasureTypes import LocationMeasure, TrialMeasure, SessionMeasure
from BTSession import BehaviorPeriod as BP
from consts import TRODES_SAMPLING_RATE


def main(plotFlags: List[str] | str = "tests", testData=False, makeCombined=True,
         runImmediate=True, numShuffles=100, excludeNoStim=False):
    print("ThesisFigures.py")
    print("plotFlags =", plotFlags)
    print("testData =", testData)
    print("makeCombined =", makeCombined)
    print("runImmediate =", runImmediate)
    print("numShuffles =", numShuffles)
    print("excludeNoStim =", excludeNoStim)

    if isinstance(plotFlags, str):
        plotFlags = [plotFlags]

    dataDir = findDataDir()
    outputDir = "ThesisFigures"
    if testData:
        outputDir += "_test"
    if excludeNoStim:
        outputDir += "_noStim"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    fontSize = 12
    labelFontSize = 18

    animalNames = ["Martin", "B13", "B14", "B16", "B17", "B18"]
    # if excludeNoStim:
    #     animalNames = [n for n in animalNames if n != "Martin"]

    infoFileName = datetime.now().strftime(
        "_".join(animalNames) + "_%Y%m%d_%H%M%S" + ".txt")
    pm = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=3)

    allSessionsByRat = {}
    for animalName in animalNames:
        animalInfo = getLoadInfo(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(
            animalInfo.output_dir, animalInfo.out_filename)
        print("loading from " + dataFilename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        if excludeNoStim:
            allSessionsByRat[animalName] = ratData.getSessions(
                lambda s: not s.isNoInterruption)
        else:
            allSessionsByRat[animalName] = ratData.getSessions()

    savedPlotFlags = plotFlags
    for ratName in animalNames:
        plotFlags = savedPlotFlags.copy()
        print("======================\n", ratName)
        sessions: List[BTSession] = allSessionsByRat[ratName]
        if testData:
            sessions = sessions[:6]
        # nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probePerformed]
        # numSessionsWithProbe = len(sessionsWithProbe)
        # sessionsWithLog = [sesh for sesh in sessions if sesh.hasActivelinkLog]

        ctrlSessionsWithProbe = [sesh for sesh in sessions if (
            not sesh.isRippleInterruption) and sesh.probePerformed]
        swrSessionsWithProbe = [
            sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probePerformed]
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)

        # ctrlSessions = [sesh for sesh in sessions if not sesh.isRippleInterruption]
        # swrSessions = [sesh for sesh in sessions if sesh.isRippleInterruption]
        # nCtrlSessions = len(ctrlSessions)
        # nSWRSessions = len(swrSessions)

        print(f"{len(sessions)} sessions ({len(sessionsWithProbe)} "
              f"with probe: {nCtrlWithProbe} Ctrl, {nSWRWithProbe} SWR)")

        data = [[
            sesh.name, sesh.conditionString(), sesh.homeWell, len(
                sesh.visitedAwayWells), sesh.fillTimeCutoff()
        ] for si, sesh in enumerate(sessions)]
        df = pd.DataFrame(data, columns=[
            "name", "condition", "home well", "num aways found", "fill time"
        ])
        s = df.to_string()
        pm.writeToInfoFile(s)
        print(s)

        # if hasattr(sessions[0], "probeFillTime"):
        #     sessionsWithProbeFillPast90 = [s for s in sessionsWithProbe if s.probeFillTime > 90]
        # else:
        #     sessionsWithProbeFillPast90 = sessionsWithProbe

        pm.pushOutputSubDir(ratName)
        if len(animalNames) > 1:
            pm.setStatCategory("rat", ratName)

        if flagCheck(plotFlags, "tests"):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")

        if flagCheck(plotFlags, "taskPerformanceOld", excludeFromAll=True):
            with pm.newFig("numWellsFound", transparent=True) as pc:
                x = range(len(sessions))
                ax = pc.ax
                assert isinstance(ax, Axes)
                ax.plot(x, [sesh.numWellsFound for sesh in sessions],
                        label="num wells found")
                ax.set_xlabel("session")
                ax.set_ylabel("num wells found")

            with pm.newFig("taskDuration", transparent=True) as pc:
                x = range(len(sessions))
                ax = pc.ax
                assert isinstance(ax, Axes)
                ax.plot(x, [sesh.taskDuration /
                        60 for sesh in sessions], label="task duration")
                ax.set_xlabel("session")
                ax.set_ylabel("task duration (min)")
                ax.tick_params(axis="y", which="both", labelcolor="tab:blue",
                               labelleft=False, labelright=True)

        if flagCheck(plotFlags, "taskPerformance"):
            # Now let's plot both of the above using two y-axes on the same plot
            with pm.newFig("taskPerformance", transparent=True) as pc:
                x = np.arange(len(sessions)) + 1
                ax = pc.ax
                assert isinstance(ax, Axes)
                ax.plot(x, [sesh.numWellsFound for sesh in sessions],
                        label="num wells found", color="tab:red")
                ax.set_xlabel("session")
                ax.set_ylabel("number of trials completed")
                ax.tick_params(axis="y", which="both", labelcolor="tab:red",
                               labelleft=True, labelright=False)
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                # Set the font size for axis tick labels
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontSize)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontSize)
                # Set the font size for the axis labels
                ax.xaxis.label.set_fontsize(labelFontSize)
                ax.yaxis.label.set_fontsize(labelFontSize)

                ax2 = ax.twinx()
                ax2.plot(x, [sesh.taskDuration / 60 for sesh in sessions],
                         label="task duration", color="tab:blue")
                ax2.set_ylabel("task duration (min)")
                ax2.tick_params(axis="y", which="both", labelcolor="tab:blue",
                                labelleft=False, labelright=True)
                ax2.set_ylim(bottom=0)
                ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                for tick in ax2.yaxis.get_major_ticks():
                    tick.label.set_fontsize(fontSize)
                ax2.yaxis.label.set_fontsize(labelFontSize)

        if flagCheck(plotFlags, "trialLatency"):
            # measureFunc(session, trialStart_posIdx, trialEnd_posIdx, trial type ("home" | "away")) -> measure value
            tm = TrialMeasure("Latency (s)", lambda sesh, start, end,
                              ttype: sesh.btPos_secs[end] - sesh.btPos_secs[start], sessions)
            tm.makeFigures(pm, plotFlags="averages", runStats=False)
            tm = TrialMeasure("Path length (ft)", lambda sesh, start, end,
                              ttype: sesh.btPos_secs[end] - sesh.btPos_secs[start], sessions)
            tm.makeFigures(pm, plotFlags="averages", runStats=False)
            tm = TrialMeasure("Normalized path length", lambda sesh, start, end,
                              ttype: sesh.normalizedPathLength(False, start, end), sessions)
            tm.makeFigures(pm, plotFlags="averages", runStats=False)

        if flagCheck(plotFlags, "morrisPaper"):
            sm = SessionMeasure("Latency to home 2-4 (sec)",
                                lambda sesh: np.sum([sesh.btPos_secs[i2] - sesh.btPos_secs[i1] for i1, i2 in zip(
                                    sesh.getAllTrialPosIdxs()[2:7:2, 0], sesh.getAllTrialPosIdxs()[2:7:2, 1])]),
                                sessions)
            sm.makeFigures(pm, plotFlags="violin", numShuffles=numShuffles)
            sm = SessionMeasure("Path length to home 2-4 (ft)",
                                lambda sesh: np.sum([sesh.pathLength(False, i1, i2) for i1, i2 in zip(
                                    sesh.getAllTrialPosIdxs()[2:7:2, 0], sesh.getAllTrialPosIdxs()[2:7:2, 1])]),
                                sessions)
            sm.makeFigures(pm, plotFlags="violin", numShuffles=numShuffles)
            sm = SessionMeasure("normalized Path length to home 2-4",
                                lambda sesh: np.sum([sesh.normalizedPathLength(False, i1, i2) for i1, i2 in zip(
                                    sesh.getAllTrialPosIdxs()[2:7:2, 0], sesh.getAllTrialPosIdxs()[2:7:2, 1])]),
                                sessions)
            sm.makeFigures(pm, plotFlags="violin", numShuffles=numShuffles)

        if flagCheck(plotFlags, "probeBehavior"):
            # home vs away probe behavior
            lm = LocationMeasure(f"number of visits to well during probe",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(
                                         BP(probe=True), pos, 0.5),
                                 ), sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pm, plotFlags=["ctrlByCondition", "measureVsCtrl",
                           "measureByCondition"], numShuffles=numShuffles)
            lm = LocationMeasure(f"total time at well during probe (sec)",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(
                                         BP(probe=True), pos, 0.5),
                                 ), sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pm, plotFlags=["ctrlByCondition", "measureVsCtrl",
                           "measureByCondition"], numShuffles=numShuffles)

        if flagCheck(plotFlags, "pseudoprobeBehavior"):
            # pseudoprobe stuff
            lm = LocationMeasure("BS pseudoprobe number of visits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(
                                         BP(probe=False, trialInterval=(0, 1)), pos, 0.5),
                                 ), sessions, smoothDist=0.0,
                                 sessionValLocation=LocationMeasure.prevSessionHomeLocation)
            lm.makeDualCategoryFigures(pm, numShuffles=numShuffles)

        if flagCheck(plotFlags, "cumulativeLFP"):
            windowSize = 5
            numSessions = len(sessions)
            bins = np.arange(0, 60*20+1, windowSize)
            stimCounts = np.empty((numSessions, len(bins)-1))
            rippleCounts = np.empty((numSessions, len(bins)-1))
            itiBins = np.arange(0, 60*1.5+1, windowSize)
            itiStimCounts = np.empty((numSessions, len(itiBins)-1))
            itiRippleCounts = np.empty((numSessions, len(itiBins)-1))
            probeBins = np.arange(0, 60*5+1, windowSize)
            probeStimCounts = np.empty((numSessions, len(probeBins)-1))
            probeRippleCounts = np.empty((numSessions, len(probeBins)-1))
            for si, sesh in enumerate(sessions):
                # Task
                stimTs = np.array(sesh.lfpBumps_ts)
                stimTs = stimTs[np.logical_and(stimTs > sesh.btPos_ts[0], stimTs <
                                               sesh.btPos_ts[-1])] - sesh.btPos_ts[0]
                stimTs = stimTs.astype(np.float64)
                stimTs /= TRODES_SAMPLING_RATE
                stimCounts[si, :], _ = np.histogram(stimTs, bins=bins)

                for i in reversed(range(len(bins)-1)):
                    if stimCounts[si, i] != 0:
                        break
                    stimCounts[si, i] = np.nan

                ripTs = np.array([r.start_ts for r in sesh.btRipsPreStats])
                ripTs = ripTs[np.logical_and(ripTs > sesh.btPos_ts[0], ripTs <
                                             sesh.btPos_ts[-1])] - sesh.btPos_ts[0]
                ripTs = ripTs.astype(np.float64)
                ripTs /= TRODES_SAMPLING_RATE
                rippleCounts[si, :], _ = np.histogram(ripTs, bins=bins)

                for i in reversed(range(len(bins)-1)):
                    if rippleCounts[si, i] != 0:
                        break
                    rippleCounts[si, i] = np.nan

                if not sesh.probePerformed:
                    itiStimCounts[si, :] = np.nan
                    itiRippleCounts[si, :] = np.nan
                    probeStimCounts[si, :] = np.nan
                    probeRippleCounts[si, :] = np.nan
                else:
                    # ITI
                    stimTs = np.array(sesh.lfpBumps_ts)
                    stimTs = stimTs[np.logical_and(stimTs > sesh.itiLfpStart_ts, stimTs <
                                                   sesh.itiLfpEnd_ts)] - sesh.itiLfpStart_ts
                    stimTs = stimTs.astype(np.float64)
                    stimTs /= TRODES_SAMPLING_RATE
                    itiStimCounts[si, :], _ = np.histogram(
                        stimTs, bins=itiBins)

                    for i in reversed(range(len(itiBins)-1)):
                        if itiStimCounts[si, i] != 0:
                            break
                        itiStimCounts[si, i] = np.nan

                    ripTs = np.array(
                        [r.start_ts for r in sesh.itiRipsPreStats])
                    ripTs = ripTs[np.logical_and(ripTs > sesh.itiLfpStart_ts, ripTs <
                                                 sesh.itiLfpEnd_ts)] - sesh.itiLfpStart_ts
                    ripTs = ripTs.astype(np.float64)
                    ripTs /= TRODES_SAMPLING_RATE
                    itiRippleCounts[si, :], _ = np.histogram(
                        ripTs, bins=itiBins)

                    for i in reversed(range(len(itiBins)-1)):
                        if itiRippleCounts[si, i] != 0:
                            break
                        itiRippleCounts[si, i] = np.nan

                    # Probe
                    stimTs = np.array(sesh.lfpBumps_ts)
                    stimTs = stimTs[np.logical_and(stimTs > sesh.probePos_ts[0], stimTs <
                                                   sesh.probePos_ts[-1])] - sesh.probePos_ts[0]
                    stimTs = stimTs.astype(np.float64)
                    stimTs /= TRODES_SAMPLING_RATE
                    probeStimCounts[si, :], _ = np.histogram(
                        stimTs, bins=probeBins)

                    for i in reversed(range(len(probeBins)-1)):
                        if probeStimCounts[si, i] != 0:
                            break
                        probeStimCounts[si, i] = np.nan

                    ripTs = np.array(
                        [r.start_ts for r in sesh.probeRipsProbeStats])
                    ripTs = ripTs[np.logical_and(ripTs > sesh.probePos_ts[0], ripTs <
                                                 sesh.probePos_ts[-1])] - sesh.probePos_ts[0]
                    ripTs = ripTs.astype(np.float64)
                    ripTs /= TRODES_SAMPLING_RATE
                    probeRippleCounts[si, :], _ = np.histogram(
                        ripTs, bins=probeBins)

                    for i in reversed(range(len(probeBins)-1)):
                        if probeRippleCounts[si, i] != 0:
                            break
                        probeRippleCounts[si, i] = np.nan

            stimCounts = np.cumsum(stimCounts, axis=1)
            rippleCounts = np.cumsum(rippleCounts, axis=1)
            itiStimCounts = np.cumsum(itiStimCounts, axis=1)
            itiRippleCounts = np.cumsum(itiRippleCounts, axis=1)
            probeStimCounts = np.cumsum(probeStimCounts, axis=1)
            probeRippleCounts = np.cumsum(probeRippleCounts, axis=1)

            swrIdx = np.array([sesh.isRippleInterruption for sesh in sessions])
            ctrlIdx = ~swrIdx
            stimCountsSWR = stimCounts[swrIdx, :]
            stimCountsCtrl = stimCounts[ctrlIdx, :]
            rippleCountsSWR = rippleCounts[swrIdx, :]
            rippleCountsCtrl = rippleCounts[ctrlIdx, :]
            itiStimCountsSWR = itiStimCounts[swrIdx, :]
            itiStimCountsCtrl = itiStimCounts[ctrlIdx, :]
            itiRippleCountsSWR = itiRippleCounts[swrIdx, :]
            itiRippleCountsCtrl = itiRippleCounts[ctrlIdx, :]
            probeStimCountsSWR = probeStimCounts[swrIdx, :]
            probeStimCountsCtrl = probeStimCounts[ctrlIdx, :]
            probeRippleCountsSWR = probeRippleCounts[swrIdx, :]
            probeRippleCountsCtrl = probeRippleCounts[ctrlIdx, :]

            with pm.newFig("lfp task stimCounts") as pc:
                ax = pc.ax
                ax.plot(bins[1:], stimCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pm.newFig("lfp task rippleCounts") as pc:
                ax = pc.ax
                ax.plot(bins[1:], rippleCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")
            with pm.newFig("lfp task stimCounts by cond") as pc:
                ax = pc.ax
                ax.plot(bins[1:], stimCountsSWR.T, c="orange", zorder=1)
                ax.plot(bins[1:], stimCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pm.newFig("lfp task rippleCounts by cond") as pc:
                ax = pc.ax
                ax.plot(bins[1:], rippleCountsSWR.T, c="orange", zorder=1)
                ax.plot(bins[1:], rippleCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")

            with pm.newFig("lfp iti stimCounts") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiStimCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pm.newFig("lfp iti rippleCounts") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiRippleCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")
            with pm.newFig("lfp iti stimCounts by cond") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiStimCountsSWR.T, c="orange", zorder=1)
                ax.plot(itiBins[1:], itiStimCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pm.newFig("lfp iti rippleCounts by cond") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiRippleCountsSWR.T,
                        c="orange", zorder=1)
                ax.plot(itiBins[1:], itiRippleCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")

            with pm.newFig("lfp probe stimCounts") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeStimCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pm.newFig("lfp probe rippleCounts") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeRippleCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")
            with pm.newFig("lfp probe stimCounts by cond") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeStimCountsSWR.T,
                        c="orange", zorder=1)
                ax.plot(probeBins[1:], probeStimCountsCtrl.T,
                        c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pm.newFig("lfp probe rippleCounts by cond") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeRippleCountsSWR.T,
                        c="orange", zorder=1)
                ax.plot(probeBins[1:], probeRippleCountsCtrl.T,
                        c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")

        # Cumulative num stims, num ripples

        if len(plotFlags) != 0:
            print("Warning, unused plot flags:", plotFlags)

        pm.popOutputSubDir()

    if len(animalNames) > 1:
        if makeCombined:
            outputSubDir = f"combo_{'_'.join(animalNames)}"
            if not testData and len(animalNames) in [5, 6]:
                pm.makeCombinedFigs(
                    outputSubDir=outputSubDir, suggestedSubPlotLayout=(2, 3))
            else:
                pm.makeCombinedFigs(outputSubDir=outputSubDir)
        if runImmediate:
            pm.runImmediateShufflesAcrossPersistentCategories(
                significantThreshold=1, numShuffles=numShuffles)


if __name__ == "__main__":
    kwargs = {}
    flags = []

    if len(sys.argv) > 1:
        flags = sys.argv[1:]

        if flagCheck(flags, "--excludeNoStim"):
            kwargs["excludeNoStim"] = True

        if flagCheck(flags, "--moreShuffles"):
            kwargs["numShuffles"] = 5000

    if len(flags) == 0:
        flags = ["all"]
    main(flags, **kwargs)
