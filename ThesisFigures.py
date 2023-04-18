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

from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getLoadInfo, plotFlagCheck
from PlotUtil import PlotManager
from BTData import BTData
from BTSession import BTSession
from MeasureTypes import LocationMeasure, TrialMeasure, SessionMeasure
from BTSession import BehaviorPeriod as BP


def main(plotFlags: List[str] | str = "tests", testData=False, makeCombined=True,
         runImmediate=True, numShuffles=100):
    if isinstance(plotFlags, str):
        plotFlags = [plotFlags]

    dataDir = findDataDir()
    outputDir = "ThesisFigures"
    if testData:
        outputDir += "_test"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    animalNames = ["Martin", "B13", "B14", "B16", "B17", "B18"]

    infoFileName = datetime.now().strftime("_".join(animalNames) + "_%Y%m%d_%H%M%S" + ".txt")
    pm = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=3)

    allSessionsByRat = {}
    for animalName in animalNames:
        animalInfo = getLoadInfo(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        print("loading from " + dataFilename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
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

        if plotFlagCheck(plotFlags, "tests"):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")

        if plotFlagCheck(plotFlags, "taskPerformanceOld"):
            # TODO combine these into one plot
            with pm.newFig("numWellsFound", transparent=True) as pc:
                x = range(len(sessions))
                ax = pc.ax
                assert isinstance(ax, Axes)
                ax.plot(x, [sesh.numWellsFound for sesh in sessions], label="num wells found")
                ax.set_xlabel("session")
                ax.set_ylabel("num wells found")

            with pm.newFig("taskDuration", transparent=True) as pc:
                x = range(len(sessions))
                ax = pc.ax
                assert isinstance(ax, Axes)
                ax.plot(x, [sesh.taskDuration / 60 for sesh in sessions], label="task duration")
                ax.set_xlabel("session")
                ax.set_ylabel("task duration (min)")
                ax.tick_params(axis="y", which="both", labelcolor="tab:blue",
                               labelleft=False, labelright=True)

        if plotFlagCheck(plotFlags, "taskPerformance"):
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

                ax2 = ax.twinx()
                ax2.plot(x, [sesh.taskDuration / 60 for sesh in sessions],
                         label="task duration", color="tab:blue")
                ax2.set_ylabel("task duration (min)")
                ax2.tick_params(axis="y", which="both", labelcolor="tab:blue",
                                labelleft=False, labelright=True)
                ax2.set_ylim(bottom=0)
                ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        if plotFlagCheck(plotFlags, "trialLatency"):
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

        if plotFlagCheck(plotFlags, "morrisPaper"):
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

        # home vs away probe behavior
        # at home, condition diff
        # pseudoprobe stuff
        # Cumulative num stims, num ripples
        # Cumulative wells visited during probe

        if len(plotFlags) != 0:
            print("Warning, unused plot flags:", plotFlags)

        pm.popOutputSubDir()

    if len(animalNames) > 1:
        if makeCombined:
            outputSubDir = f"combo_{'_'.join(animalNames)}"
            if testData or len(animalNames) != 6:
                pm.makeCombinedFigs(outputSubDir=outputSubDir)
            else:
                pm.makeCombinedFigs(
                    outputSubDir=outputSubDir, suggestedSubPlotLayout=(2, 3))
        if runImmediate:
            pm.runImmediateShufflesAcrossPersistentCategories(
                significantThreshold=1, numShuffles=numShuffles)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        flags = sys.argv[1:]
    else:
        flags = ["all"]
    main(flags)
