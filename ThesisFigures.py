from typing import List
import time
from datetime import datetime
import os
import warnings
import pandas as pd
from matplotlib.axes import Axes

from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getLoadInfo
from PlotUtil import PlotManager
from BTData import BTData
from BTSession import BTSession
from MeasureTypes import LocationMeasure
from BTSession import BehaviorPeriod as BP


def main(plotFlags: List[str] | str = "tests", testData=False):
    if isinstance(plotFlags, str):
        plotFlags = [plotFlags]

    dataDir = findDataDir()
    outputDir = "ThesisFigures"
    if testData:
        outputDir += "_test"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    animalNames = parseCmdLineAnimalNames(default=["B17"])

    infoFileName = datetime.now().strftime("_".join(animalNames) + "_%Y%m%d_%H%M%S" + ".txt")
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=3)

    allSessionsByRat = {}
    for animalName in animalNames:
        loadDebug = False
        if animalName[-1] == "d":
            loadDebug = True
            animalName = animalName[:-1]
        animalInfo = getLoadInfo(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        if loadDebug:
            dataFilename += ".debug.dat"
        print("loading from " + dataFilename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[animalName] = ratData.getSessions()

    savedPlotFlags = plotFlags
    for ratName in animalNames:
        plotFlags = savedPlotFlags.copy()
        print("======================\n", ratName)
        if ratName[-1] == "d":
            ratName = ratName[:-1]
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
        pp.writeToInfoFile(s)
        print(s)

        # if hasattr(sessions[0], "probeFillTime"):
        #     sessionsWithProbeFillPast90 = [s for s in sessionsWithProbe if s.probeFillTime > 90]
        # else:
        #     sessionsWithProbeFillPast90 = sessionsWithProbe

        pp.pushOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if "tests" in plotFlags:
            plotFlags.remove("tests")

            with warnings.catch_warnings():
                warnings.filterwarnings("error")

        if "taskPerformance" in plotFlags:
            plotFlags.remove("taskPerformance")

            pp.pushOutputSubDir("taskPerformance")

            with pp.newFig("numWellsFound", transparent=True) as pc:
                x = range(len(sessions))
                ax = pc.ax
                assert isinstance(ax, Axes)
                ax.plot(x, [sesh.numWellsFound for sesh in sessions], label="num wells found")
                ax.set_xlabel("session")
                ax.set_ylabel("num wells found")

            with pp.newFig("taskDuration", transparent=True) as pc:
                x = range(len(sessions))
                ax = pc.ax
                assert isinstance(ax, Axes)
                ax.plot(x, [sesh.taskDuration / 60 for sesh in sessions], label="task duration")
                ax.set_xlabel("session")
                ax.set_ylabel("task duration (min)")
                ax.tick_params(axis="y", which="both", labelcolor="tab:blue",
                               labelleft=False, labelright=True)

            pp.popOutputSubDir()

        if "stimLocations" in plotFlags:
            plotFlags.remove("stimLocations")

            pp.pushOutputSubDir("stimLocations")

            radius = 7/18  # roughly a pixel
            LocationMeasure("stim count",
                            lambda sesh: sesh.getValueMap(
                                lambda pos: sesh.numStimsAtLocation(pos, radius=radius),
                            ), sessions, smoothDist=0).makeFigures(pp, excludeFromCombo=True)

            LocationMeasure("stim rate",
                            lambda sesh: sesh.getValueMap(
                                lambda pos: sesh.numStimsAtLocation(
                                    pos, radius=radius) / sesh.totalTimeAtPosition(BP(probe=False), pos, radius=radius),
                            ), sessions, smoothDist=0).makeFigures(pp, excludeFromCombo=True)

            pp.popOutputSubDir()

        if len(plotFlags) != 0:
            print("Warning, unused plot flags:", plotFlags)


if __name__ == "__main__":
    plotFlags = [
        # "stimLocations",
        "taskPerformance"
    ]
    main(plotFlags)
