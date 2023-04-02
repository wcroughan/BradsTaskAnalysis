# First tasks tomorrow:
# - run makeFigures("stimsVsOccupancy") in ThesisCommittee2023.py
# - choose a set of params for those
# - Get to work on the slides ya lazy landlubber!

from datetime import datetime
import os
from typing import List
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

from UtilFunctions import findDataDir, plotFlagCheck, getLoadInfo
from PlotUtil import PlotManager
from BTData import BTData
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from MeasureTypes import SessionMeasure, LocationMeasure
from consts import TRODES_SAMPLING_RATE

# This file will take the chosen measures and run them on all the rats

# Plots to make:
# early and late task away trials. All 4 basic measures. Next/last session difference.
#   total: r1s.5
#   avg: r1s.5
#   num: r.5s.5
#   curve: r1s.5
# Late task, avg dwell time at home
#   r1s.5
# Total time across all sym during late home trials
#   r.5s.5
# Distance travelled on wall before first excursion. Start time of first excursion
#   no params to worry about
# Cumulative num wells visited in task with repeats, colored by ripple rate. And correlate slope of NWV with ripple rate
#   same
# Total time in corner (Session measure)
#   session measure boundary = 1.5

# For the following new measures, look at B17 results and choose one particular configuration
# Correlation b/w num stims in a location and occupancy in next sesh early trials or this sesh probes, and curvature next and probe.
#   r1, consider grid at half-well resolution (0.5)

# Remember for B18 and Martin should probably treat nostim sessions differently for some measures


def inCorner(pos: tuple[float, float], boundary: float = 1.5) -> bool:
    if np.isnan(pos[0]) or np.isnan(pos[1]):
        return False
    return (pos[0] < -0.5 + boundary and pos[1] < -0.5 + boundary) or \
        (pos[0] > 6.5 - boundary and pos[1] < -0.5 + boundary) or \
        (pos[0] < -0.5 + boundary and pos[1] > 6.5 - boundary) or \
        (pos[0] > 6.5 - boundary and pos[1] > 6.5 - boundary)


def totalInCornerTime(bp: BP, sesh: BTSession, boundary: float = 1.5) -> float:
    ts, xs, ys = sesh.posDuringBehaviorPeriod(bp)
    posInCorner = np.array([inCorner((x, y), boundary=boundary) for x, y in zip(xs, ys)])
    return np.sum(np.diff(ts)[posInCorner[:-1]]) / TRODES_SAMPLING_RATE


def makeFigures(plotFlags="all"):
    dataDir = findDataDir()
    outputDir = "TheRun"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    animalNames = ["Martin", "B13", "B14", "B16", "B17", "B18"]

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

    for ratName in animalNames:
        print("======================\n", ratName)
        if ratName[-1] == "d":
            ratName = ratName[:-1]
        sessions: List[BTSession] = allSessionsByRat[ratName]
        # nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probePerformed]
        # numSessionsWithProbe = len(sessionsWithProbe)
        sessionsWithLog = [sesh for sesh in sessions if sesh.hasActivelinkLog]

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

        pp.pushOutputSubDir(ratName)
        pp.setStatCategory("rat", ratName)

        if plotFlagCheck(plotFlags, "4basicNextLast"):
            # early and late task away trials. All 4 basic measures. Next/last session difference.
            #   total: r1s.5
            earlyAwayBP = BP(probe=False, trialInterval=(2, 7), inclusionFlags="awayTrial")
            lateAwayBP = BP(probe=False, trialInterval=(10, 15), inclusionFlags="awayTrial")
            radius = 1
            lm = LocationMeasure("earlyAway_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(earlyAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, excludeFromCombo=True)
            lm = LocationMeasure("lateAway_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(lateAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, excludeFromCombo=True)

            #   num: r.5s.5
            radius = .5
            lm = LocationMeasure("earlyAway_numVisits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(earlyAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, excludeFromCombo=True)
            lm = LocationMeasure("lateAway_numVisits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(lateAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, excludeFromCombo=True)

            #   avg: r1s.5
            earlyBP = BP(probe=False, trialInterval=(2, 7))
            lateBP = BP(probe=False, trialInterval=(10, 15))
            radius = 1
            lm = LocationMeasure("early_avgDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(earlyBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyBP, excludeFromCombo=True)
            lm = LocationMeasure("late_avgDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, excludeFromCombo=True)

            #   curve: r1s.5
            radius = 1
            lm = LocationMeasure("earlyAway_avgCurvature",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgCurvatureAtPosition(earlyAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, excludeFromCombo=True)
            lm = LocationMeasure("lateAway_avgCurvature",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgCurvatureAtPosition(lateAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, excludeFromCombo=True)

        if plotFlagCheck(plotFlags, "lateTaskHomeAvgDwell"):
            # Late task, avg dwell time at home
            #   r1s.5
            lateBP = BP(probe=False, trialInterval=(10, 15))
            radius = 1
            lm = LocationMeasure("lateTaskHomeAvgDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, excludeFromCombo=True)

        # Plots to make:
        if plotFlagCheck(plotFlags, "totalSymTimeLateHomeTrials"):
            # Total time across all sym during late home trials
            #   r.5s.5
            radius = 0.5
            lateBP = BP(probe=False, trialInterval=(10, 15))
            lm = LocationMeasure("late_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, excludeFromCombo=True)

        if plotFlagCheck(plotFlags, "beforeFirstExcursion"):
            # Distance travelled on wall before first excursion. Start time of first excursion
            #   no params to worry about
            sm = SessionMeasure("timeAtFirstExcursion", lambda sesh: sesh.btPos_secs[sesh.btExcursionStart_posIdx[0]],
                                sessions)
            sm.makeFigures(pp, excludeFromCombo=True)
            smDist = SessionMeasure("distBeforeFirstExcursion", lambda sesh: sesh.pathLength(False, 0, sesh.btExcursionStart_posIdx[0]),
                                    sessions)
            smDist.makeFigures(pp, excludeFromCombo=True)
            sm.makeCorrelationFigures(pp, smDist, excludeFromCombo=True)

        if plotFlagCheck(plotFlags, "timeInCorner"):
            # Total time in corner (Session measure)
            #   session measure, boundary = 1.5
            boundary = 1.5
            sm = SessionMeasure("totalTimeInCorner", lambda sesh: totalInCornerTime(
                BP(probe=False), sesh, boundary=boundary), sessions)

        if plotFlagCheck(plotFlags, "wellVisitedVsRippleRate"):
            # Cumulative num wells visited in task with repeats, colored by ripple rate. And correlate slope of NWV with ripple rate
            #   same
            maxPosIdx = np.max([len(sesh.btPos_ts) for sesh in sessions])
            cumWellsVisited = np.empty((len(sessions), maxPosIdx))
            cumWellsVisited[:] = np.nan
            cumWellsVisitedSlopes = np.empty((len(sessions), ))
            rippleRates = np.empty((len(sessions),))
            rippleRates[:] = np.nan
            for si, sesh in enumerate(sessions):
                switchedWells = np.diff(sesh.btNearestWells, prepend=-1) != 0
                vals = np.cumsum(switchedWells)
                cumWellsVisited[si, :len(vals)] = vals
                cumWellsVisitedSlopes[si] = np.polyfit(np.arange(len(vals)), vals, 1)[0]
                rippleRates[si] = len(sesh.btRipsProbeStats) / sesh.taskDuration

            def cumWellsVisitedSlope(sesh: BTSession):
                switchedWells = np.diff(sesh.btNearestWells, prepend=-1) != 0
                vals = np.cumsum(switchedWells)
                return np.polyfit(np.arange(len(vals)), vals, 1)[0]

            smRipRates = SessionMeasure("rippleRates", lambda sesh: len(sesh.btRipsProbeStats) / sesh.taskDuration,
                                        sessions)
            smWellsVisitedSlopes = SessionMeasure(
                "cumWellsVisitedSlopes", cumWellsVisitedSlope, sessions)
            smRipRates.makeFigures(pp, excludeFromCombo=True)
            smWellsVisitedSlopes.makeFigures(pp, excludeFromCombo=True)
            smRipRates.makeCorrelationFigures(pp, smWellsVisitedSlopes, excludeFromCombo=True)

            with pp.newFig("cumWellsVisitedByRipRates") as pc:
                for si, sesh in enumerate(sessions):
                    # use coolwarm colormap
                    color = plt.cm.coolwarm(
                        (rippleRates[si] - np.min(rippleRates)) / (np.max(rippleRates) - np.min(rippleRates)))
                    pc.ax.plot(cumWellsVisited[si, :], color=color)

        pp.popOutputSubDir()

    pp.runImmediateShufflesAcrossPersistentCategories()
