# First tasks tomorrow:
# - run makeFigures("stimsVsOccupancy") in ThesisCommittee2023.py
# - choose a set of params for those
# - Get to work on the slides ya lazy landlubber!

import sys
from datetime import datetime
import os
from typing import List
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl

from UtilFunctions import findDataDir, plotFlagCheck, getLoadInfo, getWellPosCoordinates, getRotatedWells
from PlotUtil import PlotManager, setupBehaviorTracePlot
from BTData import BTData
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from MeasureTypes import SessionMeasure, LocationMeasure, TrialMeasure
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

    originalPlotFlags = plotFlags

    for ratName in animalNames:
        plotFlags = [plotFlag for plotFlag in originalPlotFlags]
        # if ratName != "Martin":
        #     return
        print("======================\n", ratName)
        if ratName[-1] == "d":
            ratName = ratName[:-1]
        sessions: List[BTSession] = allSessionsByRat[ratName]
        # nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probePerformed]
        # numSessionsWithProbe = len(sessionsWithProbe)
        sessionsWithLog = [sesh for sesh in sessions if sesh.hasActivelinkLog]
        sessionsWithLFP = [sesh for sesh in sessions if not sesh.importOptions.skipLFP]
        sessionsWithLFPAndPrevLFP = [sesh for sesh in sessions if not sesh.importOptions.skipLFP and
                                     sesh.prevSession is not None and not sesh.prevSession.importOptions.skipLFP]
        sessionsWithLFPAndPrevLFPandNotNostim = [
            sesh for sesh in sessions if not sesh.importOptions.skipLFP and
            sesh.prevSession is not None and not sesh.prevSession.importOptions.skipLFP and
            not sesh.condition == BTSession.CONDITION_NO_STIM and not sesh.prevSession.condition == BTSession.CONDITION_NO_STIM]

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

        if plotFlagCheck(plotFlags, "hyp1", excludeFromAll=True):
            # num visits, trial 0-1, r0.5, s0.5
            lm = LocationMeasure("hyp1_numVisits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(BP(probe=False, trialInterval=(0, 1)), pos, radius=0.5)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            # total time spent in whole task, r1, s0
            lm = LocationMeasure("hyp1_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(BP(probe=False), pos, radius=1)),
                                 sessions, smoothDist=0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            # same but just away trials and s0.5
            lm = LocationMeasure("hyp1_totalTime_awayTrials",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(BP(probe=False, inclusionFlags="awayTrial"), pos, radius=1)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            # avgdwelltime, r1, s0.5
            lm = LocationMeasure("hyp1_avgDwellTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(BP(probe=False), pos, radius=1)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

        if plotFlagCheck(plotFlags, "hyp2", excludeFromAll=True):
            # session measure of total time spent at symmetric wells during home trials from trial 10-15
            # r.5
            def totalTimeAtSymmetricWells(sesh: BTSession):
                symWells = getRotatedWells(sesh.homeWell)
                ret = 0
                for well in symWells:
                    ret += sesh.totalTimeAtPosition(BP(probe=False, trialInterval=(
                        10, 15), inclusionFlags="homeTrial", erode=3), getWellPosCoordinates(well), radius=0.5)
                return ret
            sm = SessionMeasure("hyp2_totalTimeAtSymmetricWells", totalTimeAtSymmetricWells,
                                sessions)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

        if plotFlagCheck(plotFlags, "hyp3", excludeFromAll=True):
            # total time at home throughout session r1, s0
            lm = LocationMeasure("hyp3_totalTimeAtHome",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(BP(probe=False), pos, radius=1)),
                                 sessions, smoothDist=0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            # total time spent drinking at home throughout session r1, s0
            def totalDrinkingTime(sesh: BTSession):
                ret = 0
                for homeEnter, homeExit in zip(sesh.homeRewardEnter_ts, sesh.homeRewardExit_ts):
                    ret += (homeExit - homeEnter)
                return ret / TRODES_SAMPLING_RATE
            sm = SessionMeasure("hyp3_totalDrinkingTime", totalDrinkingTime, sessions)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            def totalOffWallTime(sesh: BTSession):
                ret = 0
                ts = sesh.btPos_secs
                for excursionStart, excursionEnd in zip(sesh.btExcursionStart_posIdx, sesh.btExcursionEnd_posIdx):
                    ret += ts[excursionEnd-1] - ts[excursionStart]
                return ret
            sm = SessionMeasure("hyp3_totalOffWallTime", totalOffWallTime, sessions)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

        if plotFlagCheck(plotFlags, "4basicNextLast", excludeFromAll=True):
            # early and late task away trials. All 4 basic measures. Next/last session difference.
            #   total: r1s.5
            earlyAwayBP = BP(probe=False, trialInterval=(2, 7), inclusionFlags="awayTrial")
            lateAwayBP = BP(probe=False, trialInterval=(10, 15), inclusionFlags="awayTrial")
            radius = 1
            lm = LocationMeasure("earlyAway_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(earlyAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            lm = LocationMeasure("lateAway_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(lateAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            #   num: r.5s.5
            radius = .5
            lm = LocationMeasure("earlyAway_numVisits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(earlyAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            lm = LocationMeasure("lateAway_numVisits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(lateAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            #   avg: r1s.5
            earlyBP = BP(probe=False, trialInterval=(2, 7))
            lateBP = BP(probe=False, trialInterval=(10, 15))
            radius = 1
            lm = LocationMeasure("earlyall_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(earlyBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            lm = LocationMeasure("lateall_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            #   num: r.5s.5
            radius = .5
            lm = LocationMeasure("earlyall_numVisits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(earlyBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            lm = LocationMeasure("lateall_numVisits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            radius = 1
            lm = LocationMeasure("early_avgDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(earlyBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            lm = LocationMeasure("late_avgDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            #   curve: r1s.5
            radius = 1
            lm = LocationMeasure("earlyAway_avgCurvature",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgCurvatureAtPosition(earlyAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            lm = LocationMeasure("lateAway_avgCurvature",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgCurvatureAtPosition(lateAwayBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

        if plotFlagCheck(plotFlags, "lateTaskHomeAvgDwell", excludeFromAll=True):
            # Late task, avg dwell time at home
            #   r1s.5
            lateBP = BP(probe=False, trialInterval=(10, 15))
            radius = 1
            lm = LocationMeasure("lateTaskHomeAvgDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

        # Plots to make:
        if plotFlagCheck(plotFlags, "totalSymTimeLateHomeTrials", excludeFromAll=True):
            # Total time across all sym during late home trials
            #   r.5s.5
            radius = 0.5
            lateBP = BP(probe=False, trialInterval=(10, 15))
            lm = LocationMeasure("late_totalTime",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(lateBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            radius = 0.5
            lateHomeBP = BP(probe=False, trialInterval=(10, 15), inclusionFlags="homeTrial")
            lm = LocationMeasure("late_totalTimeHome",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(lateHomeBP, pos, radius)),
                                 sessions, smoothDist=0.5)
            lm.makeFigures(pp, everySessionBehaviorPeriod=lateHomeBP, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

        if plotFlagCheck(plotFlags, "beforeFirstExcursion", excludeFromAll=True):
            # Distance travelled on wall before first excursion. Start time of first excursion
            #   no params to worry about
            sm = SessionMeasure("timeAtFirstExcursion", lambda sesh: sesh.btPos_secs[sesh.btExcursionStart_posIdx[0]],
                                sessions)
            sm.makeFigures(pp, plotFlags="noteverysession")
            smDist = SessionMeasure("distBeforeFirstExcursion", lambda sesh: sesh.pathLength(False, 0, sesh.btExcursionStart_posIdx[0]),
                                    sessions)
            smDist.makeFigures(pp, plotFlags="noteverysession")
            sm.makeCorrelationFigures(pp, smDist)

        if plotFlagCheck(plotFlags, "timeInCorner", excludeFromAll=True):
            # Total time in corner (Session measure)
            #   session measure, boundary = 1.5
            boundary = 1.5
            sm = SessionMeasure("totalTimeInCorner", lambda sesh: totalInCornerTime(
                BP(probe=False), sesh, boundary=boundary), sessions)

        if plotFlagCheck(plotFlags, "wellVisitedVsRippleRate", excludeFromAll=True):
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
            smRipRates.makeFigures(pp, plotFlags="noteverysession")
            smWellsVisitedSlopes.makeFigures(pp, plotFlags="noteverysession")
            smRipRates.makeCorrelationFigures(pp, smWellsVisitedSlopes)

            with pp.newFig("cumWellsVisitedByRipRates") as pc:
                for si, sesh in enumerate(sessions):
                    # use coolwarm colormap
                    color = plt.cm.coolwarm(
                        (rippleRates[si] - np.min(rippleRates)) / (np.max(rippleRates) - np.min(rippleRates)))
                    pc.ax.plot(cumWellsVisited[si, :], color=color)

        if plotFlagCheck(plotFlags, "stimsVsOccupancy", excludeFromAll=True) and ratName != "Martin":
            # Correlation b/w num stims in a location and occupancy in next sesh early trials or this sesh probes, and curvature next and probe.
            #   r1, consider grid at half-well resolution (0.5)
            radius = 1
            smoothDist = 0.5

            lmStimRate = LocationMeasure("stimRate", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.numStimsAtLocation(pos, radius=radius) / sesh.totalTimeAtPosition(BP(probe=False), pos, radius=radius)),
                sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmStimRate.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            lmStimRatePrevSession = LocationMeasure("stimRatePrevSession", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.prevSession.numStimsAtLocation(
                    pos, radius=radius) / sesh.prevSession.totalTimeAtPosition(BP(probe=False), pos, radius=radius) if sesh.prevSession is not None else np.nan),
                sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmStimRatePrevSession.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            earlyBP = BP(probe=False, trialInterval=(2, 7))
            probeBP = BP(probe=True)
            lmOccupancyEarly = LocationMeasure("occupancy_early", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.totalTimeAtPosition(earlyBP, pos, radius=radius)), sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmOccupancyEarly.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            # lmStimRate.makeLocationCorrelationFigures(pp, lmOccupancyEarly)
            lmStimRatePrevSession.makeLocationCorrelationFigures(
                pp, lmOccupancyEarly)

            lmOccupancyProbe = LocationMeasure("occupancy_probe", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.totalTimeAtPosition(probeBP, pos, radius=radius)), sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmOccupancyProbe.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            # lmStimRate.makeLocationCorrelationFigures(pp, lmOccupancyProbe)
            # lmStimRatePrevSession.makeLocationCorrelationFigures(
            #     pp, lmOccupancyProbe)

            lmAvgDwellEarly = LocationMeasure("avgDwell_early", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.avgDwellTimeAtPosition(earlyBP, pos, radius=radius)), sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmAvgDwellEarly.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            # lmAvgDwellEarly.makeLocationCorrelationFigures(pp, lmStimRate)
            # lmAvgDwellEarly.makeLocationCorrelationFigures(
            #     pp, lmStimRatePrevSession)
            lmStimRate.makeLocationCorrelationFigures(pp, lmAvgDwellEarly)
            lmStimRatePrevSession.makeLocationCorrelationFigures(
                pp, lmAvgDwellEarly)

            lmAvgDwellProbe = LocationMeasure("avgDwell_probe", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.avgDwellTimeAtPosition(probeBP, pos, radius=radius)), sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmAvgDwellProbe.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            # lmAvgDwellProbe.makeLocationCorrelationFigures(pp, lmStimRate)
            # lmAvgDwellProbe.makeLocationCorrelationFigures(
            #     pp, lmStimRatePrevSession)
            lmStimRate.makeLocationCorrelationFigures(pp, lmAvgDwellProbe)
            lmStimRatePrevSession.makeLocationCorrelationFigures(
                pp, lmAvgDwellProbe)

            lmCurvatureEarly = LocationMeasure("curvature_early", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.avgCurvatureAtPosition(earlyBP, pos, radius=radius)), sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmCurvatureEarly.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            # lmCurvatureEarly.makeLocationCorrelationFigures(pp, lmStimRate)
            # lmCurvatureEarly.makeLocationCorrelationFigures(
            #     pp, lmStimRatePrevSession)
            # lmStimRate.makeLocationCorrelationFigures(pp, lmCurvatureEarly)
            lmStimRatePrevSession.makeLocationCorrelationFigures(
                pp, lmCurvatureEarly)

            lmCurvatureProbe = LocationMeasure("curvature_probe", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.avgCurvatureAtPosition(probeBP, pos, radius=radius)), sessionsWithLFPAndPrevLFPandNotNostim, smoothDist=smoothDist)
            lmCurvatureProbe.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])
            # lmCurvatureProbe.makeLocationCorrelationFigures(pp, lmStimRate)
            # lmCurvatureProbe.makeLocationCorrelationFigures(
            #     pp, lmStimRatePrevSession)
            lmStimRate.makeLocationCorrelationFigures(pp, lmCurvatureProbe)
            # lmStimRatePrevSession.makeLocationCorrelationFigures(
            #     pp, lmCurvatureProbe)

        if plotFlagCheck(plotFlags, "trialDuration"):
            tm = TrialMeasure("duration",
                              lambda sesh, start, end, ttype: (
                                  sesh.btPos_ts[end] - sesh.btPos_ts[start]) / TRODES_SAMPLING_RATE,
                              sessions)
            tm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

        if plotFlagCheck(plotFlags, "allTraces"):
            n = len(sessions)
            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)
            with pp.newFig("probeTraces", subPlots=(ncols, nrows)) as pc:
                for i, sesh in enumerate(sessions):
                    # ax = pc.add_subplot(nrows, ncols, i + 1)
                    ax = pc.axs.flat[i]
                    # sesh.plotPositionTrace(ax)
                    wellSize = mpl.rcParams['lines.markersize']**2 / 4
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    ax.plot(sesh.probePosXs, sesh.probePosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

            with pp.newFig("taskTraces", subPlots=(ncols, nrows)) as pc:
                for i, sesh in enumerate(sessions):
                    # ax = pc.add_subplot(nrows, ncols, i + 1)
                    ax = pc.axs.flat[i]
                    # sesh.plotPositionTrace(ax)
                    wellSize = mpl.rcParams['lines.markersize']**2 / 4
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    ax.plot(sesh.btPosXs, sesh.btPosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

        if plotFlagCheck(plotFlags, "gridchyn"):
            probeBP = BP(probe=True)
            radius = 0.5
            sm = SessionMeasure("probeTotalDwell", lambda sesh: sesh.totalTimeAtPosition(
                probeBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            probeFillBP = BP(probe=True, timeInterval=BTSession.fillTimeInterval)
            sm = SessionMeasure("PFTDwell", lambda sesh: sesh.totalTimeAtPosition(
                probeFillBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            sm = SessionMeasure("probeNumEntries", lambda sesh: sesh.numVisitsToPosition(
                probeBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            sm = SessionMeasure("PFTNumEntries", lambda sesh: sesh.numVisitsToPosition(
                probeFillBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            sm = SessionMeasure("probePathOptimality", lambda sesh: sesh.pathOptimalityToPosition(
                probeBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            sm = SessionMeasure("PFTPathOptimality", lambda sesh: sesh.pathOptimalityToPosition(
                probeFillBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            sm = SessionMeasure("probePathLength", lambda sesh: sesh.pathLengthToPosition(
                probeBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

            sm = SessionMeasure("PFTPathLength", lambda sesh: sesh.pathLengthToPosition(
                probeFillBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])

        if plotFlagCheck(plotFlags, "gridchyn2"):
            probeBP = BP(probe=True)
            radius = 0.5
            lm = LocationMeasure("probeTotalDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(probeBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            probeFillBP = BP(probe=True, timeInterval=BTSession.fillTimeInterval)
            lm = LocationMeasure("PFTDwell",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(probeFillBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            lm = LocationMeasure("probeNumEntries",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(probeBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            lm = LocationMeasure("PFTNumEntries",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(probeFillBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            lm = LocationMeasure("probePathOptimality",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.pathOptimalityToPosition(probeBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            lm = LocationMeasure("PFTPathOptimality",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.pathOptimalityToPosition(probeFillBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            lm = LocationMeasure("probePathLength",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.pathLengthToPosition(probeBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

            sm = SessionMeasure("PFTPathLength", lambda sesh: sesh.pathLengthToPosition(
                probeFillBP, getWellPosCoordinates(sesh.homeWell), radius=radius), sessionsWithProbe)
            sm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])
            lm = LocationMeasure("PFTPathLength",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.pathLengthToPosition(probeFillBP, pos, radius)),
                                 sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

        if plotFlagCheck(plotFlags, "fracExcursions"):
            lm = LocationMeasure(f"fracExcursions_probe",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.fracExcursionsVisited(
                                         BP(probe=True), pos, 0.5, "session"),
                                 ), sessions, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=[
                "measureByCondition", "measureVsCtrl", "measureVsCtrlByCondition", "diff", "everysession", "everysessionoverlayatlocation", "everysessionoverlayatctrl", "everysessionoverlayatlocationbycondition", "everysessionoverlaydirect"
            ])

        pp.popOutputSubDir()

    # pp.makeCombinedFigs()
    # pp.runImmediateShufflesAcrossPersistentCategories()


def main():
    if len(sys.argv) > 1:
        plotFlags = sys.argv[1:]
    else:
        plotFlags = ["all"]
    makeFigures(plotFlags)


if __name__ == "__main__":
    main()
