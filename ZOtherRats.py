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
from matplotlib import cm

from UtilFunctions import findDataDir, plotFlagCheck, getLoadInfo, getWellPosCoordinates, getRotatedWells, numWellsVisited
from PlotUtil import PlotManager, setupBehaviorTracePlot
from BTData import BTData
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from MeasureTypes import SessionMeasure, LocationMeasure, TrialMeasure
from consts import TRODES_SAMPLING_RATE, offWallWellNames

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


def makeFigures(plotFlags="all", runImmediate=True, makeCombined=True, testData=False, animalNames=None, allRatRun="no", excludeNoStim=False):
    dataDir = findDataDir()
    outputDir = "TheRun"
    if testData:
        outputDir += "_test"
    if allRatRun != "no":
        outputDir += f"_allRats_{allRatRun}"
    if excludeNoStim:
        outputDir += "_excludeNoStim"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    if animalNames is None:
        if testData:
            # animalNames = ["Martin", "B13", "B14"]
            # animalNames = ["Martin", "B14"]
            # animalNames = ["B14"]
            animalNames = ["Martin"]
        else:
            animalNames = ["Martin", "B13", "B14", "B16", "B17", "B18"]

    infoFileName = datetime.now().strftime("_".join(animalNames) + "_%Y%m%d_%H%M%S" + ".txt")
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=3)

    totalNumSessions = 0
    allSessionsByRat = {}
    allSessionsByRat["allRats"] = []
    for animalName in animalNames:
        animalInfo = getLoadInfo(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        print("loading from " + dataFilename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[animalName] = ratData.getSessions()
        totalNumSessions += len(ratData.getSessions())

        if allRatRun != "no":
            allSessionsByRat["allRats"] = allSessionsByRat["allRats"] + ratData.getSessions()

    print("totalNumSessions =", totalNumSessions)

    originalPlotFlags = plotFlags

    animalNamesToRun = animalNames.copy()
    if allRatRun == "yes":
        animalNamesToRun.append("allRats")
    elif allRatRun == "only":
        animalNamesToRun = ["allRats"]

    for ratName in animalNamesToRun:
        plotFlags = [plotFlag for plotFlag in originalPlotFlags]
        # if ratName != "Martin":
        #     return
        print("======================\n", ratName)
        sessions: List[BTSession] = allSessionsByRat[ratName]
        if testData:
            pass
            # sessions = sessions[:7]
        if excludeNoStim:
            sessions = [sesh for sesh in sessions if sesh.condition != BTSession.CONDITION_NO_STIM]
        # nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probePerformed]
        numSessionsWithProbe = len(sessionsWithProbe)
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

        ctrlSessions = [sesh for sesh in sessions if not sesh.isRippleInterruption]
        swrSessions = [sesh for sesh in sessions if sesh.isRippleInterruption]
        # nCtrlSessions = len(ctrlSessions)
        # nSWRSessions = len(swrSessions)

        print(f"{len(sessions)} sessions ({len(sessionsWithProbe)} "
              f"with probe: {nCtrlWithProbe} Ctrl, {nSWRWithProbe} SWR)")

        data = [[
            sesh.name, sesh.conditionString(), sesh.homeWell, len(
                sesh.visitedAwayWells), sesh.fillTimeCutoff(), "None" if sesh.prevSession is None else sesh.prevSession.name,
            sesh.secondsSincePrevSession
        ] for si, sesh in enumerate(sessions)]
        df = pd.DataFrame(data, columns=[
            "name", "condition", "home well", "num aways found", "fill time", "prev session", "seconds since prev"
        ])
        s = df.to_string()
        pp.writeToInfoFile(s)
        print(s)

        pp.pushOutputSubDir(ratName)
        if len(animalNamesToRun) > 1:
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
                    color = cm.coolwarm(
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
                    ax = pc.axs.flat[i]
                    wellSize = mpl.rcParams['lines.markersize']**2 * 3
                    # setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize)
                    ax.plot(sesh.probePosXs, sesh.probePosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

            with pp.newFig("taskTraces", subPlots=(ncols, nrows)) as pc:
                for i, sesh in enumerate(sessions):
                    ax = pc.axs.flat[i]
                    wellSize = mpl.rcParams['lines.markersize']**2 * 3
                    # setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize)
                    ax.plot(sesh.btPosXs, sesh.btPosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

        if plotFlagCheck(plotFlags, "overlayTrials"):
            for i, sesh in enumerate(sessions):
                with pp.newFig(f"trialTracesOverlay_{sesh.name}", subPlots=(1, 2), excludeFromCombo=True) as pc:
                    awayAx = pc.axs.flat[0]
                    homeAx = pc.axs.flat[1]
                    wellSize = mpl.rcParams['lines.markersize']**2 * 3
                    setupBehaviorTracePlot(awayAx, sesh, wellSize=wellSize, showWells="HA")
                    setupBehaviorTracePlot(homeAx, sesh, wellSize=wellSize, showWells="HA")

                    trialPosIdxs = sesh.getAllTrialPosIdxs()
                    homeTrials = trialPosIdxs[::2, :]
                    awayTrials = trialPosIdxs[1::2, :]
                    for start, end in homeTrials:
                        homeAx.plot(
                            sesh.btPosXs[start:end], sesh.btPosYs[start:end], c="#0000007f", lw=1, zorder=1.5)
                    for start, end in awayTrials:
                        awayAx.plot(
                            sesh.btPosXs[start:end], sesh.btPosYs[start:end], c="#0000007f", lw=1, zorder=1.5)

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

        if plotFlagCheck(plotFlags, "testImmediate"):
            def testFunc(sesh: BTSession):
                val = 0
                if sesh.animalName == "Martin":
                    val = 1
                else:
                    val = int(sesh.animalName[1:])
                if sesh.isRippleInterruption:
                    val += 100
                    factor = -1
                else:
                    factor = 1
                return sesh.getTestMap() * factor + val
            lm = LocationMeasure(f"testImmediate", testFunc, sessions, smoothDist=0.5)
            lm.makeFigures(pp)

        if plotFlagCheck(plotFlags, "thesis0"):
            lm = LocationMeasure(f"probe number of visits (smoothed)",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(
                                         BP(probe=True), pos, 0.5),
                                 ), sessionsWithProbe, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])
            lm = LocationMeasure(f"probe number of visits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(
                                         BP(probe=True), pos, 0.5),
                                 ), sessionsWithProbe, smoothDist=0.0)
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])
            lm = LocationMeasure(f"probe total time (sec)",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(
                                         BP(probe=True), pos, 0.5),
                                 ), sessionsWithProbe, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])

            lm = LocationMeasure("pseudoprobe number of visits (smoothed)",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(
                                         BP(probe=False, trialInterval=(0, 1)), pos, 0.5),
                                 ), sessions, smoothDist=0.5,
                                 sessionValLocation=LocationMeasure.prevSessionHomeLocation)
            lm.makeDualCategoryFigures(pp)
            lm = LocationMeasure("pseudoprobe number of visits",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(
                                         BP(probe=False, trialInterval=(0, 1)), pos, 0.5),
                                 ), sessions, smoothDist=0.0,
                                 sessionValLocation=LocationMeasure.prevSessionHomeLocation)
            lm.makeDualCategoryFigures(pp)
            lm = LocationMeasure("pseudoprobe total time (sec)",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(
                                         BP(probe=False, trialInterval=(0, 1)), pos, 0.5),
                                 ), sessions, smoothDist=0.5,
                                 sessionValLocation=LocationMeasure.prevSessionHomeLocation)
            lm.makeDualCategoryFigures(pp)

        if plotFlagCheck(plotFlags, "thesis1"):
            tm = TrialMeasure("duration (sec)",
                              lambda sesh, start, end, ttype: (
                                  sesh.btPos_ts[end] - sesh.btPos_ts[start]) / TRODES_SAMPLING_RATE,
                              sessions)
            tm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"], runStats=False)
            tm = TrialMeasure("duration (sec) with shuffles",
                              lambda sesh, start, end, ttype: (
                                  sesh.btPos_ts[end] - sesh.btPos_ts[start]) / TRODES_SAMPLING_RATE,
                              sessions)
            tm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])
            tm = TrialMeasure("normalized path length",
                              lambda sesh, start, end, ttype: sesh.normalizedPathLength(
                                  False, start, end),
                              sessions)
            tm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"], runStats=False)

        if plotFlagCheck(plotFlags, "thesis2"):
            n = len(sessionsWithProbe)
            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)
            with pp.newFig("probeTracesGrouped", subPlots=(ncols, nrows)) as pc:
                for i, sesh in enumerate(ctrlSessionsWithProbe):
                    ax = pc.axs.flat[i]
                    wellSize = mpl.rcParams['lines.markersize']**2 * 3
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    ax.plot(sesh.probePosXs, sesh.probePosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

                for i, sesh in enumerate(swrSessionsWithProbe):
                    ax = pc.axs.flat[i + len(ctrlSessionsWithProbe)]
                    wellSize = mpl.rcParams['lines.markersize']**2 * 3
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    ax.plot(sesh.probePosXs, sesh.probePosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

                for i in range(len(sessionsWithProbe), len(pc.axs.flat)):
                    pc.axs.flat[i].axis("off")

            n = len(sessions)
            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)
            with pp.newFig("taskTracesGrouped", subPlots=(ncols, nrows)) as pc:
                for i, sesh in enumerate(ctrlSessions):
                    ax = pc.axs.flat[i]
                    wellSize = mpl.rcParams['lines.markersize']**2 * 3
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    ax.plot(sesh.btPosXs, sesh.btPosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

                for i, sesh in enumerate(swrSessions):
                    ax = pc.axs.flat[i + len(ctrlSessions)]
                    wellSize = mpl.rcParams['lines.markersize']**2 * 3
                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")
                    ax.plot(sesh.btPosXs, sesh.btPosYs, c="#0000007f", lw=1, zorder=1.5)
                    ax.set_title(sesh.name)

                for i in range(len(sessions), len(pc.axs.flat)):
                    pc.axs.flat[i].axis("off")

        if plotFlagCheck(plotFlags, "thesis3"):
            # Cumulative number of wells visited in probe across sessions
            windowSlide = 10
            t1Array = np.arange(windowSlide, 60 * 5, windowSlide)
            numVisitedOverTimeOffWall = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTimeOffWall[:] = np.nan
            numVisitedOverTimeOffWallWithRepeats = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTimeOffWallWithRepeats[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * TRODES_SAMPLING_RATE + sesh.probePos_ts[0]
                i1Array = np.searchsorted(sesh.probePos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisitedOverTimeOffWall[si, ii] = numWellsVisited(
                        sesh.probeNearestWells[0:i1], countReturns=False,
                        wellSubset=offWallWellNames)
                    numVisitedOverTimeOffWallWithRepeats[si, ii] = numWellsVisited(
                        sesh.probeNearestWells[0:i1], countReturns=True,
                        wellSubset=offWallWellNames)

            with pp.newFig("probe_numVisitedOverTime_bysidx_offwall") as pc:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisitedOverTimeOffWall.shape[1],)) * 0.5
                    pc.ax.plot(t1Array, numVisitedOverTimeOffWall[si, :] + jitter, color=cmap(si))
                pc.ax.set_xticks(np.arange(0, 60 * 5 + 1, 60))
                pc.ax.set_ylim(0, 17)

            with pp.newFig("probe_numVisitedOverTime_bysidx_offwall_withRepeats") as pc:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(
                        size=(numVisitedOverTimeOffWallWithRepeats.shape[1],)) * 0.5
                    pc.ax.plot(
                        t1Array, numVisitedOverTimeOffWallWithRepeats[si, :] + jitter, color=cmap(si))
                pc.ax.set_xticks(np.arange(0, 60 * 5 + 1, 60))

        if plotFlagCheck(plotFlags, "thesis4"):
            # num visits to position in next, trialInterval 0-1, r.5, s.5
            lm = LocationMeasure("Number of visits to position in next trial",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.numVisitsToPosition(
                                         BP(probe=False, trialInterval=(0, 1)), pos, 0.5),
                                 ), sessions, smoothDist=0.5)
            # lm.makeFigures(pp, plotFlags="ctrlByCondition")
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])

            # total time at position, r1, s0, full task
            lm = LocationMeasure("Total time at position",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.totalTimeAtPosition(
                                         BP(probe=False), pos, 1),
                                 ), sessions, smoothDist=0)
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])

            # average time at position, r1, s.5, full task
            lm = LocationMeasure("Total time at position",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(
                                         BP(probe=False), pos, 1),
                                 ), sessions, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])

        if plotFlagCheck(plotFlags, "thesis5"):
            lm = LocationMeasure(f"task gravity",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.getGravity(
                                         BP(probe=False, inclusionFlags="awayTrial"), pos),
                                 ), sessionsWithProbe, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])

            lm = LocationMeasure(f"probe gravity",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.getGravity(
                                         BP(probe=True), pos),
                                 ), sessionsWithProbe, smoothDist=0.5)
            lm.makeFigures(pp, plotFlags=["ctrlByCondition", "measureByCondition",
                           "measureVsCtrl", "measureVsCtrlByCondition", "diff"])

            lm = LocationMeasure("pseudoprobe gravity",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.getGravity(
                                         BP(probe=False, trialInterval=(0, 1)), pos),
                                 ), sessions, smoothDist=0.5,
                                 sessionValLocation=LocationMeasure.prevSessionHomeLocation)
            lm.makeDualCategoryFigures(pp)

        if plotFlagCheck(plotFlags, "thesis6"):
            # Morris paper measure: latency and norm and raw path length in home trials 2,3,4
            sm = SessionMeasure("Latency to home 2-4",
                                lambda sesh: np.sum(np.diff(sesh.getAllTrialPosIdxs()[
                                                    2:7:2, :], axis=1)) / TRODES_SAMPLING_RATE,
                                sessions)
            sm.makeFigures(pp, plotFlags="violin")

            sm = SessionMeasure("Path length to home 2-4 (ft)",
                                lambda sesh: np.sum([sesh.pathLength(False, i1, i2) for i1, i2 in zip(
                                    sesh.getAllTrialPosIdxs()[2:7:2, 0], sesh.getAllTrialPosIdxs()[2:7:2, 1])]),
                                sessions)
            sm.makeFigures(pp, plotFlags="violin")

            sm = SessionMeasure("normalized Path length to home 2-4",
                                lambda sesh: np.sum([sesh.normalizedPathLength(False, i1, i2) for i1, i2 in zip(
                                    sesh.getAllTrialPosIdxs()[2:7:2, 0], sesh.getAllTrialPosIdxs()[2:7:2, 1])]),
                                sessions)
            sm.makeFigures(pp, plotFlags="violin")

        pp.popOutputSubDir()

    if len(animalNamesToRun) > 1:
        if makeCombined:
            outputSubDir = f"combo_{'_'.join(animalNames)}"
            if testData or len(animalNames) != 6:
                pp.makeCombinedFigs(outputSubDir=outputSubDir)
            else:
                pp.makeCombinedFigs(
                    outputSubDir=outputSubDir, suggestedSubPlotLayout=(2, 3))
        if runImmediate:
            pp.runImmediateShufflesAcrossPersistentCategories(significantThreshold=1)


def main():
    testData = False
    animalNames = None
    allRatRun = "no"
    excludeNoStim = False
    if len(sys.argv) > 1:
        flags = sys.argv[1:]

        if "--test" in flags:
            testData = True
            flags.remove("--test")

        if "--no17" in flags:
            animalNames = ["Martin", "B13", "B14", "B16", "B18"]
            flags.remove("--no17")
        if "--no14" in flags:
            animalNames = ["Martin", "B13", "B16", "B17", "B18"]
            flags.remove("--no14")
        if "--no1417" in flags:
            animalNames = ["Martin", "B13", "B16", "B18"]
            flags.remove("--no1417")
        if "--just17" in flags:
            animalNames = ["B17"]
            flags.remove("--just17")
        if "--just14old" in flags:
            animalNames = ["B14_old"]
            flags.remove("--just14old")

        if "--allThesis" in flags:
            flags.remove("--allThesis")
            flags.extend(["thesis0", "thesis1", "thesis2",
                         "thesis3", "thesis4", "thesis5", "thesis6"])

        if "--alldata" in flags:
            allRatRun = "yes"
            flags.remove("--alldata")

        if "--alldataonly" in flags:
            allRatRun = "only"
            flags.remove("--alldataonly")

        if "--excludeNoStim" in flags:
            excludeNoStim = True
            flags.remove("--excludeNoStim")
            if animalNames is not None:
                if "Martin" in animalNames:
                    animalNames.remove("Martin")
            elif testData:
                animalNames = ["B13", "B14"]
            else:
                animalNames = ["B13", "B14", "B16", "B17", "B18"]

        plotFlags = flags
    else:
        plotFlags = ["all"]
    makeFigures(plotFlags, testData=testData, animalNames=animalNames,
                allRatRun=allRatRun, excludeNoStim=excludeNoStim)


if __name__ == "__main__":
    main()
