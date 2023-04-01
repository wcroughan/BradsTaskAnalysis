import os
from datetime import datetime
import time
from functools import partial
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from BTData import BTData
from PlotUtil import PlotManager, plotIndividualAndAverage
from UtilFunctions import getLoadInfo, findDataDir, getWellPosCoordinates
from MeasureTypes import LocationMeasure, SessionMeasure
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from consts import TRODES_SAMPLING_RATE

# Hypotheses:
#   Next session specificity: SWR means more visits and more total time but less avg time in next session. This is looking at full session
#       Also in away trials. Next session difference.
#           higher curvature at all wells in ctrl during away trials. More indication of a more confident memory in ctrl. Especially early away trials, home/away diff 2-7
#           And this goes away in next session in ctrl but not swr (early trials still)
#       TOCHECK: specific to early learning?
# SHOULD CHECK: Looked at next session. Now look at prev session. Can SWR interfere with recall of previous home well?
#   In whole session, more time spent in corner in Ctrl
#   General confusion in SWR: total time sym during late learning home trials
#   Time spent at home in general higher in SWR. More pronounced in late task
#   At first, Ctrl runs faster around wall, SWR goes slower out into middle
#   What is velocity at stim? Maybe more stim b/c artifacts represent more active sessions in general
#       More rips means more visits, higher total and avg dwell, and higher curvature (swr only) at aways and syms. Early learning for sure, possibly late learning too
#       In probe, more rips means more visits, more total time at aways and sym, and less curvature specificity for home (swr only)
#   Aversive stimuli in ctrl: negative correlation b/w num stims in a location an occupancy during probe? And Occupancy during next sesh early trials?
#       More stim means less curvature off wall in ctrl
#   In probe before fill, curvature home specificity higher in swr. But also more away visits. Higher curvature because wants to check lots of wells?
# SHOULD CHECK: raw values for ripple rate are low. Divide by still time instead? Double check detection logic

# Plots to make:
# early and late task away trials. All 4 basic measures. Next/last session difference.
#   total: r1s.5
#   avg: r1s.5
#   num: r.5s.5
#   curve: r1s.5
# Late task, avg dwell time at home
#   r1s.5
# Total time in corner (Session measure)
#   first plot total dwell time at rsmall, find boundary to use, then make session measure
# Total time across all sym during late home trials
#   r.5s.5
# Distance travelled on wall before first excursion. Start time of first excursion
#   no params to worry about
# Velocity PSTH around stim.
#   same
# Cumulative num wells visited in task with repeats, colored by ripple rate. And correlate slope of NWV with ripple rate
#   same
# In excursions where home was checked, how many wells checked, and what is total time of excursion? Effect of condition or correlation with ripple rate
#   with and without repeats
# Correlation b/w num stims in a location and occupancy in next sesh early trials or this sesh probes, and curvature next and probe.
#   r1, consider grid at half-well resolution (0.5)

# WORK Schedule:
#  - Make above plots
#  - Throw into slides
#  - Choose shuffles to run on full dataset
#  - Write code that will run it
#  - Make slides for thesis committee
#  - Anticipate problems with full dataset

# Some measures from other file I was thinking about:
# how often are aways checked on way to home. Specifically t2-7
# How often are previously rewarded wells checked during away trials? Specifically t2-7
# : does ripple count correlate with first trial duration? same with stim count
# : more stims = more probe OW/E/M time?
# :remember with avg time about missing values too!
# :ripple rate seems more highly correlated with task things, whereas stim rate correlates with probe things ... interpretation?
# :I think it's really important to see where the ripples/stims are happening. For each correlation explanation, can p easily think of where I think they're happening, need to test that

# :just scatter stim rate vs rip rate
# :and all these things as a function of session index


def plotFlagCheck(plotFlags: List[str], flag: str) -> bool:
    try:
        plotFlags.remove(flag)
        return True
    except ValueError:
        return "all" in plotFlags


def makeFigures(plotFlags="all"):
    if isinstance(plotFlags, str):
        plotFlags = [plotFlags]

    ratName = "B17"
    ratInfo = getLoadInfo(ratName)
    dataFilename = os.path.join(ratInfo.output_dir, ratInfo.out_filename)
    print("loading from " + dataFilename)
    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    sessions = ratData.getSessions()

    infoFileName = datetime.now().strftime("Thesis2023_%Y%m%d_%H%M%S.txt")
    dataDir = findDataDir()
    outputDir = "Thesis2023"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=3)

    if plotFlagCheck(plotFlags, "4basicNextLast"):
        # early and late task away trials. All 4 basic measures. Next/last session difference.
        #   total: r1s.5
        earlyAwayBP = BP(probe=False, trialInterval=(2, 7), inclusionFlags="awayTrial")
        lateAwayBP = BP(probe=False, trialInterval=(10, 15), inclusionFlags="awayTrial")
        radius = 1
        lm = LocationMeasure("earlyAway_totalTime", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.totalTimeAtPosition(earlyAwayBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, excludeFromCombo=True)
        lm = LocationMeasure("lateAway_totalTime", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.totalTimeAtPosition(lateAwayBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, excludeFromCombo=True)

        #   num: r.5s.5
        radius = .5
        lm = LocationMeasure("earlyAway_numVisits", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.numVisitsToPosition(earlyAwayBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, excludeFromCombo=True)
        lm = LocationMeasure("lateAway_numVisits", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.numVisitsToPosition(lateAwayBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, excludeFromCombo=True)

        #   avg: r1s.5
        earlyBP = BP(probe=False, trialInterval=(2, 7))
        lateBP = BP(probe=False, trialInterval=(10, 15))
        radius = 1
        lm = LocationMeasure("early_avgDwell", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.avgDwellTimeAtPosition(earlyBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=earlyBP, excludeFromCombo=True)
        lm = LocationMeasure("late_avgDwell", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.avgDwellTimeAtPosition(lateBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, excludeFromCombo=True)

        #   curve: r1s.5
        radius = 1
        lm = LocationMeasure("earlyAway_avgCurvature", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.avgCurvatureAtPosition(earlyAwayBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=earlyAwayBP, excludeFromCombo=True)
        lm = LocationMeasure("lateAway_avgCurvature", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.avgCurvatureAtPosition(lateAwayBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=lateAwayBP, excludeFromCombo=True)

    if plotFlagCheck(plotFlags, "lateTaskHomeAvgDwell"):
        # Late task, avg dwell time at home
        #   r1s.5
        lateBP = BP(probe=False, trialInterval=(10, 15))
        radius = 1
        lm = LocationMeasure("lateTaskHomeAvgDwell", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.avgDwellTimeAtPosition(lateBP, pos, radius)),
                             sessions, smoothDist=0.5)
        lm.makeFigures(pp, everySessionBehaviorPeriod=lateBP, excludeFromCombo=True)

    if plotFlagCheck(plotFlags, "timeInCorner"):
        # Total time in corner (Session measure)
        #   first plot total dwell time at rsmall, find boundary to use, then make session measure
        lm = LocationMeasure("totalTimeSmallR", lambda sesh:
                             lambda sesh: sesh.getValueMap(
                                 lambda pos: sesh.totalTimeAtPosition(BP(probe=False), pos, 0.1)),
                             sessions, smoothDist=0.0)
        lm.makeFigures(pp, everySessionBehaviorPeriod=BP(probe=False), excludeFromCombo=True)

        boundary = 0.5

        def inCorner(pos: tuple[float, float]) -> bool:
            if np.isnan(pos[0]) or np.isnan(pos[1]):
                return False
            return (pos[0] < -0.5 + boundary and pos[1] < -0.5 + boundary) or \
                (pos[0] > 6.5 - boundary and pos[1] < -0.5 + boundary) or \
                (pos[0] < -0.5 + boundary and pos[1] > 6.5 - boundary) or \
                (pos[0] > 6.5 - boundary and pos[1] > 6.5 - boundary)

        def totalInCornerTime(bp: BP, sesh: BTSession) -> float:
            ts, xs, ys = sesh.posDuringBehaviorPeriod(bp)
            posInCorner = np.array([inCorner((x, y)) for x, y in zip(xs, ys)])
            return np.sum(np.diff(ts)[posInCorner[:-1]]) / TRODES_SAMPLING_RATE

        sm = SessionMeasure("totalTimeInCorner", lambda sesh: totalInCornerTime(
            BP(probe=False), sesh), sessions)

    if plotFlagCheck(plotFlags, "totalSymTimeLateHomeTrials"):
        # Total time across all sym during late home trials
        #   r.5s.5
        radius = 0.5
        lateBP = BP(probe=False, trialInterval=(10, 15))
        lm = LocationMeasure("late_totalTime", lambda sesh:
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

    if plotFlagCheck(plotFlags, "velocityPSTH"):
        # Velocity PSTH around stim.
        #   same
        margin = 30
        for si, sesh in enumerate(sessions):
            psthVals = np.empty((len(sesh.btLFPBumps_posIdx), 2 * margin))
            for stimI, stim_posIdx in enumerate(sesh.btLFPBumps_posIdx):
                if stim_posIdx < margin or stim_posIdx > sesh.btPos_secs[-1] - margin:
                    psthVals[stimI, :] = np.nan
                    continue
                startIdx = stim_posIdx - margin
                endIdx = stim_posIdx + margin
                ts = sesh.btPos_secs[startIdx:endIdx] - sesh.btPos_secs[stim_posIdx]
                vs = sesh.btVelCmPerS[startIdx:endIdx]
                psthVals[stimI, :] = vs

            pp.pushOutputSubDir(sesh.name)
            with pp.newFig("velocityPSTH") as pc:
                plotIndividualAndAverage(pc.ax, psthVals, ts)
                pc.ax.set_xlabel("Time from stim (s)")
                pc.ax.set_ylabel("Velocity (cm/s)")
                pc.ax.set_title(sesh.name)
            pp.popOutputSubDir()

    if plotFlagCheck(plotFlags, "wellVisitedVsRippleRate"):
        # Cumulative num wells visited in task with repeats, colored by ripple rate. And correlate slope of NWV with ripple rate
        #   same
        maxPosIdx = np.max([len(sesh.btPos_ts) for sesh in sessions])
        cumWellsVisited = np.empty((len(sessions), maxPosIdx))
        cumWellsVisitedSlopes = np.empty((len(sessions), ))
        rippleRates = np.empty((len(sessions),))
        rippleRates[:] = np.nan
        for si, sesh in enumerate(sessions):
            switchedWells = np.diff(sesh.btNearestWells, prepend=-1) != 0
            vals = np.cumsum(switchedWells)
            cumWellsVisited[si, :len(vals)] = vals
            cumWellsVisitedSlopes[si] = np.polyfit(np.arange(len(vals)), vals, 1)[0]
            rippleRates[si] = len(sesh.btRipsProbeStats) / sesh.taskDuration

        smRipRates = SessionMeasure("rippleRates", lambda sesh: rippleRates[si], sessions)
        smWellsVisitedSlopes = SessionMeasure(
            "cumWellsVisitedSlopes", lambda sesh: cumWellsVisitedSlopes[si], sessions)
        smRipRates.makeFigures(pp, excludeFromCombo=True)
        smWellsVisitedSlopes.makeFigures(pp, excludeFromCombo=True)
        smRipRates.makeCorrelationFigures(pp, smWellsVisitedSlopes, excludeFromCombo=True)

        with pp.newFig("cumWellsVisitedByRipRates") as pc:
            for si, sesh in enumerate(sessions):
                # use coolwarm colormap
                color = plt.cm.coolwarm(
                    (rippleRates[si] - np.min(rippleRates)) / (np.max(rippleRates) - np.min(rippleRates)))
                pc.ax.plot(cumWellsVisited[si, :], color=color)

    if plotFlagCheck(plotFlags, "excursionWithHomeStats"):
        # In excursions where home was checked, how many wells checked, and what is total time of excursion? Effect of condition or correlation with ripple rate
        #   with and without repeats
        def excursionsWithHome(inProbe: bool, sesh: BTSession, radius: float = 1.0):
            xs = sesh.probePosXs if inProbe else sesh.btPosXs
            ys = sesh.probePosYs if inProbe else sesh.btPosYs
            excursionStart_posIdx = sesh.probeExcursionStart_posIdx if inProbe else sesh.btExcursionStart_posIdx
            excursionEnd_posIdx = sesh.probeExcursionEnd_posIdx if inProbe else sesh.btExcursionEnd_posIdx
            for start, end in zip(excursionStart_posIdx, excursionEnd_posIdx):
                homePosX, homePosY = getWellPosCoordinates(sesh.homeWell)
                excursionXs = xs[start:end]
                excursionYs = ys[start:end]
                distToHome = np.sqrt((excursionXs - homePosX)**2 + (excursionYs - homePosY)**2)
                if np.any(distToHome < radius):
                    # excursion contains home
                    yield start, end

        def excursionsWithHomeAvgDuration(inProbe: bool, sesh: BTSession, radius: float = 1.0):
            ts = sesh.probePos_ts if inProbe else sesh.btPos_ts
            return np.mean([(ts[end] - ts[start]) / TRODES_SAMPLING_RATE for start, end in excursionsWithHome(inProbe, sesh, radius)])

        def excursionsWithHomeWellsChecked(inProbe: bool, sesh: BTSession, radius: float = 1.0):
            nearestWells = sesh.probeNearestWells if inProbe else sesh.btNearestWells
            return np.mean([np.sum(np.diff(nearestWells[start:end]) != 0) for start, end in excursionsWithHome(inProbe, sesh, radius)])

        smProbeDur = SessionMeasure("probe_excursionsWithHomeAvgDuration",
                                    lambda sesh: excursionsWithHomeAvgDuration(True, sesh), sessions)
        smTaskDur = SessionMeasure("task_excursionsWithHomeAvgDuration",
                                   lambda sesh: excursionsWithHomeAvgDuration(False, sesh), sessions)
        smProbeWells = SessionMeasure("probe_excursionsWithHomeWellsChecked",
                                      lambda sesh: excursionsWithHomeWellsChecked(True, sesh), sessions)
        smTaskWells = SessionMeasure("task_excursionsWithHomeWellsChecked",
                                     lambda sesh: excursionsWithHomeWellsChecked(False, sesh), sessions)
        smProbeDur.makeFigures(pp, excludeFromCombo=True)
        smTaskDur.makeFigures(pp, excludeFromCombo=True)
        smProbeWells.makeFigures(pp, excludeFromCombo=True)
        smTaskWells.makeFigures(pp, excludeFromCombo=True)

        smRippleRate = SessionMeasure("rippleRate", lambda sesh: len(
            sesh.btRipsProbeStats) / sesh.taskDuration, sessions)
        smProbeDur.makeCorrelationFigures(pp, smRippleRate, excludeFromCombo=True)
        smTaskDur.makeCorrelationFigures(pp, smRippleRate, excludeFromCombo=True)
        smProbeWells.makeCorrelationFigures(pp, smRippleRate, excludeFromCombo=True)
        smTaskWells.makeCorrelationFigures(pp, smRippleRate, excludeFromCombo=True)

    if plotFlagCheck(plotFlags, "stimsVsOccupancy"):
        # Correlation b/w num stims in a location and occupancy in next sesh early trials or this sesh probes, and curvature next and probe.
        #   r1, consider grid at half-well resolution (0.5)
        radius = 1
        smoothDist = 0.5

        lmNumStims = LocationMeasure("numStims", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.numStimsAtLocation(pos, radius=radius)), sessions, smoothDist=smoothDist)
        lmNumStimsPrevSession = LocationMeasure("numStimsPrevSession", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.numStimsAtLocation(pos, radius=radius) if sesh.prevSession is not None else np.nan),
            sessions, smoothDist=smoothDist)
        earlyBP = BP(probe=False, trialInterval=(2, 7))
        probeBP = BP(probe=True)
        lmOccupancyEarly = LocationMeasure("occupancy_early", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.totalTimeAtPosition(earlyBP, pos, radius=radius)), sessions, smoothDist=smoothDist)
        lmOccupancyProbe = LocationMeasure("occupancy_probe", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.totalTimeAtPosition(probeBP, pos, radius=radius)), sessions, smoothDist=smoothDist)
        lmAvgDwellEarly = LocationMeasure("avgDwell_early", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.avgDwellTimeAtPosition(earlyBP, pos, radius=radius)), sessions, smoothDist=smoothDist)
        lmAvgDwellProbe = LocationMeasure("avgDwell_probe", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.avgDwellTimeAtPosition(probeBP, pos, radius=radius)), sessions, smoothDist=smoothDist)
        lmCurvatureEarly = LocationMeasure("curvature_early", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.avgCurvatureAtPosition(earlyBP, pos, radius=radius)), sessions, smoothDist=smoothDist)
        lmCurvatureProbe = LocationMeasure("curvature_probe", lambda sesh: sesh.getValueMap(
            lambda pos: sesh.avgCurvatureAtPosition(probeBP, pos, radius=radius)), sessions, smoothDist=smoothDist)

        lmNumStims.makeFigures(pp, excludeFromCombo=True)
        lmNumStimsPrevSession.makeFigures(pp, excludeFromCombo=True)
        lmOccupancyEarly.makeFigures(pp, excludeFromCombo=True)
        lmOccupancyProbe.makeFigures(pp, excludeFromCombo=True)
        lmAvgDwellEarly.makeFigures(pp, excludeFromCombo=True)
        lmAvgDwellProbe.makeFigures(pp, excludeFromCombo=True)
        lmCurvatureEarly.makeFigures(pp, excludeFromCombo=True)
        lmCurvatureProbe.makeFigures(pp, excludeFromCombo=True)

        lmOccupancyEarly.makeCorrelationFigures(pp, lmNumStims, excludeFromCombo=True)
        lmOccupancyProbe.makeCorrelationFigures(pp, lmNumStims, excludeFromCombo=True)
        lmAvgDwellEarly.makeCorrelationFigures(pp, lmNumStims, excludeFromCombo=True)
        lmAvgDwellProbe.makeCorrelationFigures(pp, lmNumStims, excludeFromCombo=True)
        lmCurvatureEarly.makeCorrelationFigures(pp, lmNumStims, excludeFromCombo=True)
        lmCurvatureProbe.makeCorrelationFigures(pp, lmNumStims, excludeFromCombo=True)

    if plotFlagCheck(plotFlags, "rippleLocations"):
        LocationMeasure("taskRippleCount", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.numRipplesAtLocation, sesh, BP(probe=False))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)
        LocationMeasure("probeRippleCount", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.numRipplesAtLocation, sesh, BP(probe=True))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)
        LocationMeasure("taskRippleRate", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.rippleRateAtLocation, sesh, BP(probe=False))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)
        LocationMeasure("probeRippleRate", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.rippleRateAtLocation, sesh, BP(probe=True))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)

    if plotFlagCheck(plotFlags, "ripVsStimRate"):
        stimCount = SessionMeasure("stimCount", lambda sesh: len(sesh.btLFPBumps_posIdx), sessions)
        rippleCount = SessionMeasure(
            "rippleCount", lambda sesh: len(sesh.btRipsProbeStats), sessions)
        stimRate = SessionMeasure("stimRate", lambda sesh: len(sesh.btLFPBumps_posIdx) /
                                  sesh.taskDuration, sessions)
        rippleRate = SessionMeasure("rippleRate", lambda sesh: len(
            sesh.btRipsProbeStats) / sesh.taskDuration, sessions)

        # stimCount.makeFigures(pp, excludeFromCombo=True)
        # rippleCount.makeFigures(pp, excludeFromCombo=True)
        stimCount.makeCorrelationFigures(pp, rippleCount, excludeFromCombo=True)

        stimRate.makeFigures(pp, excludeFromCombo=True)
        rippleRate.makeFigures(pp, excludeFromCombo=True)
        stimRate.makeCorrelationFigures(pp, rippleRate, excludeFromCombo=True)

    if plotFlagCheck(plotFlags, "duration"):
        probeDuration = SessionMeasure(f"totalDuration_probe",
                                       lambda sesh: sesh.probeDuration,
                                       sessions)
        probeDuration.makeFigures(pp, excludeFromCombo=True)
        taskDuration = SessionMeasure(f"totalDuration_task",
                                      lambda sesh: sesh.taskDuration,
                                      sessions)
        taskDuration.makeFigures(pp, excludeFromCombo=True)

        for ti in range(4):
            tdur = SessionMeasure(f"t{ti+1}Dur", lambda sesh: sesh.getTrialDuration(ti), sessions)
            tdur.makeFigures(pp, excludeFromCombo=True)


def main():
    makeFigures()


if __name__ == "__main__":
    main()
