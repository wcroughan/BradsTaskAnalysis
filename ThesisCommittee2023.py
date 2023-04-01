import os
from datetime import datetime
import time
from functools import partial
from typing import List

from BTData import BTData
from PlotUtil import PlotManager
from UtilFunctions import getLoadInfo, findDataDir
from MeasureTypes import LocationMeasure, SessionMeasure
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP

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
        pass

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
