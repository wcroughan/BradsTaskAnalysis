from typing import Callable, Iterable, Dict, List
from MeasureTypes import LocationMeasure, TrialMeasure, TimeMeasure, SessionMeasure
from PlotUtil import PlotManager
from DataminingFactory import runEveryCombination
from UtilFunctions import parseCmdLineAnimalNames, findDataDir, getLoadInfo, numWellsVisited
import os
from datetime import datetime
import time
from BTData import BTData
from BTSession import BTSession
import pandas as pd
import numpy as np
from BTSession import BehaviorPeriod as BP
from functools import partial
from consts import allWellNames, offWallWellNames


def runDMFOnLM(measureFunc: Callable[..., LocationMeasure], allParams: Dict[str, Iterable], pp: PlotManager, minParamsSequential=2):
    def measureFuncWrapper(params: Dict[str, Iterable]):
        lm = measureFunc(**params)
        if "bp" in params:
            lm.makeFigures(pp, excludeFromCombo=True, everySessionBehaviorPeriod=params["bp"])
        else:
            lm.makeFigures(pp, excludeFromCombo=True)

    runEveryCombination(measureFuncWrapper, allParams,
                        numParametersSequential=min(minParamsSequential, len(allParams)))


def main(plotFlags: List[str] | str = "tests"):
    if isinstance(plotFlags, str):
        plotFlags = [plotFlags]

    dataDir = findDataDir()
    outputDir = "GreatBigDatamining"
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

    for ratName in animalNames:
        print("======================\n", ratName)
        if ratName[-1] == "d":
            ratName = ratName[:-1]
        sessions: List[BTSession] = allSessionsByRat[ratName]
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

            lm = LocationMeasure("LMtest", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.fracExcursionsVisited(BP(probe=True), pos, 0.25)),
                sessions, smoothDist=0)
            # lm.makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True))
            sm = SessionMeasure("SMtest", lambda sesh: np.remainder(
                np.sqrt(float(sesh.name[-1])), 1), sessions)
            # sm.makeFigures(pp, excludeFromCombo=True)
            lm.makeCorrelationFigures(pp, sm, excludeFromCombo=True)

        if "earlyVsLate" in plotFlags:
            plotFlags.remove("earlyVsLate")
            TimeMeasure("evl_latency", lambda sesh, t0, t1, _: sesh.btPos_secs[t1] - sesh.btPos_secs[t0],
                        sessions, timePeriodsGenerator=TimeMeasure.earlyVsLateTrials).makeFigures(
                pp, excludeFromCombo=True)
            TimeMeasure("1v2_latency", lambda sesh, t0, t1, _: sesh.btPos_secs[t1] - sesh.btPos_secs[t0],
                        sessions, timePeriodsGenerator=lambda sesh: TimeMeasure.earlyVsLateTrials(sesh, earlyRange=(0, 1), lateRange=(1, 2))).makeFigures(
                pp, excludeFromCombo=True)
            TimeMeasure("1v23_latency", lambda sesh, t0, t1, _: sesh.btPos_secs[t1] - sesh.btPos_secs[t0],
                        sessions, timePeriodsGenerator=lambda sesh: TimeMeasure.earlyVsLateTrials(sesh, earlyRange=(0, 1), lateRange=(1, 3))).makeFigures(
                pp, excludeFromCombo=True)

        if "T2" in plotFlags:
            plotFlags.remove("T2")

            def T2Latency(sesh: BTSession):
                trialPosIdxs = sesh.getAllTrialPosIdxs()
                h2Start = trialPosIdxs[2][0]
                h2End = trialPosIdxs[2][1]
                return sesh.btPos_secs[h2End] - sesh.btPos_secs[h2Start]
            SessionMeasure("TH2 Latency", T2Latency, sessions).makeFigures(
                pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=False, trialInterval=(2, 3)))

        if "generalBehavior" in plotFlags:
            plotFlags.remove("generalBehavior")
            for inProbe in [True, False]:
                probeStr = "probe" if inProbe else "bt"
                TimeMeasure(f"speed_{probeStr}", lambda sesh, t0, t1, _: np.nanmean(sesh.probeVelCmPerS[t0:t1]),
                            sessions, timePeriodsGenerator=(30, 15, inProbe)).makeFigures(
                    pp, excludeFromCombo=True)
                TimeMeasure(f"fracExplore_{probeStr}", lambda sesh, t0, t1, _: np.count_nonzero(
                    sesh.evalBehaviorPeriod(BP(probe=inProbe, timeInterval=(t0, t1, "posIdx"), inclusionFlags="explore"))) / (t1 - t0),
                    sessions, timePeriodsGenerator=(30, 15, inProbe)).makeFigures(
                    pp, excludeFromCombo=True)
                TimeMeasure(f"fracOffWall_{probeStr}", lambda sesh, t0, t1, _: np.count_nonzero(
                    sesh.evalBehaviorPeriod(BP(probe=inProbe, timeInterval=(t0, t1, "posIdx"), inclusionFlags="offWall"))) / (t1 - t0),
                    sessions, timePeriodsGenerator=(30, 15, inProbe)).makeFigures(
                    pp, excludeFromCombo=True)

                def avgNumWellsVisitedPerBout(countRepeats: bool, offWallOnly: bool, sesh: BTSession):
                    nw = sesh.probeNearestWells if inProbe else sesh.btNearestWells
                    boutStarts = sesh.probeExploreBoutStart_posIdx if inProbe else sesh.btExploreBoutStart_posIdx
                    boutEnds = sesh.probeExploreBoutEnd_posIdx if inProbe else sesh.btExploreBoutEnd_posIdx
                    return np.nanmean([numWellsVisited(nw[s:b], countRepeats, offWallWellNames if offWallOnly else allWellNames) for s, b in zip(boutStarts, boutEnds)])

                SessionMeasure(f"numWellsVisitedPerExplorationBout_{probeStr}", partial(avgNumWellsVisitedPerBout, False, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))
                SessionMeasure(f"numWellsVisitedPerExplorationBout_withRepeats_{probeStr}", partial(avgNumWellsVisitedPerBout, True, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))
                SessionMeasure(f"numWellsVisitedPerExplorationBout_offWall_{probeStr}", partial(avgNumWellsVisitedPerBout, False, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))
                SessionMeasure(f"numWellsVisitedPerExplorationBout_withRepeats_offWall_{probeStr}", partial(avgNumWellsVisitedPerBout, True, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))

                def avgNumWellsVisitedPerExcursion(countRepeats: bool, offWallOnly: bool, sesh: BTSession):
                    nw = sesh.probeNearestWells if inProbe else sesh.btNearestWells
                    excursionStarts = sesh.probeExcursionStart_posIdx if inProbe else sesh.btExcursionStart_posIdx
                    excursionEnds = sesh.probeExcursionEnd_posIdx if inProbe else sesh.btExcursionEnd_posIdx
                    return np.nanmean([numWellsVisited(nw[s:b], countRepeats, offWallWellNames if offWallOnly else allWellNames)
                                       for s, b in zip(excursionStarts, excursionEnds)])

                SessionMeasure(f"numWellsVisitedPerExcursion_{probeStr}", partial(avgNumWellsVisitedPerExcursion, False, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))
                SessionMeasure(f"numWellsVisitedPerExcursion_withRepeats_{probeStr}", partial(avgNumWellsVisitedPerExcursion, True, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))
                SessionMeasure(f"numWellsVisitedPerExcursion_offWall_{probeStr}", partial(avgNumWellsVisitedPerExcursion, False, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))
                SessionMeasure(f"numWellsVisitedPerExcursion_withRepeats_offWall_{probeStr}", partial(avgNumWellsVisitedPerExcursion, True, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))

        if len(plotFlags) != 0:
            print("Warning, unused plot flags:", plotFlags)


if __name__ == "__main__":
    main()

# Measures to add for big sweep:
# T2 everything, T1 vs 2 everything, including:
#  - latency, path opt, path length, all the other stuff
# Away T1/2, and similar, measures for home memory/behavior (can do location measures and all the rest)
# Avg home minus avg away trial duration difference b/w conditions

# Note from lab meeting 2023-3-7
# In paper, possible effect in first few trials
# BP flag for early trials/trial nums
# frac excursion, add over excursions instead of session so don't weight low exploration sessions higher
# General behavior differences like speed, amt exploration, wells visited per time/per bout
# Instead of pseudo p-val, just call it p-val, take lower number, and indicate separately which direction
# Shuffle histogram below violin plots with p vals
# Should I be more neutral in choosing which control location, or bias for away fine?
# Dot product but only positive side?
# Carefully look for opposite direction of effect with different parameters
# Starting excursion near home well - diff b/w conditions?
# Average of aways with nans biases to lower values. Esp latency, worst case is b-line to home but pass through a couple aways first
#   maybe set nans to max possible value in BP? Maybe add penalty?
# Overal narrative may support strat of direct paths from far away vs roundabout searching with familiarity
# Path opt but from start of excursion
# Run stats on cumulative amt of stims

# To finish datamining B17 and apply those measures to all animals, should complete the following steps:
# - Instead of ctrl vs stim, correlate with ripple count. Stim count too? might as well
# - test out new measures and new flag arguments
# - remember to add in all new measures and parameters to the datamining factory. Can also remove results that get filtered out later based on BP
# And while that's running, figure these out:
# - look at outlier sessions according to cum stim rates. Compare to behavior measures
# - Why are ripple counts so low?
# - Should prefer away control location? Or neutral? Dependent on measure?
# - Add in check when summarizing results for difference direction. Separate significant results this way
# - Difference in stim rates b/w conditions? Probs not, but should run stats

# COMPLETED:
# - Import or move important functions so can easily run lots of measures from any file
# - Change how psudo-pval is presented. Histogram of shuffle values, line for real data, p value, and written i.e. "ctrl > swr"
# - implement the following new measures
#   - path length from start of excursion
#       - sym ctrl locations here. Unless want to subdivide by well location type or something fancy
#   - General behavior differences: speed, num wells visited per excursion/bout, frac in exploration, others from previous lab meetings/thesis meetings
# - the following mods to existing measures:
#   - dot prod just positive
#   - frac excursion, avg over excursion not session
#   - Latency nans => max value + penalty. Possibly also path opt, others?
#   - path opt from start of excursion
# - look also at specific task behavior periods like early learning
# - Try different measures of learning rate, specifically targetted at possible effect from paper
# - Think about each measure during task, what possible caveats are there?
#   - Is erode 3 enough?
# - in testing found that avg home minus avg away trial duration difference b/w conditions is significant. Should run stats on this
