from typing import Callable, Iterable, Dict, List
from MeasureTypes import LocationMeasure, TrialMeasure
from PlotUtil import PlotManager
from DataminingFactory import runEveryCombination
from UtilFunctions import parseCmdLineAnimalNames, findDataDir, getLoadInfo
import os
from datetime import datetime
import time
from BTData import BTData
from BTSession import BTSession
import pandas as pd
import numpy as np


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
            def testFunc(sesh: BTSession, trialStart_posIdx: int, trialEnd_posIdx: int, homeAwayFlag: str):
                ret = trialStart_posIdx
                if sesh.isRippleInterruption:
                    ret *= 1.1
                return ret

            def latencyFunc(sesh: BTSession, trialStart_posIdx: int, trialEnd_posIdx: int, homeAwayFlag: str):
                return sesh.btPos_secs[trialEnd_posIdx] - sesh.btPos_secs[trialStart_posIdx]
            plotFlags.remove("tests")
            tm = TrialMeasure("test", latencyFunc, sessions)
            tm.makeFigures(pp, plotFlags=["noteverytrial", "noteverysession"])
            # tm = TrialMeasure(self, name: str,
            #      measureFunc: Callable[[BTSession, int, int, str], float],
            #      sessionList: List[BTSession],
            #      trialFilter: None | Callable[[BTSession, str, int, int, int, int], bool] = None,
            #      runStats=True)

        if len(plotFlags) != 0:
            print("Warning, unused plot flags:", plotFlags)


if __name__ == "__main__":
    main()

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
# - look also at specific task behavior periods like early learning
# - Try different measures of learning rate, specifically targetted at possible effect from paper
# - implement the following new measures
#   - path length from start of excursion
#       - sym ctrl locations here. Unless want to subdivide by well location type or something fancy
#   - General behavior differences: speed, num wells visited per excursion/bout, frac in exploration, others from previous lab meetings/thesis meetings
# - Think about each measure during task, what possible caveats are there?
#   - Is erode 3 enough?
# - Instead of ctrl vs stim, correlate with ripple count. Stim count too? might as well
# - test out new measures
# - remember to add in all new measures and parameters to the datamining factory. Can also remove results that get filtered out later based on BP
# - in testing found that avg home minus avg away trial duration difference b/w conditions is significant. Should run stats on this
# And while that's running, figure these out:
# - look at outlier sessions according to cum stim rates. Compare to behavior measures
# - Why are ripple counts so low?
# - Should prefer away control location? Or neutral? Dependent on measure?
# - Add in check when summarizing results for difference direction. Separate significant results this way
# - Difference in stim rates b/w conditions? Probs not, but should run stats
# - the following mods to existing measures:
#   - dot prod just positive
#   - frac excursion, avg over excursion not session
#   - Latency nans => max value + penalty. Possibly also path opt, others?
#   - path opt from start of excursion

# COMPLETED:
# - Import or move important functions so can easily run lots of measures from any file
# - Change how psudo-pval is presented. Histogram of shuffle values, line for real data, p value, and written i.e. "ctrl > swr"
