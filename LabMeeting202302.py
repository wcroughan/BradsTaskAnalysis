from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from MeasureTypes import WellMeasure, TrialMeasure, LocationMeasure, SessionMeasure
from BTData import BTData
from PlotUtil import PlotManager, setupBehaviorTracePlot, plotIndividualAndAverage
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getLoadInfo, getRipplePower, \
    getWellPosCoordinates, offWall
import os
import time
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from typing import List, Tuple, Iterable, Callable, Dict
from matplotlib.axes import Axes
import math
import MountainViewIO
from consts import TRODES_SAMPLING_RATE
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from functools import partial
from DataminingFactory import runEveryCombination
import multiprocessing
from dataclasses import replace


# TODO
# To accomplish goal of first drafts, have the following plots made:
#  - Interruption confirmation - Try and remove artifact
#
# =================
# Data Cleanup
# =================
# Look at notes in B18's LoadInfo section, deal with all of them
# i.e. minimum date for B13
#
# Double check position tracking fill times for other rats. Especially B16 07_10 looks a bit wonky
# And B16 15_09
#
# Set a maximum gap for previous sessions, so large gaps aren't counted and prev dir goes back to None
#
# =================
# LFP Analysis
# =================
# Debug weird gap in LFP latency plots
# Remove interruption artifact from LFP
#   Simultaneous across other tetrodes
#
# Any effect on ripple power on other tetrodes of interuption?
# Where do interruptions happen?
#   How many total? What rate? etc
#
# Rebound effect?
# Just cumulative num stims by condition
# Probably better way to look also, how often after interruption is there another ripple?
#
# =================
# Behavior Analysis
# =================
# Datamine B17
# Task gravity, exclude reward?
# A measure that tries to capture the consistent paths that develop during a session, and compares that
#   to probe behavior. Something like frequency of entering a well from each angle maybe, so if
#   always approaching 35 from top and leaving from left, how consistent is that with probe?
#
# Work on DLC
#
# Better optimized clip correction region output. Save til the end so can:
#   compare to task and probe times, weed out ones that aren't during behavior
#   print separately at the end all at once


# COmpleted:
# Debug dotprod
# Anything else look fishy?
# Def glitch where B16 12_11 is giving nans
# - Actually no that one was fine, just all exploration was done after fill time
#
# Trial measure output graphs
#
# To all every-session plots, add path on top
#
# fisheye correction
#
# refactor everything to unify units, get rid of unecessary stuff, make code more readable
#
# make a dataclass that is a single ripple. Can contain start, stop, len, peak, mean, std, other stuff probs
# Then return from detectRipples can be a list of those
#
# Inspect fisheye correction more. Look at output closely, and if it seems legit (espcially at edges
# of environment), should use that for velocity, curvature, etc. Otherwise center wells will think
# they have slower speed, and curvature radius will change
#
# Look at path similarity between task and probe. Usual paths that are traced out during task don't seem to be
# followed at all during probe
#
# Datamining:
# Especially debugged dot product not divided by total time
# Add control that is rotated-similar wells instead of aways
# Tune up spotlight/dotprod measure by just looking at behavior during home trials, should see clear trend toward home
# Symetrical controls might be more important here than other measures
# Distance discount?
# Plus velocity weighting perhaps??
# Maybe exclude times when at well? At least need to way to deal with looking right pas a well
# Maybe for everysession plots try centering on home to combine, see if clear gradient situation appears
#
# Shuffles can go crazy with different measures when in datamine mode.
# Also have this output a helpful ranking of shuffles that showed up significant
#
# Histogram balance might be a good visual?
#
# focus first on getting something on which to base discussion, like very basic demonstration of memory of home during probe
#   once have measures showing home/away difference, can talk about condition difference
#
# Separate plotmanager and shuffler
# Have shuffler load a list of info files from which it grabs stats
#
# Some "well visits" are just one time point. Need to smooth more and/or just generally change logic away from this
# visit stuff
#

def generateDMFReport(resultsTableFileName: str, skipSimple: bool = False) -> None:
    df = pd.read_hdf(resultsTableFileName)

    def bpFlagsWereUseful(row):
        s = row["measure"]
        if "avgDwellTimeAtPosition" in s or \
           "numVisitsToPosition" in s or \
           "latencyToPosition" in s or \
           "fracExcursionsVisited" in s or \
           "pathOptimalityToPosition" in s:
            # These measures ignored flags
            if "moving" in s or \
               "offWall" in s or \
               "homeTrial" in s or \
               "awayTrial" in s:
                return False
        return True

    def isSimpleMeasure(row):
        s = row["measure"]
        return "simple" in s

    df["bpFlagsWereUseful"] = df.apply(bpFlagsWereUseful, axis=1)
    if skipSimple:
        df = df[~df.apply(isSimpleMeasure, axis=1)]
    df = df[df["bpFlagsWereUseful"]]
    nndf = df[df["isDiffShuf"] & ~df["isNextSeshDiffShuf"]]
    nndf = nndf.drop(columns=["bpFlagsWereUseful", "isNextSeshDiffShuf", "isDiffShuf"])

    def isProbe(row):
        return "_probe_" in row["measure"]

    nndf["isProbe"] = nndf.apply(isProbe, axis=1)

    def extractMeasureCategory(row):
        s = row["measure"]
        assert (isinstance(s, str))
        if row["isProbe"]:
            bpIndex = s.find("_probe_")
        else:
            bpIndex = s.find("_bt_")
        s = s[3:bpIndex]
        if s.startswith("simple"):
            return s.split("_")[1]
        return s

    nndf["measureCategory"] = nndf.apply(extractMeasureCategory, axis=1)

    # print(nndf.head())

    pvalThresholds = [0.1, 0.04, 0.01]
    for pvalThreshold in pvalThresholds:
        sigDf = nndf[nndf["pval"] <= pvalThreshold]
        g = sigDf.groupby(["measureCategory", "isProbe"]).agg({"plot": ["count", "unique"]})
        print(g.head())
        inFileName = ".".join(os.path.basename(resultsTableFileName).split(".")[:-1])
        outFileName = f"dmfReport_{inFileName}_pval{pvalThreshold}.csv"
        outDir = os.path.dirname(resultsTableFileName)
        g.to_csv(os.path.join(outDir, outFileName))


def runDMFOnLM(measureFunc: Callable[..., LocationMeasure], allParams: Dict[str, Iterable], pp: PlotManager, minParamsSequential=2):
    def measureFuncWrapper(params: Dict[str, Iterable]):
        lm = measureFunc(**params)
        if "bp" in params:
            lm.makeFigures(pp, excludeFromCombo=True, everySessionBehaviorPeriod=params["bp"])
        else:
            lm.makeFigures(pp, excludeFromCombo=True)

    runEveryCombination(measureFuncWrapper, allParams,
                        numParametersSequential=min(minParamsSequential, len(allParams)))


def hacky_RunOldStats(infoFileName: str):
    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "202302_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=2)
    pp.runShuffles()


def makeFigures(RUN_SHUFFLES=False, RUN_UNSPECIFIED=True, PRINT_INFO=True,
                RUN_JUST_THIS_SESSION=None, RUN_SPOTLIGHT=None, RUN_FRAC_EXCURSIONS_VISITED=None,
                RUN_OPTIMALITY=None, PLOT_DETAILED_TASK_TRACES=None,
                PLOT_DETAILED_PROBE_TRACES=None, RUN_ENTRY_EXIT_ANGLE=None,
                RUN_PATH_OCCUPANCY=None, RUN_SPOTLIGHT_EXPLORATION_TASK=None,
                RUN_DOT_PROD=None, RUN_SMOOTHING_TEST=None, RUN_MANY_DOTPROD=None,
                RUN_LFP_LATENCY=None, MAKE_INDIVIDUAL_INTERRUPTION_PLOTS=False,
                PLOT_OCCUPANCY=None, RUN_SPOTLIGHT_EXPLORATION_PROBE=None,
                RUN_OCCUPANCY_CORRELATION=None, RUN_TASK_GRAVITY=None,
                RUN_AVG_DWELL_WITHOUT_OUTLIER=None, RUN_CURVATURE=None,
                RUN_DOT_PROD_FACTORY=None, FIND_AVG_DWELL_OUTLIER=None,
                RUN_NEW_GRAVITY=None, RUN_GRAVITY_FACTORY=None, RUN_SIMPLE_MEASURES=None,
                PLOT_TASK_TRIAL_PERFORMANCE=None, RUN_PREV_SESSION_PERSEVERATION=None,
                REDO_NUM_VISITS=None, RUN_THE_FINAL_DATAMINE=None, TEST_DATA=False,
                RERUN_SIMPLE_MEASURES=None, CUMULATIVE_LFP=None, MAKE_CLEAN_LAB_MEETING_FIGURES=None,
                RUN_TESTS=False, MAKE_COMBINED=True, DATAMINE=False):
    if RUN_SPOTLIGHT is None:
        RUN_SPOTLIGHT = RUN_UNSPECIFIED
    if RUN_DOT_PROD is None:
        RUN_DOT_PROD = RUN_UNSPECIFIED
    if RUN_SMOOTHING_TEST is None:
        RUN_SMOOTHING_TEST = RUN_UNSPECIFIED
    if RUN_MANY_DOTPROD is None:
        RUN_MANY_DOTPROD = RUN_UNSPECIFIED
    if RUN_OPTIMALITY is None:
        RUN_OPTIMALITY = RUN_UNSPECIFIED
    if PLOT_DETAILED_TASK_TRACES is None:
        PLOT_DETAILED_TASK_TRACES = RUN_UNSPECIFIED
    if PLOT_DETAILED_PROBE_TRACES is None:
        PLOT_DETAILED_PROBE_TRACES = RUN_UNSPECIFIED
    if RUN_LFP_LATENCY is None:
        RUN_LFP_LATENCY = RUN_UNSPECIFIED
    if RUN_ENTRY_EXIT_ANGLE is None:
        RUN_ENTRY_EXIT_ANGLE = RUN_UNSPECIFIED
    if RUN_PATH_OCCUPANCY is None:
        RUN_PATH_OCCUPANCY = RUN_UNSPECIFIED
    if RUN_SPOTLIGHT_EXPLORATION_TASK is None:
        RUN_SPOTLIGHT_EXPLORATION_TASK = RUN_UNSPECIFIED
    if PLOT_OCCUPANCY is None:
        PLOT_OCCUPANCY = RUN_UNSPECIFIED
    if RUN_SPOTLIGHT_EXPLORATION_PROBE is None:
        RUN_SPOTLIGHT_EXPLORATION_PROBE = RUN_UNSPECIFIED
    if RUN_FRAC_EXCURSIONS_VISITED is None:
        RUN_FRAC_EXCURSIONS_VISITED = RUN_UNSPECIFIED
    if RUN_NEW_GRAVITY is None:
        RUN_NEW_GRAVITY = RUN_UNSPECIFIED
    if RUN_GRAVITY_FACTORY is None:
        RUN_GRAVITY_FACTORY = RUN_UNSPECIFIED
    if RUN_SIMPLE_MEASURES is None:
        RUN_SIMPLE_MEASURES = RUN_UNSPECIFIED
    if RUN_OCCUPANCY_CORRELATION is None:
        RUN_OCCUPANCY_CORRELATION = RUN_UNSPECIFIED
    if RUN_TASK_GRAVITY is None:
        RUN_TASK_GRAVITY = RUN_UNSPECIFIED
    if RUN_AVG_DWELL_WITHOUT_OUTLIER is None:
        RUN_AVG_DWELL_WITHOUT_OUTLIER = RUN_UNSPECIFIED
    if RUN_CURVATURE is None:
        RUN_CURVATURE = RUN_UNSPECIFIED
    if RUN_DOT_PROD_FACTORY is None:
        RUN_DOT_PROD_FACTORY = RUN_UNSPECIFIED
    if FIND_AVG_DWELL_OUTLIER is None:
        FIND_AVG_DWELL_OUTLIER = RUN_UNSPECIFIED
    if PLOT_TASK_TRIAL_PERFORMANCE is None:
        PLOT_TASK_TRIAL_PERFORMANCE = RUN_UNSPECIFIED
    if RUN_PREV_SESSION_PERSEVERATION is None:
        RUN_PREV_SESSION_PERSEVERATION = RUN_UNSPECIFIED
    if REDO_NUM_VISITS is None:
        REDO_NUM_VISITS = RUN_UNSPECIFIED
    if RUN_THE_FINAL_DATAMINE is None:
        RUN_THE_FINAL_DATAMINE = RUN_UNSPECIFIED
    if RERUN_SIMPLE_MEASURES is None:
        RERUN_SIMPLE_MEASURES = RUN_UNSPECIFIED
    if CUMULATIVE_LFP is None:
        CUMULATIVE_LFP = RUN_UNSPECIFIED
    if MAKE_CLEAN_LAB_MEETING_FIGURES is None:
        MAKE_CLEAN_LAB_MEETING_FIGURES = RUN_UNSPECIFIED

    dataDir = findDataDir()
    outputDir = "202302_labmeeting"
    if RUN_THE_FINAL_DATAMINE or RERUN_SIMPLE_MEASURES:
        if TEST_DATA:
            outputDir = "final_datamine_test"
        else:
            outputDir = "final_datamine"
    elif MAKE_CLEAN_LAB_MEETING_FIGURES:
        outputDir = "202302_labmeeting_clean"
    else:
        outputDir = "202302_labmeeting_2"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    if RUN_JUST_THIS_SESSION is None:
        animalNames = parseCmdLineAnimalNames(default=["B17"])
        if len(animalNames) > 1 and DATAMINE:
            raise Exception("Can't datamine with more than one animal. I refuse!")
        sessionToRun = None
    else:
        animalNames = [RUN_JUST_THIS_SESSION[0]]
        sessionToRun = RUN_JUST_THIS_SESSION[1]

    infoFileName = datetime.now().strftime("_".join(animalNames) + "_%Y%m%d_%H%M%S" + ".txt")
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=2 if DATAMINE else 3)

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
        if sessionToRun is not None:
            sessions: List[BTSession] = [
                s for s in sessions if sessionToRun in s.name or sessionToRun in s.infoFileName]
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
        if PRINT_INFO:
            print(s)

        # if hasattr(sessions[0], "probeFillTime"):
        #     sessionsWithProbeFillPast90 = [s for s in sessionsWithProbe if s.probeFillTime > 90]
        # else:
        #     sessionsWithProbeFillPast90 = sessionsWithProbe

        pp.pushOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if RUN_TESTS:
            infoFileName = "/media/WDC8/figures/202302_labmeeting/20230302_110034_significantShuffles.h5"
            pp.shuffler.summarizeShuffleResults(infoFileName)

            # bp1 = BP(probe=True, inclusionFlags="homeTrial", erode=10)
            # bp2 = BP(probe=True, inclusionFlags="homeTrial")
            # with pp.newFig("test nan", subPlots=(1, 2), showPlot=True, savePlot=False) as pc:
            #     sesh = sessions[0]
            #     ts, xs, ys = sesh.posDuringBehaviorPeriod(bp1)
            #     pc.axs[0].plot(xs, ys)
            #     ts, xs, ys = sesh.posDuringBehaviorPeriod(bp2)
            #     pc.axs[1].plot(xs, ys)

            # lm = LocationMeasure("test nan",
            #                      lambda sesh: sesh.getValueMap(
            #                          lambda pos: sesh.getDotProductScore(bp, pos,
            #                                                              distanceWeight=-1,
            #                                                              normalize=True)),
            #                      sessions,
            #                      smoothDist=smoothDist, parallelize=False)

        if MAKE_CLEAN_LAB_MEETING_FIGURES:
            if True:
                plotsToMake = "all"
                # plotsToMake = "measureVsCtrl"
                LocationMeasure("frac excursions 0.25 probe 0.00",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.fracExcursionsVisited(BP(probe=True), pos, 0.25)),
                                sessions, smoothDist=0).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True))
                LocationMeasure("frac excursions 0.25 probe 0 60 0.00",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.fracExcursionsVisited(BP(probe=True, timeInterval=(0, 60)), pos, 0.25)),
                                sessions, smoothDist=0).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True, timeInterval=(0, 60)))
                LocationMeasure("dot prod probe filltime offwall moving dw-1 normFalse smooth 0",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.getDotProductScore(BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags=["offWall", "moving"]),
                                                                        pos, distanceWeight=-1, normalize=False)),
                                sessions, smoothDist=0).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags=["offWall", "moving"]))
                LocationMeasure("curvature 0.5 probe 0.5",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.avgCurvatureAtPosition(BP(probe=True), pos, 0.5)),
                                sessions, smoothDist=0.5).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True))
                LocationMeasure("path optimality 1 probe 0 120 0.50",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.pathOptimalityToPosition(BP(probe=True, timeInterval=(0, 120)), pos, 1)),
                                sessions, smoothDist=0.5).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True, timeInterval=(0, 120)))
                LocationMeasure("latency 1 probe 0 120 0.5",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.latencyToPosition(BP(probe=True, timeInterval=(0, 120)), pos, 1)),
                                sessions, smoothDist=0.5).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True, timeInterval=(0, 120)))
                LocationMeasure("Total Time 1 filltime 0",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.totalTimeAtPosition(BP(probe=True, timeInterval=BTSession.fillTimeInterval), pos, 1)),
                                sessions, smoothDist=0).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True, timeInterval=BTSession.fillTimeInterval))
                LocationMeasure("Total Time 0.5 offwall 0",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.totalTimeAtPosition(BP(probe=True, inclusionFlags="offWall"), pos, 0.5)),
                                sessions, smoothDist=0).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True, inclusionFlags="offWall"))
                LocationMeasure("Gravity probe pr1.5 vrf.4 df2 smooth .5",
                                lambda sesh: sesh.getValueMap(
                                    lambda pos: sesh.getGravity(BP(probe=True), pos,
                                                                passRadius=1.5,
                                                                visitRadius=1.5 * 0.4,
                                                                passDenoiseFactor=2)),
                                sessions, smoothDist=0.5).makeFigures(pp, plotFlags=plotsToMake, everySessionBehaviorPeriod=BP(probe=True))

            else:
                # Now make behavior trace plots
                ncols = np.ceil(np.sqrt(len(sessions))).astype(int)
                nrows = np.ceil(len(sessions) / ncols).astype(int)

                allConsideredBPs = [BP(probe=True), BP(probe=False),
                                    BP(probe=True, timeInterval=(0, 60)),
                                    BP(probe=True, timeInterval=(0, 120)),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                                    BP(probe=True, inclusionFlags="moving"),
                                    BP(probe=True, inclusionFlags="offWall"),
                                    BP(probe=True, inclusionFlags=["offWall", "moving"]),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                       inclusionFlags="moving"),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                       inclusionFlags="offWall"),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                       inclusionFlags=["offWall", "moving"]),
                                    BP(probe=False, erode=3, inclusionFlags="homeTrial"),
                                    BP(probe=False, erode=3, inclusionFlags="awayTrial")]

                for bp in allConsideredBPs:
                    with pp.newFig("behavior traces " + bp.filenameString(), subPlots=(nrows, ncols)) as pc:
                        for si, sesh in enumerate(sessions):
                            ax = pc.axs.flat[si]
                            wellSize = 90
                            setupBehaviorTracePlot(ax, sesh, showWells="HA", wellSize=wellSize)
                            _, x, y = sesh.posDuringBehaviorPeriod(bp)
                            ax.plot(x, y, "k", lw=2, zorder=0.5)
                            s = "++++"
                            titleStr = f"{sesh.name} {sesh.probeFillTime} {s[0:sesh.probeFillTime // 60]}"
                            ax.set_title(titleStr)

        if CUMULATIVE_LFP:
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
                    itiStimCounts[si, :], _ = np.histogram(stimTs, bins=itiBins)

                    for i in reversed(range(len(itiBins)-1)):
                        if itiStimCounts[si, i] != 0:
                            break
                        itiStimCounts[si, i] = np.nan

                    ripTs = np.array([r.start_ts for r in sesh.itiRipsPreStats])
                    ripTs = ripTs[np.logical_and(ripTs > sesh.itiLfpStart_ts, ripTs <
                                                 sesh.itiLfpEnd_ts)] - sesh.itiLfpStart_ts
                    ripTs = ripTs.astype(np.float64)
                    ripTs /= TRODES_SAMPLING_RATE
                    itiRippleCounts[si, :], _ = np.histogram(ripTs, bins=itiBins)

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
                    probeStimCounts[si, :], _ = np.histogram(stimTs, bins=probeBins)

                    for i in reversed(range(len(probeBins)-1)):
                        if probeStimCounts[si, i] != 0:
                            break
                        probeStimCounts[si, i] = np.nan

                    ripTs = np.array([r.start_ts for r in sesh.probeRipsProbeStats])
                    ripTs = ripTs[np.logical_and(ripTs > sesh.probePos_ts[0], ripTs <
                                                 sesh.probePos_ts[-1])] - sesh.probePos_ts[0]
                    ripTs = ripTs.astype(np.float64)
                    ripTs /= TRODES_SAMPLING_RATE
                    probeRippleCounts[si, :], _ = np.histogram(ripTs, bins=probeBins)

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

            with pp.newFig("lfp task stimCounts") as pc:
                ax = pc.ax
                ax.plot(bins[1:], stimCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pp.newFig("lfp task rippleCounts") as pc:
                ax = pc.ax
                ax.plot(bins[1:], rippleCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")
            with pp.newFig("lfp task stimCounts by cond") as pc:
                ax = pc.ax
                ax.plot(bins[1:], stimCountsSWR.T, c="orange", zorder=1)
                ax.plot(bins[1:], stimCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pp.newFig("lfp task rippleCounts by cond") as pc:
                ax = pc.ax
                ax.plot(bins[1:], rippleCountsSWR.T, c="orange", zorder=1)
                ax.plot(bins[1:], rippleCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")

            with pp.newFig("lfp iti stimCounts") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiStimCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pp.newFig("lfp iti rippleCounts") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiRippleCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")
            with pp.newFig("lfp iti stimCounts by cond") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiStimCountsSWR.T, c="orange", zorder=1)
                ax.plot(itiBins[1:], itiStimCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pp.newFig("lfp iti rippleCounts by cond") as pc:
                ax = pc.ax
                ax.plot(itiBins[1:], itiRippleCountsSWR.T, c="orange", zorder=1)
                ax.plot(itiBins[1:], itiRippleCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")

            with pp.newFig("lfp probe stimCounts") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeStimCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pp.newFig("lfp probe rippleCounts") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeRippleCounts.T, c="grey", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")
            with pp.newFig("lfp probe stimCounts by cond") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeStimCountsSWR.T, c="orange", zorder=1)
                ax.plot(probeBins[1:], probeStimCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Stim Count")
            with pp.newFig("lfp probe rippleCounts by cond") as pc:
                ax = pc.ax
                ax.plot(probeBins[1:], probeRippleCountsSWR.T, c="orange", zorder=1)
                ax.plot(probeBins[1:], probeRippleCountsCtrl.T, c="cyan", zorder=1)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Cumulative Ripple Count")

        if RERUN_SIMPLE_MEASURES:
            if TEST_DATA:
                sessions = sessions[:4]

            def makeSimpleMeasure(func, radius, smoothDist, bp: BP):
                return LocationMeasure(f"simple {func.__name__} {radius:.2f} {bp.filenameString()} {smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: func(sesh, bp, pos, radius=radius),
                                       ),
                                       sessions,
                                       smoothDist=smoothDist,
                                       parallelize=False)

            allConsideredBPs = [BP(probe=True), BP(probe=False),
                                BP(probe=True, timeInterval=(0, 60)),
                                BP(probe=True, timeInterval=(0, 120)),
                                BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                                BP(probe=True, inclusionFlags="moving"),
                                BP(probe=True, inclusionFlags="offWall"),
                                BP(probe=True, inclusionFlags=["offWall", "moving"]),
                                BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                   inclusionFlags="moving"),
                                BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                   inclusionFlags="offWall"),
                                BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                   inclusionFlags=["offWall", "moving"]),
                                BP(probe=False, erode=3, inclusionFlags="homeTrial"),
                                BP(probe=False, erode=3, inclusionFlags="awayTrial")]

            if TEST_DATA:
                allConsideredBPs = [BP(probe=True), BP(probe=False),
                                    BP(probe=True, timeInterval=(0, 60)),
                                    BP(probe=False, inclusionFlags="moving"),
                                    BP(probe=True, inclusionFlags="moving")]

            simpleMeasureParams = {
                "func": [BTSession.totalTimeAtPosition,
                         BTSession.avgDwellTimeAtPosition,
                         BTSession.fracExcursionsVisited,
                         BTSession.numVisitsToPosition,
                         BTSession.latencyToPosition,
                         BTSession.avgCurvatureAtPosition,
                         BTSession.pathOptimalityToPosition],
                "radius": [0.25, 0.5, 1, 1.5],
                "smoothDist": [0, 0.5, 1.0],
                "bp": allConsideredBPs
            }

            if TEST_DATA:
                simpleMeasureParams = {
                    "func": [BTSession.totalTimeAtPosition,
                             BTSession.fracExcursionsVisited],
                    "radius": [0.25, 0.5],
                    "smoothDist": [0, 0.5],
                    "bp": allConsideredBPs
                }

            print("Rerunning simple measures")
            runDMFOnLM(makeSimpleMeasure, simpleMeasureParams, pp)

        if RUN_THE_FINAL_DATAMINE:
            if TEST_DATA:
                sessions = sessions[:4]

            def makeSimpleMeasure(func, radius, smoothDist, bp: BP):
                raise Exception("glitch here ... saving for posterity but func is ignored....")
                return LocationMeasure(f"simple {func.__name__} {radius:.2f} {bp.filenameString()} {smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.numVisitsToPosition(
                                               bp, pos, radius=radius),
                                       ),
                                       sessions,
                                       smoothDist=smoothDist,
                                       parallelize=False)

            if TEST_DATA:
                allConsideredBPs = [BP(probe=True, timeInterval=(0, 60)),
                                    BP(probe=True, timeInterval=(0, 120)),
                                    BP(probe=False)]
            else:
                allConsideredBPs = [BP(probe=True), BP(probe=False),
                                    BP(probe=True, timeInterval=(0, 60)),
                                    BP(probe=True, timeInterval=(0, 120)),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                                    BP(probe=True, inclusionFlags="moving"),
                                    BP(probe=True, inclusionFlags="offWall"),
                                    BP(probe=True, inclusionFlags=["offWall", "moving"]),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                       inclusionFlags="moving"),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                       inclusionFlags="offWall"),
                                    BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                                       inclusionFlags=["offWall", "moving"]),
                                    BP(probe=False, erode=3, inclusionFlags="homeTrial"),
                                    BP(probe=False, erode=3, inclusionFlags="awayTrial")]

            simpleMeasureParams = {
                "func": [BTSession.totalTimeAtPosition,
                         BTSession.avgDwellTimeAtPosition,
                         BTSession.fracExcursionsVisited,
                         BTSession.numVisitsToPosition,
                         BTSession.latencyToPosition,
                         BTSession.avgCurvatureAtPosition,
                         BTSession.pathOptimalityToPosition],
                "radius": [0.25, 0.5, 1, 1.5],
                "smoothDist": [0, 0.5, 1.0],
                "bp": allConsideredBPs
            }

            if not TEST_DATA:
                print("Running simple measures (1/4)")
                runDMFOnLM(makeSimpleMeasure, simpleMeasureParams, pp)

            def makeDotProdMeasure(bp: BP, distanceWeight, normalize, smoothDist):
                return LocationMeasure(f"dotprod {bp.filenameString()} "
                                       f"distWeight={distanceWeight:.2f} "
                                       f"normalize={normalize} "
                                       f"smoothDist={smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.getDotProductScore(bp, pos,
                                                                               distanceWeight=distanceWeight,
                                                                               normalize=normalize)),
                                       sessions,
                                       smoothDist=smoothDist, parallelize=False)

            if TEST_DATA:
                allDotProdParams = {
                    "bp": allConsideredBPs,
                    "distanceWeight": np.linspace(-1, 1, 3),
                    "normalize": [True, False],
                    "smoothDist": [0, 0.5],
                }
            else:
                allDotProdParams = {
                    "bp": allConsideredBPs,
                    "distanceWeight": np.linspace(-1, 1, 5),
                    "normalize": [True, False],
                    "smoothDist": [0, 0.5, 1.0],
                }

            if not TEST_DATA:
                print("Running dot product measures (2/4)")
                runDMFOnLM(makeDotProdMeasure, allDotProdParams, pp, minParamsSequential=1)

            def makeGravityMeasure(bp: BP, passRadius, visitRadiusFactor, passDenoiseFactor, measureSmoothDist):
                return LocationMeasure(f"gravity {bp.filenameString()} "
                                       f"passRadius={passRadius:.2f} "
                                       f"visitRadiusFactor={visitRadiusFactor:.2f} "
                                       f"denoiseFactor={passDenoiseFactor:.2f} "
                                       f"measureSmoothDist={measureSmoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.getGravity(bp, pos,
                                                                       passRadius=passRadius,
                                                                       visitRadius=passRadius * visitRadiusFactor,
                                                                       passDenoiseFactor=passDenoiseFactor)),
                                       sessions,
                                       smoothDist=measureSmoothDist, parallelize=False)
            if TEST_DATA:
                allParams = {
                    "passRadius": np.linspace(0.5, 1.5, 2),
                    "visitRadiusFactor": np.linspace(0.2, 0.4, 2),
                    "passDenoiseFactor": np.linspace(1.25, 2, 2),
                    "measureSmoothDist": np.linspace(0, 0.5, 2),
                    "bp": allConsideredBPs,
                }
            else:
                allParams = {
                    "passRadius": np.linspace(0.5, 1.5, 3),
                    "visitRadiusFactor": np.linspace(0.2, 0.4, 2),
                    "passDenoiseFactor": np.linspace(1.25, 2, 3),
                    "measureSmoothDist": np.linspace(0, 0.5, 3),
                    "bp": allConsideredBPs,
                }

            if not TEST_DATA:
                print("Running gravity measures (3/4)")
                runDMFOnLM(makeGravityMeasure, allParams, pp, minParamsSequential=2)

            if TEST_DATA:
                sessionsWithoutOutlier = sessions
            else:
                outlierIdx = 6
                sessionsWithoutOutlier = sessions[:outlierIdx] + sessions[outlierIdx + 1:]

            def makeAvgDwellMeasure(radius, smoothDist, bp: BP):
                return LocationMeasure(f"avgdwell_nooutlier {radius:.2f} {bp.filenameString()} {smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.avgDwellTimeAtPosition(
                                               bp, pos, radius=radius),
                                       ),
                                       sessionsWithoutOutlier,
                                       smoothDist=smoothDist,
                                       parallelize=False)

            if TEST_DATA:
                allParams = {
                    "radius": [0.25, 0.5],
                    "smoothDist": [0, 0.5],
                    "bp": allConsideredBPs
                }
            else:
                allParams = {
                    "radius": [0.25, 0.5, 1, 1.5],
                    "smoothDist": [0, 0.5, 1.0],
                    "bp": allConsideredBPs
                }

            print("Running avg dwell measures (4/4)")
            runDMFOnLM(makeAvgDwellMeasure, allParams, pp, minParamsSequential=1)

        if RUN_PREV_SESSION_PERSEVERATION:
            def nextSessionCtrlLocation(sesh: BTSession, seshIndex: int, allSessions: List[BTSession]) -> \
                    Iterable[Tuple[List[Tuple[int, float, float]], str, Tuple[str, str, str]]]:
                if seshIndex == len(allSessions) - 1:
                    return [([], "next session", ("This sesh", "Next sesh", "Session"))]
                hwx, hwy = getWellPosCoordinates(sesh.homeWell)
                return [([(seshIndex + 1, hwx, hwy)], "next session", ("This sesh", "Next sesh", "Session"))]

            LocationMeasure("Next sesh curvature", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.avgCurvatureAtPosition(BP(probe=True), pos, radius=0.75)),
                sessions,
                sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh frac visited", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.fracExcursionsVisited(pos)),
                sessions,
                sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh dot prod offwall", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.getDotProductScore(BP(probe=True, inclusionFlags=["offWall", "moving"]),
                                                    pos, distanceWeight=0)),
                            sessions, smoothDist=1.0,
                            sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh dot prod first min norm", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.getDotProductScore(BP(probe=True, timeInterval=(0, 60), inclusionFlags="moving"),
                                                    pos, distanceWeight=-1)),
                            sessions, smoothDist=0,
                            sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh task gravity", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.getGravity(BP(probe=False, inclusionFlags="awayTrial"), pos,
                                            passRadius=1.5, visitRadius=0.3, passDenoiseFactor=1.2)),
                            sessions, smoothDist=0.5,
                            sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh probe gravity", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.getGravity(BP(probe=True, timeInterval=BTSession.fillTimeInterval), pos,
                                            passRadius=1.5, visitRadius=0.3, passDenoiseFactor=1.2)),
                            sessions, smoothDist=0,
                            sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh avg dwell time", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.avgDwellTimeAtPosition(BP(probe=True, timeInterval=(0, 60)), pos, radius=0.5)),
                sessions, smoothDist=1,
                sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh num visits", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.numVisitsToPosition(BP(probe=True, timeInterval=BTSession.fillTimeInterval), pos, radius=0.25)),
                sessions, smoothDist=0.5,
                sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)
            LocationMeasure("Next sesh optimality", lambda sesh: sesh.getValueMap(
                lambda pos: sesh.pathOptimalityToPosition(BP(probe=True, timeInterval=(0, 60)), pos, radius=1.5)),
                sessions, smoothDist=0.5,
                sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)

        if REDO_NUM_VISITS:
            # def nextSessionCtrlLocation(sesh: BTSession, seshIndex: int, allSessions: List[BTSession]) -> \
            #         Iterable[Tuple[List[Tuple[int, float, float]], str, Tuple[str, str, str]]]:
            #     if seshIndex == len(allSessions) - 1:
            #         return [([], "next session", ("This sesh", "Next sesh", "Session"))]
            #     hwx, hwy = getWellPosCoordinates(sesh.homeWell)
            #     return [([(seshIndex + 1, hwx, hwy)], "next session", ("This sesh", "Next sesh", "Session"))]
            # LocationMeasure("Next sesh num visits", lambda sesh: sesh.getValueMap(
            #     lambda pos: sesh.numVisitsToPosition(BP(probe=True, timeInterval=BTSession.fillTimeInterval), pos, radius=0.25)),
            #     sessions, smoothDist=0.5,
            #     sessionCtrlLocations=nextSessionCtrlLocation).makeFigures(pp)

            def makeSimpleMeasure(radius, smoothDist, bp: BP):
                return LocationMeasure(f"simple num visits {radius:.2f} {bp.filenameString()} {smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.numVisitsToPosition(
                                               bp, pos, radius=radius),
                                       ),
                                       sessions,
                                       smoothDist=smoothDist,
                                       parallelize=False)

            allParams = {
                "radius": [0.25, 0.5],
                "smoothDist": [0, 0.5],
                "bp": [BP(probe=True), BP(probe=False)]
            }

            # allParams = {
            #     "radius": [0.25, 0.5, 0.75, 1, 1.5],
            #     "smoothDist": [0, 0.5, 1.0],
            #     "bp": [BP(probe=True), BP(probe=False),
            #            BP(probe=True, timeInterval=(0, 60)),
            #            BP(probe=True, timeInterval=BTSession.fillTimeInterval),]
            # }

            runDMFOnLM(makeSimpleMeasure, allParams, pp, minParamsSequential=1)

        if RUN_OCCUPANCY_CORRELATION:
            def occupancyCorrelation(bp: BP, sesh: BTSession) -> float:
                bpProbe = replace(bp, probe=True)
                ocprobe, _, _ = sesh.occupancyMap(bpProbe)
                bpTask = replace(bp, probe=False)
                octask, _, _ = sesh.occupancyMap(bpTask)
                return np.corrcoef(ocprobe.flatten(), octask.flatten())[0, 1]

            def occupancyCorrelationImage(bp: BP, sesh: BTSession) -> np.ndarray:
                bpProbe = replace(bp, probe=True)
                ocprobe, _, _ = sesh.occupancyMap(bpProbe)
                ocprobe = ocprobe / np.max(ocprobe)
                bpTask = replace(bp, probe=False)
                octask, _, _ = sesh.occupancyMap(bpTask)
                octask = octask / np.max(octask)
                return octask * ocprobe

            SessionMeasure("Occupancy correlation",
                           partial(occupancyCorrelation, BP(inclusionFlags="moving")),
                           sessions).makeFigures(pp, everySessionBackground=partial(occupancyCorrelationImage,
                                                                                    BP(inclusionFlags="moving")))

        if RUN_TASK_GRAVITY:
            LocationMeasure("Task gravity home trials",
                            lambda sesh: sesh.getValueMap(
                                lambda pos: sesh.getGravity(
                                    BP(inclusionFlags="homeTrial", probe=False), pos)
                            ), sessions).makeFigures(pp, excludeFromCombo=True,
                                                     everySessionBehaviorPeriod=BP(inclusionFlags="homeTrial", probe=False))
            LocationMeasure("Task gravity away trials",
                            lambda sesh: sesh.getValueMap(
                                lambda pos: sesh.getGravity(
                                    BP(inclusionFlags="awayTrial", probe=False), pos)
                            ), sessions).makeFigures(pp, excludeFromCombo=True,
                                                     everySessionBehaviorPeriod=BP(inclusionFlags="awayTrial", probe=False))
            LocationMeasure("Task gravity all",
                            lambda sesh: sesh.getValueMap(
                                lambda pos: sesh.getGravity(BP(probe=False), pos)
                            ), sessions).makeFigures(pp, excludeFromCombo=True,
                                                     everySessionBehaviorPeriod=BP(probe=False))

        if RUN_SIMPLE_MEASURES:
            def makeSimpleMeasure(func, radius, smoothDist, bp: BP):
                # print(f"test {func.__name__} {radius:.2f} {bp.filenameString()} {smoothDist:.2f}")
                return LocationMeasure(f"simple {func.__name__} {radius:.2f} {bp.filenameString()} {smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: func(sesh, bp, pos, radius=radius),
                                       ),
                                       sessions,
                                       smoothDist=smoothDist,
                                       parallelize=False)
            # allParams = {
            #     "func": [BTSession.totalTimeAtPosition,
            #              BTSession.pathOptimalityToPosition],
            #     "radius": [0.25, 0.5],
            #     "smoothDist": [0, 0.5],
            #     "bp": [BP(probe=True, timeInterval=(0, 60)),
            #            BP(probe=True, timeInterval=BTSession.fillTimeInterval),]
            # }

            allParams = {
                # "func": [BTSession.totalTimeAtPosition,
                #          BTSession.avgDwellTimeAtPosition,
                #          BTSession.numVisitsToPosition,
                #          BTSession.latencyToPosition,
                #          BTSession.pathOptimalityToPosition],
                "func": [BTSession.avgDwellTimeAtPosition,
                         BTSession.numVisitsToPosition],
                "radius": [0.25, 0.5, 0.75, 1, 1.5],
                "smoothDist": [0, 0.5, 1.0],
                "bp": [BP(probe=True), BP(probe=False),
                       BP(probe=True, timeInterval=(0, 60)),
                       BP(probe=True, timeInterval=BTSession.fillTimeInterval),]
            }

            runDMFOnLM(makeSimpleMeasure, allParams, pp, minParamsSequential=2)

        if RUN_CURVATURE:
            def makeCurvatureMeasure(radius, smoothDist, bp: BP):
                return LocationMeasure(f"curvature {radius:.2f} {bp.filenameString()} {smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.avgCurvatureAtPosition(
                                               bp, pos, radius=radius),
                                       ),
                                       sessions,
                                       smoothDist=smoothDist,
                                       parallelize=False)

            allParams = {
                "radius": [0.1, 0.25, 0.5, 0.75, 1, 1.5],
                "smoothDist": [0, 0.5, 1.0],
                "bp": [BP(probe=True), BP(probe=False),
                       BP(probe=True, timeInterval=(0, 60)),
                       BP(probe=True, timeInterval=BTSession.fillTimeInterval),]
            }

            runDMFOnLM(makeCurvatureMeasure, allParams, pp, minParamsSequential=1)

        if FIND_AVG_DWELL_OUTLIER:
            lm = LocationMeasure("avgdwell outlier finder",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.avgDwellTimeAtPosition(
                                         BP(probe=True), pos, radius=0.5),
                                 ),
                                 sessions,
                                 smoothDist=0.5)
            print("\n".join([str(v) for v in enumerate(lm.sessionValsBySession)]))

        if RUN_AVG_DWELL_WITHOUT_OUTLIER:
            outlierIdx = 6
            sessionsWithoutOutlier = sessions[:outlierIdx] + sessions[outlierIdx + 1:]

            def makeAvgDwellMeasure(radius, smoothDist, bp: BP):
                return LocationMeasure(f"avgdwell_nooutlier {radius:.2f} {bp.filenameString()} {smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.avgDwellTimeAtPosition(
                                               bp, pos, radius=radius),
                                       ),
                                       sessionsWithoutOutlier,
                                       smoothDist=smoothDist,
                                       parallelize=False)

            allParams = {
                "radius": [0.25, 0.5, 0.75, 1, 1.5],
                "smoothDist": [0, 0.5, 1.0],
                "bp": [BP(probe=True), BP(probe=False),
                       BP(probe=True, timeInterval=(0, 60)),
                       BP(probe=True, timeInterval=BTSession.fillTimeInterval),]
            }

            runDMFOnLM(makeAvgDwellMeasure, allParams, pp, minParamsSequential=1)

        if RUN_DOT_PROD_FACTORY:
            def makeDotProdMeasure(bp: BP, distanceWeight, normalize, smoothDist):
                return LocationMeasure(f"dotprod {bp.filenameString()} "
                                       f"distWeight={distanceWeight:.2f} "
                                       f"normalize={normalize} "
                                       f"smoothDist={smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.getDotProductScore(bp, pos,
                                                                               distanceWeight=distanceWeight,
                                                                               normalize=normalize)),
                                       sessions,
                                       smoothDist=smoothDist, parallelize=False)

            allParams = {
                "distanceWeight": np.linspace(-1, 1, 5),
                "normalize": [True, False],
                "smoothDist": [0, 0.5, 1.0],
                "bp": [BP(probe=False, erode=3, inclusionFlags="homeTrial"),
                       BP(probe=False, erode=3, inclusionFlags="awayTrial"),
                       BP(probe=False),
                       BP(probe=True),
                       BP(probe=True, timeInterval=(0, 60)),
                       BP(probe=True, timeInterval=(0, 120)),
                       BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                       BP(probe=True, inclusionFlags="offWall"),
                       BP(probe=True, inclusionFlags="moving"),
                       BP(probe=True, timeInterval=(0, 60), inclusionFlags="moving"),
                       BP(probe=True, timeInterval=(0, 120), inclusionFlags="moving"),
                       BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags="moving"),
                       BP(probe=True, inclusionFlags=["offWall", "moving"])]
            }

            runDMFOnLM(makeDotProdMeasure, allParams, pp, minParamsSequential=1)

        if RUN_GRAVITY_FACTORY:
            def makeGravityMeasure(bp: BP, passRadius, visitRadiusFactor, passDenoiseFactor, measureSmoothDist):
                return LocationMeasure(f"gravity {bp.filenameString()} "
                                       f"passRadius={passRadius:.2f} "
                                       f"visitRadiusFactor={visitRadiusFactor:.2f} "
                                       f"denoiseFactor={passDenoiseFactor:.2f} "
                                       f"measureSmoothDist={measureSmoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.getGravity(bp, pos,
                                                                       passRadius=passRadius,
                                                                       visitRadius=passRadius * visitRadiusFactor,
                                                                       passDenoiseFactor=passDenoiseFactor)),
                                       sessions,
                                       smoothDist=measureSmoothDist, parallelize=False)
            allParams = {
                "passRadius": np.linspace(0.5, 1.5, 3),
                "visitRadiusFactor": np.linspace(0.2, 0.4, 2),
                "passDenoiseFactor": np.linspace(1.2, 2, 3),
                "measureSmoothDist": np.linspace(0, 0.5, 3),
                "bp": [BP(probe=True), BP(probe=False), BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                       BP(probe=True, timeInterval=(0, 90)), BP(
                           probe=False, timeInterval=(60 * 5, None)),
                       BP(probe=False, inclusionFlags="homeTrial"), BP(probe=False, inclusionFlags="awayTrial")]
            }

            runDMFOnLM(makeGravityMeasure, allParams, pp, minParamsSequential=2)

        if RUN_NEW_GRAVITY:
            def gravityFigsForBP(bp: BP):
                passRadii = np.linspace(0.5, 1.5, 3)
                visitRadiusFactors = np.linspace(0.2, 0.4, 2)
                denoiseFactors = np.linspace(1.2, 2, 3)
                measureSmoothDists = np.linspace(0, 0.5, 3)
                for passRadius in passRadii:
                    for visitRadiusFactor in visitRadiusFactors:
                        visitRadius = passRadius * visitRadiusFactor
                        for denoiseFactor in denoiseFactors:
                            for measureSmoothDist in measureSmoothDists:
                                lm = LocationMeasure(f"gravity {bp.filenameString()} "
                                                     f"passRadius={passRadius:.2f} "
                                                     f"visitRadiusFactor={visitRadiusFactor:.2f} "
                                                     f"denoiseFactor={denoiseFactor:.2f}"
                                                     f"measureSmoothDist={measureSmoothDist:.2f}",
                                                     lambda sesh: sesh.getValueMap(
                                                         lambda pos: sesh.getGravity(bp, pos,
                                                                                     passRadius=passRadius,
                                                                                     visitRadius=visitRadius,
                                                                                     passDenoiseFactor=denoiseFactor)
                                                     ), sessions, smoothDist=measureSmoothDist)
                                lm.makeFigures(pp, everySessionBehaviorPeriod=bp,
                                               excludeFromCombo=True)

                # passRadius=1.25, visitRadius=0.25, passDenoiseFactor=1.1,
                lm = LocationMeasure(f"gravity {bp.filenameString()}", lambda sesh: sesh.getValueMap(
                    partial(BTSession.getGravity, sesh, bp)
                ), sessions)
                lm.makeFigures(pp, everySessionBehaviorPeriod=bp, excludeFromCombo=True)

            gravityFigsForBP(BP(probe=True))
            gravityFigsForBP(BP(probe=False))
            gravityFigsForBP(BP(probe=True, timeInterval=BTSession.fillTimeInterval))
            gravityFigsForBP(BP(probe=True, timeInterval=(0, 90)))
            gravityFigsForBP(BP(probe=False, timeInterval=(60 * 5, None)))

        if RUN_FRAC_EXCURSIONS_VISITED:
            def fracExcursionsVisited(sesh: BTSession, pos: Tuple[float, float], radius=0.25) -> float:
                if len(sesh.probeExcursionStart_posIdx) == 0:
                    return np.nan
                ret = 0
                for i0, i1 in zip(sesh.probeExcursionStart_posIdx, sesh.probeExcursionEnd_posIdx):
                    x = sesh.probePosXs[i0:i1]
                    y = sesh.probePosYs[i0:i1]
                    d = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if np.any(d < radius):
                        ret += 1
                return ret / len(sesh.probeExcursionStart_posIdx)

            LocationMeasure("frac Excursions visited", lambda sesh: sesh.getValueMap(partial(fracExcursionsVisited, sesh)),
                            sessions).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True, inclusionFlags="offWall"))

        if PLOT_OCCUPANCY:
            bp = BP(probe=False, inclusionFlags="moving", moveThreshold=20)
            LocationMeasure("moving task occupancy", lambda sesh: sesh.occupancyMap(bp)[0],
                            sessions).makeFigures(pp, everySessionBehaviorPeriod=bp)

        def dotProdFigsForBP(bp: BP):
            distFactors = [-1, -0.25, -0.1, 0, 0.1, 0.25, 1]
            # distFactors = [0]
            for distFactor in distFactors:
                sm = LocationMeasure("dot prod {} d{}".format(bp.filenameString(), distFactor), lambda sesh:
                                     sesh.getValueMap(lambda pos: sesh.getDotProductScore(
                                         bp, pos, distanceWeight=distFactor, showPlot=True)),
                                     sessions)
                sm.makeFigures(pp, everySessionBehaviorPeriod=bp, excludeFromCombo=True)

        if RUN_SPOTLIGHT_EXPLORATION_TASK:
            dotProdFigsForBP(BP(probe=False, inclusionFlags="homeTrial"))
            dotProdFigsForBP(BP(probe=False, inclusionFlags="awayTrial"))
            dotProdFigsForBP(BP(probe=False, inclusionFlags=["moving", "offWall", "notreward"]))

        if RUN_SPOTLIGHT_EXPLORATION_PROBE:
            dotProdFigsForBP(BP(probe=True, inclusionFlags="moving"))
            dotProdFigsForBP(BP(probe=True))
            dotProdFigsForBP(BP(probe=True, inclusionFlags="moving", moveThreshold=20))
            dotProdFigsForBP(BP(probe=True, inclusionFlags="moving", moveThreshold=40))
            dotProdFigsForBP(BP(probe=True, inclusionFlags="moving",
                             timeInterval=BTSession.fillTimeInterval))
            dotProdFigsForBP(BP(probe=True, timeInterval=BTSession.fillTimeInterval))
            dotProdFigsForBP(BP(probe=True, inclusionFlags="moving", timeInterval=(0, 60)))
            dotProdFigsForBP(BP(probe=True, timeInterval=(0, 60)))
            dotProdFigsForBP(BP(probe=True, inclusionFlags="moving", timeInterval=(0, 120)))
            dotProdFigsForBP(BP(probe=True, timeInterval=(0, 120)))

        if RUN_PATH_OCCUPANCY:
            def pathOccupancy(sesh: BTSession) -> float:
                resolution = 36
                occMap, occBinsX, occBinsY = sesh.occupancyMap(
                    False, resolution=resolution, movingOnly=True, moveThresh=20)
                probePosBinsX = np.digitize(sesh.probePosXs, occBinsX)
                probePosBinsY = np.digitize(sesh.probePosYs, occBinsY)
                wallMargin = 6
                offWallIdxs = (probePosBinsX > wallMargin) & (probePosBinsX < resolution - wallMargin) & \
                    (probePosBinsY > wallMargin) & (probePosBinsY < resolution - wallMargin)
                return np.nanmean(occMap[probePosBinsX[offWallIdxs], probePosBinsY[offWallIdxs]])

            def bgImgFunc(session: BTSession) -> np.ndarray:
                wallMargin = 6
                occMap, occBinsX, occBinsY = session.occupancyMap(
                    False, movingOnly=True, moveThresh=20)
                occMap[0:wallMargin, :] = np.nan
                occMap[-wallMargin:, :] = np.nan
                occMap[:, 0:wallMargin] = np.nan
                occMap[:, -wallMargin:] = np.nan
                return occMap

            def pathOccupancyWithDirection(sesh: BTSession) -> float:
                resolution = 36
                moveThresh = 20
                angleResolution = 4

                xs = sesh.btPosXs
                ys = sesh.btPosYs
                dirs = np.arctan2(np.diff(ys), np.diff(xs))
                dirs = np.append(dirs, dirs[-1])
                # Rotate a half-bin to center the bins on the angles
                dirs += np.pi / angleResolution
                dirs[dirs > np.pi] -= 2 * np.pi

                mv = sesh.btVelCmPerS > moveThresh
                mv = np.append(mv, mv[-1])
                xs = xs[mv]
                ys = ys[mv]
                dirs = dirs[mv]

                xbins = np.linspace(-0.5, 6.5, resolution + 1)
                ybins = np.linspace(-0.5, 6.5, resolution + 1)
                abins = np.linspace(-np.pi, np.pi, angleResolution + 1)
                # print(xbins, ybins, abins)
                # print(xs.shape, ys.shape, dirs.shape)
                occMap = np.histogramdd((xs, ys, dirs), bins=(xbins, ybins, abins))
                # print(occMap)
                occMap = occMap[0]
                smooth = 1
                occMap = gaussian_filter(occMap, (smooth, smooth, 0.5), mode="wrap")

                probePosBinsX = np.digitize(sesh.probePosXs, xbins)
                probePosBinsY = np.digitize(sesh.probePosYs, ybins)
                wallMargin = 6
                offWallIdxs = (probePosBinsX > wallMargin) & (probePosBinsX < resolution - wallMargin) & \
                    (probePosBinsY > wallMargin) & (probePosBinsY < resolution - wallMargin)
                return np.nanmean(occMap[probePosBinsX[offWallIdxs], probePosBinsY[offWallIdxs]])

            raise NotImplementedError(
                "TODO: This is using old version where each session is just one number")
            # Should call these SessionMeasure again and make that a new thing that's simpler
            sm = LocationMeasure("path occupancy", pathOccupancy, sessionsWithProbe)
            sm.makeFigures(pp, everySessionTraceType="probe", everySessionBackground=bgImgFunc)
            sm = LocationMeasure("path occupancy moving", pathOccupancy, sessionsWithProbe)
            sm.makeFigures(pp, everySessionTraceType="probe", everySessionBackground=bgImgFunc)
            sm = LocationMeasure("path occupancy with direction",
                                 pathOccupancyWithDirection, sessionsWithProbe)
            sm.makeFigures(pp, everySessionTraceType="probe", everySessionBackground=bgImgFunc)

        if RUN_ENTRY_EXIT_ANGLE:
            # TODO modifications:
            # - don't normalize everything
            # - try tracing whole probe path (off wall), and then seeing how it lines up with behavior somehow
            # Maybe first make off-wall occupancy map for during the task, and then for all probe timepoints,
            #   see what the value of that map is. Slight smoothing but look by eye to see if you're capturing
            #   the major pathways
            def entryExitAngleHistogram(sesh: BTSession, well: int) -> np.ndarray:
                numAngles = 8
                hs = np.zeros((2, numAngles))

                wx, wy = getWellPosCoordinates(well)

                angles = []
                angleBins = np.linspace(-np.pi, np.pi, numAngles+1)
                angleBins -= (angleBins[1] - angleBins[0]) / 2
                ents, exts = sesh.entryExitTimes(False, well, returnIdxs=True)
                for pi in np.concatenate((ents, exts)):
                    if pi == len(sesh.btPosXs):
                        pi -= 1
                    x = sesh.btPosXs[pi] - wx
                    y = sesh.btPosYs[pi] - wy
                    angle = np.arctan2(y, x)
                    if angle > angleBins[-1]:
                        angle -= 2 * np.pi
                    angles.append(angle)
                hs[0, :] = np.histogram(angles, bins=angleBins)[0]

                angles = []
                ents, exts = sesh.entryExitTimes(True, well, returnIdxs=True)
                for pi in np.concatenate((ents, exts)):
                    if pi == len(sesh.probePosXs):
                        pi -= 1
                    x = sesh.probePosXs[pi] - wx
                    y = sesh.probePosYs[pi] - wy
                    angle = np.arctan2(y, x)
                    if angle > angleBins[-1]:
                        angle -= 2 * np.pi
                    angles.append(angle)
                hs[1, :] = np.histogram(angles, bins=angleBins)[0]

                # hs /= np.sum(hs, axis=1, keepdims=True)

                # if well == 10:
                #     print(angleBins)
                #     print(angles)
                #     print(hs)

                hs = gaussian_filter1d(hs, 0.6, axis=1, mode="wrap")

                return hs

            def entryExitAngleMeasure(sesh: BTSession, well: int) -> float:
                hs = entryExitAngleHistogram(sesh, well)
                assert hs.shape[0] == 2
                return np.corrcoef(hs)[0, 1] * np.sum(hs)

            wm = WellMeasure("entryExitAngle", measureFunc=entryExitAngleMeasure,
                             sessionList=sessionsWithProbe, displayFunc=entryExitAngleHistogram)
            # for s in wm.allDisplayValsBySession:
            #     print(s[10])
            wm.makeFigures(pp, radialTraceType=["task", "probe"])

        if RUN_SMOOTHING_TEST:
            smoothVals = np.power(2.0, np.arange(-1, 5))
            for sesh in sessions:
                pp.pushOutputSubDir(sesh.name)

                with pp.newFig("probeTraceVariations", subPlots=(1, 1+len(smoothVals))) as pc:
                    setupBehaviorTracePlot(pc.ax, sesh, showWells="")
                    pc.ax.plot(sesh.probePosXs, sesh.probePosYs)
                    pc.ax.set_title("raw")

                    for si, smooth in enumerate(smoothVals):
                        ax = pc.axs[si+1]
                        setupBehaviorTracePlot(ax, sesh, showWells="")
                        x, y = sesh.probePosXs, sesh.probePosYs
                        x = gaussian_filter1d(x, smooth)
                        y = gaussian_filter1d(y, smooth)
                        ax.plot(x, y)
                        ax.set_title(f"smooth {smooth}")

                pp.popOutputSubDir()

        if RUN_SPOTLIGHT:
            WellMeasure("probe spotlight score before fill", lambda s, h: s.getDotProductScoreAtWell(
                True, h, timeInterval=[0, s.fillTimeCutoff()], binarySpotlight=True,
                boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp,
                                               everySessionTraceTimeInterval=lambda s: [
                                                   0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe")

        if RUN_DOT_PROD:
            WellMeasure("probe dotprod score before fill", lambda s, h: s.getDotProductScoreAtWell(
                True, h, timeInterval=[0, s.fillTimeCutoff()], boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceTimeInterval=lambda s: [0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe_bouts")
            WellMeasure("task dotprod score", lambda s, h: s.getDotProductScoreAtWell(
                False, h, boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceType="task_bouts")

            WellMeasure("probe dotprod score first entry", lambda s, h: s.getDotProductScoreAtWell(
                True, h, timeInterval=[0, s.getLatencyToWell(True, s.homeWell, returnUnits="secs")], boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceTimeInterval=lambda s: [0, s.getLatencyToWell(True, s.homeWell, returnUnits="secs")],
                                               everySessionTraceType="probe_bouts", plotFlags="everysession")

        if RUN_MANY_DOTPROD:
            SECTION_LEN = 5
            WellMeasure("short dotprod 1", lambda s, h: s.getDotProductScoreAtWell(
                True, h, timeInterval=[0, SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, plotFlags="everysession",
                                              everySessionTraceTimeInterval=lambda _: [
                                                  0, SECTION_LEN], everySessionTraceType="probe")
            WellMeasure("short dotprod 2", lambda s, h: s.getDotProductScoreAtWell(
                True, h, timeInterval=[SECTION_LEN, 2*SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, plotFlags="everysession",
                                              everySessionTraceTimeInterval=lambda _: [
                                                  SECTION_LEN, 2*SECTION_LEN], everySessionTraceType="probe")
            WellMeasure("short dotprod 1 and 2", lambda s, h: s.getDotProductScoreAtWell(
                True, h, timeInterval=[0, 2*SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, plotFlags="everysession",
                                              everySessionTraceTimeInterval=lambda _: [
                                                  0, 2*SECTION_LEN], everySessionTraceType="probe")

        if PLOT_TASK_TRIAL_PERFORMANCE:
            TrialMeasure("trial duration", lambda sesh, t0, t1, ttype:
                         sesh.btPos_secs[t1] - sesh.btPos_secs[t0],
                         sessions).makeFigures(pp)
            TrialMeasure("trial pathOptimality", lambda sesh, t0, t1, ttype:
                         sesh.pathOptimality(False, (t0, t1, "posIdx")),
                         sessions).makeFigures(pp)

        if PLOT_DETAILED_TASK_TRACES:
            for si, sesh in enumerate(sessions):
                pp.pushOutputSubDir(sesh.name)
                with pp.newFig("task_fullTrace") as pc:
                    setupBehaviorTracePlot(pc.ax, sesh)
                    pc.ax.plot(sesh.btPosXs, sesh.btPosYs)

                traceLen = 60
                traceStep = 30
                tStarts = np.arange(0, sesh.btPos_secs[-1] - traceLen + traceStep, traceStep)
                tEnds = tStarts + traceLen
                tStarts_posIdx = np.searchsorted(sesh.btPos_secs, tStarts)
                tEnds_posIdx = np.searchsorted(sesh.btPos_secs, tEnds)
                ncols = math.ceil(np.sqrt(len(tStarts_posIdx)))
                with pp.newFig("task_fullTraceBroken", subPlots=(ncols, ncols)) as pc:
                    for ti in range(len(tStarts_posIdx)):
                        t0 = tStarts_posIdx[ti]
                        t1 = tEnds_posIdx[ti]
                        ax = pc.axs.flat[ti]
                        ax.plot(sesh.btPosXs[t0:t1], sesh.btPosYs[t0:t1])
                        setupBehaviorTracePlot(ax, sesh)

                ncols = 13
                wellFinds_posIdx = sesh.getAllRewardPosIdxs()
                with pp.newFig("task_wellFinds", subPlots=(2, ncols)) as pc:
                    for ti in range(wellFinds_posIdx.shape[0]):
                        ai0 = ti % 2
                        ai1 = ti // 2
                        ax = pc.axs[ai0, ai1]
                        assert isinstance(ax, Axes)
                        setupBehaviorTracePlot(ax, sesh)
                        t0 = wellFinds_posIdx[ti, 0]
                        t1 = wellFinds_posIdx[ti, 1]
                        ax.plot(sesh.btPosXs[t0:t1], sesh.btPosYs[t0:t1])

                ncols = 13
                trials_posIdx = sesh.getAllTrialPosIdxs()
                with pp.newFig("task_trials", subPlots=(2, ncols)) as pc:
                    for ti in range(trials_posIdx.shape[0]):
                        ai0 = ti % 2
                        ai1 = ti // 2
                        ax = pc.axs[ai0, ai1]
                        assert isinstance(ax, Axes)
                        setupBehaviorTracePlot(ax, sesh)
                        t0 = trials_posIdx[ti, 0]
                        t1 = trials_posIdx[ti, 1]
                        ax.plot(sesh.btPosXs[t0:t1], sesh.btPosYs[t0:t1])

                ncols = math.ceil(np.sqrt(len(sesh.btExcursionStart_posIdx)))
                with pp.newFig("task_excursions", subPlots=(ncols, ncols)) as pc:
                    for ei in range(len(sesh.btExcursionStart_posIdx)):
                        ax = pc.axs.flat[ei]
                        assert isinstance(ax, Axes)
                        setupBehaviorTracePlot(ax, sesh)
                        t0 = sesh.btExcursionStart_posIdx[ei]
                        t1 = sesh.btExcursionEnd_posIdx[ei]
                        ax.plot(sesh.btPosXs[t0:t1], sesh.btPosYs[t0:t1])

                ncols = math.ceil(np.sqrt(len(sesh.btExploreBoutStart_posIdx)))
                with pp.newFig("task_bouts", subPlots=(ncols, ncols)) as pc:
                    for ei in range(len(sesh.btExploreBoutStart_posIdx)):
                        ax = pc.axs.flat[ei]
                        assert isinstance(ax, Axes)
                        setupBehaviorTracePlot(ax, sesh)
                        t0 = sesh.btExploreBoutStart_posIdx[ei]
                        t1 = sesh.btExploreBoutEnd_posIdx[ei]
                        ax.plot(sesh.btPosXs[t0:t1], sesh.btPosYs[t0:t1])
                pp.popOutputSubDir()

        if PLOT_DETAILED_PROBE_TRACES:
            for si, sesh in enumerate(sessionsWithProbe):
                pp.pushOutputSubDir(sesh.name)
                with pp.newFig("probe_fullTrace") as pc:
                    setupBehaviorTracePlot(pc.ax, sesh)
                    pc.ax.plot(sesh.probePosXs, sesh.probePosYs)

                with pp.newFig("probe_beforeFill") as pc:
                    setupBehaviorTracePlot(pc.ax, sesh)
                    ti = sesh.timeIntervalToPosIdx(
                        sesh.probePos_ts, (0, sesh.fillTimeCutoff(), "secs"))
                    pc.ax.plot(sesh.probePosXs[ti[0]:ti[1]], sesh.probePosYs[ti[0]:ti[1]])

                traceLen = 60
                traceStep = 30
                tStarts = np.arange(0, sesh.probePos_secs[-1] - traceLen + traceStep, traceStep)
                tEnds = tStarts + traceLen
                tStarts_posIdx = np.searchsorted(sesh.probePos_secs, tStarts)
                tEnds_posIdx = np.searchsorted(sesh.probePos_secs, tEnds)
                ncols = math.ceil(np.sqrt(len(tStarts_posIdx)))
                with pp.newFig("probe_fullTraceBroken", subPlots=(ncols, ncols)) as pc:
                    for ti in range(len(tStarts_posIdx)):
                        t0 = tStarts_posIdx[ti]
                        t1 = tEnds_posIdx[ti]
                        ax = pc.axs.flat[ti]
                        ax.plot(sesh.probePosXs[t0:t1], sesh.probePosYs[t0:t1])
                        setupBehaviorTracePlot(ax, sesh)

                if len(sesh.probeExcursionStart_posIdx) > 0:
                    ncols = math.ceil(np.sqrt(len(sesh.probeExcursionStart_posIdx)))
                    with pp.newFig("probe_excursions", subPlots=(ncols, ncols)) as pc:
                        for ei in range(len(sesh.probeExcursionStart_posIdx)):
                            ax = pc.axs.flat[ei]
                            assert isinstance(ax, Axes)
                            setupBehaviorTracePlot(ax, sesh)
                            t0 = sesh.probeExcursionStart_posIdx[ei]
                            t1 = sesh.probeExcursionEnd_posIdx[ei]
                            ax.plot(sesh.probePosXs[t0:t1], sesh.probePosYs[t0:t1])

                if len(sesh.probeExploreBoutStart_posIdx) > 0:
                    ncols = math.ceil(np.sqrt(len(sesh.probeExploreBoutStart_posIdx)))
                    with pp.newFig("probe_bouts", subPlots=(ncols, ncols)) as pc:
                        for ei in range(len(sesh.probeExploreBoutStart_posIdx)):
                            ax = pc.axs.flat[ei]
                            assert isinstance(ax, Axes)
                            setupBehaviorTracePlot(ax, sesh)
                            t0 = sesh.probeExploreBoutStart_posIdx[ei]
                            t1 = sesh.probeExploreBoutEnd_posIdx[ei]
                            ax.plot(sesh.probePosXs[t0:t1], sesh.probePosYs[t0:t1])

                pp.popOutputSubDir()

        if RUN_OPTIMALITY:
            # def optFromWall(sesh: BTSession, start_posIdx: int, end_posIdx: int, trialType: str) -> float:
            #     nz = np.nonzero(sesh.btExcursionStart_posIdx <= end_posIdx)
            #     excursionIdx = nz[0][-1]
            #     assert sesh.btExcursionEnd_posIdx[excursionIdx] >= end_posIdx
            #     if start_posIdx > sesh.btExcursionStart_posIdx[excursionIdx]:
            #         t1 = start_posIdx
            #     else:
            #         t1 = sesh.btExcursionStart_posIdx[excursionIdx]
            #     return sesh.pathOptimality(False, timeInterval=[t1, end_posIdx, "posIdx"])

            # TrialMeasure("optimality from wall to reward", optFromWall,
            #              sessions, lambda _, _1, _2, _3, _4, wn: offWall(wn)).makeFigures(pp)
            # The above measure doesn't work out well because of super short intervals between
            #   leaving the wall or well and finding the immediately adjacent well
            pass

        if RUN_LFP_LATENCY:
            latencyToDetectionArtifactFrom0_all = np.array([])
            latencyToDetectionArtifactFrom25_all = np.array([])
            latencyToDetectionLoggedFrom0_all = np.array([])
            latencyToDetectionLoggedFrom25_all = np.array([])

            for si, sesh in enumerate(sessionsWithLog):
                if not sesh.probePerformed:
                    continue
                pp.pushOutputSubDir(sesh.name)

                lfpFName = sesh.btLfpFnames[-1]
                print("LFP data from file {}".format(lfpFName))
                lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
                lfpV = lfpData[1]['voltage']
                lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

                lfpData = MountainViewIO.loadLFP(data_file=sesh.btLfpBaselineFname)
                baselfpV = lfpData[1]['voltage']

                combinedDeflections = np.concatenate(
                    (sesh.lfpBumps_lfpIdx, sesh.loggedDetections_lfpIdx))

                ripPower_standard, ripZPower_standard, _, _ = getRipplePower(
                    lfpV, lfpDeflections=combinedDeflections)
                baseRipPower_standard, _, _, _ = getRipplePower(
                    baselfpV, lfpDeflections=combinedDeflections)
                rp = ripPower_standard - baseRipPower_standard
                m = np.nanmean(rp)
                s = np.nanstd(rp)
                ripZPower_standardOffset = (rp - m) / s

                ripPower_causal, ripZPower_causal, _, _ = getRipplePower(lfpV, lfpDeflections=combinedDeflections,
                                                                         method="causal")
                baseRipPower_causal, _, _, _ = getRipplePower(baselfpV, lfpDeflections=combinedDeflections,
                                                              method="causal")
                rp = ripPower_causal - baseRipPower_causal
                m = np.nanmean(rp)
                s = np.nanstd(rp)
                ripZPower_causalOffset = (rp - m) / s

                _, ripZPower_activelink, _, _ = getRipplePower(
                    lfpV, meanPower=sesh.rpowmLog, stdPower=sesh.rpowsLog,
                    method="activelink", baselineLfpData=baselfpV)
                # print(f"{ np.count_nonzero(np.isnan(ripZPower_activelink)) / len(ripZPower_activelink) = }")
                # print(
                #     f"{ np.count_nonzero(np.abs(ripZPower_activelink) > 1e300) / len(ripZPower_activelink) = }")
                # print(f"{ np.count_nonzero(np.isinf(ripZPower_activelink)) / len(ripZPower_activelink) = }")
                # print(f"{ np.nanmax(np.abs(ripZPower_activelink)) = }")

                assert len(ripPower_causal) == len(ripZPower_causal) == len(ripZPower_activelink)
                # print(f"{ np.count_nonzero(np.diff(ripZPower_activelink)) = }")
                # print(f"{ len(ripZPower_activelink) = }")
                assert np.count_nonzero(np.diff(ripZPower_activelink)
                                        ) < len(ripZPower_activelink) / 2

                # plt.plot(lfpT, lfpV / 3000.0, label="LFP")
                # plt.plot(lfpT, baselfpV / 3000.0, label="base LFP")
                # plt.plot(lfpT, ripZPower_activelink, label="activelink")
                # plt.plot(lfpT, ripZPower_causal, label="causal")
                # plt.legend()
                # plt.show()

                print(f"{ len(sesh.btLFPBumps_lfpIdx) = }")
                print(f"{ len(sesh.btLoggedDetections_lfpIdx) = }")

                MARGIN_SECS = 0.25
                MARGIN_PTS = int(MARGIN_SECS * 1500)
                allRipPowsArtifact_activelink = np.empty(
                    (len(sesh.btLFPBumps_lfpIdx), 2*MARGIN_PTS))
                allRipPowsArtifact_standard = np.empty((len(sesh.btLFPBumps_lfpIdx), 2*MARGIN_PTS))
                allRipPowsArtifact_standardOffset = np.empty(
                    (len(sesh.btLFPBumps_lfpIdx), 2*MARGIN_PTS))
                allRipPowsArtifact_causal = np.empty((len(sesh.btLFPBumps_lfpIdx), 2*MARGIN_PTS))
                allRipPowsArtifact_causalOffset = np.empty(
                    (len(sesh.btLFPBumps_lfpIdx), 2*MARGIN_PTS))
                pp.pushOutputSubDir("LFP_Interruptions")
                pp.pushOutputSubDir("artifact")
                for ai, aidx in enumerate(sesh.btLFPBumps_lfpIdx):
                    i1 = aidx - MARGIN_PTS + sesh.btLfpStart_lfpIdx
                    i2 = aidx + MARGIN_PTS + sesh.btLfpStart_lfpIdx

                    allRipPowsArtifact_activelink[ai, :] = ripZPower_activelink[i1:i2]
                    allRipPowsArtifact_standard[ai, :] = ripZPower_standard[i1:i2]
                    allRipPowsArtifact_standardOffset[ai, :] = ripZPower_standardOffset[i1:i2]
                    allRipPowsArtifact_causal[ai, :] = ripZPower_causal[i1:i2]
                    allRipPowsArtifact_causalOffset[ai, :] = ripZPower_causalOffset[i1:i2]

                    if MAKE_INDIVIDUAL_INTERRUPTION_PLOTS and \
                            len(sesh.btLFPBumps_lfpIdx) // 2 < ai < len(sesh.btLFPBumps_lfpIdx) // 2 + 20:
                        with pp.newFig(f"interruption_{sesh.name}_{ai}", excludeFromCombo=True) as pc:
                            pc.ax.plot(lfpT[i1:i2], lfpV[i1:i2] / 1000, label="lfp")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_activelink[i1:i2],
                                       label="activelink power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_standard[i1:i2],
                                       label="standard power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_standardOffset[i1:i2],
                                       label="standard offset power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_causal[i1:i2],
                                       label="causal power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_causalOffset[i1:i2],
                                       label="causal offset power")
                            pc.ax.legend()
                            pc.ax.set_ylim(-3, 10)
                pp.popOutputSubDir()

                allRipPowsLogged_activelink = np.empty(
                    (len(sesh.btLoggedDetections_lfpIdx), 2*MARGIN_PTS))
                allRipPowsLogged_standard = np.empty(
                    (len(sesh.btLoggedDetections_lfpIdx), 2*MARGIN_PTS))
                allRipPowsLogged_standardOffset = np.empty(
                    (len(sesh.btLoggedDetections_lfpIdx), 2*MARGIN_PTS))
                allRipPowsLogged_causal = np.empty(
                    (len(sesh.btLoggedDetections_lfpIdx), 2*MARGIN_PTS))
                allRipPowsLogged_causalOffset = np.empty(
                    (len(sesh.btLoggedDetections_lfpIdx), 2*MARGIN_PTS))
                pp.pushOutputSubDir("logged")
                for di, didx in enumerate(sesh.btLoggedDetections_lfpIdx):
                    i1 = didx - MARGIN_PTS + sesh.btLfpStart_lfpIdx
                    i2 = didx + MARGIN_PTS + sesh.btLfpStart_lfpIdx

                    allRipPowsLogged_activelink[di, :] = ripZPower_activelink[i1:i2]
                    allRipPowsLogged_standard[di, :] = ripZPower_standard[i1:i2]
                    allRipPowsLogged_standardOffset[di, :] = ripZPower_standardOffset[i1:i2]
                    allRipPowsLogged_causal[di, :] = ripZPower_causal[i1:i2]
                    allRipPowsLogged_causalOffset[di, :] = ripZPower_causalOffset[i1:i2]

                    if MAKE_INDIVIDUAL_INTERRUPTION_PLOTS and \
                            len(sesh.btLoggedDetections_lfpIdx) // 2 < di < len(sesh.btLoggedDetections_lfpIdx) // 2 + 20:
                        with pp.newFig(f"interruption_{sesh.name}_{di}", excludeFromCombo=True) as pc:
                            pc.ax.plot(lfpT[i1:i2], lfpV[i1:i2] / 1000, label="lfp")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_activelink[i1:i2],
                                       label="activelink power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_standard[i1:i2],
                                       label="standard power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_standardOffset[i1:i2],
                                       label="standard offset power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_causal[i1:i2],
                                       label="causal power")
                            pc.ax.plot(lfpT[i1:i2], ripZPower_causalOffset[i1:i2],
                                       label="causal offset power")
                            pc.ax.legend()
                            pc.ax.set_ylim(-3, 10)
                pp.popOutputSubDir()
                pp.popOutputSubDir()

                # detTimeRev = allRipPowsArtifact[:, MARGIN_PTS:0:-1] > 2.5
                # latencyToDetectionArtifactFrom25 = np.argmax(
                #     detTimeRev, axis=1).astype(float) / 1500.0
                # with pp.newFig(f"lfp_detection_latency_to_artifact_z2-5_{sesh.name}") as pc:
                #     pc.ax.hist(latencyToDetectionArtifactFrom25)

                # detTimeRev = allRipPowsArtifact[:, MARGIN_PTS:0:-1] > 0
                # latencyToDetectionArtifactFrom0 = np.argmax(
                #     detTimeRev, axis=1).astype(float) / 1500.0
                # with pp.newFig(f"lfp_detection_latency_to_artifact_z0_{sesh.name}") as pc:
                #     pc.ax.hist(latencyToDetectionArtifactFrom0)

                # detTimeRev = allRipPowsLogged[:, MARGIN_PTS:0:-1] > 2.5
                # latencyToDetectionLoggedFrom25 = np.argmax(
                #     detTimeRev, axis=1).astype(float) / 1500.0
                # with pp.newFig(f"lfp_detection_latency_to_logged_z2-5_{sesh.name}") as pc:
                #     pc.ax.hist(latencyToDetectionLoggedFrom25)

                # detTimeRev = allRipPowsLogged[:, MARGIN_PTS:0:-1] > 0
                # latencyToDetectionLoggedFrom0 = np.argmax(detTimeRev, axis=1).astype(float) / 1500.0
                # with pp.newFig(f"lfp_detection_latency_to_logged_z0_{sesh.name}") as pc:
                #     pc.ax.hist(latencyToDetectionLoggedFrom0)

                # latencyToDetectionArtifactFrom0_all = np.append(
                #     latencyToDetectionArtifactFrom0_all, latencyToDetectionArtifactFrom0)
                # latencyToDetectionArtifactFrom25_all = np.append(
                #     latencyToDetectionArtifactFrom25_all, latencyToDetectionArtifactFrom25)
                # latencyToDetectionLoggedFrom0_all = np.append(
                #     latencyToDetectionLoggedFrom0_all, latencyToDetectionLoggedFrom0)
                # latencyToDetectionLoggedFrom25_all = np.append(
                #     latencyToDetectionLoggedFrom25_all, latencyToDetectionLoggedFrom25)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
                    warnings.filterwarnings("ignore", r"Mean of empty slice")

                    allRipPowsArtifact_activelink[np.isinf(
                        allRipPowsArtifact_activelink)] = np.nan
                    m = np.nanmean(allRipPowsArtifact_activelink, axis=0)
                    s = np.nanstd(allRipPowsArtifact_activelink, axis=0)
                    with pp.newFig(f"lfp_psth_artifact_activelink", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    with pp.newFig(f"lfp_psth_artifact_activelink_individual", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        plotIndividualAndAverage(pc.ax, allRipPowsArtifact_activelink, xvals,
                                                 individualAmt=50)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsArtifact_standard[np.isinf(allRipPowsArtifact_standard)] = np.nan
                    m = np.nanmean(allRipPowsArtifact_standard, axis=0)
                    s = np.nanstd(allRipPowsArtifact_standard, axis=0)
                    with pp.newFig(f"lfp_psth_artifact_standard", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsArtifact_standardOffset[np.isinf(
                        allRipPowsArtifact_standardOffset)] = np.nan
                    m = np.nanmean(allRipPowsArtifact_standardOffset, axis=0)
                    s = np.nanstd(allRipPowsArtifact_standardOffset, axis=0)
                    with pp.newFig(f"lfp_psth_artifact_standardOffset", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsArtifact_causal[np.isinf(allRipPowsArtifact_causal)] = np.nan
                    m = np.nanmean(allRipPowsArtifact_causal, axis=0)
                    s = np.nanstd(allRipPowsArtifact_causal, axis=0)
                    with pp.newFig(f"lfp_psth_artifact_causal", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsArtifact_causalOffset[np.isinf(
                        allRipPowsArtifact_causalOffset)] = np.nan
                    m = np.nanmean(allRipPowsArtifact_causalOffset, axis=0)
                    s = np.nanstd(allRipPowsArtifact_causalOffset, axis=0)
                    with pp.newFig(f"lfp_psth_artifact_causalOffset", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsLogged_activelink[np.isinf(allRipPowsLogged_activelink)] = np.nan
                    m = np.nanmean(allRipPowsLogged_activelink, axis=0)
                    s = np.nanstd(allRipPowsLogged_activelink, axis=0)
                    with pp.newFig(f"lfp_psth_logged_activelink", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    with pp.newFig(f"lfp_psth_logged_activelink_individual", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        plotIndividualAndAverage(pc.ax, allRipPowsLogged_activelink, xvals,
                                                 individualAmt=50)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsLogged_standard[np.isinf(allRipPowsLogged_standard)] = np.nan
                    m = np.nanmean(allRipPowsLogged_standard, axis=0)
                    s = np.nanstd(allRipPowsLogged_standard, axis=0)
                    with pp.newFig(f"lfp_psth_logged_standard", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsLogged_standardOffset[np.isinf(
                        allRipPowsLogged_standardOffset)] = np.nan
                    m = np.nanmean(allRipPowsLogged_standardOffset, axis=0)
                    s = np.nanstd(allRipPowsLogged_standardOffset, axis=0)
                    with pp.newFig(f"lfp_psth_logged_standardOffset", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsLogged_causal[np.isinf(allRipPowsLogged_causal)] = np.nan
                    m = np.nanmean(allRipPowsLogged_causal, axis=0)
                    s = np.nanstd(allRipPowsLogged_causal, axis=0)
                    with pp.newFig(f"lfp_psth_logged_causal", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                    allRipPowsLogged_causalOffset[np.isinf(
                        allRipPowsLogged_causalOffset)] = np.nan
                    m = np.nanmean(allRipPowsLogged_causalOffset, axis=0)
                    s = np.nanstd(allRipPowsLogged_causalOffset, axis=0)
                    with pp.newFig(f"lfp_psth_logged_causalOffset", excludeFromCombo=True) as pc:
                        xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                        pc.ax.plot(xvals, m)
                        pc.ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                        pc.ax.plot([0, 0], [-10, 10])
                        pc.ax.set_ylim(-2, 6)

                pp.popOutputSubDir()

            # with pp.newFig("lfp_detection_latency_to_artifact_z0_all") as pc:
            #     pc.ax.hist(latencyToDetectionArtifactFrom0_all,
            #                bins=np.linspace(0, .1, 20), density=True)
            # with pp.newFig("lfp_detection_latency_to_artifact_z25_all") as pc:
            #     pc.ax.hist(latencyToDetectionArtifactFrom25_all,
            #                bins=np.linspace(0, .1, 20), density=True)
            # with pp.newFig("lfp_detection_latency_to_logged_z0_all") as pc:
            #     pc.ax.hist(latencyToDetectionLoggedFrom0_all,
            #                bins=np.linspace(0, .1, 20), density=True)
            # with pp.newFig("lfp_detection_latency_to_logged_z25_all") as pc:
            #     pc.ax.hist(latencyToDetectionLoggedFrom25_all,
            #                bins=np.linspace(0, .1, 20), density=True)

    if len(animalNames) > 1 and MAKE_COMBINED:
        comboStr = "combo {}".format(" ".join(animalNames))
        print("======================\n", comboStr)
        suggestedLayout = None
        if len(animalNames) == 6:
            suggestedLayout = (2, 3)
        pp.makeCombinedFigs(outputSubDir=comboStr, suggestedSubPlotLayout=suggestedLayout)

    if RUN_SHUFFLES:
        pp.runImmediateShufflesAcrossPersistentCategories()

    if DATAMINE:
        if TEST_DATA:
            pp.runShuffles(numShuffles=10)
        else:
            pp.runShuffles()


if __name__ == "__main__":
    # makeFigures(RUN_UNSPECIFIED=False, RUN_THE_FINAL_DATAMINE=True, DATAMINE=True, TEST_DATA=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_THE_FINAL_DATAMINE=True, DATAMINE=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_TESTS=True)
    # makeFigures(RUN_UNSPECIFIED=False, RERUN_SIMPLE_MEASURES=True, DATAMINE=True)
    makeFigures(RUN_UNSPECIFIED=False, PLOT_DETAILED_TASK_TRACES=True)

    # hacky_RunOldStats("B17_20230210_144854.txt")
    # hacky_RunOldStats("B17_20230210_144854.txt")

    # dataDir = "/media/WDC8/figures/final_datamine/"
    # # fileName = "20230307_010237_significantShuffles.h5"
    # fileName = "20230303_194657_significantShuffles.h5"

    # inputFile = os.path.join(dataDir, fileName)
    # generateDMFReport(inputFile, skipSimple=True)
