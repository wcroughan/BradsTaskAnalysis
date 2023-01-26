from BTSession import BTSession
from MeasureTypes import WellMeasure, TrialMeasure
from BTData import BTData
from PlotUtil import PlotManager, setupBehaviorTracePlot
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getInfoForAnimal
import os
import time
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List
from matplotlib.axes import Axes
import math

# TODOs
#
# Double check position tracking fill times for other rats. Especially B16 07_10 looks a bit wonky
# And B16 15_09
#
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
# Datamine B17
# Especially debugged dot product not divided by total time
# Histogram balance might be a good visual?
# Add control that is rotated-similar wells instead of aways
# Task gravity, exclude reward?
# Tune up spotlight/dotprod measure by just looking at behavior during home trials, should see clear trend toward home
# Symetrical controls might be more important here than other measures
# Distance discount?
# Plus velocity weighting perhaps??
# Maybe exclude times when at well? At least need to way to deal with looking right pas a well
# Maybe for everysession plots try centering on home to combine, see if clear gradient situation appears
# A measure that tries to capture the consistent paths that develop during a session, and compares that
#   to probe behavior. Something like frequency of entering a well from each angle maybe, so if
#   always approaching 35 from top and leaving from left, how consistent is that with probe?
#
# Work on DLC
#
# Inspect fisheye correction more. Look at output closely, and if it seems legit (espcially at edges
# of environment), should use that for velocity, curvature, etc. Otherwise center wells will think
# they have slower speed, and curvature radius will change
#
# Do I need spline line in wellmeasure?
#
# Modify latency function in BTSession so can easily plot latency and optimality over trials
#
# Set a maximum gap for previous sessions, so large gaps aren't counted and prev dir goes back to None
# Some "well visits" are just one time point. Need to smooth more and/or just generally change logic away from this
# visit stuff
#
# Better optimized clip correction region output. Save til the end so can:
#   compare to task and probe times, weed out ones that aren't during behavior
#   print separately at the end all at once
#
# Shuffles can go crazy with different measures when in datamine mode.
# Also have this output a helpful ranking of shuffles that showed up significant


# COmpleted:
# Debug dotprod
# Anything else look fishy?
# Def glitch where B16 12_11 is giving nans
# - Actually no that one was fine, just all exploration was done after fill time
#
# Trial measure output graphs
#
# To all every-session plots, add path on top


def makeFigures(RUN_SHUFFLES=False, RUN_UNSPECIFIED=True,
                RUN_JUST_THIS_SESSION=None, RUN_SPOTLIGHT=None,
                RUN_OPTIMALITY=None, PLOT_DETAILED_TRACES=None,
                RUN_DOT_PROD=None, RUN_SMOOTHING_TEST=None, RUN_MANY_DOTPROD=None,
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
    if PLOT_DETAILED_TRACES is None:
        PLOT_DETAILED_TRACES = RUN_UNSPECIFIED

    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "202302_labmeeting")
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
                     priorityLevel=1, infoFileName=infoFileName)

    allSessionsByRat = {}
    for animalName in animalNames:
        loadDebug = False
        if animalName[-1] == "d":
            loadDebug = True
            animalName = animalName[:-1]
        animalInfo = getInfoForAnimal(animalName)
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
            sessions = [s for s in sessions if sessionToRun in s.name or sessionToRun in s.infoFileName]
        # nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probePerformed]
        # numSessionsWithProbe = len(sessionsWithProbe)

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

        # if hasattr(sessions[0], "probeFillTime"):
        #     sessionsWithProbeFillPast90 = [s for s in sessionsWithProbe if s.probeFillTime > 90]
        # else:
        #     sessionsWithProbeFillPast90 = sessionsWithProbe

        pp.pushOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if RUN_TESTS:
            TrialMeasure("test", lambda s, i1, i2, t: i2 - i1,
                         sessions).makeFigures(pp, plotFlags="averages")

        if RUN_SMOOTHING_TEST:
            smoothVals = np.power(2.0, np.arange(-1, 5))
            for sesh in sessions:
                pp.pushOutputSubDir(sesh.name)

                with pp.newFig("probeTraceVariations", subPlots=(1, 1+len(smoothVals))) \
                        as pc:
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

        if not RUN_SPOTLIGHT:
            print("Warning: skipping spotlight plots")
        else:
            WellMeasure("probe spotlight score before fill", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, s.fillTimeCutoff()], binarySpotlight=True,
                boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp,
                                               everySessionTraceTimeInterval=lambda s: [
                                                   0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe")

        if not RUN_DOT_PROD:
            print("Warning: skipping dotprod plots")
        else:
            WellMeasure("probe dotprod score before fill", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, s.fillTimeCutoff()], boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceTimeInterval=lambda s: [0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe_bouts")
            WellMeasure("task dotprod score", lambda s, h: s.getDotProductScore(
                False, h, boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceType="task_bouts")

        if not RUN_MANY_DOTPROD:
            pass
        else:
            SECTION_LEN = 5
            WellMeasure("short dotprod 1", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, makeDiffBoxPlot=False, makeMeasureBoxPlot=False,
                                              makeOtherSeshBoxPlot=False, makeOtherSeshDiffBoxPlot=False,
                                              everySessionTraceTimeInterval=lambda _: [
                                                  0, SECTION_LEN], everySessionTraceType="probe")
            WellMeasure("short dotprod 2", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[SECTION_LEN, 2*SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, makeDiffBoxPlot=False, makeMeasureBoxPlot=False,
                                              makeOtherSeshBoxPlot=False, makeOtherSeshDiffBoxPlot=False,
                                              everySessionTraceTimeInterval=lambda _: [
                                                  SECTION_LEN, 2*SECTION_LEN], everySessionTraceType="probe")
            WellMeasure("short dotprod 1 and 2", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, 2*SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, makeDiffBoxPlot=False, makeMeasureBoxPlot=False,
                                              makeOtherSeshBoxPlot=False, makeOtherSeshDiffBoxPlot=False,
                                              everySessionTraceTimeInterval=lambda _: [
                                                  0, 2*SECTION_LEN], everySessionTraceType="probe")

        if PLOT_DETAILED_TRACES:
            for si, sesh in enumerate(sessions):
                pp.pushOutputSubDir(sesh.name)
                with pp.newFig("task_fullTrace") as pc:
                    setupBehaviorTracePlot(pc.ax, sesh)
                    pc.ax.plot(sesh.btPosXs, sesh.btPosYs)

                traceLen = 60
                traceStep = 30
                tStarts = np.arange(0, sesh.btPos_secs[-1], traceStep)
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

    if len(animalNames) > 1 and MAKE_COMBINED:
        comboStr = "combo {}".format(" ".join(animalNames))
        print("======================\n", comboStr)
        suggestedLayout = None
        if len(animalNames) == 6:
            suggestedLayout = (2, 3)
        pp.makeCombinedFigs(outputSubDir=comboStr, suggestedSubPlotLayout=suggestedLayout)

    if RUN_SHUFFLES:
        pp.runImmidateShufflesAcrossPersistentCategories()

    if DATAMINE:
        pp.runShuffles()


if __name__ == "__main__":
    # makeFigures(RUN_UNSPECIFIED=False, RUN_MANY_DOTPROD=True, RUN_DOT_PROD=True, RUN_SPOTLIGHT=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_SMOOTHING_TEST=True)
    makeFigures(RUN_UNSPECIFIED=False, RUN_TESTS=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_OPTIMALITY=True)
    # makeFigures(RUN_UNSPECIFIED=False, PLOT_DETAILED_TRACES=True)
