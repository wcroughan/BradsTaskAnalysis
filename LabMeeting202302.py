from BTSession import BTSession
from MeasureTypes import WellMeasure, TrialMeasure
from BTData import BTData
from PlotUtil import PlotManager, setupBehaviorTracePlot, plotIndividualAndAverage
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getLoadInfo, getRipplePower
import os
import time
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List
from matplotlib.axes import Axes
import math
import MountainViewIO
from consts import TRODES_SAMPLING_RATE
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# TODO
# focus first on getting something on which to base discussion, like very basic demonstration of memory of home during probe
#   once have measures showing home/away difference, can talk about condition difference
#
# Quantify behavioral similarity, the typical routes taken to a well
# First idea: make circular histogram (say 8 directions, with smoothing) of direction entering and exiting each well
#   Compare dot prod or correlation between task and probe between conditions at home and maybe other wells
#   For plotting, allow measureFunc to be either a normal function as is or a tuple
#   If it's a tuple, first part is measure func, which can return either float or any other kind of value
#   If it's an array, treat like rotational histogram
#   If array of arrays, overlay rotational histograms (i.e. one for task one for probe)
#   If it's something else, don't display on the everysession plot (maybe just skip that plot then)
#   Then second element of tuple is a combining function which takes the output of measure func for a session one well
#           and other wells to compare with (i.e. aways or other seshs) and gives you a difference measure
#   should have option instead of overlaying hists to have plots show up in pairs, so can have different traces on each
#
# Look at notes in B18's LoadInfo section, deal with all of them
# i.e. minimum date for B13
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
# Do I need spline line in wellmeasure?
#
# Set a maximum gap for previous sessions, so large gaps aren't counted and prev dir goes back to None
#
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


def makeFigures(RUN_SHUFFLES=False, RUN_UNSPECIFIED=True, PRINT_INFO=True,
                RUN_JUST_THIS_SESSION=None, RUN_SPOTLIGHT=None,
                RUN_OPTIMALITY=None, PLOT_DETAILED_TASK_TRACES=None,
                PLOT_DETAILED_PROBE_TRACES=None,
                RUN_DOT_PROD=None, RUN_SMOOTHING_TEST=None, RUN_MANY_DOTPROD=None,
                RUN_LFP_LATENCY=None, MAKE_INDIVIDUAL_INTERRUPTION_PLOTS=False,
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
            WellMeasure("test", lambda s, w: w, sessions[0:5])

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

        if RUN_SPOTLIGHT:
            WellMeasure("probe spotlight score before fill", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, s.fillTimeCutoff()], binarySpotlight=True,
                boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp,
                                               everySessionTraceTimeInterval=lambda s: [
                                                   0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe")

        if RUN_DOT_PROD:
            WellMeasure("probe dotprod score before fill", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, s.fillTimeCutoff()], boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceTimeInterval=lambda s: [0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe_bouts")
            WellMeasure("task dotprod score", lambda s, h: s.getDotProductScore(
                False, h, boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceType="task_bouts")

            WellMeasure("probe dotprod score first entry", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, s.getLatencyToWell(True, s.homeWell, returnUnits="secs")], boutFlag=BTSession.BOUT_STATE_EXPLORE),
                sessionsWithProbe).makeFigures(pp, everySessionTraceTimeInterval=lambda s: [0, s.getLatencyToWell(True, s.homeWell, returnUnits="secs")],
                                               everySessionTraceType="probe_bouts", plotFlags="everysession")

        if RUN_MANY_DOTPROD:
            SECTION_LEN = 5
            WellMeasure("short dotprod 1", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, plotFlags="everysession",
                                              everySessionTraceTimeInterval=lambda _: [
                                                  0, SECTION_LEN], everySessionTraceType="probe")
            WellMeasure("short dotprod 2", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[SECTION_LEN, 2*SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, plotFlags="everysession",
                                              everySessionTraceTimeInterval=lambda _: [
                                                  SECTION_LEN, 2*SECTION_LEN], everySessionTraceType="probe")
            WellMeasure("short dotprod 1 and 2", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, 2*SECTION_LEN], boutFlag=BTSession.BOUT_STATE_EXPLORE
            ), sessionsWithProbe).makeFigures(pp, plotFlags="everysession",
                                              everySessionTraceTimeInterval=lambda _: [
                                                  0, 2*SECTION_LEN], everySessionTraceType="probe")

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
        pp.runImmidateShufflesAcrossPersistentCategories()

    if DATAMINE:
        pp.runShuffles()


if __name__ == "__main__":
    # makeFigures(RUN_UNSPECIFIED=False, RUN_MANY_DOTPROD=True, RUN_DOT_PROD=True, RUN_SPOTLIGHT=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_SMOOTHING_TEST=True)
    makeFigures(RUN_UNSPECIFIED=False, RUN_TESTS=True)
    # makeFigures(RUN_UNSPECIFIED=False)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_OPTIMALITY=True)
    # makeFigures(RUN_UNSPECIFIED=False, PLOT_DETAILED_PROBE_TRACES=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_LFP_LATENCY=True,
    #             MAKE_INDIVIDUAL_INTERRUPTION_PLOTS=True)

    # makeFigures(RUN_UNSPECIFIED=True, RUN_LFP_LATENCY=False)
