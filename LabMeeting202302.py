from consts import TRODES_SAMPLING_RATE
from BTSession import BTSession
from MeasureTypes import WellMeasure
from BTData import BTData
from PlotUtil import PlotManager, setupBehaviorTracePlot
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getInfoForAnimal, TimeThisFunction
import os
import time
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.animation as anim
from typing import List
from functools import partial
import matplotlib.style as mplstyle


# TODOs
# Debug dotprod
# Anything else look fishy?
# Def glitch where B16 12_11 is giving nans
# - Actually no that one was fine, just all exploration was done after fill time
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
# To all every-session plots, add path on top
# Add control that is rotated-similar wells instead of aways
# Task gravity, exclude reward?
# Tune up spotlight/dotprod measure by just looking at behavior during home trials, should see clear trend toward home
# Symetrical controls might be more important here than other measures
# Distance discount?
# Plus velocity weighting perhaps??
# Maybe exclude times when at well? At least need to way to deal with looking right pas a well
# Maybe for everysession plots try centering on home to combine, see if clear gradient situation appears
#
# Work on DLC
#
# Inspect fisheye correction more. Look at output closely, and if it seems legit (espcially at edges
# of environment), should use that for velocity, curvature, etc. Otherwise center wells will think
# they have slower speed, and curvature radius will change


def makeFigures(RUN_SHUFFLES=False, RUN_UNSPECIFIED=True,
                RUN_JUST_THIS_SESSION=None, RUN_SPOTLIGHT=None,
                RUN_DOT_PROD=None, RUN_SMOOTHING_TEST=None, RUN_MANY_DOTPROD=None,
                RUN_VELOCITY_INSPECTION=None,
                RUN_TESTS=False, MAKE_COMBINED=True):
    if RUN_SPOTLIGHT is None:
        RUN_SPOTLIGHT = RUN_UNSPECIFIED
    if RUN_DOT_PROD is None:
        RUN_DOT_PROD = RUN_UNSPECIFIED
    if RUN_SMOOTHING_TEST is None:
        RUN_SMOOTHING_TEST = RUN_UNSPECIFIED
    if RUN_MANY_DOTPROD is None:
        RUN_MANY_DOTPROD = RUN_UNSPECIFIED
    if RUN_VELOCITY_INSPECTION is None:
        RUN_VELOCITY_INSPECTION = RUN_UNSPECIFIED

    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "202302_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    if RUN_JUST_THIS_SESSION is None:
        animalNames = parseCmdLineAnimalNames(default=["B17"])
        sessionToRun = None
    else:
        animalNames = [RUN_JUST_THIS_SESSION[0]]
        sessionToRun = RUN_JUST_THIS_SESSION[1]

    infoFileName = datetime.now().strftime("_".join(animalNames) + "_%Y%m%d_%H%M%S" + ".txt")
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     priorityLevel=1, infoFileName=infoFileName)

    allSessionsByRat = {}
    for animalName in animalNames:
        animalInfo = getInfoForAnimal(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        print("loading from " + dataFilename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[animalName] = ratData.getSessions()

    for ratName in animalNames:
        print("======================\n", ratName)
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
            with pp.newFig("test", priority=0) as pc:
                print("I'm in the block!")
                pc.ax.plot(np.arange(5))

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
                True, h, timeInterval=[0, s.fillTimeCutoff()], binarySpotlight=True, boutFlag=BTSession.BOUT_STATE_EXPLORE),
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
            WellMeasure("task dotprod score", lambda s, h: s.getDotProductScore(False, h, boutFlag=BTSession.BOUT_STATE_EXPLORE),
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

        if not RUN_VELOCITY_INSPECTION:
            pass
        else:
            FRAME_RATE = 30
            VIDEO_SPEED = 1
            PLOT_LEN = 3
            TSTART = 30
            TEND = 34
            for si, sesh in enumerate(sessions):
                if si > 0:
                    break
                pp.pushOutputSubDir(sesh.name)

                x = sesh.probePosXs
                y = sesh.probePosYs
                mv = sesh.probeIsMv
                bout = sesh.probeBoutCategory
                vel = sesh.probeVelCmPerS
                smvel = sesh.probeSmoothVel
                t = sesh.probePos_ts / TRODES_SAMPLING_RATE
                t = t - t[0]

                frameStartTimes = np.arange(TSTART, TEND - PLOT_LEN, VIDEO_SPEED / FRAME_RATE)
                frameEndTimes = frameStartTimes + PLOT_LEN
                frameStarts_posIdx = np.searchsorted(t, frameStartTimes)
                frameEnds_posIdx = np.searchsorted(t, frameEndTimes)

                xmv1 = x.copy()
                xmv1[~mv] = np.nan
                xmv2 = x.copy()
                xmv2[mv] = np.nan
                ymv1 = y.copy()
                ymv1[~mv] = np.nan
                ymv2 = y.copy()
                ymv2[mv] = np.nan

                xbo1 = x.copy()
                xbo1[bout != BTSession.BOUT_STATE_EXPLORE] = np.nan
                xbo2 = x.copy()
                xbo2[bout != BTSession.BOUT_STATE_REST] = np.nan
                xbo3 = x.copy()
                xbo3[bout != BTSession.BOUT_STATE_REWARD] = np.nan
                ybo1 = y.copy()
                ybo1[bout != BTSession.BOUT_STATE_EXPLORE] = np.nan
                ybo2 = y.copy()
                ybo2[bout != BTSession.BOUT_STATE_REST] = np.nan
                ybo3 = y.copy()
                ybo3[bout != BTSession.BOUT_STATE_REWARD] = np.nan

                with pp.newFig("velocityAnimation", subPlots=(2, 2), showPlot=False, savePlot=False) as pc:
                    p11, = pc.axs[0, 0].plot([])
                    p12, = pc.axs[0, 0].plot([])
                    p21, = pc.axs[0, 1].plot([])
                    p22, = pc.axs[0, 1].plot([])
                    p23, = pc.axs[0, 1].plot([])
                    c1, = pc.axs[1, 0].plot(
                        t[frameStarts_posIdx[0]:frameEnds_posIdx[0]], mv[frameStarts_posIdx[0]:frameEnds_posIdx[0]])
                    c2, = pc.axs[1, 0].plot(
                        t[frameStarts_posIdx[0]:frameEnds_posIdx[0]], bout[frameStarts_posIdx[0]:frameEnds_posIdx[0]])
                    v1, = pc.axs[1, 1].plot(
                        t[frameStarts_posIdx[0]:frameEnds_posIdx[0]], vel[frameStarts_posIdx[0]:frameEnds_posIdx[0]])
                    v2, = pc.axs[1, 1].plot(
                        t[frameStarts_posIdx[0]:frameEnds_posIdx[0]], smvel[frameStarts_posIdx[0]:frameEnds_posIdx[0]])

                    p11.set_animated(True)
                    p12.set_animated(True)
                    p21.set_animated(True)
                    p22.set_animated(True)
                    p23.set_animated(True)
                    c1.set_animated(True)
                    c2.set_animated(True)
                    v1.set_animated(True)
                    v2.set_animated(True)

                    setupBehaviorTracePlot(pc.axs[0, 0], sesh)
                    setupBehaviorTracePlot(pc.axs[0, 1], sesh)

                    @TimeThisFunction
                    def animFunc(frames):
                        p11.set_data(xmv1[frames[0]:frames[1]], ymv1[frames[0]:frames[1]])
                        p12.set_data(xmv2[frames[0]:frames[1]], ymv2[frames[0]:frames[1]])
                        p21.set_data(xbo1[frames[0]:frames[1]], ybo1[frames[0]:frames[1]])
                        p22.set_data(xbo2[frames[0]:frames[1]], ybo2[frames[0]:frames[1]])
                        p23.set_data(xbo3[frames[0]:frames[1]], ybo3[frames[0]:frames[1]])
                        c1.set_data(t[frames[0]:frames[1]], mv[frames[0]:frames[1]])
                        c2.set_data(t[frames[0]:frames[1]], bout[frames[0]:frames[1]])
                        pc.axs[1, 0].set_xlim(t[frames[0]], t[frames[1]])
                        v1.set_data(t[frames[0]:frames[1]], vel[frames[0]:frames[1]])
                        v2.set_data(t[frames[0]:frames[1]], smvel[frames[0]:frames[1]])
                        pc.axs[1, 1].set_xlim(t[frames[0]], t[frames[1]])
                        return p11, p12, p21, p22, p23, c1, c2, v1, v2

                    frames = zip(frameStarts_posIdx, frameEnds_posIdx)
                    ani = anim.FuncAnimation(pc.figure, animFunc, frames, repeat=False,
                                             interval=1000/FRAME_RATE, blit=True, save_count=len(frameEnds_posIdx),
                                             init_func=partial(
                                                 animFunc, (frameStarts_posIdx[0], frameEnds_posIdx[0])))

                    start_time = time.perf_counter()
                    ani.save(pc.figName + ".mkv")
                    # plt.show()
                    end_time = time.perf_counter()
                    runTime = end_time - start_time
                    print(f"{runTime = }")
                    animFuncRunTime = animFunc.totalTime
                    print(f"{animFuncRunTime = }")

                pp.popOutputSubDir()

        pp.popOutputSubDir()

    if len(animalNames) > 1 and MAKE_COMBINED:
        comboStr = "combo {}".format(" ".join(animalNames))
        print("======================\n", comboStr)
        suggestedLayout = None
        if len(animalNames) == 6:
            suggestedLayout = (2, 3)
        pp.makeCombinedFigs(outputSubDir=comboStr, suggestedSubPlotLayout=suggestedLayout)

    if RUN_SHUFFLES:
        pp.runImmidateShufflesAcrossPersistentCategories()


if __name__ == "__main__":
    # makeFigures(RUN_UNSPECIFIED=False, RUN_MANY_DOTPROD=True, RUN_DOT_PROD=True, RUN_SPOTLIGHT=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_SMOOTHING_TEST=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_TESTS=True)
    # makeFigures(RUN_SMOOTHING_TEST=False)
    makeFigures(RUN_UNSPECIFIED=False, RUN_VELOCITY_INSPECTION=True)
