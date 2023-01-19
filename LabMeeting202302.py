import os
import time
from datetime import datetime
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getInfoForAnimal, correctFishEye
from PlotUtil import PlotCtx, setupBehaviorTracePlot
from BTData import BTData
from MeasureTypes import WellMeasure
from BTSession import BTSession

#
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
                RUN_DOT_PROD=None, RUN_FISHEYE_INSPECTION=None,
                RUN_TESTS=False, MAKE_COMBINED=True):
    if RUN_SPOTLIGHT is None:
        RUN_SPOTLIGHT = RUN_UNSPECIFIED
    if RUN_DOT_PROD is None:
        RUN_DOT_PROD = RUN_UNSPECIFIED
    if RUN_FISHEYE_INSPECTION is None:
        RUN_FISHEYE_INSPECTION = RUN_UNSPECIFIED

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
    pp = PlotCtx(outputDir=globalOutputDir, randomSeed=rseed,
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
        sessions = allSessionsByRat[ratName]
        if sessionToRun is not None:
            sessions = [s for s in sessions if sessionToRun in s.name or sessionToRun in s.infoFileName]
        # nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probe_performed]
        # numSessionsWithProbe = len(sessionsWithProbe)

        ctrlSessionsWithProbe = [sesh for sesh in sessions if (
            not sesh.isRippleInterruption) and sesh.probe_performed]
        swrSessionsWithProbe = [
            sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probe_performed]
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)

        # ctrlSessions = [sesh for sesh in sessions if not sesh.isRippleInterruption]
        # swrSessions = [sesh for sesh in sessions if sesh.isRippleInterruption]
        # nCtrlSessions = len(ctrlSessions)
        # nSWRSessions = len(swrSessions)

        print(f"{len(sessions)} sessions ({len(sessionsWithProbe)} "
              f"with probe: {nCtrlWithProbe} Ctrl, {nSWRWithProbe} SWR)")

        # if hasattr(sessions[0], "probe_fill_time"):
        #     sessionsWithProbeFillPast90 = [s for s in sessionsWithProbe if s.probe_fill_time > 90]
        # else:
        #     sessionsWithProbeFillPast90 = sessionsWithProbe

        pp.pushOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if RUN_TESTS:
            pass

        if RUN_FISHEYE_INSPECTION:
            smoothVals = np.power(2.0, np.arange(-1, 5))
            for sesh in sessions:
                assert isinstance(sesh, BTSession)
                pp.pushOutputSubDir(sesh.name)

                xs, ys = correctFishEye(sesh, sesh.probe_pos_xs, sesh.probe_pos_ys)
                with pp.newFig("probeTraceVariations", subPlots=(2, 1+len(smoothVals)), showPlot=True, savePlot=False) \
                        as axs:
                    setupBehaviorTracePlot(axs[0, 0], sesh, showAllWells=False,
                                           showHome=False, showAways=False)
                    axs[0, 0].plot(sesh.probe_pos_xs, sesh.probe_pos_ys)
                    axs[0, 0].set_title("raw")

                    for si, smooth in enumerate(smoothVals):
                        ax = axs[0, si+1]
                        setupBehaviorTracePlot(ax, sesh, showAllWells=False,
                                               showHome=False, showAways=False)
                        x, y = sesh.probe_pos_xs, sesh.probe_pos_ys
                        x = gaussian_filter1d(x, smooth)
                        y = gaussian_filter1d(y, smooth)
                        ax.plot(x, y)
                        ax.set_title(f"smooth {smooth}")

                    axs[1, 0].plot(xs, ys, c="#deac7f")
                    setupBehaviorTracePlot(axs[1, 0], sesh, showAllWells=False, showAways=False, showHome=False,
                                           extent=(-0.5, 6.5, -0.5, 6.5), reorient=False)
                    axs[1, 0].set_title("fisheye corrected")

                    for si, smooth in enumerate(smoothVals):
                        ax = axs[1, si+1]
                        setupBehaviorTracePlot(ax, sesh, showAllWells=False, showAways=False, showHome=False,
                                               extent=(-0.5, 6.5, -0.5, 6.5), reorient=False)
                        x = gaussian_filter1d(xs, smooth)
                        y = gaussian_filter1d(ys, smooth)
                        ax.plot(x, y)
                        ax.set_title(f"smooth {smooth}")

                pp.popOutputSubDir()

        if not RUN_SPOTLIGHT:
            print("Warning: skipping spotlight plots")
        else:
            WellMeasure("probe spotlight score before fill", lambda s, h: s.getSpotlightScore(
                True, h, timeInterval=[0, s.fillTimeCutoff()]),
                sessionsWithProbe).makeFigures(pp,
                                               everySessionTraceTimeInterval=lambda s: [
                                                   0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe")

        if not RUN_DOT_PROD:
            print("Warning: skipping spotlight plots")
        else:
            WellMeasure("probe dotprod score before fill", lambda s, h: s.getDotProductScore(
                True, h, timeInterval=[0, s.fillTimeCutoff()]),
                sessionsWithProbe).makeFigures(pp, everySessionTraceTimeInterval=lambda s: [0, s.fillTimeCutoff()],
                                               everySessionTraceType="probe_bouts")
            WellMeasure("task dotprod score before fill", lambda s, h: s.getDotProductScore(
                False, h),
                sessionsWithProbe).makeFigures(pp, everySessionTraceType="task_bouts")

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
    makeFigures(RUN_UNSPECIFIED=False, RUN_FISHEYE_INSPECTION=True)
    # makeFigures(RUN_UNSPECIFIED=False, RUN_TESTS=True)
