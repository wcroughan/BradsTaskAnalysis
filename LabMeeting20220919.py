import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot, ShuffSpec
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
from UtilFunctions import getInfoForAnimal, findDataDir, parseCmdLineAnimalNames, offWall, numWellsVisited
from consts import TRODES_SAMPLING_RATE, offWallWellNames
import math
import time
from matplotlib import cm
from datetime import datetime

# NOTE: Here's the example for trimming a video
# $ ffmpeg -i input.mp4 -ss 00:05:20 -t 00:10:00 -c:v copy -c:a copy output1.mp4
# Add -an to discard audio

# Cleanup steps:
# cameraModule all the position files
# trim usb videos
# Make wellLocations.csv
# Run import data, do light marking


class TrialMeasure():
    # memoDict = {}

    def __init__(self, name="", measureFunc=None, sessionList=None, forceRemake=False, trialFilter=None, skipStats=False):
        # print("initing trial measure " + name)
        # if name in TrialMeasure.memoDict and not forceRemake:
        #     existing = TrialMeasure.memoDict[name]
        #     self.measure = existing.measure
        #     self.trialCategory = existing.trialCategory
        #     self.conditionCategoryByTrial = existing.conditionCategoryByTrial
        #     self.dotColors = existing.dotColors
        #     self.name = name
        #     return

        self.measure = []
        self.trialCategory = []
        self.conditionCategoryByTrial = []
        self.dotColors = []
        self.name = name

        if measureFunc is not None:
            assert sessionList is not None
            for si, sesh in enumerate(sessionList):
                # home trials
                t1 = np.array(sesh.home_well_find_pos_idxs)
                t0 = np.array(np.hstack(([0], sesh.away_well_leave_pos_idxs)))
                if not sesh.ended_on_home:
                    t0 = t0[0:-1]
                # print(t0)
                # print(t1)
                # print(sesh.ended_on_home)
                # print(sesh.name)
                assert len(t1) == len(t0)

                for ii, (i0, i1) in enumerate(zip(t0, t1)):
                    if trialFilter is not None and not trialFilter("home", ii, i0, i1, sesh.home_well):
                        continue

                    val = measureFunc(sesh, i0, i1, "home")
                    self.measure.append(val)
                    self.trialCategory.append("home")
                    self.conditionCategoryByTrial.append(
                        "SWR" if sesh.isRippleInterruption else "Ctrl")
                    self.dotColors.append(si)

                # away trials
                t1 = np.array(sesh.away_well_find_pos_idxs)
                t0 = np.array(sesh.home_well_leave_pos_idxs)
                if sesh.ended_on_home:
                    t0 = t0[0:-1]
                assert len(t1) == len(t0)

                for ii, (i0, i1) in enumerate(zip(t0, t1)):
                    if trialFilter is not None and not trialFilter("away", ii, i0, i1, sesh.visited_away_wells[ii]):
                        continue

                    val = measureFunc(sesh, i0, i1, "away")
                    self.measure.append(val)
                    self.trialCategory.append("away")
                    self.conditionCategoryByTrial.append(
                        "SWR" if sesh.isRippleInterruption else "Ctrl")
                    self.dotColors.append(si)

        self.measure = np.array(self.measure)
        self.dotColors = np.array(self.dotColors)
        self.trialCategory = np.array(self.trialCategory)
        self.conditionCategoryByTrial = np.array(self.conditionCategoryByTrial)

        # TrialMeasure.memoDict[name] = self


class WellMeasure():
    # memoDict = {}

    def __init__(self, name="", measureFunc=None, sessionList=None, forceRemake=False, wellFilter=lambda ai, aw: offWall(aw), skipStats=False):
        # print("initing trial measure " + name)
        # if name in WellMeasure.memoDict and not forceRemake:
        #     existing = WellMeasure.memoDict[name]
        #     self.measure = existing.measure
        #     self.wellCategory = existing.wellCategory
        #     self.conditionCategoryByWell = existing.conditionCategoryByWell
        #     self.conditionCategoryBySession = existing.conditionCategoryBySession
        #     self.withinSessionMeasureDifference = existing.withinSessionMeasureDifference
        #     self.name = name
        #     self.dotColors = existing.dotColors
        #     self.dotColorsBySession = existing.dotColorsBySession
        #     return

        self.measure = []
        self.wellCategory = []
        self.conditionCategoryByWell = []
        self.conditionCategoryBySession = []
        self.name = name
        self.withinSessionMeasureDifference = []
        self.dotColors = []
        self.dotColorsBySession = []

        if measureFunc is not None:
            assert sessionList is not None
            # print(sessionList)
            for si, sesh in enumerate(sessionList):
                # print(sesh.home_well_find_times)
                homeval = measureFunc(sesh, sesh.home_well)
                self.measure.append(homeval)
                self.wellCategory.append("home")
                self.dotColors.append(si)
                self.dotColorsBySession.append(si)
                self.conditionCategoryByWell.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                self.conditionCategoryBySession.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")

                awayVals = []
                aways = sesh.visited_away_wells
                if wellFilter is not None:
                    aways = [aw for ai, aw in enumerate(aways) if wellFilter(ai, aw)]
                if len(aways) == 0:
                    print("warning: no off wall aways for session {}".format(sesh.name))
                    self.withinSessionMeasureDifference.append(np.nan)
                else:
                    for ai, aw in enumerate(aways):
                        av = measureFunc(sesh, aw)
                        awayVals.append(av)
                        self.measure.append(av)
                        self.wellCategory.append("away")
                        self.dotColors.append(si)
                        self.conditionCategoryByWell.append(self.conditionCategoryByWell[-1])

                    awayVals = np.array(awayVals)
                    self.withinSessionMeasureDifference.append(homeval - np.nanmean(awayVals))

        self.measure = np.array(self.measure)
        self.wellCategory = np.array(self.wellCategory)
        self.dotColors = np.array(self.dotColors)
        self.dotColorsBySession = np.array(self.dotColorsBySession)
        self.conditionCategoryByWell = np.array(self.conditionCategoryByWell)
        self.conditionCategoryBySession = np.array(self.conditionCategoryBySession)
        self.withinSessionMeasureDifference = np.array(self.withinSessionMeasureDifference)

        # print(self.name, self.measure)

        # WellMeasure.memoDict[name] = self


def makeFigures(
    MAKE_LAST_MEETING_FIGS=None,
    MAKE_MID_MEETING_FIGS=None,
    MAKE_SPIKE_ANALYSIS_FIGS=None,
    MAKE_CLOSE_PASS_FIGS=None,
    MAKE_INITIAL_HEADING_FIGS=None,
    MAKE_PROBE_TRACES_FIGS=None,
    MAKE_B16_BEHAVIOR_FIG=False,
    MAKE_UNSPECIFIED=True,
    RUN_SHUFFLES=False
):

    if MAKE_LAST_MEETING_FIGS is None:
        MAKE_LAST_MEETING_FIGS = MAKE_UNSPECIFIED
    if MAKE_SPIKE_ANALYSIS_FIGS is None:
        MAKE_SPIKE_ANALYSIS_FIGS = MAKE_UNSPECIFIED
    if MAKE_CLOSE_PASS_FIGS is None:
        MAKE_CLOSE_PASS_FIGS = MAKE_UNSPECIFIED
    if MAKE_INITIAL_HEADING_FIGS is None:
        MAKE_INITIAL_HEADING_FIGS = MAKE_UNSPECIFIED
    if MAKE_PROBE_TRACES_FIGS is None:
        MAKE_PROBE_TRACES_FIGS = MAKE_UNSPECIFIED

    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "20220919_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    animalNames = parseCmdLineAnimalNames(default=["B18"])

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
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probe_performed]
        numSessionsWithProbe = len(sessionsWithProbe)
        print(f"{len(sessions)} sessions ({len(sessionsWithProbe)} with probe)")

        ctrlSessionsWithProbe = [sesh for sesh in sessions if (
            not sesh.isRippleInterruption) and sesh.probe_performed]
        swrSessionsWithProbe = [
            sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probe_performed]
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)

        pp.setOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if MAKE_LAST_MEETING_FIGS:
            # All probe measures, within-session differences:
            probeWellMeasures = []
            probeWellMeasures.append(WellMeasure("avg dwell time", lambda s,
                                                 h: s.avg_dwell_time(True, h), sessionsWithProbe))
            if hasattr(sessions[0], "probe_fill_time"):
                sessionsWithProbeFillPast90 = [
                    s for s in sessionsWithProbe if s.probe_fill_time > 90]
            else:
                sessionsWithProbeFillPast90 = sessionsWithProbe
            probeWellMeasures.append(WellMeasure("avg dwell time, 90sec", lambda s, h: s.avg_dwell_time(
                True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90))
            if hasattr(sessions[0], "probe_fill_time"):
                probeWellMeasures.append(WellMeasure("avg dwell time, before fill", lambda s, h: s.avg_dwell_time(
                    True, h, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe))
                probeWellMeasures.append(WellMeasure("avg dwell time, after fill", lambda s, h: s.avg_dwell_time(
                    True, h, timeInterval=[s.probe_fill_time, 60 * 5]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("curvature", lambda s,
                                                 h: s.avg_curvature_at_well(True, h), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("curvature, 90sec", lambda s, h: s.avg_curvature_at_well(
                True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90))
            if hasattr(sessions[0], "probe_fill_time"):
                probeWellMeasures.append(WellMeasure("curvature, before fill", lambda s, h: s.avg_curvature_at_well(
                    True, h, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe))
                probeWellMeasures.append(WellMeasure("curvature, after fill", lambda s, h: s.avg_curvature_at_well(
                    True, h, timeInterval=[s.probe_fill_time, 60 * 5]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("num entries", lambda s,
                                                 h: s.num_well_entries(True, h), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("num entries, 90sec", lambda s, h: s.num_well_entries(
                True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90))
            if hasattr(sessions[0], "probe_fill_time"):
                probeWellMeasures.append(WellMeasure("num entries, before fill", lambda s, h: s.num_well_entries(
                    True, h, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe))
                probeWellMeasures.append(WellMeasure("num entries, after fill", lambda s, h: s.num_well_entries(
                    True, h, timeInterval=[s.probe_fill_time, 60 * 5]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("latency", lambda s, h: s.getLatencyToWell(
                True, h, returnSeconds=True), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("optimality", lambda s,
                                                 h: s.path_optimality(True, wellName=h), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("gravity from off wall", lambda s, h: s.gravityOfWell(
                True, h, fromWells=offWallWellNames), sessionsWithProbe))
            if hasattr(sessions[0], "probe_fill_time"):
                probeWellMeasures.append(WellMeasure("gravity from off wall, before fill", lambda s, h: s.gravityOfWell(
                    True, h, fromWells=offWallWellNames, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe))

            taskWellMeasures = []
            taskWellMeasures.append(WellMeasure("gravity from off wall", lambda s, h: s.gravityOfWell(
                False, h, fromWells=offWallWellNames), sessionsWithProbe))

            taskWellMeasures.append(WellMeasure("gravity from all neighbors, tgt5", lambda s, h: s.gravityOfWell(
                False, h, timeInterval=[s.home_well_find_times[4] / TRODES_SAMPLING_RATE, np.inf]) if len(s.home_well_find_times) >= 5 else np.nan, sessionsWithProbe))

            def ff(s, h):
                if len(s.home_well_find_times) >= 5:
                    # print("doing normal thing")
                    return s.gravityOfWell(False, h, fromWells=offWallWellNames, timeInterval=[s.home_well_find_times[4] / TRODES_SAMPLING_RATE, np.inf])
                else:
                    # print("too few homes")
                    return np.nan

            taskWellMeasures.append(WellMeasure(
                "gravity from off wall, Tgt5", ff, sessionsWithProbe))

            taskTrialMeasures = []

            def numRepeatsVisited(sesh, i0, i1, t):
                return numWellsVisited(sesh.bt_nearest_wells[i0:i1], countReturns=True) - \
                    numWellsVisited(sesh.bt_nearest_wells[i0:i1], countReturns=False)
            taskTrialMeasures.append(TrialMeasure("num repeats visited", measureFunc=numRepeatsVisited,
                                                  sessionList=sessionsWithProbe, trialFilter=lambda t, ii, i0, i1, w: offWall(w)))
            taskTrialMeasures.append(TrialMeasure("num repeats visited, Tgt5", measureFunc=numRepeatsVisited,
                                                  sessionList=sessionsWithProbe, trialFilter=lambda t, ii, i0, i1, w: offWall(w) and ii > 4))

            for wm in probeWellMeasures:
                figName = "probe_well_" + wm.name.replace(" ", "_")
                # print("Making " + figName)
                with pp.newFig(figName, withStats=True) as (ax, yvals, cats, info):
                    boxPlot(ax, wm.measure, categories2=wm.wellCategory, categories=wm.conditionCategoryByWell,
                            axesNames=["Condition", wm.name, "Well type"], violin=True, doStats=False,
                            dotColors=wm.dotColors)

                    yvals[figName] = wm.measure
                    cats["well"] = wm.wellCategory
                    cats["condition"] = wm.conditionCategoryByWell

                # print("Making diff, " + figName)
                with pp.newFig(figName + "_diff", withStats=True) as (ax, yvals, cats, info):
                    boxPlot(ax, wm.withinSessionMeasureDifference, wm.conditionCategoryBySession,
                            axesNames=["Contidion", wm.name + " with-session difference"], violin=True, doStats=False,
                            dotColors=wm.dotColorsBySession)

                    yvals[figName + "_diff"] = wm.withinSessionMeasureDifference
                    cats["condition"] = wm.conditionCategoryBySession

            for wm in taskWellMeasures:
                figName = "task_well_" + wm.name.replace(" ", "_")
                with pp.newFig(figName, withStats=True) as (ax, yvals, cats, info):
                    boxPlot(ax, wm.measure, categories2=wm.wellCategory, categories=wm.conditionCategoryByWell,
                            axesNames=["Condition", wm.name, "Well type"], violin=True, doStats=False,
                            dotColors=wm.dotColors)

                    yvals[figName] = wm.measure
                    cats["well"] = wm.wellCategory
                    cats["condition"] = wm.conditionCategoryByWell

                with pp.newFig(figName + "_diff", withStats=True) as (ax, yvals, cats, info):
                    boxPlot(ax, wm.withinSessionMeasureDifference, wm.conditionCategoryBySession,
                            axesNames=["Condition", wm.name + " with-session difference"], violin=True, doStats=False,
                            dotColors=wm.dotColorsBySession)

                    yvals[figName + "_diff"] = wm.withinSessionMeasureDifference
                    cats["condition"] = wm.conditionCategoryBySession

            for wm in taskTrialMeasures:
                figName = "task_trial_" + wm.name.replace(" ", "_")
                with pp.newFig(figName, withStats=True) as (ax, yvals, cats, info):
                    boxPlot(ax, wm.measure, categories2=wm.trialCategory, categories=wm.conditionCategoryByTrial,
                            axesNames=["Condition", wm.name, "Trial type"], violin=True, doStats=False)

                    yvals[figName] = wm.measure
                    cats["trial"] = wm.trialCategory
                    cats["condition"] = wm.conditionCategoryByTrial

            # Cumulative number of wells visited in probe across sessions
            windowSlide = 10
            t1Array = np.arange(windowSlide, 60 * 5, windowSlide)
            numVisitedOverTime = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTime[:] = np.nan
            numVisitedOverTimeOffWall = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTimeOffWall[:] = np.nan
            numVisitedOverTimeWithRepeats = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTimeWithRepeats[:] = np.nan
            numVisitedOverTimeOffWallWithRepeats = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTimeOffWallWithRepeats[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisitedOverTime[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False)
                    numVisitedOverTimeOffWall[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False,
                        wellSubset=offWallWellNames)
                    numVisitedOverTimeWithRepeats[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=True)
                    numVisitedOverTimeOffWallWithRepeats[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=True,
                        wellSubset=offWallWellNames)

            with pp.newFig("probe_numVisitedOverTime_bysidx") as ax:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisitedOverTime.shape[1],)) * 0.5
                    ax.plot(t1Array, numVisitedOverTime[si, :] + jitter, color=cmap(si))
                ax.set_xticks(np.arange(0, 60 * 5 + 1, 60))
                ax.set_ylim(0, 37)

            with pp.newFig("probe_numVisitedOverTime_bysidx_offwall") as ax:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisitedOverTimeOffWall.shape[1],)) * 0.5
                    ax.plot(t1Array, numVisitedOverTimeOffWall[si, :] + jitter, color=cmap(si))
                ax.set_xticks(np.arange(0, 60 * 5 + 1, 60))
                ax.set_ylim(0, 17)

            with pp.newFig("probe_numVisitedOverTime_bysidx_withRepeats") as ax:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisitedOverTimeWithRepeats.shape[1],)) * 0.5
                    ax.plot(t1Array, numVisitedOverTimeWithRepeats[si, :] + jitter, color=cmap(si))
                ax.set_xticks(np.arange(0, 60 * 5 + 1, 60))
                # ax.set_ylim(0, 37)

            with pp.newFig("probe_numVisitedOverTime_bysidx_offwall_withRepeats") as ax:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(
                        size=(numVisitedOverTimeOffWallWithRepeats.shape[1],)) * 0.5
                    ax.plot(
                        t1Array, numVisitedOverTimeOffWallWithRepeats[si, :] + jitter, color=cmap(si))
                ax.set_xticks(np.arange(0, 60 * 5 + 1, 60))
                # ax.set_ylim(0, 17)

            # follow-ups
            # correlation b/w trial duration and probe behavior?
            # task measures but only after learning (maybe T3-5 and onward?)
            # Evening vs morning session

        else:
            print("warning: skipping last meeting figs")

        if MAKE_SPIKE_ANALYSIS_FIGS:
            # Any cells for B13? If so, make spike calib plot
            # Latency of detection. Compare to Jadhav, or others that have shown this
            pass
        else:
            print("warning: skipping spike analysis figs")

        if not MAKE_PROBE_TRACES_FIGS:
            print("warning: skipping raw probe trace plots")
        else:
            numCols = math.ceil(math.sqrt(numSessionsWithProbe))
            with pp.newFig("allProbeTraces", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(sessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                    # ax.set_title(str(si))
                for si in range(numSessionsWithProbe, numCols * numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

            numCols = math.ceil(math.sqrt(nCtrlWithProbe))
            with pp.newFig("allProbeTraces_ctrl", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(ctrlSessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    # ax.set_title(str(si))
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                for si in range(nCtrlWithProbe, numCols * numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

            numCols = math.ceil(math.sqrt(nSWRWithProbe))
            with pp.newFig("allProbeTraces_SWR", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(swrSessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    # ax.set_title(str(si))
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                for si in range(nSWRWithProbe, numCols * numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

        if not MAKE_MID_MEETING_FIGS:
            print("Warning skipping mid meeting figs")
        else:
            pass

    if len(animalNames) > 1:
        pp.makeCombinedFigs(outputSubDir="combo {}".format(" ".join(animalNames)))

    if RUN_SHUFFLES:
        pp.runShuffles()


if __name__ == "__main__":
    # makeFigures(MAKE_UNSPECIFIED=False, MAKE_LAST_MEETING_FIGS=True, RUN_SHUFFLES=True)
    # makeFigures(MAKE_UNSPECIFIED=False, MAKE_MID_MEETING_FIGS=True)
    makeFigures()


# TODO: new analyses
# in addition to gravity modifications, should see how often goes near and doesn't get it,
# or how close goes and doesn't get it. Maybe histogram of passes where he does or doesn't find reward
#
# another possible measure, when they find away and are facing away from the home well, how often do they continue
# the direction they're facing (most of the time they do this) vs turn around immediately.
# More generally, some bias level between facing dir and home dir
#
# How close are start and end of excursion to home well? Euclidean and city block
#
# color swarm plots by trial index or date
#
# Pseudo-probe
# Any chance they return to same-condition home well during this? I.e. during pseudoprobe, will check out previous interruption home well
# on an interruption session even though there's been delay sessions in between? (and vice versa)
