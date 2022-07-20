import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
from UtilFunctions import getInfoForAnimal, findDataDir, parseCmdLineAnimalNames, offWall
from consts import TRODES_SAMPLING_RATE, offWallWellNames
import math

# NOTE: Here's the example for trimming a video
# $ ffmpeg -i input.mp4 -ss 00:05:20 -t 00:10:00 -c:v copy -c:a copy output1.mp4
# Add -an to discard audio

# Cleanup steps:
# cameraModule all the position files
# trim usb videos
# Make wellLocations.csv
# Run import data, do light marking



class TrialMeasure():
    memoDict = {}

    def __init__(self, name="", measureFunc=None, sessionList=None, forceRemake=False, trialFilter=None):
        if name in TrialMeasure.memoDict and not forceRemake:
            existing = TrialMeasure.memoDict[name]
            self.measure = existing.measure
            self.trialCategory = existing.trialCategory
            self.conditionCategoryByTrial = existing.conditionCategoryByTrial
            self.conditionCategoryBySession = existing.conditionCategoryBySession
            self.withinSessionMeasureDifference = existing.withinSessionMeasureDifference
            self.name = name
            return

        self.measure = []
        self.trialCategory = []
        self.conditionCategoryByTrial = []
        self.conditionCategoryBySession = []
        self.name = name
        self.withinSessionMeasureDifference = []

        if measureFunc is not None:
            assert sessionList is not None
            for si, sesh in enumerate(sessionList):
                # home trials
                t1 = np.array(sesh.home_well_find_pos_idxs)
                t0 = np.array(np.hstack(([0], sesh.away_well_leave_pos_idxs)))
                if not sesh.ended_on_home:
                    t0 = t0[0:-1]
                assert len(t1) == len(t0)

                for ii, (i0, i1) in enumerate(zip(t0, t1)):
                    if trialFilter is not None and not trialFilter("home", ii, i0, i1, sesh.home_well):
                        continue

                    val = measureFunc(sesh, i0, i1, "home")
                    self.measure.append(val)
                    self.trialCategory.append("home")
                    self.conditionCategoryByTrial.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                    self.conditionCategoryBySession.append("SWR" if sesh.isRippleInterruption else "Ctrl")

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
                    self.conditionCategoryByTrial.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                    self.conditionCategoryBySession.append("SWR" if sesh.isRippleInterruption else "Ctrl")

        self.measure = np.array(self.measure)
        self.trialCategory = np.array(self.trialCategory)
        self.conditionCategoryByTrial = np.array(self.conditionCategoryByTrial)
        self.conditionCategoryBySession = np.array(self.conditionCategoryBySession)
        self.withinSessionMeasureDifference = np.array(self.withinSessionMeasureDifference)

        TrialMeasure.memoDict[name] = self


class WellMeasure():
    memoDict = {}

    def __init__(self, name="", measureFunc=None, sessionList=None, forceRemake=False, wellFilter=lambda ai, aw : offWall(aw)):
        if name in WellMeasure.memoDict and not forceRemake:
            existing = WellMeasure.memoDict[name]
            self.measure = existing.measure
            self.wellCategory = existing.wellCategory
            self.conditionCategoryByWell = existing.conditionCategoryByWell
            self.conditionCategoryBySession = existing.conditionCategoryBySession
            self.withinSessionMeasureDifference = existing.withinSessionMeasureDifference
            self.name = name
            return

        self.measure = []
        self.wellCategory = []
        self.conditionCategoryByWell = []
        self.conditionCategoryBySession = []
        self.name = name
        self.withinSessionMeasureDifference = []

        if measureFunc is not None:
            assert sessionList is not None
            for si, sesh in enumerate(sessionList):
                homeval = measureFunc(sesh, sesh.home_well)
                self.measure.append(homeval)
                self.wellCategory.append("home")
                self.conditionCategoryByWell.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                self.conditionCategoryBySession.append("SWR" if sesh.isRippleInterruption else "Ctrl")

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
                        self.conditionCategoryByWell.append(self.conditionCategoryByWell[-1])

                    awayVals = np.array(awayVals)
                    self.withinSessionMeasureDifference.append(homeval - np.nanmean(awayVals))

        self.measure = np.array(self.measure)
        self.wellCategory = np.array(self.wellCategory)
        self.conditionCategoryByWell = np.array(self.conditionCategoryByWell)
        self.conditionCategoryBySession = np.array(self.conditionCategoryBySession)
        self.withinSessionMeasureDifference = np.array(self.withinSessionMeasureDifference)

        WellMeasure.memoDict[name] = self





def makeFigures(
    MAKE_LAST_MEETING_FIGS=True,
    MAKE_SPIKE_ANALYSIS_FIGS=True,
    MAKE_CLOSE_PASS_FIGS=True,
    MAKE_INITIAL_HEADING_FIGS=True
):
    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "20220607_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)
    pp = PlotCtx(outputDir=globalOutputDir, randomSeed=rseed, priorityLevel=1)

    animalNames = parseCmdLineAnimalNames(default=["B13", "B14", "Martin"])
    allSessionsByRat = {}
    for animalName in animalNames:
        animalInfo = getInfoForAnimal(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[animalName] = ratData.getSessions()

    for ratName in animalNames:
        print("======================\n", ratName)
        sessions = allSessionsByRat[ratName]
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probe_performed]

        if MAKE_LAST_MEETING_FIGS:
            # All probe measures, within-session differences:
            probeWellMeasures = []
            probeWellMeasures.append(WellMeasure("avg dwell time", lambda s, h: s.avg_dwell_time(True, h), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("avg dwell time, 90sec", lambda s, h: s.avg_dwell_time(True, h, timeInterval=[0, 90]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("avg dwell time, before fill", lambda s, h: s.avg_dwell_time(True, h, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("curvature", lambda s, h: s.avg_curvature_at_well(True, h), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("curvature, 90sec", lambda s, h: s.avg_curvature_at_well(True, h, timeInterval=[0, 90]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("curvature, before fill", lambda s, h: s.avg_curvature_at_well(True, h, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("num entries", lambda s, h: s.num_well_entries(True, h), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("num entries, 90sec", lambda s, h: s.num_well_entries(True, h, timeInterval=[0,90]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("num entries, before fill", lambda s, h: s.num_well_entries(True, h, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("latency", lambda s, h: s.getLatencyToWell(True, h, returnSeconds=True), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("optimality", lambda s, h: s.path_optimality(True, wellName=h), sessionsWithProbe))
            probeWellMeasures.append(WellMeasure("gravity from off wall", lambda s, h: s.gravityOfWell(True, h, fromWells=offWallWellNames), sessionsWithProbe))

            taskWellMeasures = []
            taskWellMeasures.append(WellMeasure("gravity from off wall", lambda s, h: s.gravityOfWell(False, h, fromWells=offWallWellNames), sessionsWithProbe))
            taskWellMeasures.append(WellMeasure("gravity from off wall, T>5", \
                lambda s, h: s.gravityOfWell(False, h, fromWells=offWallWellNames, timeInterval=[s.home_well_find_times[4], np.inf]), sessionsWithProbe))
            
            taskTrialMeasures = []
            def numRepeatsVisited(sesh, i0, i1, t):
                    numWellsVisited(sesh.bt_nearest_wells[i0:i1], countReturns=True) - \
                        numWellsVisited(sesh.bt_nearest_wells[i0:i1], countReturns=False)
            taskTrialMeasures.append(TrialMeasure("num repeats visited", measureFunc=numRepeatsVisited(sesh, i0, i1, t),
                                    sessionList=sessionsWithProbe, trialFilter=lambda t, ii, i0, i1, w: offWall(w)))
            taskTrialMeasures.append(TrialMeasure("num repeats visited, T>5", measureFunc=numRepeatsVisited(sesh, i0, i1, t),
                                    sessionList=sessionsWithProbe, trialFilter=lambda t, ii, i0, i1, w: offWall(w) and ii > 4))

            for wm in probeWellMeasures:
                figName = "probe_well_" + wm.name.replace(" ", "_")
                with pp.newFig(figName) as ax:
                    boxPlot(ax, wm.measure, wm.wellCategory, categories2=wm.conditionCategoryByWell,
                    axesNames=[wm.name, "Well type", "Condition"], violin=True, doStats=False)

                with pp.newFig(figName + "_diff") as ax:
                    boxPlot(ax, wm.withinSessionMeasureDifference, wm.conditionCategoryBySession, 
                    axesNames=[wm.name + " with-session difference", "Condition"], violin=True, doStats=False)

            for wm in taskWellMeasures:
                figName = "task_well_" + wm.name.replace(" ", "_")
                with pp.newFig(figName) as ax:
                    boxPlot(ax, wm.measure, wm.wellCategory, categories2=wm.conditionCategoryByWell,
                    axesNames=[wm.name, "Well type", "Condition"], violin=True, doStats=False)

                with pp.newFig(figName + "_diff") as ax:
                    boxPlot(ax, wm.withinSessionMeasureDifference, wm.conditionCategoryBySession, 
                    axesNames=[wm.name + " with-session difference", "Condition"], violin=True, doStats=False)

            for wm in taskTrialMeasures:
                figName = "task_trial_" + wm.name.replace(" ", "_")
                with pp.newFig(figName) as ax:
                    boxPlot(ax, wm.measure, wm.trialCategory, categories2=wm.conditionCategoryByTrial,
                    axesNames=[wm.name, "Trial type", "Condition"], violin=True, doStats=False)

                with pp.newFig(figName + "_diff") as ax:
                    boxPlot(ax, wm.withinSessionMeasureDifference, wm.conditionCategoryBySession, 
                    axesNames=[wm.name + " with-session difference", "Condition"], violin=True, doStats=False)



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


if __name__ == "__main__":
    makeFigures()


# TODO: new analyses
# in addition to gravity modifications, should see how often goes near and doesn't get it,
# or how close goes and doesn't get it. Maybe histogram of passes where he does or doesn't find reward
#
# another possible measure, when they find away and are facing away from the home well, how often do they continue
# the direction they're facing (most of the time they do this) vs turn around immediately.
# More generally, some bias level between facing dir and home dir
#
# color swarm plots by trial index or date
#
# Pseudo-probe
# Any chance they return to same-condition home well during this? I.e. during pseudoprobe, will check out previous interruption home well
# on an interruption session even though there's been delay sessions in between? (and vice versa)
