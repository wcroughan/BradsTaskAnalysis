import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot, conditionShuffle
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
import sys
from UtilFunctions import getRipplePower, numWellsVisited, onWall, getInfoForAnimal, detectRipples, findDataDir, parseCmdLineAnimalNames, fillCounts
from consts import TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE, allWellNames, offWallWellNames
import math
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# GOALS and TODOS:
# BEHAVIOR:
# -Look at just old data and just newer data. Why did the effect go away? Washed out or reveresed?
# -Look at home v away difference and other measures within session
# Look at away-from-home measures like where on the wall he was, orientation to home, dir crossed with dir to home
# -Look at gravity of home well
# Looks like there's more interruption sessions where B13 doesn't visit home at all in probe, look at behavior and see what underlying pattern is, best way to quantify
#
# LFP:
# Look at raw LFP and ripple power using different detection criteria (baseline, referenced, etc). What is difference?
#   OK a few takeaways from first look:
#       - Quite a few stims aren't getting excluded and are instead labeled as ripples with high power. Need to adjust stim detection
#       - After most stims the baseline tetrode signal actually stays way offset for about 40-80ms and then there's another artifact and it returns to mostly baseline
#               For one example, see B13, session 20220222_165339, around 755.7 seconds in
#       - Also happens to the detection tetrode sometimes, though it seems like less often and for less time when it happens
#               e.x. happens to both separately one after another in B13 session 20220222_165339, 811.7 seconds in
#       - There's another very consistent post stim aspect of the baseline tetrode voltage where it returns to normal slowly, like the PSTH seen on the reference tet but slightly bigger and different shape
# Is exportLFP output referenced?
#   yes, it is


def makeFigures():
    MAKE_EARLY_LATE_SESSION_BASIC_MEASURES = False
    MAKE_BASIC_MEASURE_PLOTS = False
    MAKE_WITHIN_SESSION_MEASURE_PLOTS = False
    MAKE_GRAVITY_PLOTS = False
    MAKE_PROBE_TRACES_FIGS = False
    MAKE_TASK_BEHAVIOR_PLOTS = False
    MAKE_PROBE_BEHAVIOR_PLOTS = False
    MAKE_RAW_LFP_POWER_PLOTS = False
    MAKE_CUMULATIVE_LFP_PLOTS = True

    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "20220524_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)
    pp = PlotCtx(outputDir=globalOutputDir, randomSeed=rseed)

    animalNames = parseCmdLineAnimalNames(default=["B13", "B14"])
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
        sessionsEarly = [sesh for sesh in sessions if int(sesh.date_str[0:4]) < 2022]
        sessionsLate = [sesh for sesh in sessions if int(sesh.date_str[0:4]) >= 2022]
        ctrlSessionsWithProbe = [sesh for sesh in sessions if (
            not sesh.isRippleInterruption) and sesh.probe_performed]
        swrSessionsWithProbe = [
            sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probe_performed]
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)
        numSessionsWithProbe = len(sessionsWithProbe)
        numSessions = len(sessions)
        sessionWithProbeIsInterruption = np.zeros((numSessionsWithProbe,)).astype(bool)
        for i in range(numSessionsWithProbe):
            if sessionsWithProbe[i].isRippleInterruption:
                sessionWithProbeIsInterruption[i] = True

        pp.setOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        # Here be plots
        if not MAKE_EARLY_LATE_SESSION_BASIC_MEASURES or MAKE_BASIC_MEASURE_PLOTS or MAKE_WITHIN_SESSION_MEASURE_PLOTS or MAKE_GRAVITY_PLOTS or MAKE_TASK_BEHAVIOR_PLOTS or MAKE_PROBE_BEHAVIOR_PLOTS:
            pass
        else:
            wellCat = []
            seshCat = []
            # seshConditionGroup = []
            avgDwell = []
            avgDwell90 = []
            curvature = []
            curvature90 = []
            seshDateCat = []
            numEntries = []
            gravity = []
            gravityFromOffWall = []

            avgDwellDifference = []
            avgDwellDifference90 = []
            curvatureDifference = []
            curvatureDifference90 = []
            numEntriesDifference = []
            seshCatBySession = []
            seshDateCatBySession = []
            gravityDifference = []
            gravityFromOffWallDifference = []

            for si, sesh in enumerate(sessionsWithProbe):
                towt = np.sum([sesh.total_dwell_time(True, w)
                               for w in offWallWellNames])

                avgDwell.append(sesh.avg_dwell_time(True, sesh.home_well))
                avgDwell90.append(sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90]))
                curvature.append(sesh.avg_curvature_at_well(True, sesh.home_well))
                curvature90.append(sesh.avg_curvature_at_well(
                    True, sesh.home_well, timeInterval=[0, 90]))
                numEntries.append(sesh.num_well_entries(True, sesh.home_well))
                gravity.append(sesh.gravityOfWell(True, sesh.home_well))
                gravityFromOffWall.append(sesh.gravityOfWell(
                    True, sesh.home_well, fromWells=offWallWellNames))

                wellCat.append("home")
                seshCat.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                # seshConditionGroup.append(sesh.conditionGroup)
                seshDateCat.append("early" if int(sesh.date_str[0:4]) < 2022 else "late")

                seshCatBySession.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                seshDateCatBySession.append("early" if int(sesh.date_str[0:4]) < 2022 else "late")

                for aw in sesh.visited_away_wells:
                    if onWall(aw):
                        continue

                    avgDwell.append(sesh.avg_dwell_time(True, aw))
                    avgDwell90.append(sesh.avg_dwell_time(True, aw, timeInterval=[0, 90]))
                    curvature.append(sesh.avg_curvature_at_well(True, aw))
                    curvature90.append(sesh.avg_curvature_at_well(True, aw, timeInterval=[0, 90]))
                    numEntries.append(sesh.num_well_entries(True, aw))
                    gravity.append(sesh.gravityOfWell(True, aw))
                    gravityFromOffWall.append(sesh.gravityOfWell(
                        True, aw, fromWells=offWallWellNames))

                    wellCat.append("away")
                    seshCat.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                    # seshConditionGroup.append(sesh.conditionGroup)
                    seshDateCat.append("early" if int(sesh.date_str[0:4]) < 2022 else "late")

                if any([not onWall(aw) for aw in sesh.visited_away_wells]):
                    homeAvgDwell = sesh.avg_dwell_time(True, sesh.home_well)
                    homeAvgDwell90 = sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                    homeCurvature = sesh.avg_curvature_at_well(True, sesh.home_well)
                    homeCurvature90 = sesh.avg_curvature_at_well(
                        True, sesh.home_well, timeInterval=[0, 90])
                    homeNumEntries = sesh.num_well_entries(True, sesh.home_well)
                    homeGravity = sesh.gravityOfWell(True, sesh.home_well)
                    homeGravityFromOffWall = sesh.gravityOfWell(
                        True, sesh.home_well, fromWells=offWallWellNames)

                    awayAvgDwell = np.nanmean([sesh.avg_dwell_time(True, aw)
                                              for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayAvgDwell90 = np.nanmean([sesh.avg_dwell_time(True, aw, timeInterval=[
                                                0, 90]) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayCurvature = np.nanmean([sesh.avg_curvature_at_well(
                        True, aw) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayCurvature90 = np.nanmean([sesh.avg_curvature_at_well(True, aw, timeInterval=[
                                                 0, 90]) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayNumEntries = np.nanmean([sesh.num_well_entries(True, aw)
                                                for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayGravity = np.nanmean([sesh.gravityOfWell(True, aw)
                                             for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayGravityFromOffWall = np.nanmean(
                        [sesh.gravityOfWell(True, aw) for aw in sesh.visited_away_wells if not onWall(aw)])

                    avgDwellDifference.append(homeAvgDwell - awayAvgDwell)
                    avgDwellDifference90.append(homeAvgDwell90 - awayAvgDwell90)
                    curvatureDifference.append(homeCurvature - awayCurvature)
                    curvatureDifference90.append(homeCurvature90 - awayCurvature90)
                    numEntriesDifference.append(homeNumEntries - awayNumEntries)
                    gravityDifference.append(homeGravity - awayGravity)
                    gravityFromOffWallDifference.append(
                        homeGravityFromOffWall - awayGravityFromOffWall)
                else:
                    print("sesh {}, rat never found any off wall away wells")
                    avgDwellDifference.append(np.nan)
                    avgDwellDifference90.append(np.nan)
                    curvatureDifference.append(np.nan)
                    curvatureDifference90.append(np.nan)
                    numEntriesDifference.append(np.nan)
                    gravityDifference.append(np.nan)
                    gravityFromOffWallDifference.append(np.nan)

            wellCat = np.array(wellCat)
            seshCat = np.array(seshCat)
            # seshConditionGroup = np.array(seshConditionGroup)
            avgDwell = np.array(avgDwell)
            avgDwell90 = np.array(avgDwell90)
            curvature = np.array(curvature)
            curvature90 = np.array(curvature90)
            numEntries = np.array(numEntries)
            seshDateCat = np.array(seshDateCat)
            gravity = np.array(gravity)
            gravityFromOffWall = np.array(gravityFromOffWall)

            avgDwellDifference = np.array(avgDwellDifference)
            avgDwellDifference90 = np.array(avgDwellDifference90)
            curvatureDifference = np.array(curvatureDifference)
            curvatureDifference90 = np.array(curvatureDifference90)
            numEntriesDifference = np.array(numEntriesDifference)
            seshCatBySession = np.array(seshCatBySession)
            seshDateCatBySession = np.array(seshDateCatBySession)
            gravityDifference = np.array(gravityDifference)
            gravityFromOffWallDifference = np.array(gravityFromOffWallDifference)

            earlyIdx = seshDateCat == "early"
            lateIdx = seshDateCat == "late"
            earlyIdxBySession = seshDateCatBySession == "early"
            lateIdxBySession = seshDateCatBySession == "late"

        if not MAKE_BASIC_MEASURE_PLOTS:
            print("warning: skipping basic measures plots")
        else:
            # all
            with pp.newFig("probe_avgdwell_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg dwell (s)", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_ofwall"] = avgDwell
                cats["well"] = wellCat
                cats["condition"] = seshCat
                # info["conditionGroup"] = seshConditionGroup
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_avgdwell_90sec_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell90, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg dwell (s), 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_90sec_ofwall"] = avgDwell90
                cats["well"] = wellCat
                cats["condition"] = seshCat
                # info["conditionGroup"] = seshConditionGroup
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg curvature", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_offwall"] = curvature
                cats["well"] = wellCat
                cats["condition"] = seshCat
                # info["conditionGroup"] = seshConditionGroup
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_90sec_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature90, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg curvature, 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_90sec_offwall"] = curvature90
                cats["well"] = wellCat
                cats["condition"] = seshCat
                # info["conditionGroup"] = seshConditionGroup
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_numentries_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntries, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "num entries", "Well Type"], violin=True, doStats=False)

                yvals["probe_num_entries_offwall"] = numEntries
                cats["well"] = wellCat
                cats["condition"] = seshCat
                # info["conditionGroup"] = seshConditionGroup
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

        if not MAKE_EARLY_LATE_SESSION_BASIC_MEASURES:
            print("warning, skipping early/late plots")
        elif len(sessionsEarly) == 0 or len(sessionsLate) == 0:
            print("warning: no early or late sessions, skipping early/late plots")
        else:
            # early
            with pp.newFig("probe_avgdwell_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell[earlyIdx], categories=seshCat[earlyIdx], categories2=wellCat[earlyIdx],
                        axesNames=["Condition", "avg dwell (s)", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_ofwall"] = avgDwell[earlyIdx]
                cats["well"] = wellCat[earlyIdx]
                cats["condition"] = seshCat[earlyIdx]
                # info["conditionGroup"] = seshConditionGroup[earlyIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_avgdwell_90sec_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell90[earlyIdx], categories=seshCat[earlyIdx], categories2=wellCat[earlyIdx],
                        axesNames=["Condition", "avg dwell (s), 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_90sec_ofwall"] = avgDwell90[earlyIdx]
                cats["well"] = wellCat[earlyIdx]
                cats["condition"] = seshCat[earlyIdx]
                # info["conditionGroup"] = seshConditionGroup[earlyIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature[earlyIdx], categories=seshCat[earlyIdx], categories2=wellCat[earlyIdx],
                        axesNames=["Condition", "avg curvature", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_offwall"] = curvature[earlyIdx]
                cats["well"] = wellCat[earlyIdx]
                cats["condition"] = seshCat[earlyIdx]
                # info["conditionGroup"] = seshConditionGroup[earlyIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_90sec_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature90[earlyIdx], categories=seshCat[earlyIdx], categories2=wellCat[earlyIdx],
                        axesNames=["Condition", "avg curvature, 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_90sec_offwall"] = curvature90[earlyIdx]
                cats["well"] = wellCat[earlyIdx]
                cats["condition"] = seshCat[earlyIdx]
                # info["conditionGroup"] = seshConditionGroup[earlyIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_numentries_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntries[earlyIdx], categories=seshCat[earlyIdx], categories2=wellCat[earlyIdx],
                        axesNames=["Condition", "num entries", "Well Type"], violin=True, doStats=False)

                yvals["probe_num_entries_offwall"] = numEntries[earlyIdx]
                cats["well"] = wellCat[earlyIdx]
                cats["condition"] = seshCat[earlyIdx]
                # info["conditionGroup"] = seshConditionGroup[earlyIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_avgdwell_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference[earlyIdxBySession], categories=seshDateCatBySession[earlyIdxBySession], axesNames=[
                        "Condition", "Avg Dwell Difference"], violin=True, doStats=False)
                yvals["probe_avgdwell_difference_offwall"] = avgDwellDifference[earlyIdxBySession]
                cats["condition"] = seshDateCatBySession[earlyIdxBySession]

            with pp.newFig("probe_avgdwell90_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference90[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=[
                        "Condition", "Avg Dwell Difference 90sec"], violin=True, doStats=False)
                yvals["probe_avgdwell90_difference_offwall"] = avgDwellDifference[earlyIdxBySession]
                cats["condition"] = seshCatBySession[earlyIdxBySession]

            with pp.newFig("probe_curvature_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=[
                        "Condition", "Curvature Difference"], violin=True, doStats=False)
                yvals["probe_curvature_difference_offwall"] = curvatureDifference[earlyIdxBySession]
                cats["condition"] = seshCatBySession[earlyIdxBySession]

            with pp.newFig("probe_curvature90_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference90[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=[
                        "Condition", "Curvature Difference 90sec"], violin=True, doStats=False)
                yvals["probe_curvature90_difference_offwall"] = curvatureDifference[earlyIdxBySession]
                cats["condition"] = seshCatBySession[earlyIdxBySession]

            with pp.newFig("probe_numEntries_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntriesDifference[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=[
                        "Condition", "Num Entries Difference"], violin=True, doStats=False)
                yvals["probe_numEntries_difference_offwall"] = numEntriesDifference[earlyIdxBySession]
                cats["condition"] = seshCatBySession[earlyIdxBySession]

            # late
            with pp.newFig("probe_avgdwell_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell[lateIdx], categories=seshCat[lateIdx], categories2=wellCat[lateIdx],
                        axesNames=["Condition", "avg dwell (s)", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_ofwall"] = avgDwell[lateIdx]
                cats["well"] = wellCat[lateIdx]
                cats["condition"] = seshCat[lateIdx]
                # info["conditionGroup"] = seshConditionGroup[lateIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_avgdwell_90sec_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell90[lateIdx], categories=seshCat[lateIdx], categories2=wellCat[lateIdx],
                        axesNames=["Condition", "avg dwell (s), 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_90sec_ofwall"] = avgDwell90[lateIdx]
                cats["well"] = wellCat[lateIdx]
                cats["condition"] = seshCat[lateIdx]
                # info["conditionGroup"] = seshConditionGroup[lateIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature[lateIdx], categories=seshCat[lateIdx], categories2=wellCat[lateIdx],
                        axesNames=["Condition", "avg curvature", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_offwall"] = curvature[lateIdx]
                cats["well"] = wellCat[lateIdx]
                cats["condition"] = seshCat[lateIdx]
                # info["conditionGroup"] = seshConditionGroup[lateIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_90sec_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature90[lateIdx], categories=seshCat[lateIdx], categories2=wellCat[lateIdx],
                        axesNames=["Condition", "avg curvature, 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_90sec_offwall"] = curvature90[lateIdx]
                cats["well"] = wellCat[lateIdx]
                cats["condition"] = seshCat[lateIdx]
                # info["conditionGroup"] = seshConditionGroup[lateIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_numentries_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntries[lateIdx], categories=seshCat[lateIdx], categories2=wellCat[lateIdx],
                        axesNames=["Condition", "num entries", "Well Type"], violin=True, doStats=False)

                yvals["probe_num_entries_offwall"] = numEntries[lateIdx]
                cats["well"] = wellCat[lateIdx]
                cats["condition"] = seshCat[lateIdx]
                # info["conditionGroup"] = seshConditionGroup[lateIdx]
                # pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_avgdwell_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference[lateIdxBySession], categories=seshDateCatBySession[lateIdxBySession], axesNames=[
                        "Condition", "Avg Dwell Difference"], violin=True, doStats=False)
                yvals["probe_avgdwell_difference_offwall"] = avgDwellDifference[lateIdxBySession]
                cats["condition"] = seshDateCatBySession[lateIdxBySession]

            with pp.newFig("probe_avgdwell90_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference90[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=[
                        "Condition", "Avg Dwell Difference 90sec"], violin=True, doStats=False)
                yvals["probe_avgdwell90_difference_offwall"] = avgDwellDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]

            with pp.newFig("probe_curvature_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=[
                        "Condition", "Curvature Difference"], violin=True, doStats=False)
                yvals["probe_curvature_difference_offwall"] = curvatureDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]

            with pp.newFig("probe_curvature90_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference90[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=[
                        "Condition", "Curvature Difference 90sec"], violin=True, doStats=False)
                yvals["probe_curvature90_difference_offwall"] = curvatureDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]

            with pp.newFig("probe_numEntries_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntriesDifference[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=[
                        "Condition", "Num Entries Difference"], violin=True, doStats=False)
                yvals["probe_numEntries_difference_offwall"] = numEntriesDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]

        if not MAKE_WITHIN_SESSION_MEASURE_PLOTS:
            print("warning: skipping within session measure plots")
        else:
            with pp.newFig("probe_avgdwell_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference, categories=seshCatBySession, axesNames=[
                        "Condition", "Avg Dwell Difference"], violin=True, doStats=False)
                yvals["probe_avgdwell_difference_offwall"] = avgDwellDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_avgdwell90_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference90, categories=seshCatBySession, axesNames=[
                        "Condition", "Avg Dwell Difference 90sec"], violin=True, doStats=False)
                yvals["probe_avgdwell90_difference_offwall"] = avgDwellDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_curvature_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference, categories=seshCatBySession, axesNames=[
                        "Condition", "Curvature Difference"], violin=True, doStats=False)
                yvals["probe_curvature_difference_offwall"] = curvatureDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_curvature90_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference90, categories=seshCatBySession, axesNames=[
                        "Condition", "Curvature Difference 90sec"], violin=True, doStats=False)
                yvals["probe_curvature90_difference_offwall"] = curvatureDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_numEntries_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntriesDifference, categories=seshCatBySession, axesNames=[
                        "Condition", "Num Entries Difference"], violin=True, doStats=False)
                yvals["probe_numEntries_difference_offwall"] = numEntriesDifference
                cats["condition"] = seshCatBySession

        if not MAKE_GRAVITY_PLOTS:
            print("warning: skipping gravity plots")
        else:
            with pp.newFig("probe_gravity_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=gravity, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "gravity", "Well Type"], violin=True, doStats=False)
                yvals["probe_gravity_offwall"] = gravity
                cats["well"] = wellCat
                cats["condition"] = seshCat

            with pp.newFig("probe_gravityFromOffWall_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=gravityFromOffWall, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "gravityFromOffWall", "Well Type"], violin=True, doStats=False)
                yvals["probe_gravityFromOffWall_offwall"] = gravityFromOffWall
                cats["well"] = wellCat
                cats["condition"] = seshCat

            with pp.newFig("probe_gravity_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=gravityDifference, categories=seshCatBySession, axesNames=[
                        "Condition", "Gravity Difference"], violin=True, doStats=False)
                yvals["probe_gravity_difference_offwall"] = gravityDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_gravityFromOffWall_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=gravityFromOffWallDifference, categories=seshCatBySession, axesNames=[
                        "Condition", "gravityFromOffWall Difference"], violin=True, doStats=False)
                yvals["probe_gravityFromOffWall_difference_offwall"] = gravityFromOffWallDifference
                cats["condition"] = seshCatBySession
        if not MAKE_PROBE_TRACES_FIGS:
            print("warning: skipping raw probe trace plots")
        else:
            numCols = math.ceil(math.sqrt(numSessionsWithProbe))
            with pp.newFig("allProbeTraces", subPlots=(numCols, numCols), figScale=0.3, priority=10) as axs:
                for si, sesh in enumerate(sessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                    # ax.set_title(str(si))
                for si in range(numSessionsWithProbe, numCols*numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

            numCols = math.ceil(math.sqrt(nCtrlWithProbe))
            with pp.newFig("allProbeTraces_ctrl", subPlots=(numCols, numCols), figScale=0.3, priority=10) as axs:
                for si, sesh in enumerate(ctrlSessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    # ax.set_title(str(si))
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                for si in range(nCtrlWithProbe, numCols*numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

            numCols = math.ceil(math.sqrt(nSWRWithProbe))
            with pp.newFig("allProbeTraces_SWR", subPlots=(numCols, numCols), figScale=0.3, priority=10) as axs:
                for si, sesh in enumerate(swrSessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    # ax.set_title(str(si))
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                for si in range(nSWRWithProbe, numCols*numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

        if not MAKE_TASK_BEHAVIOR_PLOTS:
            print("warning: skipping task behavior plots")
        else:
            sessionIsInterruption = np.zeros((numSessions,)).astype(bool)
            for i in range(numSessions):
                if sessions[i].isRippleInterruption:
                    sessionIsInterruption[i] = True

            homeFindLatencies = np.empty((numSessions, 10))
            homeFindLatencies[:] = np.nan
            for si, sesh in enumerate(sessions):
                t1 = np.array(sesh.home_well_find_times)
                t0 = np.array(np.hstack(([sesh.bt_pos_ts[0]], sesh.away_well_leave_times)))
                if not sesh.ended_on_home:
                    t0 = t0[0:-1]
                times = (t1 - t0) / BTSession.TRODES_SAMPLING_RATE
                homeFindLatencies[si, 0:sesh.num_home_found] = times

            awayFindLatencies = np.empty((numSessions, 10))
            awayFindLatencies[:] = np.nan
            for si, sesh in enumerate(sessions):
                t1 = np.array(sesh.away_well_find_times)
                t0 = np.array(sesh.home_well_leave_times)
                if sesh.ended_on_home:
                    t0 = t0[0:-1]
                times = (t1 - t0) / BTSession.TRODES_SAMPLING_RATE
                awayFindLatencies[si, 0:sesh.num_away_found] = times

            ymax = 100
            swrHomeFindLatencies = homeFindLatencies[sessionIsInterruption, :]
            ctrlHomeFindLatencies = homeFindLatencies[np.logical_not(sessionIsInterruption), :]
            swrAwayFindLatencies = awayFindLatencies[sessionIsInterruption, :]
            ctrlAwayFindLatencies = awayFindLatencies[np.logical_not(sessionIsInterruption), :]
            pltx = np.arange(10)+1

            with pp.newFig("task_latency_to_home_by_condition", priority=5) as ax:
                plotIndividualAndAverage(ax, swrHomeFindLatencies, pltx,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, ctrlHomeFindLatencies, pltx,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_ylim(0, ymax)
                ax.set_xlim(1, 10)
                ax.set_xticks(np.arange(0, 10, 2) + 1)

            with pp.newFig("task_latency_to_away_by_condition", priority=5) as ax:
                plotIndividualAndAverage(ax, swrAwayFindLatencies, pltx,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, ctrlAwayFindLatencies, pltx,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_ylim(0, ymax)
                ax.set_xlim(1, 10)
                ax.set_xticks(np.arange(0, 10, 2) + 1)

            numAwaysFound = np.array([sesh.num_away_found for sesh in sessions])
            condition = np.array([("SWR" if sesh.isRippleInterruption else "Ctrl")
                                 for sesh in sessions])
            dateCat = np.array([("early" if int(sesh.date_str[0:4]) < 2022 else "late")
                               for sesh in sessions])
            earlyIdxAllSessions = dateCat == "early"
            lateIdxAllSessions = dateCat == "late"
            with pp.newFig("task_num_aways_found", priority=5) as ax:
                boxPlot(ax, yvals=numAwaysFound, categories=condition,
                        axesNames=["Condition", "num aways found"], violin=True, doStats=False)

            if len(sessionsEarly) > 0 and len(sessionsLate) > 0:
                with pp.newFig("task_num_aways_found_early", priority=5) as ax:
                    boxPlot(ax, yvals=numAwaysFound[earlyIdxAllSessions], categories=condition[earlyIdxAllSessions],
                            axesNames=["Condition", "num aways found"], violin=True, doStats=False)

                with pp.newFig("task_num_aways_found_late", priority=5) as ax:
                    boxPlot(ax, yvals=numAwaysFound[lateIdxAllSessions], categories=condition[lateIdxAllSessions],
                            axesNames=["Condition", "num aways found"], violin=True, doStats=False)

        if not MAKE_PROBE_BEHAVIOR_PLOTS:
            print("warning: skipping probe behavior plots")
        else:
            # =============
            # average vel in probe
            windowSize = 30
            windowsSlide = 6
            t0Array = np.arange(0, 60*5-windowSize+windowsSlide, windowsSlide)

            avgVels = np.empty((numSessionsWithProbe, len(t0Array)))
            avgVels[:] = np.nan
            fracExplores = np.empty((numSessionsWithProbe, len(t0Array)))
            fracExplores[:] = np.nan

            for si, sesh in enumerate(sessionsWithProbe):
                for ti, t0 in enumerate(t0Array):
                    avgVels[si, ti] = sesh.mean_vel(True, timeInterval=[t0, t0+windowSize])
                    fracExplores[si, ti] = sesh.prop_time_in_bout_state(
                        True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[t0, t0+windowSize])

            with pp.newFig("probe_avg_vel", priority=5) as ax:
                plotIndividualAndAverage(ax, avgVels, t0Array)
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_avg_vel_by_cond", priority=5) as ax:
                plotIndividualAndAverage(ax, avgVels[sessionWithProbeIsInterruption, :], t0Array,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, avgVels[~ sessionWithProbeIsInterruption, :], t0Array,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_frac_explore", priority=5) as ax:
                plotIndividualAndAverage(ax, fracExplores, t0Array)
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_frac_explore_by_cond", priority=5) as ax:
                plotIndividualAndAverage(ax, fracExplores[sessionWithProbeIsInterruption, :], t0Array,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, fracExplores[~ sessionWithProbeIsInterruption, :], t0Array,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            # =============
            # probe num wells visited by condition
            numVisited = [numWellsVisited(s.probe_nearest_wells) for s in sessionsWithProbe]
            numVisitedOffWall = [numWellsVisited(
                s.probe_nearest_wells, wellSubset=offWallWellNames) for s in sessionsWithProbe]
            with pp.newFig("probe_num_wells_visited_by_condition", priority=5) as ax:
                boxPlot(ax, numVisited, seshCatBySession, axesNames=[
                        "Condition", "Number of wells visited in probe"], doStats=False)
            with pp.newFig("probe_num_wells_visited_offwall_by_condition", priority=5) as ax:
                boxPlot(ax, numVisited, seshCatBySession, axesNames=[
                        "Condition", "Number of wells visited in probe"], doStats=False)

            windowSlide = 5
            t1Array = np.arange(windowSlide, 60*5, windowSlide)
            numVisitedOverTime = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTime[:] = np.nan
            numVisitedOverTimeOffWall = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOverTimeOffWall[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * BTSession.TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisitedOverTime[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False)
                    numVisitedOverTimeOffWall[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False,
                        wellSubset=offWallWellNames)

            with pp.newFig("probe_cumulative_num_wells_visited", priority=5) as ax:
                plotIndividualAndAverage(ax, numVisitedOverTime, t1Array)
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_cumulative_num_wells_visited_by_condition", priority=5) as ax:
                plotIndividualAndAverage(ax, numVisitedOverTime[sessionWithProbeIsInterruption, :], t1Array,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, numVisitedOverTime[~ sessionWithProbeIsInterruption, :], t1Array,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_cumulative_num_wells_visited_offwall", priority=5) as ax:
                plotIndividualAndAverage(ax, numVisitedOverTimeOffWall, t1Array)
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_cumulative_num_wells_visited_by_condition_offwall", priority=5) as ax:
                plotIndividualAndAverage(ax, numVisitedOverTimeOffWall[sessionWithProbeIsInterruption, :], t1Array,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, numVisitedOverTimeOffWall[~ sessionWithProbeIsInterruption, :], t1Array,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))

        if not MAKE_RAW_LFP_POWER_PLOTS:
            print("warning: skipping raw LFP power plots")
        else:
            firstSeshToView = 25
            for seshi, sesh in enumerate(sessionsWithProbe):
                if seshi < firstSeshToView:
                    continue

                # load raw LFP on detection and baseline tetrodes
                lfpFName = sesh.bt_lfp_fnames[-1]
                print("LFP data from file {}".format(lfpFName))
                lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
                lfpV = lfpData[1]['voltage']
                lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

                lfpData = MountainViewIO.loadLFP(data_file=sesh.bt_lfp_baseline_fname)
                baselfpV = lfpData[1]['voltage']
                baselfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

                # get ripple power for both
                btLFPData = lfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
                btRipPower, btRipZPower, _, _ = getRipplePower(
                    btLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs, meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower)
                probeLFPData = lfpV[sesh.probeLfpStart_idx:sesh.probeLfpEnd_idx]
                probeRipPower, _, _, _ = getRipplePower(probeLFPData, omit_artifacts=False)

                baselineProbeLFPData = baselfpV[sesh.probeLfpStart_idx:sesh.probeLfpEnd_idx]
                probeBaselinePower, _, baselineProbeMeanRipplePower, baselineProbeStdRipplePower = getRipplePower(
                    baselineProbeLFPData, omit_artifacts=False)
                btBaselineLFPData = baselfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
                btBaselineRipplePower, _, _, _ = getRipplePower(
                    btBaselineLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs, meanPower=baselineProbeMeanRipplePower, stdPower=baselineProbeStdRipplePower)

                probeRawPowerDiff = probeRipPower - probeBaselinePower
                zmean = np.nanmean(probeRawPowerDiff)
                zstd = np.nanstd(probeRawPowerDiff)

                rawPowerDiff = btRipPower - btBaselineRipplePower
                zPowerDiff = (rawPowerDiff - zmean) / zstd

                # Go through places detected by either and look at raw lfp, power by both measures
                # session.btWithBaseRipStartIdx, session.btWithBaseRipLens, session.btWithBaseRipPeakIdx, session.btWithBaseRipPeakAmps, session.btWithBaseRipCrossThreshIdxs = detectRipples(
                # session.btRipStartIdxsProbeStats, session.btRipLensProbeStats, session.btRipPeakIdxsProbeStats, session.btRipPeakAmpsProbeStats, session.btRipCrossThreshIdxsProbeStats = \

                ripStarts = np.array([], dtype=int)
                ripLens = np.array([], dtype=int)
                ripPeakIdxs = np.array([], dtype=int)
                ripPeakAmps = np.array([])
                crossThreshIdxs = np.array([], dtype=int)
                source = np.array([])
                sourceIdx = np.array([], dtype=int)
                sourceNumPts = {}

                INCLUDE_RAW_POWER_DETECTIONS = True
                INCLUDE_ZDIFF_POWER_DETECTIONS = True
                INCLUDE_DEFLECTIONS = False
                if INCLUDE_DEFLECTIONS:
                    l = len(sesh.bt_lfp_artifact_idxs)
                    ripStarts = np.concatenate((ripStarts, sesh.bt_lfp_artifact_idxs))
                    ripLens = np.concatenate((ripLens, 3*np.ones((l,)).astype(int)))
                    ripPeakIdxs = np.concatenate((ripPeakIdxs, sesh.bt_lfp_artifact_idxs + 2))
                    ripPeakAmps = np.concatenate((ripPeakAmps, np.ones((l,))))
                    crossThreshIdxs = np.concatenate(
                        (crossThreshIdxs, sesh.bt_lfp_artifact_idxs + 1))
                    source = np.concatenate((source, np.array(["deflection"] * l)))
                    sourceIdx = np.concatenate((sourceIdx, np.arange(l)))
                    sourceNumPts["deflection"] = l
                if INCLUDE_RAW_POWER_DETECTIONS:
                    l = len(sesh.btRipStartIdxsProbeStats)
                    ripStarts = np.concatenate((ripStarts, sesh.btRipStartIdxsProbeStats))
                    ripLens = np.concatenate((ripLens, sesh.btRipLensProbeStats))
                    ripPeakIdxs = np.concatenate((ripPeakIdxs, sesh.btRipPeakIdxsProbeStats))
                    ripPeakAmps = np.concatenate((ripPeakAmps, sesh.btRipPeakAmpsProbeStats))
                    crossThreshIdxs = np.concatenate(
                        (crossThreshIdxs, sesh.btRipCrossThreshIdxsProbeStats))
                    source = np.concatenate((source, np.array(["rawz"] * l)))
                    sourceIdx = np.concatenate((sourceIdx, np.arange(l)))
                    sourceNumPts["rawz"] = l
                if INCLUDE_ZDIFF_POWER_DETECTIONS:
                    l = len(sesh.btWithBaseRipStartIdx)
                    ripStarts = np.concatenate((ripStarts, sesh.btWithBaseRipStartIdx))
                    ripLens = np.concatenate((ripLens, sesh.btWithBaseRipLens))
                    ripPeakIdxs = np.concatenate((ripPeakIdxs, sesh.btWithBaseRipPeakIdx))
                    ripPeakAmps = np.concatenate((ripPeakAmps, sesh.btWithBaseRipPeakAmps))
                    crossThreshIdxs = np.concatenate(
                        (crossThreshIdxs, sesh.btWithBaseRipCrossThreshIdxs))
                    source = np.concatenate((source, np.array(["zdiff"] * l)))
                    sourceIdx = np.concatenate((sourceIdx, np.arange(l)))
                    sourceNumPts["zdiff"] = l

                # ripStarts = np.concatenate(
                #     (sesh.btWithBaseRipStartIdx, sesh.btRipStartIdxsProbeStats))
                sortInd = ripStarts.argsort()
                ripStarts = ripStarts[sortInd]
                ripLens = ripLens[sortInd]
                ripPeakIdxs = ripPeakIdxs[sortInd]
                ripPeakAmps = ripPeakAmps[sortInd]
                crossThreshIdxs = crossThreshIdxs[sortInd]
                source = source[sortInd]
                sourceIdx = sourceIdx[sortInd]
                # ripLens = np.concatenate(
                #     (sesh.btWithBaseRipLens, sesh.btRipLensProbeStats))[sortInd]
                # ripPeakIdxs = np.concatenate(
                #     (sesh.btWithBaseRipPeakIdx, sesh.btRipPeakIdxsProbeStats))[sortInd]
                # ripPeakAmps = np.concatenate(
                #     (sesh.btWithBaseRipPeakAmps, sesh.btRipPeakAmpsProbeStats))[sortInd]
                # crossThreshIdxs = np.concatenate(
                #     (sesh.btWithBaseRipCrossThreshIdxs, sesh.btRipCrossThreshIdxsProbeStats))[sortInd]

                MARGIN_SEC = 0.25
                MARGIN_PTS = int(MARGIN_SEC * 1500)
                btT = lfpT[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]

                for i in range(len(ripStarts)):
                    ripStart = ripStarts[i]
                    ripLen = ripLens[i]
                    ripPeakIdx = ripPeakIdxs[i]
                    ripPeakAmp = ripPeakAmps[i]
                    crossThreshIdx = crossThreshIdxs[i]
                    print(ripStart, ripLen, ripPeakIdx, ripPeakAmp, crossThreshIdx)

                    i0 = max(0, ripStart - MARGIN_PTS)
                    i1 = min(ripStart + ripLen + MARGIN_PTS, len(btT)-1)
                    rawX = btT[i0:i1]
                    rawY = np.diff(btLFPData[i0:i1] / 70.0, prepend=0)

                    rawBaseY = btBaselineLFPData[i0:i1] / 70.0

                    rect = Rectangle((btT[ripStart], min(rawY)), btT[ripStart+ripLen] -
                                     btT[ripStart], max(rawY)-min(rawY))
                    pc = PatchCollection([rect], facecolor="grey", alpha=0.5)

                    crossX = btT[crossThreshIdx]
                    crossY = ripPeakAmp
                    peakX = btT[ripPeakIdx]
                    peakY = ripPeakAmp

                    with pp.newFig("rawLFP_{}".format(i), priority=10, showPlot=True, savePlot=False, figScale=2) as ax:
                        ax.add_collection(pc)
                        ax.plot(rawX, rawY, label="raw detection LFP", linewidth=1.5)
                        ax.plot(rawX, rawBaseY, label="raw baseline LFP", linewidth=1.5)
                        ax.plot(rawX, zPowerDiff[i0:i1], label="diff z pow")
                        ax.plot(rawX, btRipZPower[i0:i1], label="detection tet zpow")
                        ax.plot([crossX, peakX], [crossY, peakY],
                                linewidth=3, label="Cross to peak")

                        # handles, labels = ax.get_legend_handles_labels()
                        # ax.legend(handles[1:3], labels[1:3], fontsize=6).set_zorder(2)
                        ax.legend()
                        ax.set_title(
                            "{}: {}/{}".format(source[i], sourceIdx[i], sourceNumPts[source[i]]))

        if not MAKE_CUMULATIVE_LFP_PLOTS:
            print("warning: skipping LFP cumulative rate plots")
        else:
            windowSize = 5
            # note these will have a big tail of nans that I could chop off if it becomes an issue
            taskL = 60*30 // windowSize
            itiL = 60 * 5 // windowSize
            probeL = 60 * 10 // windowSize
            stimCounts = np.empty((numSessions, taskL))
            rippleCounts = np.empty((numSessions, taskL))
            rippleCountsProbeStats = np.empty((numSessions, taskL))
            rippleCountsWithBaseline = np.empty((numSessions, taskL))
            itiStimCounts = np.empty((numSessions, itiL))
            itiRippleCounts = np.empty((numSessions, itiL))
            probeStimCounts = np.empty((numSessions, probeL))
            probeRippleCounts = np.empty((numSessions, probeL))
            for si, sesh in enumerate(sessions):
                # Task
                fillCounts(stimCounts[si, :], sesh.interruption_timestamps,
                           sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1], windowSize)

                fillCounts(rippleCounts[si, :], sesh.btRipStartTimestampsPreStats,
                           sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1], windowSize)

                if not sesh.probe_performed:
                    itiStimCounts[si, :] = np.nan
                    itiRippleCounts[si, :] = np.nan
                    probeStimCounts[si, :] = np.nan
                    probeRippleCounts[si, :] = np.nan
                    rippleCountsProbeStats[si, :] = np.nan
                    rippleCountsWithBaseline[si, :] = np.nan
                else:
                    # ITI
                    fillCounts(itiStimCounts[si, :], sesh.interruption_timestamps,
                               sesh.itiLfpStart_ts, sesh.itiLfpEnd_ts, windowSize)
                    fillCounts(itiRippleCounts[si, :], sesh.ITIRipStartTimestamps,
                               sesh.itiLfpStart_ts, sesh.itiLfpEnd_ts, windowSize)

                    # Probe
                    fillCounts(probeStimCounts[si, :], sesh.interruption_timestamps,
                               sesh.probeLfpStart_ts, sesh.probeLfpEnd_ts, windowSize)
                    fillCounts(probeRippleCounts[si, :], sesh.probeRipStartTimestamps,
                               sesh.probeLfpStart_ts, sesh.probeLfpEnd_ts, windowSize)

                    # Task (other stats)
                    if sesh.bt_lfp_baseline_fname is not None:
                        fillCounts(rippleCountsWithBaseline[si, :], sesh.btWithBaseRipStartTimestamps,
                                   sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1], windowSize)
                    else:
                        rippleCountsWithBaseline[si, :] = np.nan

                    fillCounts(rippleCountsProbeStats[si, :], sesh.btRipStartTimestampsProbeStats,
                               sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1], windowSize)

            stimCounts = np.cumsum(stimCounts, axis=1)
            rippleCounts = np.cumsum(rippleCounts, axis=1)
            itiStimCounts = np.cumsum(itiStimCounts, axis=1)
            itiRippleCounts = np.cumsum(itiRippleCounts, axis=1)
            probeStimCounts = np.cumsum(probeStimCounts, axis=1)
            probeRippleCounts = np.cumsum(probeRippleCounts, axis=1)
            rippleCountsProbeStats = np.cumsum(rippleCountsProbeStats, axis=1)
            rippleCountsWithBaseline = np.cumsum(rippleCountsWithBaseline, axis=1)

            swrProbeIdx = [s.probe_performed and s.isRippleInterruption for s in sessions]
            ctrlProbeIdx = [s.probe_performed and not s.isRippleInterruption for s in sessions]
            swrNoProbeIdx = [(not s.probe_performed) and s.isRippleInterruption for s in sessions]
            ctrlNoProbeIdx = [(not s.probe_performed)
                              and not s.isRippleInterruption for s in sessions]

            taskX = np.arange(windowSize, taskL*windowSize+windowSize/2, windowSize)
            itiX = np.arange(windowSize, itiL*windowSize+windowSize/2, windowSize)
            probeX = np.arange(windowSize, probeL*windowSize+windowSize/2, windowSize)

            with pp.newFig("lfp_task_cumStimCounts") as ax:
                ax.plot(taskX, stimCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(taskX, stimCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(taskX, stimCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(taskX, stimCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_task_cumRippleCounts_withBaseline") as ax:
                ax.plot(taskX, rippleCountsWithBaseline[swrProbeIdx, :].T, color="orange")
                ax.plot(taskX, rippleCountsWithBaseline[swrNoProbeIdx, :].T,
                        '--', color="orange")
                ax.plot(taskX, rippleCountsWithBaseline[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(taskX, rippleCountsWithBaseline[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_task_cumRippleCounts_probestats") as ax:
                ax.plot(taskX, rippleCountsProbeStats[swrProbeIdx, :].T, color="orange")
                ax.plot(taskX, rippleCountsProbeStats[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(taskX, rippleCountsProbeStats[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(taskX, rippleCountsProbeStats[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_task_cumRippleCounts") as ax:
                ax.plot(taskX, rippleCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(taskX, rippleCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(taskX, rippleCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(taskX, rippleCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_iti_cumStimCounts") as ax:
                ax.plot(itiX, itiStimCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(itiX, itiStimCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(itiX, itiStimCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(itiX, itiStimCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_iti_cumRippleCounts") as ax:
                ax.plot(itiX, itiRippleCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(itiX, itiRippleCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(itiX, itiRippleCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(itiX, itiRippleCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_probe_cumStimCounts") as ax:
                ax.plot(probeX, probeStimCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(probeX, probeStimCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(probeX, probeStimCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(probeX, probeStimCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_probe_cumRippleCounts") as ax:
                ax.plot(probeX, probeRippleCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(probeX, probeRippleCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(probeX, probeRippleCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(probeX, probeRippleCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")


if __name__ == "__main__":
    makeFigures()
