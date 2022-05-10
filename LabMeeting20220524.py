import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot, conditionShuffle
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
import sys
from UtilFunctions import getRipplePower, numWellsVisited, onWall, getInfoForAnimal, detectRipples, findDataDir, parseCmdLineAnimalNames
from consts import TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE, allWellNames, offWallWellNames
import math
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
import time

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
# Is exportLFP output referenced?

def makeFigures():
    MAKE_EARLY_LATE_SESSION_BASIC_MEASURES = False
    MAKE_BASIC_MEASURE_PLOTS = False
    MAKE_WITHIN_SESSION_MEASURE_PLOTS = False
    MAKE_GRAVITY_PLOTS = False
    MAKE_RAW_LFP_POWER_PLOTS = False
    MAKE_PROBE_TRACES_FIGS = False
    MAKE_TASK_BEHAVIOR_PLOTS = False
    MAKE_PROBE_BEHAVIOR_PLOTS = True

    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "20220524_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)
    pp = PlotCtx(outputDir=globalOutputDir, randomSeed=rseed)

    animalNames = parseCmdLineAnimalNames(default=["B13", "B14", "Martin"])
    allSessionsByRat = {}
    for animalName in animalNames:
        animalInfo = getInfoForAnimal(animalName)
        dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
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
        sessionWithProbeIsInterruption = np.zeros((numSessionsWithProbe,)).astype(bool)
        for i in range(numSessionsWithProbe):
            if sessionsWithProbe[i].isRippleInterruption:
                sessionWithProbeIsInterruption[i] = True


        pp.setOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)
    
        # Here be plots
        if False and not (MAKE_EARLY_LATE_SESSION_BASIC_MEASURES or MAKE_BASIC_MEASURE_PLOTS or MAKE_WITHIN_SESSION_MEASURE_PLOTS or MAKE_GRAVITY_PLOTS):
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
                gravityFromOffWall.append(sesh.gravityOfWell(True, sesh.home_well, fromWells=offWallWellNames))

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
                    gravityFromOffWall.append(sesh.gravityOfWell(True, aw, fromWells=offWallWellNames))

                    wellCat.append("away")
                    seshCat.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                    # seshConditionGroup.append(sesh.conditionGroup)
                    seshDateCat.append("early" if int(sesh.date_str[0:4]) < 2022 else "late")

                if any([not onWall(aw) for aw in sesh.visited_away_wells]):
                    homeAvgDwell = sesh.avg_dwell_time(True, sesh.home_well)
                    homeAvgDwell90 = sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                    homeCurvature = sesh.avg_curvature_at_well(True, sesh.home_well)
                    homeCurvature90 = sesh.avg_curvature_at_well(True, sesh.home_well, timeInterval=[0, 90])
                    homeNumEntries = sesh.num_well_entries(True, sesh.home_well)
                    homeGravity = sesh.gravityOfWell(True, sesh.home_well)
                    homeGravityFromOffWall = sesh.gravityOfWell(True, sesh.home_well, fromWells=offWallWellNames)

                    awayAvgDwell = np.nanmean([sesh.avg_dwell_time(True, aw) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayAvgDwell90 = np.nanmean([sesh.avg_dwell_time(True, aw, timeInterval=[0, 90]) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayCurvature = np.nanmean([sesh.avg_curvature_at_well(True, aw) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayCurvature90 = np.nanmean([sesh.avg_curvature_at_well(True, aw, timeInterval=[0, 90]) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayNumEntries = np.nanmean([sesh.num_well_entries(True, aw) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayGravity = np.nanmean([sesh.gravityOfWell(True, aw) for aw in sesh.visited_away_wells if not onWall(aw)])
                    awayGravityFromOffWall = np.nanmean([sesh.gravityOfWell(True, aw) for aw in sesh.visited_away_wells if not onWall(aw)])

                    avgDwellDifference.append(homeAvgDwell - awayAvgDwell)
                    avgDwellDifference90.append(homeAvgDwell90 - awayAvgDwell90)
                    curvatureDifference.append(homeCurvature - awayCurvature)
                    curvatureDifference90.append(homeCurvature90 - awayCurvature90)
                    numEntriesDifference.append(homeNumEntries - awayNumEntries)
                    gravityDifference.append(homeGravity - awayGravity)
                    gravityFromOffWallDifference.append(homeGravityFromOffWall - awayGravityFromOffWall)
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
                boxPlot(ax, yvals=avgDwellDifference[earlyIdxBySession], categories=seshDateCatBySession[earlyIdxBySession], axesNames=["Condition", "Avg Dwell Difference"], violin=True, doStats=False)
                yvals["probe_avgdwell_difference_offwall"] = avgDwellDifference[earlyIdxBySession]
                cats["condition"] = seshDateCatBySession[earlyIdxBySession]

            with pp.newFig("probe_avgdwell90_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference90[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=["Condition", "Avg Dwell Difference 90sec"], violin=True, doStats=False)
                yvals["probe_avgdwell90_difference_offwall"] = avgDwellDifference[earlyIdxBySession]
                cats["condition"] = seshCatBySession[earlyIdxBySession]

            with pp.newFig("probe_curvature_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=["Condition", "Curvature Difference"], violin=True, doStats=False)
                yvals["probe_curvature_difference_offwall"] = curvatureDifference[earlyIdxBySession]
                cats["condition"] = seshCatBySession[earlyIdxBySession]

            with pp.newFig("probe_curvature90_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference90[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=["Condition", "Curvature Difference 90sec"], violin=True, doStats=False)
                yvals["probe_curvature90_difference_offwall"] = curvatureDifference[earlyIdxBySession]
                cats["condition"] = seshCatBySession[earlyIdxBySession]

            with pp.newFig("probe_numEntries_difference_offwall_early", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntriesDifference[earlyIdxBySession], categories=seshCatBySession[earlyIdxBySession], axesNames=["Condition", "Num Entries Difference"], violin=True, doStats=False)
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
                boxPlot(ax, yvals=avgDwellDifference[lateIdxBySession], categories=seshDateCatBySession[lateIdxBySession], axesNames=["Condition", "Avg Dwell Difference"], violin=True, doStats=False)
                yvals["probe_avgdwell_difference_offwall"] = avgDwellDifference[lateIdxBySession]
                cats["condition"] = seshDateCatBySession[lateIdxBySession]

            with pp.newFig("probe_avgdwell90_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference90[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=["Condition", "Avg Dwell Difference 90sec"], violin=True, doStats=False)
                yvals["probe_avgdwell90_difference_offwall"] = avgDwellDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]

            with pp.newFig("probe_curvature_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=["Condition", "Curvature Difference"], violin=True, doStats=False)
                yvals["probe_curvature_difference_offwall"] = curvatureDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]

            with pp.newFig("probe_curvature90_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference90[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=["Condition", "Curvature Difference 90sec"], violin=True, doStats=False)
                yvals["probe_curvature90_difference_offwall"] = curvatureDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]

            with pp.newFig("probe_numEntries_difference_offwall_late", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntriesDifference[lateIdxBySession], categories=seshCatBySession[lateIdxBySession], axesNames=["Condition", "Num Entries Difference"], violin=True, doStats=False)
                yvals["probe_numEntries_difference_offwall"] = numEntriesDifference[lateIdxBySession]
                cats["condition"] = seshCatBySession[lateIdxBySession]


        if not MAKE_WITHIN_SESSION_MEASURE_PLOTS:
            print("warning: skipping within session measure plots")
        else:
            with pp.newFig("probe_avgdwell_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference, categories=seshCatBySession, axesNames=["Condition", "Avg Dwell Difference"], violin=True, doStats=False)
                yvals["probe_avgdwell_difference_offwall"] = avgDwellDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_avgdwell90_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellDifference90, categories=seshCatBySession, axesNames=["Condition", "Avg Dwell Difference 90sec"], violin=True, doStats=False)
                yvals["probe_avgdwell90_difference_offwall"] = avgDwellDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_curvature_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference, categories=seshCatBySession, axesNames=["Condition", "Curvature Difference"], violin=True, doStats=False)
                yvals["probe_curvature_difference_offwall"] = curvatureDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_curvature90_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureDifference90, categories=seshCatBySession, axesNames=["Condition", "Curvature Difference 90sec"], violin=True, doStats=False)
                yvals["probe_curvature90_difference_offwall"] = curvatureDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_numEntries_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=numEntriesDifference, categories=seshCatBySession, axesNames=["Condition", "Num Entries Difference"], violin=True, doStats=False)
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
                boxPlot(ax, yvals=gravityDifference, categories=seshCatBySession, axesNames=["Condition", "Gravity Difference"], violin=True, doStats=False)
                yvals["probe_gravity_difference_offwall"] = gravityDifference
                cats["condition"] = seshCatBySession

            with pp.newFig("probe_gravityFromOffWall_difference_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=gravityFromOffWallDifference, categories=seshCatBySession, axesNames=["Condition", "gravityFromOffWall Difference"], violin=True, doStats=False)
                yvals["probe_gravityFromOffWall_difference_offwall"] = gravityFromOffWallDifference
                cats["condition"] = seshCatBySession

        if not MAKE_RAW_LFP_POWER_PLOTS:
            print("warning: skipping raw LFP power plots")
        else:
            pass

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
                    ax.set_title(sesh.name, fontdict={'fontsize':8})
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
                    ax.set_title(sesh.name, fontdict={'fontsize':8})
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
                    ax.set_title(sesh.name, fontdict={'fontsize':8})
                for si in range(nSWRWithProbe, numCols*numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

        if not MAKE_TASK_BEHAVIOR_PLOTS:
            print("warning: skipping task behavior plots")
        else:
            numSessions = len(sessions)
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
            condition = np.array([("SWR" if sesh.isRippleInterruption else "Ctrl") for sesh in sessions])
            dateCat = np.array([("early" if int(sesh.date_str[0:4]) < 2022 else "late") for sesh in sessions])
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
                plotIndividualAndAverage(ax, avgVels[sessionWithProbeIsInterruption,:], t0Array,
                                        individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, avgVels[~ sessionWithProbeIsInterruption, :], t0Array,
                                        individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_frac_explore", priority=5) as ax:
                plotIndividualAndAverage(ax, fracExplores, t0Array)
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_frac_explore_by_cond", priority=5) as ax:
                plotIndividualAndAverage(ax, fracExplores[sessionWithProbeIsInterruption,:], t0Array,
                                        individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, fracExplores[~ sessionWithProbeIsInterruption, :], t0Array,
                                        individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            # =============
            # probe num wells visited by condition
            numVisited = [numWellsVisited(s.probe_nearest_wells) for s in sessionsWithProbe]
            numVisitedOffWall = [numWellsVisited(s.probe_nearest_wells, wellSubset=offWallWellNames) for s in sessionsWithProbe]
            with pp.newFig("probe_num_wells_visited_by_condition", priority=5) as ax:
                boxPlot(ax, numVisited, seshCatBySession, axesNames=["Condition", "Number of wells visited in probe"], doStats=False)
            with pp.newFig("probe_num_wells_visited_offwall_by_condition", priority=5) as ax:
                boxPlot(ax, numVisited, seshCatBySession, axesNames=["Condition", "Number of wells visited in probe"], doStats=False)

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
                plotIndividualAndAverage(ax, numVisitedOverTime[sessionWithProbeIsInterruption,:], t1Array,
                                        individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, numVisitedOverTime[~ sessionWithProbeIsInterruption, :], t1Array,
                                        individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_cumulative_num_wells_visited_offwall", priority=5) as ax:
                plotIndividualAndAverage(ax, numVisitedOverTimeOffWall, t1Array)
                ax.set_xticks(np.arange(0, 60*5+1, 60))

            with pp.newFig("probe_cumulative_num_wells_visited_by_condition_offwall", priority=5) as ax:
                plotIndividualAndAverage(ax, numVisitedOverTimeOffWall[sessionWithProbeIsInterruption,:], t1Array,
                                        individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, numVisitedOverTimeOffWall[~ sessionWithProbeIsInterruption, :], t1Array,
                                        individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_xticks(np.arange(0, 60*5+1, 60))






if __name__ == "__main__":
    makeFigures()

