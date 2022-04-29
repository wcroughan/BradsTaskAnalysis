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
# Look at just old data and just newer data. Why did the effect go away? Washed out or reveresed?
# Look at home v away difference and other measures within session
# Look at away-from-home measures like where on the wall he was, orientation to home, dir crossed with dir to home
# Look at gravity of home well
# 
# LFP:
# Look at raw LFP and ripple power using different detection criteria (baseline, referenced, etc). What is difference?
# Is exportLFP output referenced?

def makeFigures():
    MAKE_EARLY_LATE_SESSION_BASIC_MEASURES = True
    MAKE_BASIC_MEASURE_PLOTS = True

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

        pp.setOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)
    
        # Here be plots
        if not (MAKE_EARLY_LATE_SESSION_BASIC_MEASURES or MAKE_BASIC_MEASURE_PLOTS):
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

            for si, sesh in enumerate(sessionsWithProbe):
                towt = np.sum([sesh.total_dwell_time(True, w)
                               for w in offWallWellNames])

                avgDwell.append(sesh.avg_dwell_time(True, sesh.home_well))
                avgDwell90.append(sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90]))
                curvature.append(sesh.avg_curvature_at_well(True, sesh.home_well))
                curvature90.append(sesh.avg_curvature_at_well(
                    True, sesh.home_well, timeInterval=[0, 90]))
                numEntries.append(sesh.num_well_entries(True, sesh.home_well))

                wellCat.append("home")
                seshCat.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                # seshConditionGroup.append(sesh.conditionGroup)
                seshDateCat.append("early" if int(sesh.date_str[0:4]) < 2022 else "late")

                for aw in sesh.visited_away_wells:
                    if onWall(aw):
                        continue

                    avgDwell.append(sesh.avg_dwell_time(True, aw))
                    avgDwell90.append(sesh.avg_dwell_time(True, aw, timeInterval=[0, 90]))
                    curvature.append(sesh.avg_curvature_at_well(True, aw))
                    curvature90.append(sesh.avg_curvature_at_well(True, aw, timeInterval=[0, 90]))
                    numEntries.append(sesh.num_well_entries(True, aw))

                    wellCat.append("away")
                    seshCat.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                    # seshConditionGroup.append(sesh.conditionGroup)
                    seshDateCat.append("early" if int(sesh.date_str[0:4]) < 2022 else "late")

            wellCat = np.array(wellCat)
            seshCat = np.array(seshCat)
            # seshConditionGroup = np.array(seshConditionGroup)
            avgDwell = np.array(avgDwell)
            avgDwell90 = np.array(avgDwell90)
            curvature = np.array(curvature)
            curvature90 = np.array(curvature90)
            numEntries = np.array(numEntries)
            seshDateCat = np.array(seshDateCat)

            earlyIdx = seshDateCat == "early"
            lateIdx = seshDateCat == "late"

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





if __name__ == "__main__":
    makeFigures()

