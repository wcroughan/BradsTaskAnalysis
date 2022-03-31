import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot, conditionShuffle
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
import sys
from UtilFunctions import getRipplePower, numWellsVisited, onWall, weekIdxForDateStr, getInfoForAnimal, detectRipples
from consts import TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE, allWellNames, offWallWellNames
import math
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt

# NOTES AFTER MEETING:
# STATS:
# Can change "global" to "main" to be in line with usual stats nomenclature
#
# BEHAVIOR:
# Look at interaction effect instead of just home
# Even better, for each session define a variable to be the different b/w home measure and all the away measures for that session
#
# Some data points can be gathered far from home: orientation toward home, nearness to home along wall, direction vector crossed with dir to home
#
# Should look at gravity of home well (how often when passing by is it visited)
#
# LFP:
# Size of ripples all small that are making it through detection in SWR condition?
#
# DBL check LFP output from exportLFP is referenced or not? Also is the data activelink gets referenced or not?
#
# Baseline is one way to weed out noise, but most offline detection uses other ways like min/max duration. Am I looking at those? If not, do.
#
# Ripple power stim artifact might carry over past 100ms refrac period?
#
# Just plot LFP and power with different criteria and check it out by eye. See where the differences are
#
# Run same stats just on the data that was presented during thesis committee
#
# GENERAL:
# Treat this data set as a data-mining expedition since have lots of rats coming. Want to establish hypotheses here, look at all the variations, then test them rigorously on future rats


def makeFigures():
    MAKE_RAW_LFP_FIGS = False
    RUN_RIPPLE_DETECTION_COMPARISON = False
    RUN_RIPPLE_REFRAC_PERIOD_CHECK = False
    RUN_RIPPLE_BASELINE_TET_ANALYSIS = False
    MAKE_PROBE_EXPLORATION_OVER_TIME_FIGS = False
    MAKE_PROBE_TRACES_FIGS = False
    MAKE_PROBE_OFFWALL_TIME_FIGS = True
    MAKE_NORMAL_PERSEV_MEASURES_FIGS = False
    MAKE_PERSEV_WITH_EXPLORATION_CUTOFF_FIGS = False
    MAKE_CUMULATIVE_LFP_FIGS = False
    MAKE_TASK_LATENCIES_FIGS = False
    RUN_SHUFFLES = True

    possibleDataDirs = ["/media/WDC6/", "/media/fosterlab/WDC6/", "/home/wcroughan/data/"]
    dataDir = None
    for dd in possibleDataDirs:
        if os.path.exists(dd):
            dataDir = dd
            break

    if dataDir == None:
        print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
        exit()

    globalOutputDir = os.path.join(dataDir, "figures", "20220315_labmeeting")

    if len(sys.argv) >= 2:
        animalNames = sys.argv[1:]
    else:
        animalNames = ['B13', 'B14', 'Martin']
    print("Plotting data for animals ", animalNames)

    allSessionsByRat = {}
    allSessionsWithProbeByRat = {}

    for an in animalNames:
        print("loading {}".format(an))
        animalInfo = getInfoForAnimal(an)
        # print(animalInfo)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        # if an == "B13":
        #     dataFilename = os.path.join(dataDir, "B13/processed_data/B13_bradtask.dat")
        # elif an == "B14":
        #     dataFilename = os.path.join(dataDir, "B14/processed_data/B14_bradtask.dat")
        # elif an == "Martin":
        #     dataFilename = os.path.join(dataDir, "Martin/processed_data/martin_bradtask.dat")
        # else:
        #     raise Exception("Unknown rat " + an)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[an] = ratData.getSessions()
        allSessionsWithProbeByRat[an] = ratData.getSessions(lambda s: s.probe_performed)

    pp = PlotCtx(globalOutputDir, priorityLevel=None, randomSeed=94702)
    for ratName in animalNames:
        print("===============================\nRunning rat", ratName)
        pp.setOutputSubDir(ratName)
        sessions = allSessionsByRat[ratName]
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probe_performed]
        ctrlSessionsWithProbe = [sesh for sesh in sessions if (
            not sesh.isRippleInterruption) and sesh.probe_performed]
        swrSessionsWithProbe = [
            sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probe_performed]
        numSessionsWithProbe = len(sessionsWithProbe)
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)
        numSessions = len(sessions)
        nCtrlNoProbe = np.count_nonzero(
            [(not s.probe_performed) and (not s.isRippleInterruption) for s in sessions])
        nSwrNoProbe = np.count_nonzero(
            [(not s.probe_performed) and s.isRippleInterruption for s in sessions])

        print("N: {}\n\twith probe: {} ({} ctrl, {} swr)\n\twithout probe: {} ({} ctrl, {} swr)".format(numSessions, numSessionsWithProbe, nCtrlWithProbe, nSWRWithProbe,
                                                                                                        numSessions - numSessionsWithProbe, nCtrlNoProbe, nSwrNoProbe))
        pp.writeToInfoFile("N: {}\n\twith probe: {} ({} ctrl, {} swr)\n\twithout probe: {} ({} ctrl, {} swr)".format(numSessions, numSessionsWithProbe, nCtrlWithProbe, nSWRWithProbe,
                                                                                                                     numSessions - numSessionsWithProbe, nCtrlNoProbe, nSwrNoProbe))

        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if MAKE_TASK_LATENCIES_FIGS:
            homeFindTimes = np.empty((numSessions, 10))
            homeFindTimes[:] = np.nan
            for si, sesh in enumerate(sessions):
                t1 = np.array(sesh.home_well_find_times)
                t0 = np.array(np.hstack(([sesh.bt_pos_ts[0]], sesh.away_well_leave_times)))
                if not sesh.ended_on_home:
                    t0 = t0[0:-1]
                times = (t1 - t0) / TRODES_SAMPLING_RATE
                homeFindTimes[si, 0:sesh.num_home_found] = times

            awayFindTimes = np.empty((numSessions, 10))
            awayFindTimes[:] = np.nan
            for si, sesh in enumerate(sessions):
                t1 = np.array(sesh.away_well_find_times)
                t0 = np.array(sesh.home_well_leave_times)
                if sesh.ended_on_home:
                    t0 = t0[0:-1]
                times = (t1 - t0) / TRODES_SAMPLING_RATE
                awayFindTimes[si, 0:sesh.num_away_found] = times

            swrIdx = np.array([sesh.isRippleInterruption for sesh in sessions])
            swrHomeFindTimes = homeFindTimes[swrIdx]
            ctrlHomeFindTimes = homeFindTimes[~swrIdx]
            swrAwayFindTimes = awayFindTimes[swrIdx]
            ctrlAwayFindTimes = awayFindTimes[~swrIdx]

            with pp.newFig("task_well_latencies_home") as ax:
                pltx = np.arange(10)+1
                plotIndividualAndAverage(ax, swrHomeFindTimes, pltx,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, ctrlHomeFindTimes, pltx,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_ylim(0, 300)
                ax.set_xlim(1, 10)
                ax.set_xticks(np.arange(0, 10, 2) + 1)

            with pp.newFig("task_well_latencies_away") as ax:
                pltx = np.arange(10)+1
                plotIndividualAndAverage(ax, swrAwayFindTimes, pltx,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(ax, ctrlAwayFindTimes, pltx,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                ax.set_ylim(0, 300)
                ax.set_xlim(1, 10)
                ax.set_xticks(np.arange(0, 10, 2) + 1)

        if MAKE_NORMAL_PERSEV_MEASURES_FIGS or MAKE_PERSEV_WITH_EXPLORATION_CUTOFF_FIGS or MAKE_PROBE_OFFWALL_TIME_FIGS:
            wellCat = []
            seshCat = []
            seshConditionGroup = []
            avgDwell = []
            avgDwell90 = []
            curvature = []
            curvature90 = []
            seshIdx = []
            propWellTime = []
            totalOffWallTime = []

            for si, sesh in enumerate(sessionsWithProbe):
                towt = np.sum([sesh.total_dwell_time(True, w)
                               for w in offWallWellNames])

                avgDwell.append(sesh.avg_dwell_time(True, sesh.home_well))
                avgDwell90.append(sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90]))
                curvature.append(sesh.avg_curvature_at_well(True, sesh.home_well))
                curvature90.append(sesh.avg_curvature_at_well(
                    True, sesh.home_well, timeInterval=[0, 90]))
                propWellTime.append(sesh.total_dwell_time(True, sesh.home_well) / towt)

                wellCat.append("home")
                seshCat.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                seshConditionGroup.append(sesh.conditionGroup)
                seshIdx.append(si)

                totalOffWallTime.append(towt)

                for aw in sesh.visited_away_wells:
                    if onWall(aw):
                        continue

                    avgDwell.append(sesh.avg_dwell_time(True, aw))
                    avgDwell90.append(sesh.avg_dwell_time(True, aw, timeInterval=[0, 90]))
                    curvature.append(sesh.avg_curvature_at_well(True, aw))
                    curvature90.append(sesh.avg_curvature_at_well(True, aw, timeInterval=[0, 90]))
                    propWellTime.append(sesh.total_dwell_time(True, aw) / towt)

                    wellCat.append("away")
                    seshCat.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                    seshConditionGroup.append(sesh.conditionGroup)
                    seshIdx.append(si)

            wellCat = np.array(wellCat)
            seshCat = np.array(seshCat)
            seshConditionGroup = np.array(seshConditionGroup)
            seshIdx = np.array(seshIdx)
            avgDwell = np.array(avgDwell)
            avgDwell90 = np.array(avgDwell90)
            curvature = np.array(curvature)
            curvature90 = np.array(curvature90)
            propWellTime = np.array(propWellTime)
            totalOffWallTime = np.array(totalOffWallTime)

        if MAKE_NORMAL_PERSEV_MEASURES_FIGS:
            with pp.newFig("probe_avgdwell_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg dwell (s)", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_ofwall"] = avgDwell
                cats["well"] = wellCat
                cats["condition"] = seshCat
                info["conditionGroup"] = seshConditionGroup
                pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_avgdwell_90sec_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell90, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg dwell (s), 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_90sec_ofwall"] = avgDwell90
                cats["well"] = wellCat
                cats["condition"] = seshCat
                info["conditionGroup"] = seshConditionGroup
                pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg curvature", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_offwall"] = curvature
                cats["well"] = wellCat
                cats["condition"] = seshCat
                info["conditionGroup"] = seshConditionGroup
                pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_90sec_offwall", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature90, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "avg curvature, 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_90sec_offwall"] = curvature90
                cats["well"] = wellCat
                cats["condition"] = seshCat
                info["conditionGroup"] = seshConditionGroup
                pp.setCustomShuffleFunction("condition", conditionShuffle)

        if MAKE_PROBE_TRACES_FIGS:
            numCols = math.ceil(math.sqrt(numSessionsWithProbe))
            with pp.newFig("allProbeTraces", subPlots=(numCols, numCols), figScale=0.3, priority=10) as axs:
                for si, sesh in enumerate(sessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    # ax.set_title(sesh.name)
                    ax.set_title(str(si))
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
                    ax.set_title(str(si))
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
                    ax.set_title(str(si))
                for si in range(nSWRWithProbe, numCols*numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

        if MAKE_PROBE_EXPLORATION_OVER_TIME_FIGS or MAKE_PERSEV_WITH_EXPLORATION_CUTOFF_FIGS or MAKE_PROBE_OFFWALL_TIME_FIGS:
            windowSlide = 5
            t1Array = np.arange(windowSlide, 60*5, windowSlide)
            numVisited = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisited[:] = np.nan
            numVisitedOffWall = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisitedOffWall[:] = np.nan
            weeks = np.empty((numSessionsWithProbe,), dtype=int)
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisited[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False)
                    numVisitedOffWall[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False,
                        wellSubset=[w for w in allWellNames if not onWall(w)])

                datestr = sesh.name.split("_")[0]
                if "test" in datestr:
                    datestr = datestr[4:]
                weeks[si] = weekIdxForDateStr(datestr)

            weeks = weeks - np.min(weeks)
            numVisitedAUC = np.mean(numVisited, axis=1) / 36.0
            numVisitedOffWallAUC = np.mean(numVisitedOffWall, axis=1) / 16.0

        if MAKE_PROBE_EXPLORATION_OVER_TIME_FIGS:
            with pp.newFig("probe_numVisitedOverTime_bysidx", priority=10) as ax:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisited.shape[1],)) * 0.5
                    ax.plot(t1Array, numVisited[si, :] + jitter, color=cmap(si))
                ax.set_xticks(np.arange(0, 60*5+1, 60))
                ax.set_ylim(0, 37)

            with pp.newFig("probe_numVisitedOverTime_bysidx_offwall", priority=10) as ax:
                cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisitedOffWall.shape[1],)) * 0.5
                    ax.plot(t1Array, numVisitedOffWall[si, :] + jitter, color=cmap(si))
                ax.set_xticks(np.arange(0, 60*5+1, 60))
                ax.set_ylim(0, 17)

            with pp.newFig("probe_numVisitedOverTime_byweek_offwall", priority=10) as ax:
                cmap = cm.get_cmap("coolwarm", np.max(weeks))
                for si in range(numSessionsWithProbe):
                    jitter = np.random.uniform(size=(numVisitedOffWall.shape[1],)) * 0.5
                    ax.plot(t1Array, numVisitedOffWall[si, :] + jitter, color=cmap(weeks[si]))
                ax.set_xticks(np.arange(0, 60*5+1, 60))
                ax.set_ylim(0, 17)

            numWeekGroups = np.count_nonzero(np.diff(weeks) >= 2) + 1
            if numWeekGroups > 1:
                with pp.newFig("probe_numVisitedOverTime_byweek_offwall_separate", priority=10, subPlots=(1, numWeekGroups)) as axs:
                    cmap = cm.get_cmap("coolwarm", np.max(weeks))
                    for si in range(numSessionsWithProbe):
                        jitter = np.random.uniform(size=(numVisitedOffWall.shape[1],)) * 0.5
                        wgi = np.count_nonzero(np.diff(weeks[:si]) >= 2)
                        ax = axs[wgi]
                        ax.plot(t1Array, numVisitedOffWall[si, :] + jitter, color=cmap(weeks[si]))
                    for wgi in range(numWeekGroups):
                        ax = axs[wgi]
                        ax.set_xticks(np.arange(0, 60*5+1, 60))
                        ax.set_ylim(0, 17)

            with pp.newFig("probe_numVisitedOverTime_AUC_bysidx", priority=8) as ax:
                ax.plot(numVisitedAUC)
                ax.scatter(x=np.arange(len(numVisitedAUC)), y=numVisitedAUC)
                bigSplits = np.nonzero(np.diff(weeks) >= 2)
                for bs in bigSplits:
                    ax.plot([bs+0.5, bs+0.5], [0, 1], color="red")
                ax.set_ylim(0, 1)

            with pp.newFig("probe_numVisitedOverTime_AUC_bysidx_offwall", priority=8) as ax:
                ax.plot(numVisitedOffWallAUC)
                ax.scatter(x=np.arange(len(numVisitedOffWallAUC)), y=numVisitedOffWallAUC)
                bigSplits = np.nonzero(np.diff(weeks) >= 2)
                for bs in bigSplits:
                    ax.plot([bs+0.5, bs+0.5], [0, 1], color="red")
                ax.set_ylim(0, 1)

            with pp.newFig("probe_numVisitedOverTime_AUC_bysidx_offwall_cutoff", priority=8) as ax:
                ax.plot(numVisitedOffWallAUC)
                ax.scatter(x=np.arange(len(numVisitedOffWallAUC)), y=numVisitedOffWallAUC)
                bigSplits = np.nonzero(np.diff(weeks) >= 2)
                for bs in bigSplits:
                    ax.plot([bs+0.5, bs+0.5], [0, 1], color="red")
                ax.set_ylim(0, 1)
                ax.plot([0, len(numVisitedOffWallAUC)], [0.1, 0.1])

        if MAKE_PERSEV_WITH_EXPLORATION_CUTOFF_FIGS or MAKE_PROBE_OFFWALL_TIME_FIGS:
            cutoff = 0.1
            seshMadeCutoff = numVisitedOffWallAUC > cutoff
            if np.count_nonzero(seshMadeCutoff) > 0:
                seshIdxMadeCutoff = seshMadeCutoff[seshIdx]
                seshCatCut = seshCat[seshIdxMadeCutoff]
                wellCatCut = wellCat[seshIdxMadeCutoff]
                seshConditionGroupCut = seshConditionGroup[seshIdxMadeCutoff]
                avgDwellCut = avgDwell[seshIdxMadeCutoff]
                avgDwell90Cut = avgDwell90[seshIdxMadeCutoff]
                curvatureCut = curvature[seshIdxMadeCutoff]
                curvature90Cut = curvature90[seshIdxMadeCutoff]
                propWellTimeCut = propWellTime[seshIdxMadeCutoff]

                totalOffWallTimeCut = totalOffWallTime[seshMadeCutoff]

                anyMadeAUCCutoff = True
            else:
                anyMadeAUCCutoff = False

        if MAKE_PERSEV_WITH_EXPLORATION_CUTOFF_FIGS and anyMadeAUCCutoff:
            with pp.newFig("probe_avgdwell_90sec_offwall_auccutoff", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwell90Cut, categories=seshCatCut, categories2=wellCatCut,
                        axesNames=["Condition", "avg dwell (s), 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_90sec_ofwall_auccutoff"] = avgDwell90Cut
                cats["well"] = wellCatCut
                cats["condition"] = seshCatCut
                info["conditionGroup"] = seshConditionGroupCut
                pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_avgdwell_offwall_auccutoff", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=avgDwellCut, categories=seshCatCut, categories2=wellCatCut,
                        axesNames=["Condition", "avg dwell (s)", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 30)

                yvals["probe_avgdwell_ofwall_auccutoff"] = avgDwellCut
                cats["well"] = wellCatCut
                cats["condition"] = seshCatCut
                info["conditionGroup"] = seshConditionGroupCut
                pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_offwall_auccutoff", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvatureCut, categories=seshCatCut, categories2=wellCatCut,
                        axesNames=["Condition", "avg curvature", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_offwall_auccutoff"] = curvatureCut
                cats["well"] = wellCatCut
                cats["condition"] = seshCatCut
                info["conditionGroup"] = seshConditionGroupCut
                pp.setCustomShuffleFunction("condition", conditionShuffle)

            with pp.newFig("probe_curvature_90sec_offwall_auccutoff", priority=5, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=curvature90Cut, categories=seshCatCut, categories2=wellCatCut,
                        axesNames=["Condition", "avg curvature, 90sec", "Well Type"], violin=True, doStats=False)
                ax.set_ylim(0, 3)

                yvals["probe_curvature_90sec_offwall_auccutoff"] = curvature90Cut
                cats["well"] = wellCatCut
                cats["condition"] = seshCatCut
                info["conditionGroup"] = seshConditionGroupCut
                pp.setCustomShuffleFunction("condition", conditionShuffle)

        if MAKE_PROBE_OFFWALL_TIME_FIGS:
            with pp.newFig("probe_totalOffWallTime_bysidx", priority=10) as ax:
                x = np.arange(numSessionsWithProbe)
                swrWithProbeIdx = np.array(
                    [sesh.isRippleInterruption for sesh in sessionsWithProbe])
                ax.scatter(x[swrWithProbeIdx], totalOffWallTime[swrWithProbeIdx], color="orange")
                ax.scatter(x[~swrWithProbeIdx], totalOffWallTime[~swrWithProbeIdx], color="cyan")

            with pp.newFig("probe_propOffWallTimeNearWell", priority=10, withStats=True) as (ax, yvals, cats, info):
                boxPlot(ax, yvals=propWellTime, categories=seshCat, categories2=wellCat,
                        axesNames=["Condition", "frac off-wall time at well", "Well Type"], violin=True, doStats=False)

                yvals["probe_propOffWallTimeNearWell"] = propWellTime
                cats["well"] = wellCat
                cats["condition"] = seshCat
                info["conditionGroup"] = seshConditionGroup
                pp.setCustomShuffleFunction("condition", conditionShuffle)
                ax.set_ylim(0, 1)

            if anyMadeAUCCutoff:
                with pp.newFig("probe_propOffWallTimeNearWell_auccutoff", priority=8, withStats=True) as (ax, yvals, cats, info):
                    boxPlot(ax, yvals=propWellTimeCut, categories=seshCatCut, categories2=wellCatCut,
                            axesNames=["Condition", "frac off-wall time at well", "Well Type"], violin=True, doStats=False)

                    yvals["probe_propOffWallTimeNearWell_auccutoff"] = propWellTimeCut
                    cats["well"] = wellCatCut
                    cats["condition"] = seshCatCut
                    info["conditionGroup"] = seshConditionGroupCut
                    pp.setCustomShuffleFunction("condition", conditionShuffle)
                    ax.set_ylim(0, 1)

        if RUN_RIPPLE_DETECTION_COMPARISON:
            preBTThreshs = np.array([sesh.prebtMeanRipplePower + 3.0 *
                                    sesh.prebtStdRipplePower for sesh in sessionsWithProbe])
            ITIThreshs = np.array([sesh.ITIMeanRipplePower + 3.0 *
                                   sesh.ITIStdRipplePower for sesh in sessionsWithProbe])
            probeThreshs = np.array([sesh.probeMeanRipplePower + 3.0 *
                                    sesh.probeStdRipplePower for sesh in sessionsWithProbe])
            preBTNAThreshs = np.array([sesh.prebtMeanRipplePowerArtifactsRemoved + 3.0 *
                                       sesh.prebtStdRipplePowerArtifactsRemoved for sesh in sessionsWithProbe])
            preBTNAFHThreshs = np.array([sesh.prebtMeanRipplePowerArtifactsRemovedFirstHalf + 3.0 *
                                        sesh.prebtStdRipplePowerArtifactsRemovedFirstHalf for sesh in sessionsWithProbe])
            # maxval = max(np.max(preBTThreshs), np.max(ITIThreshs),
            #              np.max(probeThreshs), np.max(preBTNAThreshs),
            #              np.max(preBTNAFHThreshs))

            threshArrs = [preBTThreshs, ITIThreshs, probeThreshs, preBTNAThreshs, preBTNAFHThreshs]
            maxval = max(*[np.max(arr) for arr in threshArrs])
            threshArrLabels = ["Pre task", "ITI", "Probe",
                               "Pre task, no artifacts", "Pre task first half, no artifacts"]

            with pp.newFig("rippleThreshComparison", subPlots=(1, 3)) as axs:
                axs[0].scatter(preBTThreshs, ITIThreshs)
                axs[0].plot([0, maxval], [0, maxval])
                axs[0].set_xlabel("preBT")
                axs[0].set_ylabel("ITI")
                axs[1].scatter(preBTThreshs, probeThreshs)
                axs[1].plot([0, maxval], [0, maxval])
                axs[1].set_xlabel("preBT")
                axs[1].set_ylabel("probe")
                axs[2].scatter(ITIThreshs, probeThreshs)
                axs[2].plot([0, maxval], [0, maxval])
                axs[2].set_xlabel("ITI")
                axs[2].set_ylabel("probe")

            with pp.newFig("rippleThreshComparison_preBT", subPlots=(1, 3)) as axs:
                axs[0].scatter(preBTThreshs, preBTNAThreshs)
                axs[0].plot([0, maxval], [0, maxval])
                axs[0].set_xlabel("preBT")
                axs[0].set_ylabel("preBT Artifacts Removed")
                axs[1].scatter(preBTThreshs, preBTNAFHThreshs)
                axs[1].plot([0, maxval], [0, maxval])
                axs[1].set_xlabel("preBT")
                axs[1].set_ylabel("preBT first half Artifacts Removed")
                axs[2].scatter(preBTNAThreshs, preBTNAFHThreshs)
                axs[2].plot([0, maxval], [0, maxval])
                axs[2].set_xlabel("preBT Artifacts Removed")
                axs[2].set_ylabel("preBT first half Artifacts Removed")

            with pp.newFig("rippleThreshComparison_all", subPlots=(5, 5)) as axs:
                for row in range(5):
                    yvals = threshArrs[row]
                    for col in range(row, 5):
                        xvals = threshArrs[col]
                        ax = axs[row, col]
                        # axi = 5*row + col
                        ax.scatter(xvals, yvals)
                        ax.plot([0, maxval], [0, maxval])

                        if row == 0:
                            ax.set_title(threshArrLabels[col])
                        if col == 0:
                            axs[row, 0].set_ylabel(threshArrLabels[row])

        if MAKE_CUMULATIVE_LFP_FIGS:
            windowSize = 5
            bins = np.arange(0, 60*20+1, windowSize)
            stimCounts = np.empty((numSessions, len(bins)-1))
            rippleCounts = np.empty((numSessions, len(bins)-1))
            rippleCountsProbeStats = np.empty((numSessions, len(bins)-1))
            rippleCountsWithBaseline = np.empty((numSessions, len(bins)-1))
            itiBins = np.arange(0, 60*1.5+1, windowSize)
            itiStimCounts = np.empty((numSessions, len(itiBins)-1))
            itiRippleCounts = np.empty((numSessions, len(itiBins)-1))
            probeBins = np.arange(0, 60*5+1, windowSize)
            probeStimCounts = np.empty((numSessions, len(probeBins)-1))
            probeRippleCounts = np.empty((numSessions, len(probeBins)-1))
            for si, sesh in enumerate(sessions):
                # Task
                stimTs = np.array(sesh.interruption_timestamps)
                stimTs = stimTs[np.logical_and(stimTs > sesh.bt_pos_ts[0], stimTs <
                                               sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                stimTs /= TRODES_SAMPLING_RATE
                stimCounts[si, :], _ = np.histogram(stimTs, bins=bins)

                for i in reversed(range(len(bins)-1)):
                    if stimCounts[si, i] != 0:
                        break
                    stimCounts[si, i] = np.nan

                ripTs = np.array(sesh.btRipStartTimestampsPreStats)
                ripTs = ripTs[np.logical_and(ripTs > sesh.bt_pos_ts[0], ripTs <
                                             sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                ripTs /= TRODES_SAMPLING_RATE
                rippleCounts[si, :], _ = np.histogram(ripTs, bins=bins)

                for i in reversed(range(len(bins)-1)):
                    if rippleCounts[si, i] != 0:
                        break
                    rippleCounts[si, i] = np.nan

                if not sesh.probe_performed:
                    itiStimCounts[si, :] = np.nan
                    itiRippleCounts[si, :] = np.nan
                    probeStimCounts[si, :] = np.nan
                    probeRippleCounts[si, :] = np.nan
                    rippleCountsProbeStats[si, :] = np.nan
                    rippleCountsWithBaseline[si, :] = np.nan
                else:
                    # ITI
                    stimTs = np.array(sesh.interruption_timestamps)
                    stimTs = stimTs[np.logical_and(stimTs > sesh.itiLfpStart_ts, stimTs <
                                                   sesh.itiLfpEnd_ts)] - sesh.itiLfpStart_ts
                    stimTs /= TRODES_SAMPLING_RATE
                    itiStimCounts[si, :], _ = np.histogram(stimTs, bins=itiBins)

                    for i in reversed(range(len(itiBins)-1)):
                        if itiStimCounts[si, i] != 0:
                            break
                        itiStimCounts[si, i] = np.nan

                    ripTs = np.array(sesh.ITIRipStartTimestamps)
                    ripTs = ripTs[np.logical_and(ripTs > sesh.itiLfpStart_ts, ripTs <
                                                 sesh.itiLfpEnd_ts)] - sesh.itiLfpStart_ts
                    ripTs /= TRODES_SAMPLING_RATE
                    itiRippleCounts[si, :], _ = np.histogram(ripTs, bins=itiBins)

                    for i in reversed(range(len(itiBins)-1)):
                        if itiRippleCounts[si, i] != 0:
                            break
                        itiRippleCounts[si, i] = np.nan

                    # Probe
                    stimTs = np.array(sesh.interruption_timestamps)
                    stimTs = stimTs[np.logical_and(stimTs > sesh.probeLfpStart_ts, stimTs <
                                                   sesh.probeLfpEnd_ts)] - sesh.probeLfpStart_ts
                    stimTs /= TRODES_SAMPLING_RATE
                    probeStimCounts[si, :], _ = np.histogram(stimTs, bins=probeBins)

                    for i in reversed(range(len(probeBins)-1)):
                        if probeStimCounts[si, i] != 0:
                            break
                        probeStimCounts[si, i] = np.nan

                    ripTs = np.array(sesh.probeRipStartTimestamps)
                    ripTs = ripTs[np.logical_and(ripTs > sesh.probeLfpStart_ts, ripTs <
                                                 sesh.probeLfpEnd_ts)] - sesh.probeLfpStart_ts
                    ripTs /= TRODES_SAMPLING_RATE
                    probeRippleCounts[si, :], _ = np.histogram(ripTs, bins=probeBins)

                    for i in reversed(range(len(probeBins)-1)):
                        if probeRippleCounts[si, i] != 0:
                            break
                        probeRippleCounts[si, i] = np.nan

                    # Task (other stats)
                    if sesh.bt_lfp_baseline_fname is not None:
                        ripTs = np.array(sesh.btWithBaseRipStartTimestamps)
                        ripTs = ripTs[np.logical_and(ripTs > sesh.bt_pos_ts[0], ripTs <
                                                     sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                        ripTs /= TRODES_SAMPLING_RATE
                        rippleCountsWithBaseline[si, :], _ = np.histogram(ripTs, bins=bins)

                        for i in reversed(range(len(bins)-1)):
                            if rippleCountsWithBaseline[si, i] != 0:
                                break
                            rippleCountsWithBaseline[si, i] = np.nan
                    else:
                        rippleCountsWithBaseline[si, :] = np.nan

                    ripTs = np.array(sesh.btRipStartTimestampsProbeStats)
                    ripTs = ripTs[np.logical_and(ripTs > sesh.bt_pos_ts[0], ripTs <
                                                 sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                    ripTs /= TRODES_SAMPLING_RATE
                    rippleCountsProbeStats[si, :], _ = np.histogram(ripTs, bins=bins)

                    for i in reversed(range(len(bins)-1)):
                        if rippleCountsProbeStats[si, i] != 0:
                            break
                        rippleCountsProbeStats[si, i] = np.nan

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

            with pp.newFig("lfp_task_cumStimCounts") as ax:
                ax.plot(bins[1:], stimCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(bins[1:], stimCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(bins[1:], stimCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(bins[1:], stimCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_task_cumRippleCounts_withBaseline") as ax:
                ax.plot(bins[1:], rippleCountsWithBaseline[swrProbeIdx, :].T, color="orange")
                ax.plot(bins[1:], rippleCountsWithBaseline[swrNoProbeIdx, :].T,
                        '--', color="orange")
                ax.plot(bins[1:], rippleCountsWithBaseline[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(bins[1:], rippleCountsWithBaseline[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_task_cumRippleCounts_probestats") as ax:
                ax.plot(bins[1:], rippleCountsProbeStats[swrProbeIdx, :].T, color="orange")
                ax.plot(bins[1:], rippleCountsProbeStats[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(bins[1:], rippleCountsProbeStats[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(bins[1:], rippleCountsProbeStats[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_task_cumRippleCounts") as ax:
                ax.plot(bins[1:], rippleCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(bins[1:], rippleCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(bins[1:], rippleCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(bins[1:], rippleCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_iti_cumStimCounts") as ax:
                ax.plot(itiBins[1:], itiStimCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(itiBins[1:], itiStimCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(itiBins[1:], itiStimCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(itiBins[1:], itiStimCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_iti_cumRippleCounts") as ax:
                ax.plot(itiBins[1:], itiRippleCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(itiBins[1:], itiRippleCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(itiBins[1:], itiRippleCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(itiBins[1:], itiRippleCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_probe_cumStimCounts") as ax:
                ax.plot(probeBins[1:], probeStimCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(probeBins[1:], probeStimCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(probeBins[1:], probeStimCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(probeBins[1:], probeStimCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

            with pp.newFig("lfp_probe_cumRippleCounts") as ax:
                ax.plot(probeBins[1:], probeRippleCounts[swrProbeIdx, :].T, color="orange")
                ax.plot(probeBins[1:], probeRippleCounts[swrNoProbeIdx, :].T, '--', color="orange")
                ax.plot(probeBins[1:], probeRippleCounts[ctrlProbeIdx, :].T, color="cyan")
                ax.plot(probeBins[1:], probeRippleCounts[ctrlNoProbeIdx, :].T, '--', color="cyan")

        for sesh in sessionsWithProbe:
            print("Running session", sesh.name)
            pp.setOutputSubDir(ratName + "/" + sesh.name)

            if RUN_RIPPLE_REFRAC_PERIOD_CHECK:
                # interruptionI = 0
                # interruptionIdx = sesh.interruptionIdxs[interruptionI]
                ripCrossIdxs = np.array(sesh.btRipCrossThreshIdxsProbeStats)
                mostRecentInterruption = np.searchsorted(
                    sesh.interruptionIdxs, ripCrossIdxs)
                notTooEarly = mostRecentInterruption > 0
                nteMostRecentInterruption = mostRecentInterruption[notTooEarly] - 1
                nteRipCrossIdxs = ripCrossIdxs[notTooEarly]
                interruptionToRipInterval = (
                    nteRipCrossIdxs - sesh.interruptionIdxs[nteMostRecentInterruption]).astype(float) / float(LFP_SAMPLING_RATE)

                with pp.newFig("interruption_to_ripple_delay") as ax:
                    ax.hist(interruptionToRipInterval, bins=np.linspace(0, 1, 40))

                refracPeriod = 0.1  # 100ms
                numWithinRefrac = np.count_nonzero(interruptionToRipInterval < refracPeriod)
                if len(interruptionToRipInterval) > 0:
                    pp.writeToInfoFile("interruption to rip: numWithinRefracPeriod: {}/{} ripples ({})".format(numWithinRefrac,
                                                                                                               len(interruptionToRipInterval), numWithinRefrac / len(interruptionToRipInterval)))
                    print("interruption to rip: numWithinRefracPeriod: {}/{} ripples ({})".format(numWithinRefrac,
                                                                                                  len(interruptionToRipInterval), numWithinRefrac / len(interruptionToRipInterval)))

                nextInterruption = mostRecentInterruption + 1
                notTooLate = nextInterruption < len(sesh.interruptionIdxs)
                ntlNextInterruption = nextInterruption[notTooLate]
                ntlRipCrossIdxs = ripCrossIdxs[notTooLate]
                ripToInterruptionInterval = (
                    sesh.interruptionIdxs[ntlNextInterruption] - ntlRipCrossIdxs).astype(float) / float(LFP_SAMPLING_RATE)

                with pp.newFig("ripple_to_interruption_delay") as ax:
                    ax.hist(ripToInterruptionInterval, bins=np.linspace(0, 1, 40))

                refracPeriod = 0.05  # 50ms
                numWithinRefrac = np.count_nonzero(ripToInterruptionInterval < refracPeriod)
                if len(ripToInterruptionInterval) > 0:
                    pp.writeToInfoFile("rip to interruption: numWithinRefracPeriod: {}/{} ripples ({})".format(numWithinRefrac,
                                                                                                               len(ripToInterruptionInterval), numWithinRefrac / len(ripToInterruptionInterval)))
                    print("rip to interruption: numWithinRefracPeriod: {}/{} ripples ({})".format(numWithinRefrac,
                                                                                                  len(ripToInterruptionInterval), numWithinRefrac / len(ripToInterruptionInterval)))

            if RUN_RIPPLE_BASELINE_TET_ANALYSIS and sesh.bt_lfp_baseline_fname is not None:
                # lfpFName = sesh.bt_lfp_fnames[-1]
                # # print("LFP data from file {}".format(lfpFName))
                # lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
                # lfpV = lfpData[1]['voltage']
                # lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

                # # print("LFP data from file {}".format(sesh.bt_lfp_baseline_fname))
                # lfpData = MountainViewIO.loadLFP(data_file=sesh.bt_lfp_baseline_fname)
                # baselfpV = lfpData[1]['voltage']
                # baselfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

                # btLFPData = lfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
                # btRipPower, _, _, _ = getRipplePower(
                #     btLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs, meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower, showPlot=False)
                # probeLFPData = lfpV[sesh.probeLfpStart_idx:sesh.probeLfpEnd_idx]
                # probeRipPower, _, _, _ = getRipplePower(probeLFPData, omit_artifacts=False)

                # baselineProbeLFPData = baselfpV[sesh.probeLfpStart_idx:sesh.probeLfpEnd_idx]
                # probeBaselinePower, _, baselineProbeMeanRipplePower, baselineProbeStdRipplePower = getRipplePower(
                #     baselineProbeLFPData, omit_artifacts=False)
                # btBaselineLFPData = baselfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
                # btBaselineRipplePower, _, _, _ = getRipplePower(
                #     btBaselineLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs, meanPower=baselineProbeMeanRipplePower, stdPower=baselineProbeStdRipplePower, showPlot=False)

                # probeRawPowerDiff = probeRipPower - probeBaselinePower
                # zmean = np.nanmean(probeRawPowerDiff)
                # zstd = np.nanstd(probeRawPowerDiff)

                # rawPowerDiff = btRipPower - btBaselineRipplePower
                # zPowerDiff = (rawPowerDiff - zmean) / zstd

                # withBaseRipStartIdx, withBaseRipLens, withBaseRipPeakIdx, withBaseRipPeakAmps, withBaseRipCrossThreshIdxs = detectRipples(
                #     zPowerDiff)
                # print("with base: {} rips detected".format(len(withBaseRipStartIdx)))

                cfn = np.zeros((2, 2))
                if len(sesh.btWithBaseRipLens) == 0:
                    print("hmmmm ... no rips detected with baseline on session", sesh.name)
                    pp.writeToInfoFile(
                        "hmmmm ... no rips detected with baseline on session", sesh.name)
                else:
                    # How many without base were detected with base?
                    nextWithBase = np.searchsorted(
                        sesh.btWithBaseRipStartIdx, sesh.btRipStartIdxsProbeStats)
                    prevWithBase = nextWithBase - 1
                    prevWithBase[prevWithBase == -1] = 0
                    nextWithBase[nextWithBase == len(sesh.btWithBaseRipStartIdx)] = len(
                        sesh.btWithBaseRipStartIdx) - 1
                    for ri, rs in enumerate(sesh.btRipStartIdxsProbeStats):
                        wbStartNext = sesh.btWithBaseRipStartIdx[nextWithBase[ri]]
                        wbLenNext = sesh.btWithBaseRipLens[nextWithBase[ri]]
                        wbStartPrev = sesh.btWithBaseRipStartIdx[prevWithBase[ri]]
                        wbLenPrev = sesh.btWithBaseRipLens[prevWithBase[ri]]
                        l = sesh.btRipLensProbeStats[ri]
                        if (rs + l > wbStartPrev and wbStartPrev + wbLenPrev > rs) or \
                                (rs + l > wbStartNext and wbStartNext + wbLenNext > rs):
                            # overlapping
                            cfn[0, 0] += 1
                        else:
                            cfn[0, 1] += 1

                    # How many with base were detected without base?
                    nextWithoutBase = np.searchsorted(
                        sesh.btRipStartIdxsProbeStats, sesh.btWithBaseRipStartIdx)
                    prevWithoutBase = nextWithoutBase - 1
                    prevWithoutBase[prevWithoutBase == -1] = 0
                    nextWithoutBase[nextWithoutBase == len(sesh.btRipStartIdxsProbeStats)] = len(
                        sesh.btRipStartIdxsProbeStats) - 1
                    for ri, rs in enumerate(sesh.btWithBaseRipStartIdx):
                        wbStartNext = sesh.btRipStartIdxsProbeStats[nextWithoutBase[ri]]
                        wbLenNext = sesh.btRipLensProbeStats[nextWithoutBase[ri]]
                        wbStartPrev = sesh.btRipStartIdxsProbeStats[prevWithoutBase[ri]]
                        wbLenPrev = sesh.btRipLensProbeStats[prevWithoutBase[ri]]
                        l = sesh.btRipLensProbeStats[ri]
                        if (rs + l > wbStartPrev and wbStartPrev + wbLenPrev > rs) or \
                                (rs + l > wbStartNext and wbStartNext + wbLenNext > rs):
                            # overlapping
                            cfn[1, 0] += 1
                        else:
                            cfn[1, 1] += 1

                    df = pd.DataFrame(data={"detected by other": cfn[:, 0], "not detected": cfn[:, 1]}, index=[
                        "Without baseline", "With Baseline"])
                    print(df)
                    pp.writeToInfoFile(str(df))
                    with pp.newFig("lfp_basenobasecfn") as ax:
                        ax.imshow(cfn)

                    with pp.newFig("lfp_basenobaseovertime") as ax:
                        x = np.repeat(np.vstack((sesh.btWithBaseRipStartIdx, np.array(sesh.btWithBaseRipStartIdx) + np.array(sesh.btWithBaseRipLens))
                                                ).T.flatten() / float(LFP_SAMPLING_RATE), 2)
                        y = np.array([0, 1, 1, 0] * len(sesh.btWithBaseRipStartIdx))
                        ax.plot(x, y)

                        x = np.repeat(np.vstack((sesh.btRipStartIdxsProbeStats, np.array(sesh.btRipStartIdxsProbeStats) + np.array(sesh.btRipLensProbeStats))
                                                ).T.flatten() / float(LFP_SAMPLING_RATE), 2)
                        y = np.array([0, 2, 2, 0] * len(sesh.btRipStartIdxsProbeStats))
                        ax.plot(x, y)
                        ax.legend(["with base", "no base"])

                # TODO store idxs of ripples that contribute to cfn[:,1] and plot raw lfp for those

            if MAKE_RAW_LFP_FIGS:
                lfpFName = sesh.bt_lfp_fnames[-1]
                print("LFP data from file {}".format(lfpFName))
                lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
                lfpV = lfpData[1]['voltage']
                lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

                amps = sesh.btRipPeakAmpsProbeStats

                for i in range(100):
                    maxRipIdx = np.argmax(amps)

                    ripStartIdx = sesh.btRipStartIdxsProbeStats[maxRipIdx]
                    ripLen = sesh.btRipLensProbeStats[maxRipIdx]
                    ripPk = sesh.btRipPeakIdxsProbeStats[maxRipIdx]
                    margin = int(0.15 * 1500)
                    i1 = max(0, ripStartIdx - margin)
                    i2 = min(ripStartIdx + ripLen + margin, len(lfpV))
                    x = lfpT[i1:i2] - lfpT[ripPk]
                    y = lfpV[i1:i2]

                    xStart = lfpT[ripStartIdx] - lfpT[ripPk]
                    xEnd = lfpT[ripStartIdx + ripLen] - lfpT[ripPk]
                    ymin = np.min(y)
                    ymax = np.max(y)

                    with pp.newFig("rawRip_task_{}".format(i)) as ax:
                        ax.plot(x, y, zorder=1)
                        ax.plot([xStart, xStart], [ymin, ymax], c="red", zorder=0)
                        ax.plot([0, 0], [ymin, ymax], c="red", zorder=0)
                        ax.plot([xEnd, xEnd], [ymin, ymax], c="red", zorder=0)

                    amps[maxRipIdx] = 0

            if RUN_RIPPLE_DETECTION_COMPARISON:
                pp.writeToInfoFile("preBTmean={}\npreBTStd={}".format(
                    sesh.prebtMeanRipplePower, sesh.prebtStdRipplePower))
                pp.writeToInfoFile("ITImean={}\nITIStd={}".format(
                    sesh.ITIMeanRipplePower, sesh.ITIStdRipplePower))
                pp.writeToInfoFile("probemean={}\nprobeStd={}".format(
                    sesh.probeMeanRipplePower, sesh.probeStdRipplePower))

    if RUN_SHUFFLES:
        pp.runShuffles(numShuffles=100)


if __name__ == "__main__":
    makeFigures()
