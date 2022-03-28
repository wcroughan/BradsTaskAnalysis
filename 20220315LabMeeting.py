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

MAKE_RAW_LFP_FIGS = False
RUN_RIPPLE_DETECTION_COMPARISON = True
RUN_RIPPLE_REFRAC_PERIOD_CHECK = True
RUN_RIPPLE_BASELINE_TET_ANALYSIS = True
MAKE_PROBE_EXPLORATION_OVER_TIME_FIGS = True
MAKE_PROBE_TRACES_FIGS = True
MAKE_PROBE_OFFWALL_TIME_FIGS = True
MAKE_NORMAL_PERSEV_MEASURES_FIGS = True
MAKE_PERSEV_WITH_EXPLORATION_CUTOFF_FIGS = True

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


pp = PlotCtx(globalOutputDir, priorityLevel=8, randomSeed=94702)
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

    if len(animalNames) > 1:
        pp.setStatCategory("rat", ratName)

    if MAKE_NORMAL_PERSEV_MEASURES_FIGS or MAKE_PERSEV_WITH_EXPLORATION_CUTOFF_FIGS or MAKE_PROBE_OFFWALL_TIME_FIGS:
        wellCat = []
        seshCat = []
        seshConditionGroup = []
        avgDwell90 = []
        curvature = []
        curvature90 = []
        seshIdx = []
        propWellTime = []
        totalOffWallTime = []

        for si, sesh in enumerate(sessionsWithProbe):
            towt = np.sum([sesh.total_dwell_time(True, w)
                           for w in offWallWellNames])

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
        avgDwell90 = np.array(avgDwell90)
        curvature = np.array(curvature)
        curvature90 = np.array(curvature90)
        propWellTime = np.array(propWellTime)
        totalOffWallTime = np.array(totalOffWallTime)

    if MAKE_NORMAL_PERSEV_MEASURES_FIGS:
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
                ax.set_title(sesh.name)
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
            ax.set_ylim(0, 36)

        with pp.newFig("probe_numVisitedOverTime_bysidx_offwall", priority=10) as ax:
            cmap = cm.get_cmap("coolwarm", numSessionsWithProbe)
            for si in range(numSessionsWithProbe):
                jitter = np.random.uniform(size=(numVisitedOffWall.shape[1],)) * 0.5
                ax.plot(t1Array, numVisitedOffWall[si, :] + jitter, color=cmap(si))
            ax.set_xticks(np.arange(0, 60*5+1, 60))
            ax.set_ylim(0, 16)

        with pp.newFig("probe_numVisitedOverTime_byweek_offwall", priority=10) as ax:
            cmap = cm.get_cmap("coolwarm", np.max(weeks))
            for si in range(numSessionsWithProbe):
                jitter = np.random.uniform(size=(numVisitedOffWall.shape[1],)) * 0.5
                ax.plot(t1Array, numVisitedOffWall[si, :] + jitter, color=cmap(weeks[si]))
            ax.set_xticks(np.arange(0, 60*5+1, 60))
            ax.set_ylim(0, 16)

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
                    ax.set_ylim(0, 16)

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
            swrIdx = np.array([sesh.isRippleInterruption for sesh in sessionsWithProbe])
            ax.scatter(x[swrIdx], totalOffWallTime[swrIdx], color="orange")
            ax.scatter(x[~swrIdx], totalOffWallTime[~swrIdx], color="cyan")

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

                yvals["probe_propOffWallTimeNearWell"] = propWellTimeCut
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
            mostRecentInterruption = mostRecentInterruption[notTooEarly] - 1
            ripCrossIdxs = ripCrossIdxs[notTooEarly]
            interruptionToRipInterval = (
                ripCrossIdxs - sesh.interruptionIdxs[mostRecentInterruption]).astype(float) / float(LFP_SAMPLING_RATE)

            with pp.newFig("interruption_to_ripple_delay") as ax:
                ax.hist(interruptionToRipInterval, bins=np.linspace(0, 1, 40))

        if RUN_RIPPLE_BASELINE_TET_ANALYSIS:
            lfpFName = sesh.bt_lfp_fnames[-1]
            print("LFP data from file {}".format(lfpFName))
            lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
            lfpV = lfpData[1]['voltage']
            lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

            print("LFP data from file {}".format(sesh.bt_lfp_baseline_fname))
            lfpData = MountainViewIO.loadLFP(data_file=sesh.bt_lfp_baseline_fname)
            baselfpV = lfpData[1]['voltage']
            baselfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

            btLFPData = lfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
            _, btRipPower, _, _ = getRipplePower(
                btLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs, meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower, showPlot=False)

            baselineProbeLFPData = baselfpV[sesh.probeLfpStart_idx:sesh.probeLfpEnd_idx]
            _, _, baselineProbeMeanRipplePower, baselineProbeStdRipplePower = getRipplePower(
                baselineProbeLFPData, omit_artifacts=False)
            btBaselineLFPData = baselfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
            _, btBaselineRipplePower, _, _ = getRipplePower(
                btBaselineLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs, meanPower=baselineProbeMeanRipplePower, stdPower=baselineProbeStdRipplePower, showPlot=False)

            withBaseRipStartIdx, withBaseRipLens, withBaseRipPeakIdx, withBaseRipPeakAmps, withBaseRipCrossThreshIdxs = detectRipples(
                btRipPower - btBaselineRipplePower)

            cfn = np.zeros((2, 2))
            # How many without base were detected with base?
            nextWithBase = np.searchsorted(withBaseRipStartIdx, sesh.btRipStartIdxsProbeStats)
            prevWithBase = nextWithBase - 1
            prevWithBase[prevWithBase == -1] = 0
            nextWithBase[nextWithBase == len(withBaseRipStartIdx)] = len(withBaseRipStartIdx) - 1
            for ri, rs in enumerate(sesh.btRipStartIdxsProbeStats):
                wbStartNext = withBaseRipStartIdx[nextWithBase[ri]]
                wbLenNext = withBaseRipLens[nextWithBase[ri]]
                wbStartPrev = withBaseRipStartIdx[prevWithBase[ri]]
                wbLenPrev = withBaseRipLens[prevWithBase[ri]]
                l = sesh.btRipLensProbeStats[ri]
                if (rs + l > wbStartPrev and wbStartPrev + wbLenPrev > rs) or \
                        (rs + l > wbStartNext and wbStartNext + wbLenNext > rs):
                    # overlapping
                    cfn[0, 0] += 1
                else:
                    cfn[0, 1] += 1

            # How many with base were detected without base?
            nextWithoutBase = np.searchsorted(sesh.btRipStartIdxsProbeStats, withBaseRipStartIdx)
            prevWithoutBase = nextWithoutBase - 1
            prevWithoutBase[prevWithoutBase == -1] = 0
            nextWithoutBase[nextWithoutBase == len(sesh.btRipStartIdxsProbeStats)] = len(
                sesh.btRipStartIdxsProbeStats) - 1
            for ri, rs in enumerate(withBaseRipStartIdx):
                wbStartNext = sesh.btRipStartIdxsProbeStats[nextWithoutBase[ri]]
                wbLenNext = sesh.btRipStartIdxsProbeStats[nextWithoutBase[ri]]
                wbStartPrev = sesh.btRipStartIdxsProbeStats[prevWithoutBase[ri]]
                wbLenPrev = sesh.btRipStartIdxsProbeStats[prevWithoutBase[ri]]
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
            with pp.newFig("lfp_basenobasecfn") as ax:
                ax.imshow(cfn)

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

pp.runShuffles(numShuffles=50)
