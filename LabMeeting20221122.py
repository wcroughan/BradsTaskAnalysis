import os
import math
import time

from MeasureTypes import WellMeasure
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getInfoForAnimal
from PlotUtil import PlotCtx, setupBehaviorTracePlot
from BTData import BTData
from datetime import datetime
from consts import offWallWellNames
from BTSession import BTSession


def makeFigures(RUN_SHUFFLES=False, RUN_UNSPECIFIED=True,
                RUN_AWAY_FROM_HOME_PLOTS=None, RUN_MISC_PERSEV_PLOTS=None,
                RUN_BASIC_PLOTS=None, RUN_BEHAVIOR_TRACE=None):
    if RUN_BASIC_PLOTS is None:
        RUN_BASIC_PLOTS = RUN_UNSPECIFIED
    if RUN_BEHAVIOR_TRACE is None:
        RUN_BEHAVIOR_TRACE = RUN_UNSPECIFIED
    if RUN_AWAY_FROM_HOME_PLOTS is None:
        RUN_AWAY_FROM_HOME_PLOTS = RUN_UNSPECIFIED
    if RUN_MISC_PERSEV_PLOTS is None:
        RUN_MISC_PERSEV_PLOTS = RUN_UNSPECIFIED

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

        ctrlSessionsWithProbe = [sesh for sesh in sessions if (not sesh.isRippleInterruption) and sesh.probe_performed]
        swrSessionsWithProbe = [sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probe_performed]
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)

        if hasattr(sessions[0], "probe_fill_time"):
            sessionsWithProbeFillPast90 = [s for s in sessionsWithProbe if s.probe_fill_time > 90]
        else:
            sessionsWithProbeFillPast90 = sessionsWithProbe

        pp.setOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if not RUN_BASIC_PLOTS:
            print("WARNING SKPPING BASIC PLOTS")
        else:
            WellMeasure("probe avg dwell time 90sec", lambda s, h: s.avg_dwell_time(True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90).makeFigures(pp)
            WellMeasure("probe avg curvature 90sec", lambda s, h: s.avg_curvature_at_well(True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90).makeFigures(pp)
            WellMeasure("probe gravity from off wall", lambda s, h: s.gravityOfWell(True, h, fromWells=offWallWellNames), sessionsWithProbe).makeFigures(pp)
            if hasattr(sessions[0], "probe_fill_time"):
                WellMeasure("probe gravity from off wall before fill", lambda s, h:
                            s.gravityOfWell(True, h, fromWells=offWallWellNames, timeInterval=[0, s.probe_fill_time]), sessionsWithProbe).makeFigures(pp)
            WellMeasure("task gravity from off wall", lambda s, h: s.gravityOfWell(False, h, fromWells=offWallWellNames), sessionsWithProbe).makeFigures(pp)

        if not RUN_BEHAVIOR_TRACE:
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

        if not RUN_AWAY_FROM_HOME_PLOTS:
            print("warning: skpping away from home plots")
        else:
            pass

        if not RUN_MISC_PERSEV_PLOTS:
            print("warning: skppping misc perseve plts")
        else:
            def fracAtWellBoutAvg(s: BTSession, h: int):
                fracSum = 0.0
                for i1, i2 in zip(s.probe_excursion_starts, s.probe_excursion_ends):
                    totalDur = s.probe_pos_ts[i2] - s.probe_pos_ts[i1]
                    durAtWell = 0.0
                    for ent, ext in zip(s.probe_well_entry_idxs, s.probe_well_exit_idxs):
                        if ext < i1:
                            continue
                        if ent > i2:
                            continue
                        durAtWell += s.probe_pos_ts[ext] - s.probe_pos_ts[ent] 

                    fracSum += durAtWell / totalDur
                return fracSum / len(s.probe_excursion_starts)

            def fracAtWellTimeAvg(s: BTSession, h: int):
                totalDur = 0.0
                durAtWell = 0.0
                for i1, i2 in zip(s.probe_excursion_starts, s.probe_excursion_ends):
                    totalDur += s.probe_pos_ts[i2] - s.probe_pos_ts[i1]
                    for ent, ext in zip(s.probe_well_entry_idxs, s.probe_well_exit_idxs):
                        if ext < i1:
                            continue
                        if ent > i2:
                            continue
                        durAtWell += s.probe_pos_ts[ext] - s.probe_pos_ts[ent] 

                return durAtWell / totalDur

            WellMeasure("probe frac excursion at well over bouts", fracAtWellBoutAvg, sessionsWithProbeFillPast90).makeFigures(pp)
            WellMeasure("probe frac excursion at well over time", fracAtWellTimeAvg, sessionsWithProbeFillPast90).makeFigures(pp)


if __name__ == "__main__":
    makeFigures()

# ======================================================================
# Running tally of suggested analyses:
#
# Latency to interruption
#
# Some perseveration measure during away trials to see if there's an effect during the actual task
#   maybe: only on excursions where no well found, look at pct time spent at home
#
# difference in effect magnitude by distance from starting location to home well?
#
# Based on speed or exploration, is there a more principled way to
# choose period of probe that is measured? B/c could vary by rat
#
# Measure away from home well: how often oriented toward home (especially during exploration)
# try with and without distance discount
#
# In general, should run these analyses against shuffled home/away selections
#
# How often during task does he make a u-turn when facing away from home well?
#
# excursions start/end near home?
#
# Within each excursion, pct of time spent at home
#
# pseudoprobe, do they return to previous home? Any relation to last probe behavior?
# What about previous home-of-same-condition, i.e. on interruption they go to last interruption home?
#
# curvature during task
#
# gravity but 90seconds
#
# Is gravity correctly calculated? What about wall -> home -> wall?
#
# DLC: direction from two LEDs or from head and back of body
# DLC: multiple views already built in somehow?
#      YES! https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/Overviewof3D.md
# Yartsev lab mesh tracking software
#
# DLC: is alignment frame-accurate?
# ======================================================================
