import os
import math
import time

from MeasureTypes import WellMeasure
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getInfoForAnimal
from PlotUtil import PlotCtx, setupBehaviorTracePlot
from BTData import BTData
from datetime import datetime
from consts import offWallWellNames, all_well_names
from BTSession import BTSession
import numpy as np

# TODOs
# Add shuffles to all measures
#
# Away from home measures: spotlight, cross product
#
# Use dist from well and quantify each individually instead of dividing by nearest well
#
# Improved gravity logic.
#
# Latency to interruption


def makeFigures(RUN_SHUFFLES=False, RUN_UNSPECIFIED=True,
                RUN_MISC_PERSEV_PLOTS=True, RUN_OLD_GRAVITY_PLOTS=True,
                RUN_BASIC_PLOTS=None, RUN_BEHAVIOR_TRACE=None):
    if RUN_BASIC_PLOTS is None:
        RUN_BASIC_PLOTS = RUN_UNSPECIFIED
    if RUN_BEHAVIOR_TRACE is None:
        RUN_BEHAVIOR_TRACE = RUN_UNSPECIFIED
    if RUN_MISC_PERSEV_PLOTS is None:
        RUN_MISC_PERSEV_PLOTS = RUN_UNSPECIFIED
    if RUN_OLD_GRAVITY_PLOTS is None:
        RUN_OLD_GRAVITY_PLOTS = RUN_UNSPECIFIED

    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "20230117_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    # animalNames = parseCmdLineAnimalNames(default=["B16", "B17", "B18", "B13", "B14", "Martin"])
    animalNames = parseCmdLineAnimalNames(default=["B13", "B14", "Martin"])
    # animalNames = parseCmdLineAnimalNames(default=["B13"])

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
            # WellMeasure("probe avg dwell time 90sec", lambda s, h: s.avg_dwell_time(
            #     True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90).makeFigures(pp)
            # WellMeasure("probe avg curvature 90sec", lambda s, h: s.avg_curvature_at_well(
            #     True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90).makeFigures(pp)
            WellMeasure("probe gravity from off wall", lambda s, h: s.gravityOfWell(
                True, h, fromWells=offWallWellNames), sessionsWithProbe).makeFigures(pp)
            WellMeasure("probe gravity from all", lambda s, h: s.gravityOfWell(
                True, h), sessionsWithProbe).makeFigures(pp)
            if hasattr(sessions[0], "probe_fill_time"):
                WellMeasure("probe gravity from off wall before fill", lambda s, h:
                            s.gravityOfWell(True, h, fromWells=offWallWellNames,
                                            timeInterval=[0, s.probe_fill_time]),
                            sessionsWithProbe).makeFigures(pp)
            WellMeasure("task gravity from off wall", lambda s, h: s.gravityOfWell(
                False, h, fromWells=offWallWellNames), sessionsWithProbe).makeFigures(pp)
            WellMeasure("task gravity from all", lambda s, h: s.gravityOfWell(
                False, h), sessionsWithProbe).makeFigures(pp)

        if not RUN_OLD_GRAVITY_PLOTS:
            print("WARNING SKPPING OLD GRAVITY PLOTS")
        else:
            WellMeasure("probe gravity from off wall (old)", lambda s, h: s.gravityOfWell_old(
                True, h, fromWells=offWallWellNames), sessionsWithProbe).makeFigures(pp)
            WellMeasure("probe gravity from all (old)", lambda s, h: s.gravityOfWell_old(
                True, h), sessionsWithProbe).makeFigures(pp)
            WellMeasure("task gravity from off wall (old)", lambda s, h: s.gravityOfWell_old(
                False, h, fromWells=offWallWellNames), sessionsWithProbe).makeFigures(pp)
            WellMeasure("task gravity from all (old)", lambda s, h: s.gravityOfWell_old(
                False, h), sessionsWithProbe).makeFigures(pp)

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

        if not RUN_MISC_PERSEV_PLOTS:
            print("warning: skppping misc perseve plts")
        else:
            def fracAtWellBoutAvg(s: BTSession, h: int):
                if len(s.probe_excursion_starts) == 0:
                    print(f"Warning: session {s.name} has no probe excursions")
                    return np.nan
                wellIdx = np.argmax(all_well_names == h)
                fracSum = 0.0
                for i1, i2 in zip(s.probe_excursion_starts, s.probe_excursion_ends):
                    if i2 == i1 + 1:
                        continue
                    # print(i1, i2, s.probe_pos_ts[i1], s.probe_pos_ts[i2-1])
                    # print(np.count_nonzero(np.diff(s.probe_pos_ts) == 0))
                    # print(len(s.probe_pos_ts))
                    totalDur = s.probe_pos_ts[i2-1] - s.probe_pos_ts[i1]
                    durAtWell = 0.0
                    for ent, ext in zip(s.probe_well_entry_idxs[wellIdx], s.probe_well_exit_idxs[wellIdx]):
                        if ext < i1:
                            continue
                        if ent > i2-1:
                            continue
                        durAtWell += s.probe_pos_ts[ext] - s.probe_pos_ts[ent]

                    fracSum += durAtWell / totalDur
                return fracSum / len(s.probe_excursion_starts)

            def fracAtWellTimeAvg(s: BTSession, h: int):
                if len(s.probe_excursion_starts) == 0:
                    print(f"Warning: session {s.name} has no probe excursions")
                    return np.nan
                wellIdx = np.argmax(all_well_names == h)
                totalDur = 0.0
                durAtWell = 0.0
                for i1, i2 in zip(s.probe_excursion_starts, s.probe_excursion_ends):
                    totalDur += s.probe_pos_ts[i2-1] - s.probe_pos_ts[i1]
                    for ent, ext in zip(s.probe_well_entry_idxs[wellIdx], s.probe_well_exit_idxs[wellIdx]):
                        if ext < i1:
                            continue
                        if ent > i2-1:
                            continue
                        durAtWell += s.probe_pos_ts[ext] - s.probe_pos_ts[ent]

                return durAtWell / totalDur

            WellMeasure("probe frac excursion at well over bouts", fracAtWellBoutAvg,
                        sessionsWithProbeFillPast90).makeFigures(pp)
            WellMeasure("probe frac excursion at well over time", fracAtWellTimeAvg,
                        sessionsWithProbeFillPast90).makeFigures(pp)

    if len(animalNames) > 1:
        comboStr = "combo {}".format(" ".join(animalNames))
        print("======================\n", comboStr)
        pp.makeCombinedFigs(outputSubDir=comboStr)

    # if RUN_SHUFFLES:
        # pp.runShuffles()


if __name__ == "__main__":
    makeFigures(RUN_MISC_PERSEV_PLOTS=False, RUN_BASIC_PLOTS=True,
                RUN_BEHAVIOR_TRACE=False, RUN_OLD_GRAVITY_PLOTS=False)
    # makeFigures()
