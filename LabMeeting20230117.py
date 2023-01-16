import os
import math
import time
import MountainViewIO

from MeasureTypes import WellMeasure
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getInfoForAnimal, offWall, getRipplePower
from PlotUtil import PlotCtx, setupBehaviorTracePlot
from BTData import BTData
from datetime import datetime
from consts import offWallWellNames, all_well_names, TRODES_SAMPLING_RATE
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
                RUN_MISC_PERSEV_PLOTS=None, RUN_OLD_GRAVITY_PLOTS=None,
                RUN_WELL_PASS_TRACES=None, RUN_PASS_HIST=None,
                RUN_SPOTLIGHT=None, RUN_SWR_LATENCY_PLOTS=None,
                MAKE_INDIVIDUAL_INTERRUPTION_PLOTS=None,
                RUN_BASIC_PLOTS=None, RUN_BEHAVIOR_TRACE=None):
    if RUN_BASIC_PLOTS is None:
        RUN_BASIC_PLOTS = RUN_UNSPECIFIED
    if RUN_BEHAVIOR_TRACE is None:
        RUN_BEHAVIOR_TRACE = RUN_UNSPECIFIED
    if RUN_MISC_PERSEV_PLOTS is None:
        RUN_MISC_PERSEV_PLOTS = RUN_UNSPECIFIED
    if RUN_OLD_GRAVITY_PLOTS is None:
        RUN_OLD_GRAVITY_PLOTS = RUN_UNSPECIFIED
    if RUN_WELL_PASS_TRACES is None:
        RUN_WELL_PASS_TRACES = RUN_UNSPECIFIED
    if RUN_PASS_HIST is None:
        RUN_PASS_HIST = RUN_UNSPECIFIED
    if RUN_SPOTLIGHT is None:
        RUN_SPOTLIGHT = RUN_UNSPECIFIED
    if RUN_SWR_LATENCY_PLOTS is None:
        RUN_SWR_LATENCY_PLOTS = RUN_UNSPECIFIED

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
        nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probe_performed]
        numSessionsWithProbe = len(sessionsWithProbe)
        print(f"{len(sessions)} sessions ({len(sessionsWithProbe)} with probe)")

        ctrlSessionsWithProbe = [sesh for sesh in sessions if (
            not sesh.isRippleInterruption) and sesh.probe_performed]
        swrSessionsWithProbe = [
            sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probe_performed]
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)

        ctrlSessions = [sesh for sesh in sessions if not sesh.isRippleInterruption]
        swrSessions = [sesh for sesh in sessions if sesh.isRippleInterruption]
        nCtrlSessions = len(ctrlSessions)
        nSWRSessions = len(swrSessions)

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
            WellMeasure("probe avg dwell time 90sec", lambda s, h: s.avg_dwell_time(
                True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90).makeFigures(pp)
            WellMeasure("probe avg curvature 90sec", lambda s, h: s.avg_curvature_at_well(
                True, h, timeInterval=[0, 90]), sessionsWithProbeFillPast90).makeFigures(pp)
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
            numCols = math.ceil(math.sqrt(nSessions))
            with pp.newFig("allTaskTraces", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(sessions):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                    # ax.set_title(str(si))
                for si in range(nSessions, numCols * numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

            numCols = math.ceil(math.sqrt(nSessions))
            with pp.newFig("allTaskTraces_moving", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(sessions):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.bt_mv_xs, sesh.bt_mv_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                    # ax.set_title(str(si))
                for si in range(nSessions, numCols * numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

            for si, sesh in enumerate(sessions):
                nbouts = len(sesh.bt_explore_bout_starts)
                numCols = math.ceil(math.sqrt(nbouts))
                with pp.newFig(f"explore_bouts_task_{sesh.name}", subPlots=(numCols, numCols), figScale=0.3) as axs:
                    for bi in range(nbouts):
                        ax = axs[bi // numCols, bi % numCols]
                        i1 = sesh.bt_explore_bout_starts[bi]
                        i2 = sesh.bt_explore_bout_ends[bi]
                        ax.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], c="#deac7f")
                        c = "orange" if sesh.isRippleInterruption else "cyan"
                        setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                        ax.set_title(sesh.name, fontdict={'fontsize': 8})
                        # ax.set_title(str(bi))
                    for bi in range(nbouts, numCols * numCols):
                        ax = axs[bi // numCols, bi % numCols]
                        ax.cla()
                        ax.tick_params(axis="both", which="both", label1On=False,
                                       label2On=False, tick1On=False, tick2On=False)

                nbouts = len(sesh.probe_explore_bout_starts)
                if nbouts > 0:
                    numCols = math.ceil(math.sqrt(nbouts))
                    with pp.newFig(f"explore_bouts_probe_{sesh.name}", subPlots=(numCols, numCols), figScale=0.3) \
                            as axs:
                        for bi in range(nbouts):
                            if numCols == 1:
                                ax = axs
                            else:
                                ax = axs[bi // numCols, bi % numCols]
                            i1 = sesh.probe_explore_bout_starts[bi]
                            i2 = sesh.probe_explore_bout_ends[bi]
                            ax.plot(sesh.probe_pos_xs[i1:i2], sesh.probe_pos_ys[i1:i2], c="#deac7f")
                            c = "orange" if sesh.isRippleInterruption else "cyan"
                            setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                            ax.set_title(sesh.name, fontdict={'fontsize': 8})
                            # ax.set_title(str(bi))
                        for bi in range(nbouts, numCols * numCols):
                            ax = axs[bi // numCols, bi % numCols]
                            ax.cla()
                            ax.tick_params(axis="both", which="both", label1On=False,
                                           label2On=False, tick1On=False, tick2On=False)

            numCols = math.ceil(math.sqrt(nCtrlSessions))
            with pp.newFig("allTaskTraces_ctrl", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(ctrlSessions):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    # ax.set_title(str(si))
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                for si in range(nCtrlSessions, numCols * numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

            numCols = math.ceil(math.sqrt(nSWRSessions))
            with pp.newFig("allTaskTraces_SWR", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(swrSessions):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, c="#deac7f")
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=2)
                    # ax.set_title(str(si))
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})
                for si in range(nSWRSessions, numCols * numCols):
                    ax = axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

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

            numCols = math.ceil(math.sqrt(numSessionsWithProbe))
            with pp.newFig("allProbeTraces_moving", subPlots=(numCols, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(sessionsWithProbe):
                    ax = axs[si // numCols, si % numCols]
                    ax.plot(sesh.probe_mv_xs, sesh.probe_mv_ys, c="#deac7f")
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

        if not RUN_WELL_PASS_TRACES:
            print("Warning: Skippin well pass traces")
        else:
            numRows = numSessionsWithProbe
            numCols = 20
            with pp.newFig("allPasses", subPlots=(numRows, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(sessionsWithProbe):
                    # ax = axs[si // numCols, si % numCols]
                    ax = axs[si, 0]
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    for v in ax.spines.values():
                        v.set_color(c)
                        v.set_linewidth(3)
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})

                    ents, exts, wellCoords = sesh.getPasses(sesh.home_well, False)
                    for enti, exti in zip(ents, exts):
                        ax.plot(np.array(sesh.bt_pos_xs[enti:exti]) - wellCoords[0],
                                np.array(sesh.bt_pos_ys[enti:exti]) - wellCoords[1], c="#deac7f")
                    ax.set_xlim(-200, 200)
                    ax.set_ylim(-200, 200)

                    ax = axs[si, 1]
                    ents, exts, wellCoords = sesh.getPasses(sesh.home_well, True)
                    for enti, exti in zip(ents, exts):
                        ax.plot(np.array(sesh.probe_pos_xs[enti:exti]) - wellCoords[0],
                                np.array(sesh.probe_pos_ys[enti:exti]) - wellCoords[1], c="#deac7f")
                    ax.set_xlim(-200, 200)
                    ax.set_ylim(-200, 200)
                    ax.set_title("probe home", fontdict={'fontsize': 8})

                    visitedOffWallAways = [aw for aw in sesh.visited_away_wells if offWall(aw)]
                    for welli in range(len(visitedOffWallAways)):
                        ax = axs[si, 2+welli*2]
                        well = visitedOffWallAways[welli]
                        ents, exts, wellCoords = sesh.getPasses(well, False)
                        for enti, exti in zip(ents, exts):
                            ax.plot(np.array(sesh.bt_pos_xs[enti:exti]) - wellCoords[0],
                                    np.array(sesh.bt_pos_ys[enti:exti]) - wellCoords[1], c="#deac7f")
                        ax.set_xlim(-200, 200)
                        ax.set_ylim(-200, 200)
                        ax.set_title(f"{well}, task", fontdict={'fontsize': 8})
                        ax.tick_params(axis="both", which="both", label1On=False,
                                       label2On=False, tick1On=False, tick2On=False)

                        ax = axs[si, 2+welli*2+1]
                        ents, exts, wellCoords = sesh.getPasses(well, True)
                        for enti, exti in zip(ents, exts):
                            ax.plot(np.array(sesh.probe_pos_xs[enti:exti]) - wellCoords[0],
                                    np.array(sesh.probe_pos_ys[enti:exti]) - wellCoords[1], c="#deac7f")
                        ax.set_xlim(-200, 200)
                        ax.set_ylim(-200, 200)
                        ax.set_title(f"{well}, probe", fontdict={'fontsize': 8})
                        ax.tick_params(axis="both", which="both", label1On=False,
                                       label2On=False, tick1On=False, tick2On=False)

                    for axc in range(2+2*len(visitedOffWallAways), numCols):
                        ax = axs[si, axc]
                        ax.cla()
                        ax.tick_params(axis="both", which="both", label1On=False,
                                       label2On=False, tick1On=False, tick2On=False)

        if not RUN_PASS_HIST:
            print("Warning not making pass histograms")
        else:
            numRows = numSessionsWithProbe
            numCols = 10
            passDist = 250
            bins = np.linspace(0, passDist, 15)
            with pp.newFig("allPassHistograms", subPlots=(numRows, numCols), figScale=0.3) as axs:
                for si, sesh in enumerate(sessionsWithProbe):
                    # ax = axs[si // numCols, si % numCols]
                    ax = axs[si, 0]
                    c = "orange" if sesh.isRippleInterruption else "cyan"
                    for v in ax.spines.values():
                        v.set_color(c)
                        v.set_linewidth(3)
                    ax.set_title(sesh.name, fontdict={'fontsize': 8})

                    ents, exts, wellCoords = sesh.getPasses(
                        sesh.home_well, False, distance=passDist)
                    allPassDists = np.empty((len(ents),))
                    pi = 0
                    for enti, exti in zip(ents, exts):
                        x = np.array(sesh.bt_pos_xs[enti:exti]) - wellCoords[0]
                        y = np.array(sesh.bt_pos_ys[enti:exti]) - wellCoords[1]
                        mags = np.power(x, 2) + np.power(y, 2)
                        allPassDists[pi] = np.sqrt(np.min(mags))
                        pi += 1
                    ax.hist(allPassDists, bins, label='task', histtype='step')

                    ents, exts, wellCoords = sesh.getPasses(sesh.home_well, True, distance=passDist)
                    allPassDists = np.empty((len(ents),))
                    pi = 0
                    for enti, exti in zip(ents, exts):
                        x = np.array(sesh.probe_pos_xs[enti:exti]) - wellCoords[0]
                        y = np.array(sesh.probe_pos_ys[enti:exti]) - wellCoords[1]
                        mags = np.power(x, 2) + np.power(y, 2)
                        allPassDists[pi] = np.sqrt(np.min(mags))
                        pi += 1
                    ax.hist(allPassDists, bins, label='probe', histtype='step')
                    ax.legend()

                    visitedOffWallAways = [aw for aw in sesh.visited_away_wells if offWall(aw)]
                    for welli in range(len(visitedOffWallAways)):
                        ax = axs[si, 1+welli]
                        well = visitedOffWallAways[welli]
                        ax.set_title(f"{well}", fontdict={'fontsize': 8})

                        ents, exts, wellCoords = sesh.getPasses(well, False, distance=passDist)
                        allPassDists = np.empty((len(ents),))
                        pi = 0
                        for enti, exti in zip(ents, exts):
                            x = np.array(sesh.bt_pos_xs[enti:exti]) - wellCoords[0]
                            y = np.array(sesh.bt_pos_ys[enti:exti]) - wellCoords[1]
                            mags = np.power(x, 2) + np.power(y, 2)
                            allPassDists[pi] = np.sqrt(np.min(mags))
                            pi += 1
                        ax.hist(allPassDists, bins, label='task', histtype='step')

                        ents, exts, wellCoords = sesh.getPasses(well, True, distance=passDist)
                        allPassDists = np.empty((len(ents),))
                        pi = 0
                        for enti, exti in zip(ents, exts):
                            x = np.array(sesh.probe_pos_xs[enti:exti]) - wellCoords[0]
                            y = np.array(sesh.probe_pos_ys[enti:exti]) - wellCoords[1]
                            mags = np.power(x, 2) + np.power(y, 2)
                            allPassDists[pi] = np.sqrt(np.min(mags))
                            pi += 1
                        ax.hist(allPassDists, bins, label='probe', histtype='step')

                    for axc in range(1+len(visitedOffWallAways), numCols):
                        ax = axs[si, axc]
                        ax.cla()
                        ax.tick_params(axis="both", which="both", label1On=False,
                                       label2On=False, tick1On=False, tick2On=False)

            taskHomeCtrlPassDists = np.array([])
            taskAwayCtrlPassDists = np.array([])
            taskHomeSWRPassDists = np.array([])
            taskAwaySWRPassDists = np.array([])
            probeHomeCtrlPassDists = np.array([])
            probeAwayCtrlPassDists = np.array([])
            probeHomeSWRPassDists = np.array([])
            probeAwaySWRPassDists = np.array([])

            for si, sesh in enumerate(sessionsWithProbe):
                ents, exts, wellCoords = sesh.getPasses(sesh.home_well, False, distance=passDist)
                allPassDists = np.empty((len(ents),))
                pi = 0
                for enti, exti in zip(ents, exts):
                    x = np.array(sesh.bt_pos_xs[enti:exti]) - wellCoords[0]
                    y = np.array(sesh.bt_pos_ys[enti:exti]) - wellCoords[1]
                    mags = np.power(x, 2) + np.power(y, 2)
                    allPassDists[pi] = np.sqrt(np.min(mags))
                    pi += 1
                if sesh.isRippleInterruption:
                    taskHomeSWRPassDists = np.hstack((taskHomeSWRPassDists, allPassDists))
                else:
                    taskHomeCtrlPassDists = np.hstack((taskHomeCtrlPassDists, allPassDists))

                ents, exts, wellCoords = sesh.getPasses(sesh.home_well, True, distance=passDist)
                allPassDists = np.empty((len(ents),))
                pi = 0
                for enti, exti in zip(ents, exts):
                    x = np.array(sesh.probe_pos_xs[enti:exti]) - wellCoords[0]
                    y = np.array(sesh.probe_pos_ys[enti:exti]) - wellCoords[1]
                    mags = np.power(x, 2) + np.power(y, 2)
                    allPassDists[pi] = np.sqrt(np.min(mags))
                    pi += 1
                if sesh.isRippleInterruption:
                    probeHomeSWRPassDists = np.hstack((probeHomeSWRPassDists, allPassDists))
                else:
                    probeHomeCtrlPassDists = np.hstack((probeHomeCtrlPassDists, allPassDists))

                visitedOffWallAways = [aw for aw in sesh.visited_away_wells if offWall(aw)]
                for welli in range(len(visitedOffWallAways)):
                    ents, exts, wellCoords = sesh.getPasses(well, False, distance=passDist)
                    allPassDists = np.empty((len(ents),))
                    pi = 0
                    for enti, exti in zip(ents, exts):
                        x = np.array(sesh.bt_pos_xs[enti:exti]) - wellCoords[0]
                        y = np.array(sesh.bt_pos_ys[enti:exti]) - wellCoords[1]
                        mags = np.power(x, 2) + np.power(y, 2)
                        allPassDists[pi] = np.sqrt(np.min(mags))
                        pi += 1
                    if sesh.isRippleInterruption:
                        taskAwaySWRPassDists = np.hstack((taskAwaySWRPassDists, allPassDists))
                    else:
                        taskAwayCtrlPassDists = np.hstack((taskAwayCtrlPassDists, allPassDists))

                    ents, exts, wellCoords = sesh.getPasses(well, True, distance=passDist)
                    allPassDists = np.empty((len(ents),))
                    pi = 0
                    for enti, exti in zip(ents, exts):
                        x = np.array(sesh.probe_pos_xs[enti:exti]) - wellCoords[0]
                        y = np.array(sesh.probe_pos_ys[enti:exti]) - wellCoords[1]
                        mags = np.power(x, 2) + np.power(y, 2)
                        allPassDists[pi] = np.sqrt(np.min(mags))
                        pi += 1
                    if sesh.isRippleInterruption:
                        probeAwaySWRPassDists = np.hstack((probeAwaySWRPassDists, allPassDists))
                    else:
                        probeAwayCtrlPassDists = np.hstack((probeAwayCtrlPassDists, allPassDists))

            with pp.newFig("allPassHistCombo_condition", withStats=False, subPlots=(2, 2)) as axs:
                ax = axs[0, 0]
                ax.set_title("Home, task")
                ax.hist(taskHomeSWRPassDists, bins, label='SWR', histtype='step', density=True)
                ax.hist(taskHomeCtrlPassDists, bins, label='Ctrl', histtype='step', density=True)
                ax.legend()

                ax = axs[1, 0]
                ax.set_title("Away, task")
                ax.hist(taskAwaySWRPassDists, bins, label='SWR', histtype='step', density=True)
                ax.hist(taskAwayCtrlPassDists, bins, label='Ctrl', histtype='step', density=True)
                ax.legend()

                ax = axs[0, 1]
                ax.set_title("Home, probe")
                ax.hist(probeHomeSWRPassDists, bins, label='SWR', histtype='step', density=True)
                ax.hist(probeHomeCtrlPassDists, bins, label='Ctrl', histtype='step', density=True)
                ax.legend()

                ax = axs[1, 1]
                ax.set_title("Away, probe")
                ax.hist(probeAwaySWRPassDists, bins, label='SWR', histtype='step', density=True)
                ax.hist(probeAwayCtrlPassDists, bins, label='Ctrl', histtype='step', density=True)
                ax.legend()

            with pp.newFig("allPassHistCombo_taskvprobe", withStats=False, subPlots=(2, 2)) as axs:
                ax = axs[0, 0]
                ax.set_title("SWR, Home")
                ax.hist(taskHomeSWRPassDists, bins, label='task', histtype='step', density=True)
                ax.hist(probeHomeSWRPassDists, bins, label='probe', histtype='step', density=True)
                ax.legend()

                ax = axs[1, 0]
                ax.set_title("Ctrl, Home")
                ax.hist(taskHomeCtrlPassDists, bins, label='task', histtype='step', density=True)
                ax.hist(probeHomeCtrlPassDists, bins, label='probe', histtype='step', density=True)
                ax.legend()

                ax = axs[0, 1]
                ax.set_title("SWR, Away")
                ax.hist(taskAwaySWRPassDists, bins, label='task', histtype='step', density=True)
                ax.hist(probeAwaySWRPassDists, bins, label='probe', histtype='step', density=True)
                ax.legend()

                ax = axs[1, 1]
                ax.set_title("Ctrl, Away")
                ax.hist(taskAwayCtrlPassDists, bins, label='task', histtype='step', density=True)
                ax.hist(probeAwayCtrlPassDists, bins, label='probe', histtype='step', density=True)
                ax.legend()

        if not RUN_SPOTLIGHT:
            print("Warning skpping spotlight plots")
        else:
            WellMeasure("probe spotlight score", lambda s, h: s.getSpotlightScore(
                True, h), sessionsWithProbe).makeFigures(pp)
            WellMeasure("task spotlight score", lambda s, h: s.getSpotlightScore(
                False, h), sessionsWithProbe).makeFigures(pp)
            WellMeasure("probe dotprod score", lambda s, h: s.getDotProductScore(
                True, h), sessionsWithProbe).makeFigures(pp)
            WellMeasure("task dotprod score", lambda s, h: s.getDotProductScore(
                False, h), sessionsWithProbe).makeFigures(pp)

        if not RUN_SWR_LATENCY_PLOTS:
            print("Warning: skipping swr latency plots")
        else:
            for si, sesh in enumerate(swrSessionsWithProbe):
                lfpFName = sesh.bt_lfp_fnames[-1]
                print("LFP data from file {}".format(lfpFName))
                lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
                lfpV = lfpData[1]['voltage']
                lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

                lfpData = MountainViewIO.loadLFP(data_file=sesh.bt_lfp_baseline_fname)
                baselfpV = lfpData[1]['voltage']

                btLFPData = lfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
                btRipPower, btRipZPower, _, _ = getRipplePower(
                    btLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs, meanPower=sesh.probeMeanRipplePower,
                    stdPower=sesh.probeStdRipplePower, skipTimePointsBackward=5, skipTimePointsFoward=5)
                probeLFPData = lfpV[sesh.probeLfpStart_idx:sesh.probeLfpEnd_idx]
                probeRipPower, _, _, _ = getRipplePower(
                    probeLFPData, omit_artifacts=False, skipTimePointsBackward=5, skipTimePointsFoward=5)

                baselineProbeLFPData = baselfpV[sesh.probeLfpStart_idx:sesh.probeLfpEnd_idx]
                probeBaselinePower, _, baselineProbeMeanRipplePower, baselineProbeStdRipplePower = getRipplePower(
                    baselineProbeLFPData, omit_artifacts=False, skipTimePointsBackward=5, skipTimePointsFoward=5)
                btBaselineLFPData = baselfpV[sesh.bt_lfp_start_idx:sesh.bt_lfp_end_idx]
                btBaselineRipplePower, _, _, _ = getRipplePower(
                    btBaselineLFPData, lfp_deflections=sesh.bt_lfp_artifact_idxs,
                    meanPower=baselineProbeMeanRipplePower, stdPower=baselineProbeStdRipplePower,
                    skipTimePointsBackward=5, skipTimePointsFoward=5)

                probeRawPowerDiff = probeRipPower - probeBaselinePower
                zmean = np.nanmean(probeRawPowerDiff)
                zstd = np.nanstd(probeRawPowerDiff)
                rawPowerDiff = btRipPower - btBaselineRipplePower
                zPowerDiff = (rawPowerDiff - zmean) / zstd

                MARGIN_SECS = 0.25
                MARGIN_PTS = int(MARGIN_SECS * 1500)
                allRipPows = np.empty((len(sesh.bt_lfp_artifact_idxs), 2*MARGIN_PTS))
                if MAKE_INDIVIDUAL_INTERRUPTION_PLOTS:
                    pp.setOutputSubDir(os.path.join(ratName, "LFP_Interruptions"))
                for ai, aidx in enumerate(sesh.bt_lfp_artifact_idxs):
                    # This is just annoying to deal with and shouldn't affect much, can add back in later
                    if aidx < MARGIN_PTS or aidx > len(btLFPData) - MARGIN_PTS:
                        continue

                    i1 = aidx - MARGIN_PTS
                    i2 = aidx + MARGIN_PTS

                    try:
                        allRipPows[ai, :] = zPowerDiff[i1:i2]
                    except Exception as e:
                        print(i1, i2, allRipPows.shape, zPowerDiff.shape,
                              i2 - i1, len(lfpT), len(btLFPData))
                        raise e

                    if MAKE_INDIVIDUAL_INTERRUPTION_PLOTS and ai < 10:
                        with pp.newFig(f"interruption_{sesh.name}_{ai}") as ax:
                            ax.plot(lfpT[i1:i2] - lfpT[sesh.bt_lfp_start_idx],
                                    btLFPData[i1:i2] / 1000, c="blue", label="lfp")
                            ax.plot(lfpT[i1:i2] - lfpT[sesh.bt_lfp_start_idx], zPowerDiff[i1:i2],
                                    c="orange", label="ripple power")
                            ax.set_ylim(-3, 10)

                if MAKE_INDIVIDUAL_INTERRUPTION_PLOTS:
                    pp.setOutputSubDir(ratName)

                detTimeRev = allRipPows[:, MARGIN_PTS:0:-1] > 2.5
                latencyToDetection = np.argmax(detTimeRev, axis=1).astype(float) / 1500.0
                with pp.newFig(f"lfp_detection_latency_z2-5_{sesh.name}") as ax:
                    ax.hist(latencyToDetection)

                detTimeRev = allRipPows[:, MARGIN_PTS:0:-1] > 0
                latencyToDetection = np.argmax(detTimeRev, axis=1).astype(float) / 1500.0
                with pp.newFig(f"lfp_detection_latency_z0_{sesh.name}") as ax:
                    ax.hist(latencyToDetection)

                m = np.nanmean(allRipPows, axis=0)
                s = np.nanstd(allRipPows, axis=0)
                with pp.newFig(f"lfp_psth_{sesh.name}") as ax:
                    xvals = np.linspace(-MARGIN_SECS, MARGIN_SECS, len(m))
                    ax.plot(xvals, m)
                    ax.fill_between(xvals, m - s, m + s, alpha=0.3)
                    y1, y2 = ax.get_ylim()
                    ax.plot([0, 0], [y1, y2])

    if len(animalNames) > 1:
        comboStr = "combo {}".format(" ".join(animalNames))
        print("======================\n", comboStr)
        pp.makeCombinedFigs(outputSubDir=comboStr)

    # if RUN_SHUFFLES:
        # pp.runShuffles()


if __name__ == "__main__":
    makeFigures(RUN_UNSPECIFIED=False,
                RUN_SWR_LATENCY_PLOTS=True, MAKE_INDIVIDUAL_INTERRUPTION_PLOTS=True)
    # makeFigures()
