import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
import seaborn as sns
import os
from scipy.stats import pearsonr, ttest_ind_from_stats
import statsmodels.api as sm
from matplotlib.lines import Line2D
from datetime import datetime
import math

from BTData import BTData
from BTSession import BTSession
from BTRestSession import BTRestSession


def makeThesisCommitteeFigs():
    SAVE_OUTPUT_PLOTS = True
    SHOW_OUTPUT_PLOTS = False
    SKIP_SINGLE_SESSION_PLOTS = True

    possibleDataDirs = ["/media/WDC7/", "/media/fosterlab/WDC7/", "/home/wcroughan/data/"]
    dataDir = None
    for dd in possibleDataDirs:
        if os.path.exists(dd):
            dataDir = dd
            break

    if dataDir == None:
        print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
        exit()

    globalOutputDir = os.path.join(dataDir, "figures", "thesisCommittee20220208")
    if not os.path.exists(globalOutputDir):
        os.makedirs(globalOutputDir)

    allWellNames = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])

    animalNames = ["B13", "B14", "Martin"]
    # animalNames = ["B13"]
    allSessions = []
    allSessionsByRat = {}
    allSessionsWithProbe = []
    allSessionsWithProbeByRat = {}

    for an in animalNames:
        if an == "B13":
            dataFilename = os.path.join(dataDir, "B13/processed_data/B13_bradtask.dat")
        elif an == "B14":
            dataFilename = os.path.join(dataDir, "B14/processed_data/B14_bradtask.dat")
        elif an == "Martin":
            dataFilename = os.path.join(dataDir, "Martin/processed_data/martin_bradtask.dat")
        else:
            raise Exception("Unknown rat " + an)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[an] = ratData.getSessions()
        allSessionsWithProbeByRat[an] = ratData.getSessions(lambda s: s.probe_performed)
        allSessions += ratData.getSessions()
        allSessionsWithProbe += ratData.getSessions(lambda s: s.probe_performed)

    def setupBehaviorTracePlot(axs, sesh, showAllWells=True, showHome=True, showAways=True, zorder=2):
        if isinstance(axs, np.ndarray):
            axs = axs.flat
        elif not isinstance(axs, list):
            axs = [axs]
        x1 = np.min(sesh.bt_pos_xs)
        x2 = np.max(sesh.bt_pos_xs)
        y1 = np.min(sesh.bt_pos_ys)
        y2 = np.max(sesh.bt_pos_ys)
        for ax in axs:
            if showAllWells:
                for w in allWellNames:
                    wx, wy = sesh.well_coords_map[str(w)]
                    ax.scatter(wx, wy, c="black", zorder=zorder)
            if showAways:
                for w in sesh.visited_away_wells:
                    wx, wy = sesh.well_coords_map[str(w)]
                    ax.scatter(wx, wy, c="blue", zorder=zorder)
            if showHome:
                wx, wy = sesh.well_coords_map[str(sesh.home_well)]
                ax.scatter(wx, wy, c="red", zorder=zorder)

            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)
            ax.tick_params(axis="both", which="both", label1On=False,
                           label2On=False, tick1On=False, tick2On=False)

    def saveOrShow(fname, outputDir=globalOutputDir, clearAxes=True):
        if SAVE_OUTPUT_PLOTS:
            if fname[0] != "/":
                fname = os.path.join(outputDir, fname)
            plt.savefig(fname, bbox_inches="tight")
            # plt.savefig(fname, dpi=800)
        if SHOW_OUTPUT_PLOTS:
            plt.show()

        if clearAxes:
            plt.cla()

    def plotIndividualAndAverage(ax, dataPoints, xvals, individualColor="black", avgColor="blue"):
        hm = np.nanmean(dataPoints, axis=0)
        hs = np.nanstd(dataPoints, axis=0)
        h1 = hm - hs
        h2 = hm + hs
        axs.plot(xvals, dataPoints.T, c=individualColor, lw=0.5)
        axs.plot(xvals, hm, color=avgColor)
        axs.fill_between(xvals, h1, h2, facecolor=avgColor, alpha=0.2)

    FIG_SCALE = 5
    fig = plt.figure(figsize=(FIG_SCALE, FIG_SCALE))
    axs = fig.subplots()
    for animalName in animalNames:
        print("==========================\n" + animalName)
        animalOutputDir = os.path.join(globalOutputDir, animalName)
        if not os.path.exists(animalOutputDir):
            os.makedirs(animalOutputDir)

        numSessions = len(allSessionsByRat[animalName])
        sessionIsInterruption = np.zeros((numSessions,)).astype(bool)
        for i in range(numSessions):
            if allSessionsByRat[animalName][i].isRippleInterruption:
                sessionIsInterruption[i] = True

        numSessionsWithProbe = len(allSessionsWithProbeByRat[animalName])
        sessionWithProbeIsInterruption = np.zeros((numSessionsWithProbe,)).astype(bool)
        for i in range(numSessionsWithProbe):
            if allSessionsWithProbeByRat[animalName][i].isRippleInterruption:
                sessionWithProbeIsInterruption[i] = True

        # =============
        # well find latencies
        homeFindTimes = np.empty((numSessions, 10))
        homeFindTimes[:] = np.nan
        for si, sesh in enumerate(allSessionsByRat[animalName]):
            t1 = np.array(sesh.home_well_find_times)
            t0 = np.array(np.hstack(([sesh.bt_pos_ts[0]], sesh.away_well_leave_times)))
            if not sesh.ended_on_home:
                t0 = t0[0:-1]
            times = (t1 - t0) / BTSession.TRODES_SAMPLING_RATE
            homeFindTimes[si, 0:sesh.num_home_found] = times

        awayFindTimes = np.empty((numSessions, 10))
        awayFindTimes[:] = np.nan
        for si, sesh in enumerate(allSessionsByRat[animalName]):
            t1 = np.array(sesh.away_well_find_times)
            t0 = np.array(sesh.home_well_leave_times)
            if sesh.ended_on_home:
                t0 = t0[0:-1]
            times = (t1 - t0) / BTSession.TRODES_SAMPLING_RATE
            awayFindTimes[si, 0:sesh.num_away_found] = times

        fig.set_figheight(FIG_SCALE / 2)
        fig.set_figwidth(FIG_SCALE / 2)

        pltx = np.arange(10)+1
        # ymax = max(np.nanmax(homeFindTimes), np.nanmax(awayFindTimes))
        ymax = 300
        plotIndividualAndAverage(axs, homeFindTimes, pltx)
        axs.set_ylim(0, ymax)
        axs.set_xlim(1, 10)
        axs.set_xticks(np.arange(0, 10, 2) + 1)
        saveOrShow("latency_to_home", outputDir=animalOutputDir)

        plotIndividualAndAverage(axs, awayFindTimes, pltx)
        axs.set_ylim(0, ymax)
        axs.set_xlim(1, 10)
        axs.set_xticks(np.arange(0, 10, 2) + 1)
        saveOrShow("latency_to_away", outputDir=animalOutputDir)

        # =============
        # well find latencies by condition
        ymax = 100
        swrHomeFindTimes = homeFindTimes[sessionIsInterruption, :]
        ctrlHomeFindTimes = homeFindTimes[np.logical_not(sessionIsInterruption), :]
        plotIndividualAndAverage(axs, swrHomeFindTimes, pltx,
                                 individualColor="orange", avgColor="orange")
        plotIndividualAndAverage(axs, ctrlHomeFindTimes, pltx,
                                 individualColor="cyan", avgColor="cyan")
        axs.set_ylim(0, ymax)
        axs.set_xlim(1, 10)
        axs.set_xticks(np.arange(0, 10, 2) + 1)
        saveOrShow("latency_to_home_by_condition", outputDir=animalOutputDir)

        swrAwayFindTimes = awayFindTimes[sessionIsInterruption, :]
        ctrlAwayFindTimes = awayFindTimes[np.logical_not(sessionIsInterruption), :]
        plotIndividualAndAverage(axs, swrAwayFindTimes, pltx,
                                 individualColor="orange", avgColor="orange")
        plotIndividualAndAverage(axs, ctrlAwayFindTimes, pltx,
                                 individualColor="cyan", avgColor="cyan")
        axs.set_ylim(0, ymax)
        axs.set_xlim(1, 10)
        axs.set_xticks(np.arange(0, 10, 2) + 1)
        saveOrShow("latency_to_away_by_condition", outputDir=animalOutputDir)

        # =============
        # average vel in probe
        windowSize = 15
        windowsSlide = 3
        t0Array = np.arange(0, 60*5-windowSize+windowsSlide, windowsSlide)

        avgVels = np.empty((numSessionsWithProbe, len(t0Array)))
        avgVels[:] = np.nan

        for si, sesh in enumerate(allSessionsWithProbeByRat[animalName]):
            for ti, t0 in enumerate(t0Array):
                avgVels[si, ti] = sesh.mean_vel(True, timeInterval=[t0, t0+windowSize])

        # ymax = np.max(avgVels)
        plotIndividualAndAverage(axs, avgVels, t0Array)
        # axs.set_ylim(0, ymax)
        # axs.set_xlim(1, 10)
        # axs.set_xticks(np.arange(0, 10, 2) + 1)
        saveOrShow("avgVel", outputDir=animalOutputDir)

        if SKIP_SINGLE_SESSION_PLOTS:
            continue

        fig.set_figheight(FIG_SCALE)
        fig.set_figwidth(FIG_SCALE)

        for sesh in allSessionsByRat[animalName]:
            print(sesh.name)
            seshOutputDir = os.path.join(animalOutputDir, sesh.name)
            if not os.path.exists(seshOutputDir):
                os.makedirs(seshOutputDir)

            # =============
            # Raw behavioral trace (task)
            axs.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, c="#deac7f")
            setupBehaviorTracePlot(axs, sesh)
            saveOrShow("raw_task", outputDir=seshOutputDir)

            # =============
            # Trace split by trial
            numHomes = sesh.num_home_found
            if numHomes > 1:
                fig.set_figheight(2 * FIG_SCALE / 2)
                fig.set_figwidth(numHomes * FIG_SCALE / 2)
                axs = fig.subplots(2, numHomes)
                for fti, ft in enumerate(sesh.home_well_find_pos_idxs):
                    i1 = 0 if fti == 0 else sesh.away_well_leave_pos_idxs[fti-1]
                    i2 = ft
                    axs[0, fti].plot(sesh.bt_pos_xs[i1:i2],
                                     sesh.bt_pos_ys[i1:i2], c="#deac7f", zorder=3, lw=3)
                    if fti > 0:
                        awx, awy = sesh.well_coords_map[str(sesh.visited_away_wells[fti-1])]
                        axs[0, fti].scatter(awx, awy, c="blue", zorder=2)
                for fti, ft in enumerate(sesh.away_well_find_pos_idxs):
                    i1 = sesh.home_well_leave_pos_idxs[fti]
                    i2 = ft
                    axs[1, fti].plot(sesh.bt_pos_xs[i1:i2],
                                     sesh.bt_pos_ys[i1:i2], c="#deac7f", zorder=3, lw=3)

                    awx, awy = sesh.well_coords_map[str(sesh.visited_away_wells[fti])]
                    axs[1, fti].scatter(awx, awy, c="blue", zorder=2)
                setupBehaviorTracePlot(axs, sesh, showAllWells=True, showAways=False, zorder=1)
                if sesh.ended_on_home:
                    axs[1, -1].cla()
                saveOrShow("all_trials", outputDir=seshOutputDir)
                axs = fig.subplots(1, 1)
                fig.set_figheight(FIG_SCALE)
                fig.set_figwidth(FIG_SCALE)

            if False:
                for fti, ft in enumerate(sesh.home_well_find_pos_idxs):
                    i1 = 0 if fti == 0 else sesh.away_well_leave_pos_idxs[fti-1]
                    i2 = ft
                    axs.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], c="#deac7f")
                    setupBehaviorTracePlot(axs, sesh)
                    saveOrShow("home_find_{}".format(fti), outputDir=seshOutputDir)

                for fti, ft in enumerate(sesh.away_well_find_pos_idxs):
                    i1 = sesh.home_well_leave_pos_idxs[fti]
                    i2 = ft
                    axs.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], c="#deac7f")
                    setupBehaviorTracePlot(axs, sesh)
                    saveOrShow("away_find_{}".format(fti), outputDir=seshOutputDir)

        for sesh in allSessionsWithProbeByRat[animalName]:
            print(sesh.name)
            seshOutputDir = os.path.join(animalOutputDir, sesh.name)
            if not os.path.exists(seshOutputDir):
                os.makedirs(seshOutputDir)

            # =============
            # Raw behavioral trace (probe)
            axs.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
            setupBehaviorTracePlot(axs, sesh)
            saveOrShow("raw_probe".format(sesh.name), outputDir=seshOutputDir)

    # if isinstance(allData, list):
    #     self.all_sessions = []
    #     self.all_rest_sessions = []
    #     for ad in allData:
    #         self.all_sessions += ad.getSessions()
    #         self.all_rest_sessions += ad.getRestSessions()
    # else:
    #     self.all_sessions = allData.getSessions()
    #     self.all_rest_sessions = allData.getRestSessions()
    # self.all_sessions_with_probe = [s for s in self.all_sessions if s.probe_performed]
    # self.tlbls = [self.trial_label(sesh) for sesh in self.all_sessions]
    # self.tlbls_with_probe = [self.trial_label(sesh) for sesh in self.all_sessions_with_probe]

    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         statsFileName = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S_stats.txt"))
    #         self.statsFile = open(statsFileName, "w")
    #         idxKeyFileName = os.path.join(
    #             output_dir, datetime.now().strftime("%Y%m%d_%H%M%S_idx_key.txt"))
    #         with open(idxKeyFileName, "w") as idxKeyFile:
    #             idxKeyFile.write("All sessions:\n")
    #             for si, s in enumerate(self.all_sessions):
    #                 idxKeyFile.write("{}: {} - {} (home {}, {} stims)\n".format(si, s.name,
    #                                  "SWR " if s.isRippleInterruption else "CTRL", s.home_well, len(s.bt_interruption_pos_idxs)))

    #             idxKeyFile.write("All sessions with probe:\n")
    #             for si, s in enumerate(self.all_sessions_with_probe):
    #                 idxKeyFile.write("{}: {} - {} (home {}, {} stims)\n".format(si, s.name,
    #                                  "SWR " if s.isRippleInterruption else "CTRL", s.home_well, len(s.bt_interruption_pos_idxs)))

    #             idxKeyFile.write("All rest sessions\n")
    #             for si, s in enumerate(self.all_rest_sessions):
    #                 idxKeyFile.write(str(si) + ": " + s.name + "\n")


if __name__ == "__main__":
    makeThesisCommitteeFigs()
