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
from itertools import groupby
import warnings

from BTData import BTData
from BTSession import BTSession
from BTRestSession import BTRestSession


def makeThesisCommitteeFigs():
    SAVE_OUTPUT_PLOTS = True
    SHOW_OUTPUT_PLOTS = False
    FIG_SCALE = 5

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

    def saveOrShow(fname, outputDir=globalOutputDir, clearAxes=True, statsFile=None):
        if SAVE_OUTPUT_PLOTS:
            if fname[0] != "/":
                fname = os.path.join(outputDir, fname)
            plt.savefig(fname, bbox_inches="tight", dpi=200)
            # plt.savefig(fname, dpi=800)
        if SHOW_OUTPUT_PLOTS:
            plt.show()

        if clearAxes:
            plt.clf()

        if statsFile is not None:
            statsFile.write("Saved {}\n\n".format(fname))

        ax = plt.subplot(111)
        ax.cla()
        return ax

    def plotIndividualAndAverage(ax, dataPoints, xvals, individualColor="grey", avgColor="blue", spread="std", individualZOrder=1, averageZOrder=2):
        hm = np.nanmean(dataPoints, axis=0)
        hs = np.nanstd(dataPoints, axis=0)
        if spread == "sem":
            n = dataPoints.shape[0] - np.count_nonzero(np.isnan(dataPoints), axis=0)
            hs = hs / np.sqrt(n)
        h1 = hm - hs
        h2 = hm + hs
        ax.plot(xvals, dataPoints.T, c=individualColor, lw=0.5, zorder=individualZOrder)
        ax.plot(xvals, hm, color=avgColor, zorder=averageZOrder)
        ax.fill_between(xvals, h1, h2, facecolor=avgColor, alpha=0.3, zorder=averageZOrder)

    def runTwoWayShuffle(vals, cat1, cat2, cat1Name="A", cat2Name="B", dataName="data", numShuffles=500, statsFile=None):
        if statsFile is not None:
            statsFile.write("=======================\nTwo way shuffle for {}\n\tcat1: {}\n\tcat2: {}\n\tnumShuffles: {}\n".format(
                dataName, cat1Name, cat2Name, numShuffles))
        df = pd.DataFrame(data={dataName: vals, cat1Name: cat1, cat2Name: cat2})

        ma = df.groupby(cat1Name).mean()
        observedADiff = ma.iat[1, 0] - ma.iat[0, 0]
        mb = df.groupby(cat2Name).mean()
        observedBDiff = mb.iat[1, 0] - mb.iat[0, 0]

        sdf1 = df[df[cat2Name] == df.at[df.index[0], cat2Name]].drop(
            columns=cat2Name).reset_index(drop=True)
        sdf2 = df[df[cat2Name] != df.at[df.index[0], cat2Name]].drop(
            columns=cat2Name).reset_index(drop=True)
        mab1 = sdf1.groupby(cat1Name).mean()
        mab2 = sdf2.groupby(cat1Name).mean()
        observedABDiff = (mab2.iat[1, 0] - mab1.iat[1, 0]) - (mab2.iat[0, 0] - mab1.iat[0, 0])

        sdf1 = df[df[cat1Name] == df.at[df.index[0], cat1Name]].drop(
            columns=cat1Name).reset_index(drop=True)
        sdf2 = df[df[cat1Name] != df.at[df.index[0], cat1Name]].drop(
            columns=cat1Name).reset_index(drop=True)
        mab1 = sdf1.groupby(cat2Name).mean()
        mab2 = sdf2.groupby(cat2Name).mean()
        observedBADiff = (mab2.iat[1, 0] - mab1.iat[1, 0]) - (mab2.iat[0, 0] - mab1.iat[0, 0])

        # global effect of A: ignore cat2, run 1-way shuffle on cat1
        sdf = df.drop(columns=cat2Name)
        shuffleValues = np.empty((numShuffles,))
        for si in range(numShuffles):
            sdf[cat1Name] = sdf[cat1Name].sample(frac=1, random_state=1).reset_index(drop=True)
            ma = sdf.groupby(cat1Name).mean()
            shuffleValues[si] = ma.iat[1, 0] - ma.iat[0, 0]
        pctile = np.count_nonzero(shuffleValues < observedADiff) / numShuffles
        if statsFile is not None:
            shufMean = np.nanmean(shuffleValues)
            shufStd = np.nanstd(shuffleValues)
            statsFile.write("Global effect of {}\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                cat1Name, shufMean, shufStd, observedADiff, pctile))

        # global effect of B: same but drop cat1, run shuffle on cat2
        sdf = df.drop(columns=cat1Name)
        shuffleValues = np.empty((numShuffles,))
        for si in range(numShuffles):
            sdf[cat2Name] = sdf[cat2Name].sample(frac=1, random_state=1).reset_index(drop=True)
            ma = sdf.groupby(cat2Name).mean()
            shuffleValues[si] = ma.iat[1, 0] - ma.iat[0, 0]
        pctile = np.count_nonzero(shuffleValues < observedBDiff) / numShuffles
        if statsFile is not None:
            shufMean = np.nanmean(shuffleValues)
            shufStd = np.nanstd(shuffleValues)
            statsFile.write("Global effect of {}\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                cat2Name, shufMean, shufStd, observedBDiff, pctile))

        # global effect of A-B: shuf cat1
        sdf1 = df[df[cat2Name] == df.at[df.index[0], cat2Name]].drop(
            columns=cat2Name).reset_index(drop=True)
        sdf2 = df[df[cat2Name] != df.at[df.index[0], cat2Name]].drop(
            columns=cat2Name).reset_index(drop=True)
        shuffleValues = np.empty((numShuffles,))
        for si in range(numShuffles):
            sdf1[cat1Name] = sdf1[cat1Name].sample(frac=1, random_state=1).reset_index(drop=True)
            sdf2[cat1Name] = sdf2[cat1Name].sample(frac=1, random_state=1).reset_index(drop=True)
            mab1 = sdf1.groupby(cat1Name).mean()
            mab2 = sdf2.groupby(cat1Name).mean()
            shuffleValues[si] = (mab2.iat[1, 0] - mab1.iat[1, 0]) - \
                (mab2.iat[0, 0] - mab1.iat[0, 0])
        pctile = np.count_nonzero(shuffleValues < observedABDiff) / numShuffles
        if statsFile is not None:
            shufMean = np.nanmean(shuffleValues)
            shufStd = np.nanstd(shuffleValues)
            statsFile.write("Interaction effect (shuffling {})\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                cat1Name, shufMean, shufStd, observedABDiff, pctile))

        # global effect of A-B: shuf cat2
        sdf1 = df[df[cat1Name] == df.at[df.index[0], cat1Name]].drop(
            columns=cat1Name).reset_index(drop=True)
        sdf2 = df[df[cat1Name] != df.at[df.index[0], cat1Name]].drop(
            columns=cat1Name).reset_index(drop=True)
        shuffleValues = np.empty((numShuffles,))
        for si in range(numShuffles):
            sdf1[cat2Name] = sdf1[cat2Name].sample(frac=1, random_state=1).reset_index(drop=True)
            sdf2[cat2Name] = sdf2[cat2Name].sample(frac=1, random_state=1).reset_index(drop=True)
            mab1 = sdf1.groupby(cat2Name).mean()
            mab2 = sdf2.groupby(cat2Name).mean()
            shuffleValues[si] = (mab2.iat[1, 0] - mab1.iat[1, 0]) - \
                (mab2.iat[0, 0] - mab1.iat[0, 0])
        pctile = np.count_nonzero(shuffleValues < observedBADiff) / numShuffles
        if statsFile is not None:
            shufMean = np.nanmean(shuffleValues)
            shufStd = np.nanstd(shuffleValues)
            statsFile.write("Interaction effect (shuffling {})\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                cat2Name, shufMean, shufStd, observedBADiff, pctile))

        # subgroup effect of A: separate df by cat2, for each unique cat2 value shuffle cat1
        uniqueVals = df[cat2Name].unique()
        for uv in uniqueVals:
            sdf = df[df[cat2Name] == uv].drop(columns=cat2Name).reset_index(drop=True)
            ma = sdf.groupby(cat1Name).mean()
            observedADiffUV = ma.iat[1, 0] - ma.iat[0, 0]

            shuffleValues = np.empty((numShuffles,))
            for si in range(numShuffles):
                sdf[cat1Name] = sdf[cat1Name].sample(frac=1, random_state=1).reset_index(drop=True)
                ma = sdf.groupby(cat1Name).mean()
                shuffleValues[si] = ma.iat[1, 0] - ma.iat[0, 0]
            pctile = np.count_nonzero(shuffleValues < observedADiffUV) / numShuffles
            if statsFile is not None:
                shufMean = np.nanmean(shuffleValues)
                shufStd = np.nanstd(shuffleValues)
                statsFile.write("Shuffling {} just in subset where {} == {}\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                    cat1Name, cat2Name, uv, shufMean, shufStd, observedADiffUV, pctile))

        # subgroup effect of B: separate df by cat1, for each unique cat1 value shuffle cat2
        uniqueVals = df[cat1Name].unique()
        for uv in uniqueVals:
            sdf = df[df[cat1Name] == uv].drop(columns=cat1Name).reset_index(drop=True)
            ma = sdf.groupby(cat2Name).mean()
            observedBDiffUV = ma.iat[1, 0] - ma.iat[0, 0]

            shuffleValues = np.empty((numShuffles,))
            for si in range(numShuffles):
                sdf[cat2Name] = sdf[cat2Name].sample(frac=1, random_state=1).reset_index(drop=True)
                ma = sdf.groupby(cat2Name).mean()
                shuffleValues[si] = ma.iat[1, 0] - ma.iat[0, 0]
            pctile = np.count_nonzero(shuffleValues < observedBDiffUV) / numShuffles
            if statsFile is not None:
                shufMean = np.nanmean(shuffleValues)
                shufStd = np.nanstd(shuffleValues)
                statsFile.write("Shuffling {} just in subset where {} == {}\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                    cat2Name, cat1Name, uv, shufMean, shufStd, observedBDiffUV, pctile))

        if statsFile is not None:
            statsFile.write("\n")

    def runOneWayShuffle(vals, cat, catName="A", dataName="data", numShuffles=500, statsFile=None):
        if statsFile is not None:
            statsFile.write("=======================\nOne way shuffle for {}\n\tcat: {}\n\tnumShuffles: {}\n".format(
                dataName, catName, numShuffles))
        df = pd.DataFrame(data={dataName: vals, catName: cat})

        ma = df.groupby(catName).mean()
        observedADiff = ma.iat[1, 0] - ma.iat[0, 0]

        # global effect of category
        sdf = df.copy()
        shuffleValues = np.empty((numShuffles,))
        for si in range(numShuffles):
            sdf[catName] = sdf[catName].sample(frac=1, random_state=1).reset_index(drop=True)
            ma = sdf.groupby(catName).mean()
            shuffleValues[si] = ma.iat[1, 0] - ma.iat[0, 0]
        pctile = np.count_nonzero(shuffleValues < observedADiff) / numShuffles
        if statsFile is not None:
            shufMean = np.nanmean(shuffleValues)
            shufStd = np.nanstd(shuffleValues)
            statsFile.write("Global effect of {}\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                catName, shufMean, shufStd, observedADiff, pctile))
            statsFile.write("\n")

    def boxPlot(ax, yvals, categories, categories2=None, axesNames=None, violin=False, doStats=True, statsFile=None):
        if categories2 is None:
            categories2 = ["a" for _ in categories]
            cat2IsFake = True
            if axesNames is not None:
                axesNames.append("Z")
        else:
            cat2IsFake = False

        # Same sorting here as in perseveration plot function so colors are always the same
        sortList = [x + str(xi) for xi, x in enumerate(categories)]
        categories = [x for _, x in sorted(zip(sortList, categories))]
        yvals = [x for _, x in sorted(zip(sortList, yvals))]
        categories2 = [x for _, x in sorted(zip(sortList, categories2))]

        if axesNames is None:
            hideAxes = True
            axesNames = ["X", "Y", "Z"]
        else:
            hideAxes = False

        axesNamesNoSpaces = [a.replace(" ", "_") for a in axesNames]
        s = pd.Series([categories, yvals, categories2], index=axesNamesNoSpaces)

        ax.cla()
        pal = sns.color_palette(palette=["cyan", "orange"])
        if violin:
            plotWorked = False
            swarmDotSize = 3
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=UserWarning)
                while not plotWorked:
                    try:
                        p1 = sns.violinplot(ax=ax, hue=axesNamesNoSpaces[0],
                                            y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, palette=pal, linewidth=0.2, cut=0, zorder=1)
                        p2 = sns.swarmplot(ax=ax, hue=axesNamesNoSpaces[0],
                                           y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, color="0.25", size=swarmDotSize, dodge=True, zorder=3)
                        plotWorked = True
                    except UserWarning as e:
                        swarmDotSize /= 2
                        print("reduing dots to {}".format(swarmDotSize))

                        ax.cla()

                        if swarmDotSize < 0.1:
                            raise e
        else:
            p1 = sns.boxplot(
                ax=ax, hue=axesNamesNoSpaces[0], y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, palette=pal, zorder=1)
            p2 = sns.swarmplot(ax=ax, hue=axesNamesNoSpaces[0],
                               y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, color="0.25", dodge=True, zorder=3)

        if cat2IsFake:
            p1.set(xticklabels=[])
        # lg = p2.get_legend()
        # lg.remove()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], fontsize=6).set_zorder(2)

        if not hideAxes:
            if not cat2IsFake:
                ax.set_xlabel(axesNames[2])
            ax.set_ylabel(axesNames[1])

        if doStats:
            if cat2IsFake:
                runOneWayShuffle(
                    yvals, categories, catName=axesNamesNoSpaces[0], dataName=axesNamesNoSpaces[1], statsFile=statsFile)
            else:
                runTwoWayShuffle(yvals, categories, categories2,
                                 cat1Name=axesNamesNoSpaces[0], cat2Name=axesNamesNoSpaces[2], dataName=axesNamesNoSpaces[1], statsFile=statsFile)

    def numWellsVisited(nearestWells, countReturns=False, wellSubset=None):
        g = groupby(nearestWells)
        if wellSubset is None:
            wellSubset = allWellNames
        if countReturns:
            return len([k for k, _ in g if k in wellSubset])
        else:
            return len(set([k for k, _ in g if k in wellSubset]))

    def makeFiguresForSessions(sessions, outputDir, SKIP_SINGLE_SESSION_PLOTS=True, SKIP_NO_PROBE_SESSION_PLOTS=False, SKIP_PROBE_SESSION_PLOTS=False):
        fig = plt.figure(figsize=(FIG_SCALE, FIG_SCALE))
        axs = fig.subplots()

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        statsFileName = os.path.join(outputDir, datetime.now().strftime("%Y%m%d_%H%M%S_stats.txt"))
        statsFile = open(statsFileName, "w")

        numSessions = len(sessions)
        sessionIsInterruption = np.zeros((numSessions,)).astype(bool)
        for i in range(numSessions):
            if sessions[i].isRippleInterruption:
                sessionIsInterruption[i] = True

        sessionsWithProbe = [s for s in sessions if s.probe_performed]
        numSessionsWithProbe = len(sessionsWithProbe)
        sessionWithProbeIsInterruption = np.zeros((numSessionsWithProbe,)).astype(bool)
        for i in range(numSessionsWithProbe):
            if sessionsWithProbe[i].isRippleInterruption:
                sessionWithProbeIsInterruption[i] = True

        if True:
            if False:
                # =============
                # well find latencies
                homeFindTimes = np.empty((numSessions, 10))
                homeFindTimes[:] = np.nan
                for si, sesh in enumerate(sessions):
                    t1 = np.array(sesh.home_well_find_times)
                    t0 = np.array(np.hstack(([sesh.bt_pos_ts[0]], sesh.away_well_leave_times)))
                    if not sesh.ended_on_home:
                        t0 = t0[0:-1]
                    times = (t1 - t0) / BTSession.TRODES_SAMPLING_RATE
                    homeFindTimes[si, 0:sesh.num_home_found] = times

                awayFindTimes = np.empty((numSessions, 10))
                awayFindTimes[:] = np.nan
                for si, sesh in enumerate(sessions):
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
                axs = saveOrShow("latency_to_home", outputDir=outputDir, statsFile=statsFile)

                plotIndividualAndAverage(axs, awayFindTimes, pltx)
                axs.set_ylim(0, ymax)
                axs.set_xlim(1, 10)
                axs.set_xticks(np.arange(0, 10, 2) + 1)
                axs = saveOrShow("latency_to_away", outputDir=outputDir, statsFile=statsFile)

                # =============
                # well find latencies by condition
                ymax = 100
                swrHomeFindTimes = homeFindTimes[sessionIsInterruption, :]
                ctrlHomeFindTimes = homeFindTimes[np.logical_not(sessionIsInterruption), :]
                plotIndividualAndAverage(axs, swrHomeFindTimes, pltx,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(axs, ctrlHomeFindTimes, pltx,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                axs.set_ylim(0, ymax)
                axs.set_xlim(1, 10)
                axs.set_xticks(np.arange(0, 10, 2) + 1)
                axs = saveOrShow("latency_to_home_by_condition",
                                 outputDir=outputDir, statsFile=statsFile)

                swrAwayFindTimes = awayFindTimes[sessionIsInterruption, :]
                ctrlAwayFindTimes = awayFindTimes[np.logical_not(sessionIsInterruption), :]
                plotIndividualAndAverage(axs, swrAwayFindTimes, pltx,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(axs, ctrlAwayFindTimes, pltx,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                axs.set_ylim(0, ymax)
                axs.set_xlim(1, 10)
                axs.set_xticks(np.arange(0, 10, 2) + 1)
                axs = saveOrShow("latency_to_away_by_condition",
                                 outputDir=outputDir, statsFile=statsFile)

                # =============
                # average vel in probe
                windowSize = 15
                windowsSlide = 3
                t0Array = np.arange(0, 60*5-windowSize+windowsSlide, windowsSlide)
                avgVels = np.empty((numSessionsWithProbe, len(t0Array)))
                avgVels[:] = np.nan
                for si, sesh in enumerate(sessionsWithProbe):
                    for ti, t0 in enumerate(t0Array):
                        avgVels[si, ti] = sesh.mean_vel(True, timeInterval=[t0, t0+windowSize])
                plotIndividualAndAverage(axs, avgVels, t0Array)
                axs.set_xticks(np.arange(0, 60*5+1, 60))
                axs = saveOrShow("avgVel_probe", outputDir=outputDir, statsFile=statsFile)

                # =============
                # average vel in probe by condition
                swrAvgVels = avgVels[sessionWithProbeIsInterruption, :]
                ctrlAvgVels = avgVels[np.logical_not(sessionWithProbeIsInterruption), :]
                plotIndividualAndAverage(axs, swrAvgVels, t0Array,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(axs, ctrlAvgVels, t0Array,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                axs.set_xticks(np.arange(0, 60*5+1, 60))
                axs = saveOrShow("avgVel_probe_by_condition",
                                 outputDir=outputDir, statsFile=statsFile)

                # =============
                # average vel in task
                windowSize = 60
                windowsSlide = 12
                t0Array = np.arange(0, 60*20-windowSize+windowsSlide, windowsSlide)
                avgVels = np.empty((numSessions, len(t0Array)))
                avgVels[:] = np.nan
                for si, sesh in enumerate(sessions):
                    duration = (sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]) / \
                        BTSession.TRODES_SAMPLING_RATE
                    for ti, t0 in enumerate(t0Array):
                        if t0+windowSize > duration:
                            axs.scatter(t0, avgVels[si, ti-1], c="black", zorder=0, s=1)
                            break
                        avgVels[si, ti] = sesh.mean_vel(False, timeInterval=[t0, t0+windowSize])
                plotIndividualAndAverage(axs, avgVels, t0Array)
                axs.set_xticks(np.arange(0, 60*20+1, 60*5))
                axs = saveOrShow("avgVel_task", outputDir=outputDir, statsFile=statsFile)

                # =============
                # average vel in task by condition
                swrAvgVels = avgVels[sessionIsInterruption, :]
                ctrlAvgVels = avgVels[np.logical_not(sessionIsInterruption), :]
                plotIndividualAndAverage(axs, swrAvgVels, t0Array,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(axs, ctrlAvgVels, t0Array,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                axs.set_xticks(np.arange(0, 60*20+1, 60*5))
                axs = saveOrShow("avgVel_task_by_condition",
                                 outputDir=outputDir, statsFile=statsFile)

                # =============
                # frac time exploring in probe
                windowSize = 15
                windowsSlide = 3
                t0Array = np.arange(0, 60*5-windowSize+windowsSlide, windowsSlide)
                fracExplores = np.empty((numSessionsWithProbe, len(t0Array)))
                fracExplores[:] = np.nan
                for si, sesh in enumerate(sessionsWithProbe):
                    for ti, t0 in enumerate(t0Array):
                        fracExplores[si, ti] = sesh.prop_time_in_bout_state(
                            True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[t0, t0+windowSize])
                plotIndividualAndAverage(axs, fracExplores, t0Array)
                axs.set_xticks(np.arange(0, 60*5+1, 60))
                axs = saveOrShow("fracExplore_probe", outputDir=outputDir, statsFile=statsFile)

                # =============
                # frac time exploring in probe by condition
                swrfracExplores = fracExplores[sessionWithProbeIsInterruption, :]
                ctrlfracExplores = fracExplores[np.logical_not(sessionWithProbeIsInterruption), :]
                plotIndividualAndAverage(axs, swrfracExplores, t0Array,
                                         individualColor="orange", avgColor="orange", spread="sem")
                plotIndividualAndAverage(axs, ctrlfracExplores, t0Array,
                                         individualColor="cyan", avgColor="cyan", spread="sem")
                axs.set_xticks(np.arange(0, 60*5+1, 60))
                axs = saveOrShow("fracExplore_probe_by_condition",
                                 outputDir=outputDir, statsFile=statsFile)

                if False:
                    # =============
                    # frac time exploring in task
                    windowSize = 60
                    windowsSlide = 12
                    t0Array = np.arange(0, 60*20-windowSize+windowsSlide, windowsSlide)
                    fracExplores = np.empty((numSessions, len(t0Array)))
                    fracExplores[:] = np.nan
                    for si, sesh in enumerate(sessions):
                        duration = (sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]) / \
                            BTSession.TRODES_SAMPLING_RATE
                        for ti, t0 in enumerate(t0Array):
                            if t0+windowSize > duration:
                                axs.scatter(t0, fracExplores[si, ti-1], c="black", zorder=0, s=1)
                                break
                            fracExplores[si, ti] = sesh.prop_time_in_bout_state(
                                False, BTSession.BOUT_STATE_EXPLORE, timeInterval=[t0, t0+windowSize])
                    plotIndividualAndAverage(axs, fracExplores, t0Array)
                    axs.set_xticks(np.arange(0, 60*20+1, 60*5))
                    axs = saveOrShow("fracExplore_task", outputDir=outputDir, statsFile=statsFile)

                    # =============
                    # frac time exploring in task by condition
                    swrfracExplores = fracExplores[sessionIsInterruption, :]
                    ctrlfracExplores = fracExplores[np.logical_not(sessionIsInterruption), :]
                    plotIndividualAndAverage(axs, swrfracExplores, t0Array,
                                             individualColor="orange", avgColor="orange", spread="sem")
                    plotIndividualAndAverage(axs, ctrlfracExplores, t0Array,
                                             individualColor="cyan", avgColor="cyan", spread="sem")
                    axs.set_xticks(np.arange(0, 60*20+1, 60*5))
                    axs = saveOrShow("fracExplore_task_by_condition",
                                     outputDir=outputDir, statsFile=statsFile)

                # =============
                # Interruption condition probe traces
                numSWRSessionsWithProbe = np.count_nonzero(sessionWithProbeIsInterruption)
                fig.set_figheight(math.ceil(numSWRSessionsWithProbe / 2) * FIG_SCALE / 2)
                fig.set_figwidth(2 * FIG_SCALE / 2)
                axs = fig.subplots(math.ceil(numSWRSessionsWithProbe / 2), 2)
                for si, sesh in enumerate(np.array(sessionsWithProbe)[sessionWithProbeIsInterruption]):
                    axs[si // 2, si % 2].plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    setupBehaviorTracePlot(axs[si // 2, si % 2], sesh)
                if numSWRSessionsWithProbe % 2 == 1:
                    axs[-1, 1].cla()
                    axs[-1, 1].tick_params(axis="both", which="both", label1On=False,
                                           label2On=False, tick1On=False, tick2On=False)
                axs = saveOrShow("swr_probe_traces", outputDir=outputDir, statsFile=statsFile)
                fig.set_figheight(FIG_SCALE)
                fig.set_figwidth(FIG_SCALE)

                # =============
                # control condition probe traces
                numCtrlSessionsWithProbe = np.count_nonzero(
                    np.logical_not(sessionWithProbeIsInterruption))
                fig.set_figheight(math.ceil(numCtrlSessionsWithProbe / 2) * FIG_SCALE / 2)
                fig.set_figwidth(2 * FIG_SCALE / 2)
                axs = fig.subplots(math.ceil(numCtrlSessionsWithProbe / 2), 2)
                for si, sesh in enumerate(np.array(sessionsWithProbe)[np.logical_not(sessionWithProbeIsInterruption)]):
                    axs[si // 2, si % 2].plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
                    setupBehaviorTracePlot(axs[si // 2, si % 2], sesh)
                if numCtrlSessionsWithProbe % 2 == 1:
                    axs[-1, 1].cla()
                    axs[-1, 1].tick_params(axis="both", which="both", label1On=False,
                                           label2On=False, tick1On=False, tick2On=False)
                axs = saveOrShow("ctrl_probe_traces", outputDir=outputDir, statsFile=statsFile)
                fig.set_figheight(FIG_SCALE)
                fig.set_figwidth(FIG_SCALE)

            # =============
            # probe latency to home
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            probeLatHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                else "Control" for i in range(numSessionsWithProbe)]
            probeLatHome = [min(s.getLatencyToWell(
                True, s.home_well, returnSeconds=True, emptyVal=300), 300) for s in sessionsWithProbe]
            boxPlot(axs, probeLatHome, probeLatHomeCats,
                    axesNames=["Condition", "Latency to Home (s)"], statsFile=statsFile)
            axs = saveOrShow("probe_latency_to_home", outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe number of visits to home
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            probeNumVisitsHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                      else "Control" for i in range(numSessionsWithProbe)]
            probeNumVisitsHome = [s.num_well_entries(True, s.home_well)
                                  for s in sessionsWithProbe]
            boxPlot(axs, probeNumVisitsHome, probeNumVisitsHomeCats, axesNames=[
                    "Condition", "Number of visits to home well"], statsFile=statsFile)
            axs = saveOrShow("probe_num_visits_to_home", outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe latency to all aways
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            probeLatAwaysCats = []
            probeLatAways = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    probeLatAwaysCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    probeLatAways.append(min(sesh.getLatencyToWell(
                        True, aw, returnSeconds=True, emptyVal=300), 300))
            boxPlot(axs, probeLatAways, probeLatAwaysCats, axesNames=[
                    "Condition", "Latency to Aways (s)"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_latency_to_aways", outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe num visits to aways
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            probeNumVisitsAwaysCats = []
            probeNumVisitsAways = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    probeNumVisitsAwaysCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    probeNumVisitsAways.append(sesh.num_well_entries(True, aw))
            boxPlot(axs, probeNumVisitsAways, probeNumVisitsAwaysCats, axesNames=[
                    "Condition", "Number of visits to away well"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_num_visits_to_away", outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe latency to all others
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            probeLatOthersCats = []
            probeLatOthers = []
            for sesh in sessionsWithProbe:
                otherWells = set(allWellNames) - set(sesh.visited_away_wells) - \
                    set([sesh.home_well])
                for ow in otherWells:
                    probeLatOthersCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    probeLatOthers.append(min(sesh.getLatencyToWell(
                        True, ow, returnSeconds=True, emptyVal=300), 300))
            boxPlot(axs, probeLatOthers, probeLatOthersCats, axesNames=[
                    "Condition", "Latency to Others (s)"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_latency_to_others", outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe num visits to others
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            probeNumVisitsOthersCats = []
            probeNumVisitsOthers = []
            for sesh in sessionsWithProbe:
                otherWells = set(allWellNames) - set(sesh.visited_away_wells) - \
                    set([sesh.home_well])
                for ow in otherWells:
                    probeNumVisitsOthersCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    probeNumVisitsOthers.append(sesh.num_well_entries(True, ow))
            boxPlot(axs, probeNumVisitsOthers, probeNumVisitsOthersCats, axesNames=[
                    "Condition", "Number of visits to other wells"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_num_visits_to_others", outputDir=outputDir, statsFile=statsFile)

            # =============
            # Combined box plots from above
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = probeLatHomeCats + probeLatAwaysCats
            yvals = probeLatHome + probeLatAways
            wellCats = ["home"] * len(probeLatHomeCats) + ["away"] * len(probeLatAwaysCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Latency to well (s)", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_latency_to_wells", outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = probeNumVisitsHomeCats + probeNumVisitsAwaysCats
            yvals = probeNumVisitsHome + probeNumVisitsAways
            wellCats = ["home"] * len(probeNumVisitsHomeCats) + \
                ["away"] * len(probeNumVisitsAwaysCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Number of visits", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_num_visits_to_wells", outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe num wells visited by condition
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            categories = ["Interruption" if sessionWithProbeIsInterruption[i]
                          else "Control" for i in range(numSessionsWithProbe)]
            yvals = [numWellsVisited(s.probe_nearest_wells)
                     for s in sessionsWithProbe]
            boxPlot(axs, yvals, categories, axesNames=[
                    "Condition", "Number of wells visited in probe"], statsFile=statsFile)
            axs = saveOrShow("probe_num_wells_visited", outputDir=outputDir, statsFile=statsFile)

            # =============
            # frac time exploring in probe by condition (box plot)
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            categories = ["Interruption" if sessionWithProbeIsInterruption[i]
                          else "Control" for i in range(numSessionsWithProbe)]
            yvals = [s.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE)
                     for s in sessionsWithProbe]
            boxPlot(axs, yvals, categories, axesNames=[
                    "Condition", "Frac of probe exploring"], statsFile=statsFile)
            axs = saveOrShow("probe_frac_explore_box", outputDir=outputDir, statsFile=statsFile)

            # =============
            # Num explore bouts in probe by condition
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            categories = ["Interruption" if sessionWithProbeIsInterruption[i]
                          else "Control" for i in range(numSessionsWithProbe)]
            yvals = [len(s.probe_explore_bout_ends)
                     for s in sessionsWithProbe]
            boxPlot(axs, yvals, categories, axesNames=[
                    "Condition", "Num explore bouts"], statsFile=statsFile)
            axs = saveOrShow("probe_num_explore_bouts", outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe num wells visited per bout with repeats by condition
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            categories = []
            yvals = []
            for sesh in sessionsWithProbe:
                for i1, i2 in zip(sesh.probe_explore_bout_starts, sesh.probe_explore_bout_ends):
                    categories.append("Interruption" if sesh.isRippleInterruption else "Control")
                    yvals.append(numWellsVisited(
                        sesh.probe_nearest_wells[i1:i2], countReturns=True))
            boxPlot(axs, yvals, categories, axesNames=[
                    "Condition", "Number of wells visited per explore bout with repeats"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_num_wells_visited_per_explore_bout_with_repeats",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # probe num wells visited per bout without repeats by condition
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            categories = []
            yvals = []
            for sesh in sessionsWithProbe:
                for i1, i2 in zip(sesh.probe_explore_bout_starts, sesh.probe_explore_bout_ends):
                    categories.append("Interruption" if sesh.isRippleInterruption else "Control")
                    yvals.append(numWellsVisited(
                        sesh.probe_nearest_wells[i1:i2], countReturns=False))
            boxPlot(axs, yvals, categories, axesNames=[
                    "Condition", "Number of wells visited per explore bout without repeats"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_num_wells_visited_per_explore_bout_without_repeats",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # num wells visited in probe by time
            windowSlide = 5
            t1Array = np.arange(windowSlide, 60*5, windowSlide)
            numVisited = np.empty((numSessionsWithProbe, len(t1Array)))
            numVisited[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                t1s = t1Array * BTSession.TRODES_SAMPLING_RATE + sesh.probe_pos_ts[0]
                i1Array = np.searchsorted(sesh.probe_pos_ts, t1s)
                for ii, i1 in enumerate(i1Array):
                    numVisited[si, ii] = numWellsVisited(
                        sesh.probe_nearest_wells[0:i1], countReturns=False)
            plotIndividualAndAverage(axs, numVisited, t1Array)
            axs.set_xticks(np.arange(0, 60*5+1, 60))
            axs = saveOrShow("numVisitedOverTime", outputDir=outputDir, statsFile=statsFile)

            # =============
            # num wells visited in probe by time by condition
            swrfracExplores = numVisited[sessionWithProbeIsInterruption, :]
            ctrlfracExplores = numVisited[np.logical_not(sessionWithProbeIsInterruption), :]
            plotIndividualAndAverage(axs, swrfracExplores, t1Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, ctrlfracExplores, t1Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xticks(np.arange(0, 60*5+1, 60))
            axs = saveOrShow("numVisitedOverTime_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # total dwell time, full probe
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            totalDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                      else "Control" for i in range(numSessionsWithProbe)]
            totalDwellTimeHome = [s.total_dwell_time(True, s.home_well)
                                  for s in sessionsWithProbe]
            boxPlot(axs, totalDwellTimeHome, totalDwellTimeHomeCats, axesNames=[
                    "Condition", "total dwell time (s)"], statsFile=statsFile)
            axs = saveOrShow("probe_total_dwell_time_home",
                             outputDir=outputDir, statsFile=statsFile)

            totalDwellTimeAwayCats = []
            totalDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    totalDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    totalDwellTimeAway.append(sesh.total_dwell_time(True, aw))
            boxPlot(axs, totalDwellTimeAway, totalDwellTimeAwayCats, axesNames=[
                    "Condition", "total dwell time (s)"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_total_dwell_time_aways",
                             outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = totalDwellTimeHomeCats + totalDwellTimeAwayCats
            yvals = totalDwellTimeHome + totalDwellTimeAway
            wellCats = ["home"] * len(totalDwellTimeHomeCats) + \
                ["away"] * len(totalDwellTimeAwayCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "total dwell time (s)", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_total_dwell_time", outputDir=outputDir, statsFile=statsFile)

            # =============
            # total dwell time, 90sec
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            totalDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                      else "Control" for i in range(numSessionsWithProbe)]
            totalDwellTimeHome = [s.total_dwell_time(True, s.home_well, timeInterval=[0, 90])
                                  for s in sessionsWithProbe]
            boxPlot(axs, totalDwellTimeHome, totalDwellTimeHomeCats, axesNames=[
                    "Condition", "total dwell time (s)"], statsFile=statsFile)
            axs = saveOrShow("probe_total_dwell_time_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            totalDwellTimeAwayCats = []
            totalDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    totalDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    totalDwellTimeAway.append(sesh.total_dwell_time(
                        True, aw, timeInterval=[0, 90]))
            boxPlot(axs, totalDwellTimeAway, totalDwellTimeAwayCats, axesNames=[
                    "Condition", "total dwell time (s)"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_total_dwell_time_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = totalDwellTimeHomeCats + totalDwellTimeAwayCats
            yvals = totalDwellTimeHome + totalDwellTimeAway
            wellCats = ["home"] * len(totalDwellTimeHomeCats) + \
                ["away"] * len(totalDwellTimeAwayCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "total dwell time (s)", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_total_dwell_time_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # Avg dwell time, full probe
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            avgDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                    else "Control" for i in range(numSessionsWithProbe)]
            avgDwellTimeHome = [s.avg_dwell_time(True, s.home_well, emptyVal=np.nan)
                                for s in sessionsWithProbe]
            boxPlot(axs, avgDwellTimeHome, avgDwellTimeHomeCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], statsFile=statsFile)
            axs.set_ylim(top=15)
            axs = saveOrShow("probe_avg_dwell_time_home", outputDir=outputDir, statsFile=statsFile)

            avgDwellTimeAwayCats = []
            avgDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    avgDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    avgDwellTimeAway.append(sesh.avg_dwell_time(True, aw, emptyVal=np.nan))
            boxPlot(axs, avgDwellTimeAway, avgDwellTimeAwayCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], violin=True, statsFile=statsFile)
            axs.set_ylim(top=15)
            axs = saveOrShow("probe_avg_dwell_time_aways", outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = avgDwellTimeHomeCats + avgDwellTimeAwayCats
            yvals = avgDwellTimeHome + avgDwellTimeAway
            wellCats = ["home"] * len(avgDwellTimeHomeCats) + ["away"] * len(avgDwellTimeAwayCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Avg dwell time (s)", "Well Type"], violin=True, statsFile=statsFile)
            axs.set_ylim(top=15)
            axs = saveOrShow("probe_avg_dwell_time", outputDir=outputDir, statsFile=statsFile)

            # =============
            # Avg dwell time, 90sec
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            avgDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                    else "Control" for i in range(numSessionsWithProbe)]
            avgDwellTimeHome = [s.avg_dwell_time(True, s.home_well, emptyVal=np.nan, timeInterval=[0, 90])
                                for s in sessionsWithProbe]
            boxPlot(axs, avgDwellTimeHome, avgDwellTimeHomeCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], statsFile=statsFile)
            axs = saveOrShow("probe_avg_dwell_time_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            avgDwellTimeAwayCats = []
            avgDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    avgDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    avgDwellTimeAway.append(sesh.avg_dwell_time(
                        True, aw, emptyVal=np.nan, timeInterval=[0, 90]))
            boxPlot(axs, avgDwellTimeAway, avgDwellTimeAwayCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_avg_dwell_time_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = avgDwellTimeHomeCats + avgDwellTimeAwayCats
            yvals = avgDwellTimeHome + avgDwellTimeAway
            wellCats = ["home"] * len(avgDwellTimeHomeCats) + ["away"] * len(avgDwellTimeAwayCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Avg dwell time (s)", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_avg_dwell_time_90sec", outputDir=outputDir, statsFile=statsFile)

            # =============
            # num entries, 90sec
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            numEntriesHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                  else "Control" for i in range(numSessionsWithProbe)]
            numEntriesHome = [s.num_well_entries(True, s.home_well, timeInterval=[0, 90])
                              for s in sessionsWithProbe]
            boxPlot(axs, numEntriesHome, numEntriesHomeCats, axesNames=[
                    "Condition", "Num entries"], statsFile=statsFile)
            axs = saveOrShow("probe_num_entries_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            numEntriesAwayCats = []
            numEntriesAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    numEntriesAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    numEntriesAway.append(sesh.num_well_entries(
                        True, aw, timeInterval=[0, 90]))
            boxPlot(axs, numEntriesAway, numEntriesAwayCats, axesNames=[
                    "Condition", "Num entries"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_num_entries_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = numEntriesHomeCats + numEntriesAwayCats
            yvals = numEntriesHome + numEntriesAway
            wellCats = ["home"] * len(numEntriesHomeCats) + ["away"] * len(numEntriesAwayCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Num entries", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_num_entries_90sec", outputDir=outputDir, statsFile=statsFile)

            # =============
            # curvature, full probe
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            curvatureHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                 else "Control" for i in range(numSessionsWithProbe)]
            curvatureHome = [s.avg_curvature_at_well(True, s.home_well)
                             for s in sessionsWithProbe]
            boxPlot(axs, curvatureHome, curvatureHomeCats, axesNames=[
                    "Condition", "Curvature"], statsFile=statsFile)
            axs = saveOrShow("probe_curvature_at_home", outputDir=outputDir, statsFile=statsFile)

            curvatureAwayCats = []
            curvatureAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    curvatureAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureAway.append(sesh.avg_curvature_at_well(True, aw))
            boxPlot(axs, curvatureAway, curvatureAwayCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_curvature_at_aways", outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = curvatureHomeCats + curvatureAwayCats
            yvals = curvatureHome + curvatureAway
            wellCats = ["home"] * len(curvatureHomeCats) + ["away"] * len(curvatureAwayCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_curvature", outputDir=outputDir, statsFile=statsFile)

            # =============
            # curvature, 90sec
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            curvatureHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                 else "Control" for i in range(numSessionsWithProbe)]
            curvatureHome = [s.avg_curvature_at_well(True, s.home_well, timeInterval=[0, 90])
                             for s in sessionsWithProbe]
            boxPlot(axs, curvatureHome, curvatureHomeCats, axesNames=[
                    "Condition", "Curvature"], statsFile=statsFile)
            axs = saveOrShow("probe_curvature_at_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            curvatureAwayCats = []
            curvatureAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    curvatureAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureAway.append(sesh.avg_curvature_at_well(
                        True, aw, timeInterval=[0, 90]))
            boxPlot(axs, curvatureAway, curvatureAwayCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_curvature_at_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            seshCats = curvatureHomeCats + curvatureAwayCats
            yvals = curvatureHome + curvatureAway
            wellCats = ["home"] * len(curvatureHomeCats) + ["away"] * len(curvatureAwayCats)
            boxPlot(axs, yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile)
            axs = saveOrShow("probe_curvature_90sec", outputDir=outputDir, statsFile=statsFile)

            # =============
            # Num wells checked, centered on reward find times
            windowSlide = 3
            numCheckedDuration = 60*5
            t1Array = np.arange(windowSlide, numCheckedDuration, windowSlide)
            numWellsChecked = np.empty((0, len(t1Array)))
            numWellsCheckedWithRepeats = np.empty((0, len(t1Array)))
            emptyArr = np.empty((1, len(t1Array)))
            emptyArr[:] = np.nan
            seshCat = []
            wellCat = []
            terminationCondition = []
            trialIdx = []

            numTimesHomeWellChecked = np.empty((0, len(t1Array)))
            homeWellCheckSeshCat = []

            for si, sesh in enumerate(sessions):
                for fti, ft in enumerate(sesh.home_well_find_times):
                    t1s = t1Array * BTSession.TRODES_SAMPLING_RATE + ft
                    if fti < sesh.num_away_found:
                        tend = sesh.away_well_find_times[fti]
                        terminationCondition.append("found")
                    else:
                        tend = sesh.bt_pos_ts[-1]
                        terminationCondition.append("end")
                    t1s = t1s[t1s <= tend]
                    i1Array = np.searchsorted(sesh.bt_pos_ts, t1s)
                    i0 = np.searchsorted(sesh.bt_pos_ts, ft)

                    numWellsChecked = np.vstack((numWellsChecked, emptyArr))
                    numWellsCheckedWithRepeats = np.vstack((numWellsCheckedWithRepeats, emptyArr))
                    seshCat.append("Interruption" if sesh.isRippleInterruption else "Control")
                    wellCat.append("home")
                    trialIdx.append(fti)

                    numTimesHomeWellChecked = np.vstack((numTimesHomeWellChecked, emptyArr))
                    homeWellCheckSeshCat.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")

                    for ii, i1 in enumerate(i1Array):
                        numWellsChecked[-1, ii] = numWellsVisited(sesh.bt_nearest_wells[i0:i1])
                        numWellsCheckedWithRepeats[-1, ii] = numWellsVisited(
                            sesh.bt_nearest_wells[i0:i1], countReturns=True)
                        numTimesHomeWellChecked[-1, ii] = numWellsVisited(
                            sesh.bt_nearest_wells[i0:i1], countReturns=True, wellSubset=[sesh.home_well])

                for fti, ft in enumerate(sesh.away_well_find_times):
                    t1s = t1Array * BTSession.TRODES_SAMPLING_RATE + ft
                    if fti+1 < len(sesh.home_well_find_times):
                        tend = sesh.home_well_find_times[fti+1]
                        terminationCondition.append("found")
                    else:
                        tend = sesh.bt_pos_ts[-1]
                        terminationCondition.append("end")
                    t1s = t1s[t1s <= tend]
                    i1Array = np.searchsorted(sesh.bt_pos_ts, t1s)
                    i0 = np.searchsorted(sesh.bt_pos_ts, ft)

                    numWellsChecked = np.vstack((numWellsChecked, emptyArr))
                    numWellsCheckedWithRepeats = np.vstack((numWellsCheckedWithRepeats, emptyArr))
                    seshCat.append("Interruption" if sesh.isRippleInterruption else "Control")
                    wellCat.append("away")
                    trialIdx.append(fti)

                    for ii, i1 in enumerate(i1Array):
                        numWellsChecked[-1, ii] = numWellsVisited(sesh.bt_nearest_wells[i0:i1])
                        numWellsCheckedWithRepeats[-1, ii] = numWellsVisited(
                            sesh.bt_nearest_wells[i0:i1], countReturns=True)

            seshCat = np.array(seshCat)
            wellCat = np.array(wellCat)
            terminationCondition = np.array(terminationCondition)
            trialIdx = np.array(trialIdx)
            homeWellCheckSeshCat = np.array(homeWellCheckSeshCat)

            plotIndividualAndAverage(axs, numWellsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Interruption", wellCat == "home")), :], t1Array,
                individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numWellsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Control", wellCat == "home")), :], t1Array,
                individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 60)
            axs.set_ylim(0, 36)
            axs = saveOrShow("wellsCheckedAfterRewardHome",
                             outputDir=outputDir, statsFile=statsFile)
            plotIndividualAndAverage(axs, numWellsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Interruption", wellCat == "away")), :], t1Array,
                individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numWellsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Control", wellCat == "away")), :], t1Array,
                individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 60)
            axs.set_ylim(0, 36)
            axs = saveOrShow("wellsCheckedAfterRewardAway",
                             outputDir=outputDir, statsFile=statsFile)

            plotIndividualAndAverage(axs, numTimesHomeWellChecked[homeWellCheckSeshCat == "Interruption", :], t1Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numTimesHomeWellChecked[homeWellCheckSeshCat == "Control", :], t1Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 60)
            # axs.set_ylim(0, 36)
            axs = saveOrShow("numTimesHomeWellChecked_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            numRepeatsChecked = numWellsCheckedWithRepeats - numWellsChecked
            plotIndividualAndAverage(axs, numRepeatsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Interruption", wellCat == "home")), :], t1Array,
                individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numRepeatsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Control", wellCat == "home")), :], t1Array,
                individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 120)
            axs.set_ylim(0, 36)
            axs = saveOrShow("repeatsCheckedAfterRewardHome",
                             outputDir=outputDir, statsFile=statsFile)
            plotIndividualAndAverage(axs, numRepeatsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Interruption", wellCat == "away")), :], t1Array,
                individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numRepeatsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Control", wellCat == "away")), :], t1Array,
                individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 120)
            axs.set_ylim(0, 36)
            axs = saveOrShow("repeatsCheckedAfterRewardAway",
                             outputDir=outputDir, statsFile=statsFile)

            # check more wells or repeats on way to home well in one condition?
            idx = terminationCondition == "found"
            cat1 = seshCat[idx]
            cat2 = wellCat[idx]
            cat2 = ["home" if c == "away" else "away" for c in cat2]
            yvals = numWellsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            boxPlot(axs, yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num checked before next reward", "Trial Type"], statsFile=statsFile)
            axs = saveOrShow("num_checked_before_next_reward",
                             outputDir=outputDir, statsFile=statsFile)

            yvals = numRepeatsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            boxPlot(axs, yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num repeats checked before next reward", "Trial Type"], statsFile=statsFile)
            axs = saveOrShow("num_repeats_checked_before_next_reward",
                             outputDir=outputDir, statsFile=statsFile)

            idx = np.logical_and(terminationCondition == "found", trialIdx > 4)
            cat1 = seshCat[idx]
            cat2 = wellCat[idx]
            cat2 = ["home" if c == "away" else "away" for c in cat2]
            yvals = numWellsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            boxPlot(axs, yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num checked before next reward", "Trial Type"], statsFile=statsFile)
            axs = saveOrShow("num_checked_before_next_reward_secondhalf",
                             outputDir=outputDir, statsFile=statsFile)

            yvals = numRepeatsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            boxPlot(axs, yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num repeats checked before next reward", "Trial Type"], statsFile=statsFile)
            axs = saveOrShow("num_repeats_checked_before_next_reward_secondhalf",
                             outputDir=outputDir, statsFile=statsFile)

        # =============
        # stim rate at different well types
        wellCats = []
        seshCats = []
        vals = []
        for sesh in sessions:
            seshCats.append(
                "Interruption" if sesh.isRippleInterruption else "Control")
            wellCats.append("home")
            vals.append(sesh.numStimsAtWell(sesh.home_well) /
                        sesh.total_dwell_time(False, sesh.home_well))

            for aw in sesh.visited_away_wells:
                seshCats.append(
                    "Interruption" if sesh.isRippleInterruption else "Control")
                wellCats.append("away")
                vals.append(sesh.numStimsAtWell(aw) / sesh.total_dwell_time(False, aw))

        boxPlot(axs, vals, seshCats, categories2=wellCats, violin=True, axesNames=[
                "Condition", "Stim rate (Hz)", "Well type"], statsFile=statsFile)
        axs = saveOrShow("stim_rate_at_wells", outputDir=outputDir, statsFile=statsFile)

        if False:
            # =============
            # cumulative number of stims and ripples
            windowSize = 5
            bins = np.arange(0, 60*20+1, windowsSize)
            stimCounts = np.empty((numSessions, len(bins)-1))
            rippleCounts = np.empty((numSessions, len(bins)-1))
            for si, sesh in enumerate(sessions):
                stimTs = np.array(sesh.interruption_timestamps)
                stimTs = stimTs[np.logical_and(stimTs > sesh.bt_pos_ts[0], stimTs <
                                               sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                stimTs /= BTSession.TRODES_SAMPLING_RATE
                stimCounts[si, :] = np.histogram(stimTs, bins=bins)

                ripTs = np.array(sesh.btRipStartTimstampsPreStats)
                ripTs = ripTs[np.logical_and(ripTs > sesh.bt_pos_ts[0], ripTs <
                                             sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                ripTs /= BTSession.TRODES_SAMPLING_RATE
                rippleCounts[si, :] = np.histogram(ripTs, bins=bins)

            plotIndividualAndAverage(axs, stimCounts, bins[1:])
            axs = saveOrShow("stimCounts", outputDir=outputDir, statsFile=statsFile)
            plotIndividualAndAverage(axs, rippleCounts, bins[1:])
            axs = saveOrShow("rippleCounts", outputDir=outputDir, statsFile=statsFile)

            stimCountsSWR = stimCounts[sessionIsInterruption, :]
            stimCountsCtrl = stimCounts[np.logical_not(sessionIsInterruption), :]
            plotIndividualAndAverage(axs, stimCountsSWR, bins[1:],
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, stimCountsCtrl, bins[1:],
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs = saveOrShow("stimCounts_by_condition", outputDir=outputDir, statsFile=statsFile)

            rippleCountsSWR = rippleCounts[sessionIsInterruption, :]
            rippleCountsCtrl = rippleCounts[np.logical_not(sessionIsInterruption), :]
            plotIndividualAndAverage(axs, rippleCountsSWR, bins[1:],
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, rippleCountsCtrl, bins[1:],
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs = saveOrShow("rippleCounts_by_condition", outputDir=outputDir, statsFile=statsFile)

            # =============
            # cumulative number of ripples (just when using probe stats)
            windowSize = 5
            bins = np.arange(0, 60*20+1, windowsSize)
            rippleCounts = np.empty((numSessionsWithProbe, len(bins)-1))
            for si, sesh in enumerate(sessionsWithProbe):
                ripTs = np.array(sesh.btRipStartTimstampsProbeStats)
                ripTs = ripTs[np.logical_and(ripTs > sesh.bt_pos_ts[0], ripTs <
                                             sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                ripTs /= BTSession.TRODES_SAMPLING_RATE
                rippleCounts[si, :] = np.histogram(ripTs, bins=bins)

            plotIndividualAndAverage(axs, rippleCounts, bins[1:])
            axs = saveOrShow("rippleCountsProbeStats", outputDir=outputDir, statsFile=statsFile)

            rippleCountsSWR = rippleCounts[sessionWithProbeIsInterruption, :]
            rippleCountsCtrl = rippleCounts[np.logical_not(sessionWithProbeIsInterruption), :]
            plotIndividualAndAverage(axs, rippleCountsSWR, bins[1:],
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, rippleCountsCtrl, bins[1:],
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs = saveOrShow("rippleCountsProbeStats_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

        # s.numStimsAtWell(w)
        #  w: s.numStimsAtWell(w) / s.total_dwell_time(False, w))
        # len(s.bt_interruption_pos_idxs) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE)
        # len(s.btRipStartIdxsProbeStats) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE)

        # ==========================================
        if SKIP_SINGLE_SESSION_PLOTS:
            return

        fig.set_figheight(FIG_SCALE)
        fig.set_figwidth(FIG_SCALE)

        for sesh in sessions:
            if SKIP_NO_PROBE_SESSION_PLOTS:
                break

            print(sesh.name)
            seshOutputDir = os.path.join(outputDir, sesh.name)
            if not os.path.exists(seshOutputDir):
                os.makedirs(seshOutputDir)

            # =============
            # Raw behavioral trace (task)
            axs.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, c="#deac7f")
            setupBehaviorTracePlot(axs, sesh)
            axs = saveOrShow("raw_task", outputDir=seshOutputDir, statsFile=statsFile)

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
                axs = saveOrShow("all_trials", outputDir=seshOutputDir, statsFile=statsFile)
                fig.set_figheight(FIG_SCALE)
                fig.set_figwidth(FIG_SCALE)

            for fti, ft in enumerate(sesh.home_well_find_pos_idxs):
                i1 = 0 if fti == 0 else sesh.away_well_leave_pos_idxs[fti-1]
                i2 = ft
                axs.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], c="#deac7f")
                setupBehaviorTracePlot(axs, sesh)
                axs = saveOrShow("home_find_{}".format(
                    fti), outputDir=seshOutputDir, statsFile=statsFile)

            for fti, ft in enumerate(sesh.away_well_find_pos_idxs):
                i1 = sesh.home_well_leave_pos_idxs[fti]
                i2 = ft
                axs.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], c="#deac7f")
                setupBehaviorTracePlot(axs, sesh)
                axs = saveOrShow("away_find_{}".format(
                    fti), outputDir=seshOutputDir, statsFile=statsFile)

        for sesh in sessionsWithProbe:
            if SKIP_PROBE_SESSION_PLOTS:
                break

            print(sesh.name)
            seshOutputDir = os.path.join(outputDir, sesh.name)
            if not os.path.exists(seshOutputDir):
                os.makedirs(seshOutputDir)

            # =============
            # Raw behavioral trace (probe)
            axs.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
            setupBehaviorTracePlot(axs, sesh)
            axs = saveOrShow("raw_probe".format(sesh.name),
                             outputDir=seshOutputDir, statsFile=statsFile)

    for animalName in animalNames:
        # if animalName == "B14":
        # continue
        print("==========================\n" + animalName)
        animalOutputDir = os.path.join(globalOutputDir, animalName)
        sessions = allSessionsByRat[animalName]
        makeFiguresForSessions(sessions, animalOutputDir)

    print("==========================\nPrince Martin")
    princeMartinOutputDir = os.path.join(globalOutputDir, "PrinceMartin")
    sessions = allSessionsByRat["B13"] + allSessionsByRat["Martin"]
    makeFiguresForSessions(sessions, princeMartinOutputDir)


if __name__ == "__main__":
    makeThesisCommitteeFigs()
