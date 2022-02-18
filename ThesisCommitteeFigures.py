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
import textwrap as twp
from matplotlib.transforms import Bbox
import matplotlib

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
        print("Saved {}".format(fname))

        ax = plt.subplot(111)
        ax.cla()
        return ax

    def plotIndividualAndAverage(ax, dataPoints, xvals, individualColor="grey", avgColor="blue", spread="std", individualZOrder=1, averageZOrder=2):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
            warnings.filterwarnings("ignore", r"Mean of empty slice")
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

        retDict = {}

        # global effect of A: ignore cat2, run 1-way shuffle on cat1
        sdf = df.drop(columns=cat2Name)
        shuffleValues = np.empty((numShuffles,))
        for si in range(numShuffles):
            sdf[cat1Name] = sdf[cat1Name].sample(frac=1, random_state=1).reset_index(drop=True)
            ma = sdf.groupby(cat1Name).mean()
            shuffleValues[si] = ma.iat[1, 0] - ma.iat[0, 0]
        pctile = np.count_nonzero(shuffleValues < observedADiff) / numShuffles
        retDict['Global ' + cat1Name] = pctile
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
        retDict['Global ' + cat2Name] = pctile
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
        retDict[cat2Name + ' diff (' + cat1Name + ' effect)'] = pctile
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
        retDict[cat1Name + ' diff (' + cat2Name + ' effect)'] = pctile
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
            retDict['within ' + str(uv)] = pctile
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
            retDict['within ' + str(uv)] = pctile
            if statsFile is not None:
                shufMean = np.nanmean(shuffleValues)
                shufStd = np.nanstd(shuffleValues)
                statsFile.write("Shuffling {} just in subset where {} == {}\n\tShuffle mean: {}\tstd: {}\n\tObserved Value: {}\n\tpval={}\n".format(
                    cat2Name, cat1Name, uv, shufMean, shufStd, observedBDiffUV, pctile))

        if statsFile is not None:
            statsFile.write("\n")

        return retDict

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

        return pctile

    def pctilePvalSig(val):
        if val > 0.1 and val < 0.9:
            return 0
        if val > 0.05 and val < 0.95:
            return 1
        if val > 0.01 and val < 0.99:
            return 2
        if val > 0.001 and val < 0.999:
            return 2
        return 3

    def boxPlot(ax, yvals, categories, categories2=None, axesNames=None, violin=False, doStats=True, statsFile=None, statsAx=None):
        if categories2 is None:
            categories2 = ["a" for _ in categories]
            sortingCategories2 = categories2
            cat2IsFake = True
            if axesNames is not None:
                axesNames.append("Z")
        else:
            cat2IsFake = False
            if len(set(categories2)) == 2 and "home" in categories2 and "away" in categories2:
                sortingCategories2 = ["aaahome" if c == "home" else "away" for c in categories2]
            else:
                sortingCategories2 = categories2

        # Same sorting here as in perseveration plot function so colors are always the same
        sortList = ["{}__{}__{}".format(x, y, xi)
                    for xi, (x, y) in enumerate(zip(categories, sortingCategories2))]
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
            # print(axesNames)
            plotWorked = False
            swarmDotSize = 3
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=UserWarning)
                while not plotWorked:
                    try:
                        # print("trying...")
                        p1 = sns.violinplot(ax=ax, hue=axesNamesNoSpaces[0],
                                            y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, palette=pal, linewidth=0.2, cut=0, zorder=1)
                        p2 = sns.swarmplot(ax=ax, hue=axesNamesNoSpaces[0],
                                           y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, color="0.25", size=swarmDotSize, dodge=True, zorder=3)
                        # print("worked")
                        plotWorked = True
                    except UserWarning as e:
                        swarmDotSize /= 2
                        # print("reduing dots to {}".format(swarmDotSize))

                        # ax.cla()
                        # print(ax.figure)
                        # try:
                        #     p1.remove()
                        # except:
                        #     pass

                        # try:
                        #     p2.remove()
                        # except:
                        #     pass
                        p1.cla()

                        # print(ax.figure)

                        if swarmDotSize < 0.1:
                            raise e
        else:
            p1 = sns.boxplot(
                ax=ax, hue=axesNamesNoSpaces[0], y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, palette=pal, zorder=1)
            p2 = sns.swarmplot(ax=ax, hue=axesNamesNoSpaces[0],
                               y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, color="0.25", dodge=True, zorder=3)

        if cat2IsFake:
            p1.set(xticklabels=[])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], fontsize=6).set_zorder(2)

        if not hideAxes:
            if not cat2IsFake:
                ax.set_xlabel(axesNames[2])
            ax.set_ylabel(axesNames[1])

        if doStats:
            if cat2IsFake:
                pval = runOneWayShuffle(
                    yvals, categories, catName=axesNamesNoSpaces[0], dataName=axesNamesNoSpaces[1], statsFile=statsFile)

                if statsAx is not None:
                    statsAx.remove()
                    r = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none',
                                                     visible=False)
                    pvalDisplay = str(min(pval, 1.0-pval))
                    if len(pvalDisplay) > 5:
                        pvalDisplay = pvalDisplay[:5]

                    pvalLabel = "".join(["*"] * pctilePvalSig(pval) +
                                        ["p=" + pvalDisplay])
                    # print(pvalLabel)
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:2] + [r], labels[:2] + [pvalLabel], fontsize=6).set_zorder(2)
            else:
                statsDict = runTwoWayShuffle(yvals, categories, categories2,
                                             cat1Name=axesNamesNoSpaces[0], cat2Name=axesNamesNoSpaces[2], dataName=axesNamesNoSpaces[1], statsFile=statsFile)
                if statsAx is not None:
                    statsAx.tick_params(axis="both", which="both", label1On=False,
                                        label2On=False, tick1On=False, tick2On=False)
                    txtwidth = 20
                    fontsizelabel = 7
                    fontsizepval = 8

                    for li, (ll, lp) in enumerate(reversed(list(zip([twp.fill(str(s), txtwidth) for s in statsDict.keys()], [
                            twp.fill(str(s), txtwidth) for s in statsDict.values()])))):

                        pvalDisplay = str(min(float(lp), 1.0-float(lp)))
                        if len(pvalDisplay) > 5:
                            pvalDisplay = pvalDisplay[:5]
                        yp = (li + 0.75) / (len(statsDict) + 0.5)
                        statsAx.text(0.15, yp, ll, fontsize=fontsizelabel, va="center")
                        statsAx.text(0.75, yp, pvalDisplay, fontsize=fontsizepval, va="center")
                        statsAx.text(0.13, yp, ''.join(
                            ['*'] * pctilePvalSig(float(lp))), fontsize=fontsizepval, va="center", ha="right")

                        # if li > 0:
                        #     statsAx.plot([0, 1], [(li + 0.25) / (len(statsDict) + 0.5)] * 2,
                        #                  c="grey", lw=0.5)

    def numWellsVisited(nearestWells, countReturns=False, wellSubset=None):
        g = groupby(nearestWells)
        if wellSubset is None:
            wellSubset = allWellNames
        if countReturns:
            return len([k for k, _ in g if k in wellSubset])
        else:
            return len(set([k for k, _ in g if k in wellSubset]))

    def makeFiguresForSessions(sessions, outputDir, MAKE_STATS_FILE=False,
                               SKIP_ALL_SESSIONS_PLOTS=False, SKIP_COMPARISON_PLOTS=False,
                               SKIP_LFP_PLOTS=False,
                               SKIP_SINGLE_SESSION_PLOTS=True, SKIP_NO_PROBE_SESSION_PLOTS=False,
                               SKIP_PROBE_SESSION_PLOTS=False):
        fig = plt.figure(figsize=(FIG_SCALE, FIG_SCALE))
        axs = fig.subplots()

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        if MAKE_STATS_FILE:
            statsFileName = os.path.join(
                outputDir, datetime.now().strftime("%Y%m%d_%H%M%S_stats.txt"))
            statsFile = open(statsFileName, "w")
        else:
            statsFile = None

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

        sessionsWithoutProbe = [s for s in sessions if not s.probe_performed]
        numSessionsWithoutProbe = len(sessionsWithoutProbe)
        sessionWithoutProbeIsInterruption = np.zeros((numSessionsWithoutProbe,)).astype(bool)
        for i in range(numSessionsWithoutProbe):
            if sessionsWithoutProbe[i].isRippleInterruption:
                sessionWithoutProbeIsInterruption[i] = True

        boxPlotWidthRatio = 1
        doubleBoxPlotWidthRatio = 2
        boxPlotStatsRatio = 1

        if not SKIP_ALL_SESSIONS_PLOTS:
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
            axs = saveOrShow("task_latency_to_home", outputDir=outputDir, statsFile=statsFile)

            plotIndividualAndAverage(axs, awayFindTimes, pltx)
            axs.set_ylim(0, ymax)
            axs.set_xlim(1, 10)
            axs.set_xticks(np.arange(0, 10, 2) + 1)
            axs = saveOrShow("task_latency_to_away", outputDir=outputDir, statsFile=statsFile)

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
            axs = saveOrShow("task_latency_to_home_by_condition",
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
            axs = saveOrShow("task_latency_to_away_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # average vel in probe
            windowSize = 30
            windowsSlide = 6
            t0Array = np.arange(0, 60*5-windowSize+windowsSlide, windowsSlide)
            avgVels = np.empty((numSessionsWithProbe, len(t0Array)))
            avgVels[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                for ti, t0 in enumerate(t0Array):
                    avgVels[si, ti] = sesh.mean_vel(True, timeInterval=[t0, t0+windowSize])
            plotIndividualAndAverage(axs, avgVels, t0Array)
            axs.set_xticks(np.arange(0, 60*5+1, 60))
            axs = saveOrShow("probe_avgVel", outputDir=outputDir, statsFile=statsFile)

            # =============
            # average vel in probe by condition
            swrAvgVels = avgVels[sessionWithProbeIsInterruption, :]
            ctrlAvgVels = avgVels[np.logical_not(sessionWithProbeIsInterruption), :]
            plotIndividualAndAverage(axs, swrAvgVels, t0Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, ctrlAvgVels, t0Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xticks(np.arange(0, 60*5+1, 60))
            axs = saveOrShow("probe_avgVel_by_condition",
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
            axs = saveOrShow("task_avgVel", outputDir=outputDir, statsFile=statsFile)

            # =============
            # average vel in task by condition
            swrAvgVels = avgVels[sessionIsInterruption, :]
            ctrlAvgVels = avgVels[np.logical_not(sessionIsInterruption), :]
            plotIndividualAndAverage(axs, swrAvgVels, t0Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, ctrlAvgVels, t0Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xticks(np.arange(0, 60*20+1, 60*5))
            axs = saveOrShow("task_avgVel_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # frac time exploring in probe
            windowSize = 30
            windowsSlide = 6
            t0Array = np.arange(0, 60*5-windowSize+windowsSlide, windowsSlide)
            fracExplores = np.empty((numSessionsWithProbe, len(t0Array)))
            fracExplores[:] = np.nan
            for si, sesh in enumerate(sessionsWithProbe):
                for ti, t0 in enumerate(t0Array):
                    fracExplores[si, ti] = sesh.prop_time_in_bout_state(
                        True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[t0, t0+windowSize])
            plotIndividualAndAverage(axs, fracExplores, t0Array)
            axs.set_xticks(np.arange(0, 60*5+1, 60))
            axs = saveOrShow("probe_fracExplore", outputDir=outputDir, statsFile=statsFile)

            # =============
            # frac time exploring in probe by condition
            swrfracExplores = fracExplores[sessionWithProbeIsInterruption, :]
            ctrlfracExplores = fracExplores[np.logical_not(sessionWithProbeIsInterruption), :]
            plotIndividualAndAverage(axs, swrfracExplores, t0Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, ctrlfracExplores, t0Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xticks(np.arange(0, 60*5+1, 60))
            axs = saveOrShow("probe_fracExplore_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

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
            axs = saveOrShow("task_fracExplore", outputDir=outputDir, statsFile=statsFile)

            # =============
            # frac time exploring in task by condition
            swrfracExplores = fracExplores[sessionIsInterruption, :]
            ctrlfracExplores = fracExplores[np.logical_not(sessionIsInterruption), :]
            plotIndividualAndAverage(axs, swrfracExplores, t0Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, ctrlfracExplores, t0Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xticks(np.arange(0, 60*20+1, 60*5))
            axs = saveOrShow("task_fracExplore_by_condition",
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
            axs = saveOrShow("probe_traces_swr", outputDir=outputDir, statsFile=statsFile)
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
            axs = saveOrShow("probe_traces_ctrl", outputDir=outputDir, statsFile=statsFile)
            fig.set_figheight(FIG_SCALE)
            fig.set_figwidth(FIG_SCALE)

            # =============
            # probe latency to home
            probeLatHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                else "Control" for i in range(numSessionsWithProbe)]
            probeLatHome = [min(s.getLatencyToWell(
                True, s.home_well, returnSeconds=True, emptyVal=300), 300) for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], probeLatHome, probeLatHomeCats,
                    axesNames=["Condition", "Latency to Home (s)"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_latency_to_home", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe number of visits to home
            probeNumVisitsHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                      else "Control" for i in range(numSessionsWithProbe)]
            probeNumVisitsHome = [s.num_well_entries(True, s.home_well)
                                  for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], probeNumVisitsHome, probeNumVisitsHomeCats, axesNames=[
                    "Condition", "Number of visits to home well"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_visits_to_home",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe latency to all aways
            probeLatAwaysCats = []
            probeLatAways = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    probeLatAwaysCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    probeLatAways.append(min(sesh.getLatencyToWell(
                        True, aw, returnSeconds=True, emptyVal=300), 300))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], probeLatAways, probeLatAwaysCats, axesNames=[
                    "Condition", "Latency to Aways (s)"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_latency_to_aways", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe num visits to aways
            probeNumVisitsAwaysCats = []
            probeNumVisitsAways = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    probeNumVisitsAwaysCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    probeNumVisitsAways.append(sesh.num_well_entries(True, aw))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], probeNumVisitsAways, probeNumVisitsAwaysCats, axesNames=[
                    "Condition", "Number of visits to away well"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_visits_to_away",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe latency to all others
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
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], probeLatOthers, probeLatOthersCats, axesNames=[
                    "Condition", "Latency to Others (s)"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_latency_to_others",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe num visits to others
            probeNumVisitsOthersCats = []
            probeNumVisitsOthers = []
            for sesh in sessionsWithProbe:
                otherWells = set(allWellNames) - set(sesh.visited_away_wells) - \
                    set([sesh.home_well])
                for ow in otherWells:
                    probeNumVisitsOthersCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    probeNumVisitsOthers.append(sesh.num_well_entries(True, ow))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], probeNumVisitsOthers, probeNumVisitsOthersCats, axesNames=[
                    "Condition", "Number of visits to other wells"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_visits_to_others",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # Combined box plots from above
            seshCats = probeLatHomeCats + probeLatAwaysCats
            yvals = probeLatHome + probeLatAways
            wellCats = ["home"] * len(probeLatHomeCats) + ["away"] * len(probeLatAwaysCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Latency to well (s)", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_latency_to_wells", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = probeNumVisitsHomeCats + probeNumVisitsAwaysCats
            yvals = probeNumVisitsHome + probeNumVisitsAways
            wellCats = ["home"] * len(probeNumVisitsHomeCats) + \
                ["away"] * len(probeNumVisitsAwaysCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Number of visits", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_visits_to_wells",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe num wells visited by condition
            categories = ["Interruption" if sessionWithProbeIsInterruption[i]
                          else "Control" for i in range(numSessionsWithProbe)]
            yvals = [numWellsVisited(s.probe_nearest_wells)
                     for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, categories, axesNames=[
                    "Condition", "Number of wells visited in probe"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_wells_visited",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # frac time exploring in probe by condition (box plot)
            categories = ["Interruption" if sessionWithProbeIsInterruption[i]
                          else "Control" for i in range(numSessionsWithProbe)]
            yvals = [s.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE)
                     for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, categories, axesNames=[
                    "Condition", "Frac of probe exploring"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_frac_explore_box", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # Num explore bouts in probe by condition
            categories = ["Interruption" if sessionWithProbeIsInterruption[i]
                          else "Control" for i in range(numSessionsWithProbe)]
            yvals = [len(s.probe_explore_bout_ends)
                     for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, categories, axesNames=[
                    "Condition", "Num explore bouts"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_explore_bouts",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe num wells visited per bout with repeats by condition
            categories = []
            yvals = []
            for sesh in sessionsWithProbe:
                for i1, i2 in zip(sesh.probe_explore_bout_starts, sesh.probe_explore_bout_ends):
                    categories.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    yvals.append(numWellsVisited(
                        sesh.probe_nearest_wells[i1:i2], countReturns=True))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, categories, axesNames=[
                    "Condition", "Number of wells visited per explore bout with repeats"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_wells_visited_per_explore_bout_with_repeats",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # probe num wells visited per bout without repeats by condition
            categories = []
            yvals = []
            for sesh in sessionsWithProbe:
                for i1, i2 in zip(sesh.probe_explore_bout_starts, sesh.probe_explore_bout_ends):
                    categories.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    yvals.append(numWellsVisited(
                        sesh.probe_nearest_wells[i1:i2], countReturns=False))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, categories, axesNames=[
                    "Condition", "Number of wells visited per explore bout without repeats"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_wells_visited_per_explore_bout_without_repeats",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

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
            axs = saveOrShow("probe_numVisitedOverTime",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # num wells visited in probe by time by condition
            swrfracExplores = numVisited[sessionWithProbeIsInterruption, :]
            ctrlfracExplores = numVisited[np.logical_not(sessionWithProbeIsInterruption), :]
            plotIndividualAndAverage(axs, swrfracExplores, t1Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, ctrlfracExplores, t1Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xticks(np.arange(0, 60*5+1, 60))
            axs = saveOrShow("probe_numVisitedOverTime_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # total dwell time, full probe
            totalDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                      else "Control" for i in range(numSessionsWithProbe)]
            totalDwellTimeHome = [s.total_dwell_time(True, s.home_well)
                                  for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], totalDwellTimeHome, totalDwellTimeHomeCats, axesNames=[
                    "Condition", "total dwell time (s)"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_total_dwell_time_home",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            totalDwellTimeAwayCats = []
            totalDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    totalDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    totalDwellTimeAway.append(sesh.total_dwell_time(True, aw))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], totalDwellTimeAway, totalDwellTimeAwayCats, axesNames=[
                    "Condition", "total dwell time (s)"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_total_dwell_time_aways",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = totalDwellTimeHomeCats + totalDwellTimeAwayCats
            yvals = totalDwellTimeHome + totalDwellTimeAway
            wellCats = ["home"] * len(totalDwellTimeHomeCats) + \
                ["away"] * len(totalDwellTimeAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "total dwell time (s)", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_total_dwell_time", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # total dwell time, 90sec
            totalDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                      else "Control" for i in range(numSessionsWithProbe)]
            totalDwellTimeHome = [s.total_dwell_time(True, s.home_well, timeInterval=[0, 90])
                                  for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], totalDwellTimeHome, totalDwellTimeHomeCats, axesNames=[
                    "Condition", "total dwell time (s)"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_total_dwell_time_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            totalDwellTimeAwayCats = []
            totalDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    totalDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    totalDwellTimeAway.append(sesh.total_dwell_time(
                        True, aw, timeInterval=[0, 90]))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], totalDwellTimeAway, totalDwellTimeAwayCats, axesNames=[
                    "Condition", "total dwell time (s)"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_total_dwell_time_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = totalDwellTimeHomeCats + totalDwellTimeAwayCats
            yvals = totalDwellTimeHome + totalDwellTimeAway
            wellCats = ["home"] * len(totalDwellTimeHomeCats) + \
                ["away"] * len(totalDwellTimeAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "total dwell time (s)", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_total_dwell_time_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # Avg dwell time, full probe
            avgDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                    else "Control" for i in range(numSessionsWithProbe)]
            avgDwellTimeHome = [s.avg_dwell_time(True, s.home_well, emptyVal=np.nan)
                                for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], avgDwellTimeHome, avgDwellTimeHomeCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], statsFile=statsFile, statsAx=axs[1])
            axs[0].set_ylim(top=15)
            axs = saveOrShow("probe_avg_dwell_time_home",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            avgDwellTimeAwayCats = []
            avgDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    avgDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    avgDwellTimeAway.append(sesh.avg_dwell_time(True, aw, emptyVal=np.nan))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], avgDwellTimeAway, avgDwellTimeAwayCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs[0].set_ylim(top=15)
            axs = saveOrShow("probe_avg_dwell_time_aways",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = avgDwellTimeHomeCats + avgDwellTimeAwayCats
            yvals = avgDwellTimeHome + avgDwellTimeAway
            wellCats = ["home"] * len(avgDwellTimeHomeCats) + \
                ["away"] * len(avgDwellTimeAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Avg dwell time (s)", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs[0].set_ylim(top=15)
            axs = saveOrShow("probe_avg_dwell_time", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # Avg dwell time, 90sec
            avgDwellTimeHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                    else "Control" for i in range(numSessionsWithProbe)]
            avgDwellTimeHome = [s.avg_dwell_time(True, s.home_well, emptyVal=np.nan, timeInterval=[0, 90])
                                for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], avgDwellTimeHome, avgDwellTimeHomeCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_avg_dwell_time_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            avgDwellTimeAwayCats = []
            avgDwellTimeAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    avgDwellTimeAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    avgDwellTimeAway.append(sesh.avg_dwell_time(
                        True, aw, emptyVal=np.nan, timeInterval=[0, 90]))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], avgDwellTimeAway, avgDwellTimeAwayCats, axesNames=[
                    "Condition", "Avg dwell time (s)"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_avg_dwell_time_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = avgDwellTimeHomeCats + avgDwellTimeAwayCats
            yvals = avgDwellTimeHome + avgDwellTimeAway
            wellCats = ["home"] * len(avgDwellTimeHomeCats) + \
                ["away"] * len(avgDwellTimeAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Avg dwell time (s)", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_avg_dwell_time_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # num entries, 90sec
            numEntriesHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                  else "Control" for i in range(numSessionsWithProbe)]
            numEntriesHome = [s.num_well_entries(True, s.home_well, timeInterval=[0, 90])
                              for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], numEntriesHome, numEntriesHomeCats, axesNames=[
                    "Condition", "Num entries"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_entries_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            numEntriesAwayCats = []
            numEntriesAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    numEntriesAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    numEntriesAway.append(sesh.num_well_entries(
                        True, aw, timeInterval=[0, 90]))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], numEntriesAway, numEntriesAwayCats, axesNames=[
                    "Condition", "Num entries"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_entries_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = numEntriesHomeCats + numEntriesAwayCats
            yvals = numEntriesHome + numEntriesAway
            wellCats = ["home"] * len(numEntriesHomeCats) + ["away"] * len(numEntriesAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Num entries", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_num_entries_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # curvature, full probe
            curvatureHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                 else "Control" for i in range(numSessionsWithProbe)]
            curvatureHome = [s.avg_curvature_at_well(True, s.home_well)
                             for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureHome, curvatureHomeCats, axesNames=[
                    "Condition", "Curvature"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_home",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            curvatureAwayCats = []
            curvatureAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    curvatureAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureAway.append(sesh.avg_curvature_at_well(True, aw))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureAway, curvatureAwayCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_aways",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = curvatureHomeCats + curvatureAwayCats
            yvals = curvatureHome + curvatureAway
            wellCats = ["home"] * len(curvatureHomeCats) + ["away"] * len(curvatureAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            def onWall(well):
                return well < 9 or well > 40 or well % 8 in [2, 7]

            curvatureOffWallAwayCats = []
            curvatureOffWallAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    if onWall(aw):
                        continue
                    curvatureOffWallAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureOffWallAway.append(sesh.avg_curvature_at_well(True, aw))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureOffWallAway, curvatureOffWallAwayCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_off_wall_aways",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = curvatureHomeCats + curvatureOffWallAwayCats
            yvals = curvatureHome + curvatureOffWallAway
            wellCats = ["home"] * len(curvatureHomeCats) + \
                ["OffWallAway"] * len(curvatureOffWallAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_offwall", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            curvatureOffWallOtherCats = []
            curvatureOffWallOther = []
            for sesh in sessionsWithProbe:
                for ow in allWellNames:
                    if ow == sesh.home_well or ow in sesh.visited_away_wells or onWall(ow):
                        continue
                    curvatureOffWallOtherCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureOffWallOther.append(sesh.avg_curvature_at_well(True, ow))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureOffWallOther, curvatureOffWallOtherCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_off_wall_Others",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = curvatureOffWallOtherCats + curvatureOffWallAwayCats
            yvals = curvatureOffWallOther + curvatureOffWallAway
            wellCats = ["OffWallOther"] * len(curvatureOffWallOtherCats) + \
                ["OffWallAway"] * len(curvatureOffWallAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_offwall_away_vs_Other",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # curvature, 90sec
            curvatureHomeCats = ["Interruption" if sessionWithProbeIsInterruption[i]
                                 else "Control" for i in range(numSessionsWithProbe)]
            curvatureHome = [s.avg_curvature_at_well(True, s.home_well, timeInterval=[0, 90])
                             for s in sessionsWithProbe]
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureHome, curvatureHomeCats, axesNames=[
                    "Condition", "Curvature"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_home_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            curvatureAwayCats = []
            curvatureAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    curvatureAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureAway.append(sesh.avg_curvature_at_well(
                        True, aw, timeInterval=[0, 90]))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureAway, curvatureAwayCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = curvatureHomeCats + curvatureAwayCats
            yvals = curvatureHome + curvatureAway
            wellCats = ["home"] * len(curvatureHomeCats) + ["away"] * len(curvatureAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_90sec", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            curvatureOffWallAwayCats = []
            curvatureOffWallAway = []
            for sesh in sessionsWithProbe:
                for aw in sesh.visited_away_wells:
                    if onWall(aw):
                        continue
                    curvatureOffWallAwayCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureOffWallAway.append(
                        sesh.avg_curvature_at_well(True, aw, timeInterval=[0, 90]))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureOffWallAway, curvatureOffWallAwayCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_off_wall_aways_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = curvatureHomeCats + curvatureOffWallAwayCats
            yvals = curvatureHome + curvatureOffWallAway
            wellCats = ["home"] * len(curvatureHomeCats) + \
                ["OffWallAway"] * len(curvatureOffWallAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_offwall_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            curvatureOffWallOtherCats = []
            curvatureOffWallOther = []
            for sesh in sessionsWithProbe:
                for ow in allWellNames:
                    if ow == sesh.home_well or ow in sesh.visited_away_wells or onWall(ow):
                        continue
                    curvatureOffWallOtherCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    curvatureOffWallOther.append(
                        sesh.avg_curvature_at_well(True, ow, timeInterval=[0, 90]))
            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], curvatureOffWallOther, curvatureOffWallOtherCats, axesNames=[
                    "Condition", "Curvature"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_at_off_wall_Other_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            seshCats = curvatureOffWallOtherCats + curvatureOffWallAwayCats
            yvals = curvatureOffWallOther + curvatureOffWallAway
            wellCats = ["OffWallOther"] * len(curvatureOffWallOtherCats) + \
                ["OffWallAway"] * len(curvatureOffWallAwayCats)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals=yvals, categories=seshCats, categories2=wellCats, axesNames=[
                    "Condition", "Curvature", "Well Type"], violin=True, statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("probe_curvature_offwall_away_vs_Other_90sec",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

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
                    numWellsCheckedWithRepeats = np.vstack(
                        (numWellsCheckedWithRepeats, emptyArr))
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
                    numWellsCheckedWithRepeats = np.vstack(
                        (numWellsCheckedWithRepeats, emptyArr))
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
            axs = saveOrShow("task_wellsCheckedAfterRewardHome",
                             outputDir=outputDir, statsFile=statsFile)
            plotIndividualAndAverage(axs, numWellsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Interruption", wellCat == "away")), :], t1Array,
                individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numWellsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Control", wellCat == "away")), :], t1Array,
                individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 60)
            axs.set_ylim(0, 36)
            axs = saveOrShow("task_wellsCheckedAfterRewardAway",
                             outputDir=outputDir, statsFile=statsFile)

            plotIndividualAndAverage(axs, numTimesHomeWellChecked[homeWellCheckSeshCat == "Interruption", :], t1Array,
                                     individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numTimesHomeWellChecked[homeWellCheckSeshCat == "Control", :], t1Array,
                                     individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 60)
            # axs.set_ylim(0, 36)
            axs = saveOrShow("task_numTimesHomeWellChecked_by_condition",
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
            axs = saveOrShow("task_repeatsCheckedAfterRewardHome",
                             outputDir=outputDir, statsFile=statsFile)
            plotIndividualAndAverage(axs, numRepeatsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Interruption", wellCat == "away")), :], t1Array,
                individualColor="orange", avgColor="orange", spread="sem")
            plotIndividualAndAverage(axs, numRepeatsChecked[np.logical_and(terminationCondition == "found", np.logical_and(
                seshCat == "Control", wellCat == "away")), :], t1Array,
                individualColor="cyan", avgColor="cyan", spread="sem")
            axs.set_xlim(0, 120)
            axs.set_ylim(0, 36)
            axs = saveOrShow("task_repeatsCheckedAfterRewardAway",
                             outputDir=outputDir, statsFile=statsFile)

            # check more wells or repeats on way to home well in one condition?
            idx = terminationCondition == "found"
            cat1 = seshCat[idx]
            cat2 = wellCat[idx]
            cat2 = ["home" if c == "away" else "away" for c in cat2]
            yvals = numWellsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num checked before next reward", "Trial Type"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("task_num_checked_before_next_reward",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            yvals = numRepeatsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num repeats checked before next reward", "Trial Type"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("task_num_repeats_checked_before_next_reward",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            idx = np.logical_and(terminationCondition == "found", trialIdx > 4)
            cat1 = seshCat[idx]
            cat2 = wellCat[idx]
            cat2 = ["home" if c == "away" else "away" for c in cat2]
            yvals = numWellsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num checked before next reward", "Trial Type"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("task_num_checked_before_next_reward_secondhalf",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            yvals = numRepeatsChecked[idx]
            yvals = np.nanmax(yvals, axis=1)
            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], yvals, cat1, categories2=cat2, violin=True, axesNames=[
                    "Condition", "Num repeats checked before next reward", "Trial Type"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("task_num_repeats_checked_before_next_reward_secondhalf",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # How correlated are dwell time and curvature
            wellCats = []
            wallCats = []
            curvature = []
            avgDwell = []

            def onWall(well):
                return well < 9 or well > 40 or well % 8 in [2, 7]

            for sesh in sessionsWithProbe:
                for w in allWellNames:
                    if w == sesh.home_well:
                        wellCats.append("home")
                    elif w in sesh.visited_away_wells:
                        wellCats.append("away")
                    else:
                        wellCats.append("other")

                    wallCats.append(onWall(w))

                    curvature.append(sesh.avg_curvature_at_well(True, w))
                    avgDwell.append(sesh.avg_dwell_time(True, w))

            wellCats = np.array(wellCats)
            wallCats = np.array(wallCats)
            curvature = np.array(curvature)
            avgDwell = np.array(avgDwell)

            offWallIdx = np.logical_not(wallCats)
            awayIdx = wellCats == "away"

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)
            axs.scatter(avgDwell[offWallIdx], curvature[offWallIdx])
            axs = saveOrShow("avgDwellVsCurvature_offwall",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)
            axs.scatter(avgDwell[awayIdx], curvature[awayIdx])
            axs = saveOrShow("avgDwellVsCurvature_away", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)
            axs.scatter(avgDwell[awayIdx & offWallIdx], curvature[awayIdx & offWallIdx])
            axs = saveOrShow("avgDwellVsCurvature_away_offwall",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

        if not SKIP_LFP_PLOTS:
            # =============
            # stim rate at different well types
            stimRateAtWellWellCats = []
            stimRateAtWellSeshCats = []
            stimRateAtWell = []
            for sesh in sessions:
                stimRateAtWellSeshCats.append(
                    "Interruption" if sesh.isRippleInterruption else "Control")
                stimRateAtWellWellCats.append("home")
                if sesh.total_dwell_time(False, sesh.home_well) == 0:
                    stimRateAtWell.append(np.nan)
                else:
                    stimRateAtWell.append(sesh.numStimsAtWell(sesh.home_well) /
                                          sesh.total_dwell_time(False, sesh.home_well))

                for aw in sesh.visited_away_wells:
                    stimRateAtWellSeshCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    stimRateAtWellWellCats.append("away")
                    stimRateAtWell.append(sesh.numStimsAtWell(
                        aw) / sesh.total_dwell_time(False, aw))

            stimRateAtWell = np.array(stimRateAtWell)
            stimRateAtWellWellCats = np.array(stimRateAtWellWellCats)
            stimRateAtWellSeshCats = np.array(stimRateAtWellSeshCats)

            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], stimRateAtWell, stimRateAtWellSeshCats, categories2=stimRateAtWellWellCats, violin=True, axesNames=[
                    "Condition", "Stim rate (Hz)", "Well type"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("lfp_stim_rate_at_wells", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            stimRateAtWellSWR = stimRateAtWell[stimRateAtWellSeshCats == "Interruption"]
            stimRateAtWellWellCatsSWR = stimRateAtWellWellCats[stimRateAtWellSeshCats == "Interruption"]
            boxPlot(axs[0], stimRateAtWellSWR, stimRateAtWellWellCatsSWR, violin=True, axesNames=[
                    "Well type", "Stim rate (Hz)"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("lfp_stim_rate_at_wells_just_swr",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # ripple rate at different well types
            rippleRateAtWellWellCats = []
            rippleRateAtWellSeshCats = []
            rippleRateAtWell = []
            for sesh in sessions:
                rippleRateAtWellSeshCats.append(
                    "Interruption" if sesh.isRippleInterruption else "Control")
                rippleRateAtWellWellCats.append("home")
                if sesh.total_dwell_time(False, sesh.home_well) == 0:
                    rippleRateAtWell.append(np.nan)
                else:
                    rippleRateAtWell.append(sesh.numRipplesAtWell(False, sesh.home_well) /
                                            sesh.total_dwell_time(False, sesh.home_well))

                for aw in sesh.visited_away_wells:
                    rippleRateAtWellSeshCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    rippleRateAtWellWellCats.append("away")
                    rippleRateAtWell.append(sesh.numRipplesAtWell(False, aw) /
                                            sesh.total_dwell_time(False, aw))

            rippleRateAtWell = np.array(rippleRateAtWell)
            rippleRateAtWellWellCats = np.array(rippleRateAtWellWellCats)
            rippleRateAtWellSeshCats = np.array(rippleRateAtWellSeshCats)

            plt.close(fig)
            fig, axs = plt.subplots(
                1, 2, gridspec_kw={'width_ratios': [doubleBoxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((doubleBoxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            boxPlot(axs[0], rippleRateAtWell, rippleRateAtWellSeshCats, categories2=rippleRateAtWellWellCats, violin=True, axesNames=[
                    "Condition", "Ripple Rate (Hz)", "Well type"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("lfp_ripple_rate_at_wells", outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            plt.close(fig)
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [
                boxPlotWidthRatio, boxPlotStatsRatio]})
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth((boxPlotWidthRatio+boxPlotStatsRatio) * FIG_SCALE / 2)
            rippleRateAtWellSWR = rippleRateAtWell[rippleRateAtWellSeshCats == "Interruption"]
            rippleRateAtWellWellCatsSWR = rippleRateAtWellWellCats[rippleRateAtWellSeshCats == "Interruption"]
            boxPlot(axs[0], rippleRateAtWellSWR, rippleRateAtWellWellCatsSWR, violin=True, axesNames=[
                    "Well type", "Ripple Rate (Hz)"], statsFile=statsFile, statsAx=axs[1])
            axs = saveOrShow("lfp_ripple_rate_at_wells_just_swr",
                             outputDir=outputDir, statsFile=statsFile)
            fig.clf()
            axs = plt.subplot(111)

            # =============
            # cumulative number of stims and ripples
            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE)
            windowSize = 5
            bins = np.arange(0, 60*20+1, windowSize)
            stimCounts = np.empty((numSessions, len(bins)-1))
            rippleCounts = np.empty((numSessions, len(bins)-1))
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
                stimTs /= BTSession.TRODES_SAMPLING_RATE
                stimCounts[si, :], _ = np.histogram(stimTs, bins=bins)

                for i in reversed(range(len(bins)-1)):
                    if stimCounts[si, i] != 0:
                        break
                    stimCounts[si, i] = np.nan

                ripTs = np.array(sesh.btRipStartTimestampsPreStats)
                ripTs = ripTs[np.logical_and(ripTs > sesh.bt_pos_ts[0], ripTs <
                                             sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                ripTs /= BTSession.TRODES_SAMPLING_RATE
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
                else:
                    # ITI
                    stimTs = np.array(sesh.interruption_timestamps)
                    stimTs = stimTs[np.logical_and(stimTs > sesh.itiLfpStart_ts, stimTs <
                                                   sesh.itiLfpEnd_ts)] - sesh.itiLfpStart_ts
                    stimTs /= BTSession.TRODES_SAMPLING_RATE
                    itiStimCounts[si, :], _ = np.histogram(stimTs, bins=itiBins)

                    for i in reversed(range(len(itiBins)-1)):
                        if itiStimCounts[si, i] != 0:
                            break
                        itiStimCounts[si, i] = np.nan

                    ripTs = np.array(sesh.ITIRipStartTimestamps)
                    ripTs = ripTs[np.logical_and(ripTs > sesh.itiLfpStart_ts, ripTs <
                                                 sesh.itiLfpEnd_ts)] - sesh.itiLfpStart_ts
                    ripTs /= BTSession.TRODES_SAMPLING_RATE
                    itiRippleCounts[si, :], _ = np.histogram(ripTs, bins=itiBins)

                    for i in reversed(range(len(itiBins)-1)):
                        if itiRippleCounts[si, i] != 0:
                            break
                        itiRippleCounts[si, i] = np.nan

                    # Probe
                    stimTs = np.array(sesh.interruption_timestamps)
                    stimTs = stimTs[np.logical_and(stimTs > sesh.probeLfpStart_ts, stimTs <
                                                   sesh.probeLfpEnd_ts)] - sesh.probeLfpStart_ts
                    stimTs /= BTSession.TRODES_SAMPLING_RATE
                    probeStimCounts[si, :], _ = np.histogram(stimTs, bins=probeBins)

                    for i in reversed(range(len(probeBins)-1)):
                        if probeStimCounts[si, i] != 0:
                            break
                        probeStimCounts[si, i] = np.nan

                    ripTs = np.array(sesh.probeRipStartTimestamps)
                    ripTs = ripTs[np.logical_and(ripTs > sesh.probeLfpStart_ts, ripTs <
                                                 sesh.probeLfpEnd_ts)] - sesh.probeLfpStart_ts
                    ripTs /= BTSession.TRODES_SAMPLING_RATE
                    probeRippleCounts[si, :], _ = np.histogram(ripTs, bins=probeBins)

                    for i in reversed(range(len(probeBins)-1)):
                        if probeRippleCounts[si, i] != 0:
                            break
                        probeRippleCounts[si, i] = np.nan

            stimCounts = np.cumsum(stimCounts, axis=1)
            rippleCounts = np.cumsum(rippleCounts, axis=1)
            itiStimCounts = np.cumsum(itiStimCounts, axis=1)
            itiRippleCounts = np.cumsum(itiRippleCounts, axis=1)
            probeStimCounts = np.cumsum(probeStimCounts, axis=1)
            probeRippleCounts = np.cumsum(probeRippleCounts, axis=1)

            axs.plot(bins[1:], stimCounts.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_task_stimCounts", outputDir=outputDir, statsFile=statsFile)
            axs.plot(bins[1:], rippleCounts.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_task_rippleCounts", outputDir=outputDir, statsFile=statsFile)

            stimCountsSWR = stimCounts[sessionIsInterruption, :]
            stimCountsCtrl = stimCounts[np.logical_not(sessionIsInterruption), :]
            axs.plot(bins[1:], stimCountsSWR.T, c="orange", zorder=1)
            axs.plot(bins[1:], stimCountsCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_task_stimCounts_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            rippleCountsSWR = rippleCounts[sessionIsInterruption, :]
            rippleCountsCtrl = rippleCounts[np.logical_not(sessionIsInterruption), :]
            axs.plot(bins[1:], rippleCountsSWR.T, c="orange", zorder=1)
            axs.plot(bins[1:], rippleCountsCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_task_rippleCounts_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            axs.plot(itiBins[1:], itiStimCounts.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_iti_stimCounts", outputDir=outputDir, statsFile=statsFile)
            axs.plot(itiBins[1:], itiRippleCounts.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_iti_rippleCounts", outputDir=outputDir, statsFile=statsFile)

            itiStimCountsSWR = itiStimCounts[sessionIsInterruption, :]
            itiStimCountsCtrl = itiStimCounts[np.logical_not(sessionIsInterruption), :]
            axs.plot(itiBins[1:], itiStimCountsSWR.T, c="orange", zorder=1)
            axs.plot(itiBins[1:], itiStimCountsCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_iti_stimCounts_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            itiRippleCountsSWR = itiRippleCounts[sessionIsInterruption, :]
            itiRippleCountsCtrl = itiRippleCounts[np.logical_not(sessionIsInterruption), :]
            axs.plot(itiBins[1:], itiRippleCountsSWR.T, c="orange", zorder=1)
            axs.plot(itiBins[1:], itiRippleCountsCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_iti_rippleCounts_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            axs.plot(probeBins[1:], probeStimCounts.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_probe_stimCounts", outputDir=outputDir, statsFile=statsFile)
            axs.plot(probeBins[1:], probeRippleCounts.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_probe_rippleCounts", outputDir=outputDir, statsFile=statsFile)

            probeStimCountsSWR = probeStimCounts[sessionIsInterruption, :]
            probeStimCountsCtrl = probeStimCounts[np.logical_not(sessionIsInterruption), :]
            axs.plot(probeBins[1:], probeStimCountsSWR.T, c="orange", zorder=1)
            axs.plot(probeBins[1:], probeStimCountsCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_probe_stimCounts_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            probeRippleCountsSWR = probeRippleCounts[sessionIsInterruption, :]
            probeRippleCountsCtrl = probeRippleCounts[np.logical_not(sessionIsInterruption), :]
            axs.plot(probeBins[1:], probeRippleCountsSWR.T, c="orange", zorder=1)
            axs.plot(probeBins[1:], probeRippleCountsCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_probe_rippleCounts_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # cumulative number of stims and ripples but just for sessions with probe
            probeIdx = np.array([s.probe_performed for s in sessions])
            stimCountsWithProbe = stimCounts[probeIdx, :]
            rippleCountsWithProbe = rippleCounts[probeIdx, :]
            itiStimCountsWithProbe = itiStimCounts[probeIdx, :]
            itiRippleCountsWithProbe = itiRippleCounts[probeIdx, :]
            probeStimCountsWithProbe = probeStimCounts[probeIdx, :]
            probeRippleCountsWithProbe = probeRippleCounts[probeIdx, :]

            axs.plot(bins[1:], stimCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_task_stimCounts_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            axs.plot(bins[1:], rippleCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_task_rippleCounts_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            stimCountsWithProbeSWR = stimCountsWithProbe[sessionWithProbeIsInterruption, :]
            stimCountsWithProbeCtrl = stimCountsWithProbe[np.logical_not(
                sessionWithProbeIsInterruption), :]
            axs.plot(bins[1:], stimCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(bins[1:], stimCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_task_stimCounts_by_condition_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            rippleCountsWithProbeSWR = rippleCountsWithProbe[sessionWithProbeIsInterruption, :]
            rippleCountsWithProbeCtrl = rippleCountsWithProbe[np.logical_not(
                sessionWithProbeIsInterruption), :]
            axs.plot(bins[1:], rippleCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(bins[1:], rippleCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_task_rippleCounts_by_condition_probeonly",
                             outputDir=outputDir, statsFile=statsFile)

            axs.plot(itiBins[1:], itiStimCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_iti_stimCounts_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            axs.plot(itiBins[1:], itiRippleCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_iti_rippleCounts_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            itiStimCountsWithProbeSWR = itiStimCountsWithProbe[sessionWithProbeIsInterruption, :]
            itiStimCountsWithProbeCtrl = itiStimCountsWithProbe[np.logical_not(
                sessionWithProbeIsInterruption), :]
            axs.plot(itiBins[1:], itiStimCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(itiBins[1:], itiStimCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_iti_stimCounts_by_condition_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            itiRippleCountsWithProbeSWR = itiRippleCountsWithProbe[sessionWithProbeIsInterruption, :]
            itiRippleCountsWithProbeCtrl = itiRippleCountsWithProbe[np.logical_not(
                sessionWithProbeIsInterruption), :]
            axs.plot(itiBins[1:], itiRippleCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(itiBins[1:], itiRippleCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_iti_rippleCounts_by_condition_probeonly",
                             outputDir=outputDir, statsFile=statsFile)

            axs.plot(probeBins[1:], probeStimCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_probe_stimCounts_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            axs.plot(probeBins[1:], probeRippleCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_probe_rippleCounts_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            probeStimCountsWithProbeSWR = probeStimCountsWithProbe[sessionWithProbeIsInterruption, :]
            probeStimCountsWithProbeCtrl = probeStimCountsWithProbe[np.logical_not(
                sessionWithProbeIsInterruption), :]
            axs.plot(probeBins[1:], probeStimCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(probeBins[1:], probeStimCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_probe_stimCounts_by_condition_probeonly",
                             outputDir=outputDir, statsFile=statsFile)
            probeRippleCountsWithProbeSWR = probeRippleCountsWithProbe[sessionWithProbeIsInterruption, :]
            probeRippleCountsWithProbeCtrl = probeRippleCountsWithProbe[np.logical_not(
                sessionWithProbeIsInterruption), :]
            axs.plot(probeBins[1:], probeRippleCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(probeBins[1:], probeRippleCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_probe_rippleCounts_by_condition_probeonly",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # same but plot both probe and non, with non as dashed
            noProbeIdx = np.logical_not(probeIdx)
            stimCountsWithoutProbe = stimCounts[noProbeIdx, :]
            rippleCountsWithoutProbe = rippleCounts[noProbeIdx, :]
            itiStimCountsWithoutProbe = itiStimCounts[noProbeIdx, :]
            itiRippleCountsWithoutProbe = itiRippleCounts[noProbeIdx, :]
            probeStimCountsWithoutProbe = probeStimCounts[noProbeIdx, :]
            probeRippleCountsWithoutProbe = probeRippleCounts[noProbeIdx, :]

            axs.plot(bins[1:], stimCountsWithoutProbe.T, '--', c="grey", zorder=1)
            axs.plot(bins[1:], stimCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_task_stimCounts_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            axs.plot(bins[1:], rippleCountsWithoutProbe.T, '--', c="grey", zorder=1)
            axs.plot(bins[1:], rippleCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_task_rippleCounts_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            stimCountsWithoutProbeSWR = stimCountsWithoutProbe[sessionWithoutProbeIsInterruption, :]
            stimCountsWithoutProbeCtrl = stimCountsWithoutProbe[np.logical_not(
                sessionWithoutProbeIsInterruption), :]
            axs.plot(bins[1:], stimCountsWithoutProbeSWR.T, '--', c="orange", zorder=1)
            axs.plot(bins[1:], stimCountsWithoutProbeCtrl.T, '--', c="cyan", zorder=1)
            axs.plot(bins[1:], stimCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(bins[1:], stimCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_task_stimCounts_by_condition_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            rippleCountsWithoutProbeSWR = rippleCountsWithoutProbe[sessionWithoutProbeIsInterruption, :]
            rippleCountsWithoutProbeCtrl = rippleCountsWithoutProbe[np.logical_not(
                sessionWithoutProbeIsInterruption), :]
            axs.plot(bins[1:], rippleCountsWithoutProbeSWR.T, '--', c="orange", zorder=1)
            axs.plot(bins[1:], rippleCountsWithoutProbeCtrl.T, '--', c="cyan", zorder=1)
            axs.plot(bins[1:], rippleCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(bins[1:], rippleCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs.set_ylim(0, 700)
            axs = saveOrShow("lfp_task_rippleCounts_by_condition_nondash",
                             outputDir=outputDir, statsFile=statsFile)

            axs.plot(itiBins[1:], itiStimCountsWithoutProbe.T, '--', c="grey", zorder=1)
            axs.plot(itiBins[1:], itiStimCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_iti_stimCounts_nondash", outputDir=outputDir, statsFile=statsFile)
            axs.plot(itiBins[1:], itiRippleCountsWithoutProbe.T, '--', c="grey", zorder=1)
            axs.plot(itiBins[1:], itiRippleCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_iti_rippleCounts_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            itiStimCountsWithoutProbeSWR = itiStimCountsWithoutProbe[sessionWithoutProbeIsInterruption, :]
            itiStimCountsWithoutProbeCtrl = itiStimCountsWithoutProbe[np.logical_not(
                sessionWithoutProbeIsInterruption), :]
            axs.plot(itiBins[1:], itiStimCountsWithoutProbeSWR.T, '--', c="orange", zorder=1)
            axs.plot(itiBins[1:], itiStimCountsWithoutProbeCtrl.T, '--', c="cyan", zorder=1)
            axs.plot(itiBins[1:], itiStimCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(itiBins[1:], itiStimCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_iti_stimCounts_by_condition_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            itiRippleCountsWithoutProbeSWR = itiRippleCountsWithoutProbe[sessionWithoutProbeIsInterruption, :]
            itiRippleCountsWithoutProbeCtrl = itiRippleCountsWithoutProbe[np.logical_not(
                sessionWithoutProbeIsInterruption), :]
            axs.plot(itiBins[1:], itiRippleCountsWithoutProbeSWR.T, '--', c="orange", zorder=1)
            axs.plot(itiBins[1:], itiRippleCountsWithoutProbeCtrl.T, '--', c="cyan", zorder=1)
            axs.plot(itiBins[1:], itiRippleCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(itiBins[1:], itiRippleCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_iti_rippleCounts_by_condition_nondash",
                             outputDir=outputDir, statsFile=statsFile)

            axs.plot(probeBins[1:], probeStimCountsWithoutProbe.T, '--', c="grey", zorder=1)
            axs.plot(probeBins[1:], probeStimCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_probe_stimCounts_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            axs.plot(probeBins[1:], probeRippleCountsWithoutProbe.T, '--', c="grey", zorder=1)
            axs.plot(probeBins[1:], probeRippleCountsWithProbe.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_probe_rippleCounts_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            probeStimCountsWithoutProbeSWR = probeStimCountsWithoutProbe[sessionWithoutProbeIsInterruption, :]
            probeStimCountsWithoutProbeCtrl = probeStimCountsWithoutProbe[np.logical_not(
                sessionWithoutProbeIsInterruption), :]
            axs.plot(probeBins[1:], probeStimCountsWithoutProbeSWR.T, '--', c="orange", zorder=1)
            axs.plot(probeBins[1:], probeStimCountsWithoutProbeCtrl.T, '--', c="cyan", zorder=1)
            axs.plot(probeBins[1:], probeStimCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(probeBins[1:], probeStimCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Stim Count")
            axs = saveOrShow("lfp_probe_stimCounts_by_condition_nondash",
                             outputDir=outputDir, statsFile=statsFile)
            probeRippleCountsWithoutProbeSWR = probeRippleCountsWithoutProbe[sessionWithoutProbeIsInterruption, :]
            probeRippleCountsWithoutProbeCtrl = probeRippleCountsWithoutProbe[np.logical_not(
                sessionWithoutProbeIsInterruption), :]
            axs.plot(probeBins[1:], probeRippleCountsWithoutProbeSWR.T, '--', c="orange", zorder=1)
            axs.plot(probeBins[1:], probeRippleCountsWithoutProbeCtrl.T, '--', c="cyan", zorder=1)
            axs.plot(probeBins[1:], probeRippleCountsWithProbeSWR.T, c="orange", zorder=1)
            axs.plot(probeBins[1:], probeRippleCountsWithProbeCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_probe_rippleCounts_by_condition_nondash",
                             outputDir=outputDir, statsFile=statsFile)

            # =============
            # cumulative number of ripples (just when using probe stats)
            windowSize = 5
            bins = np.arange(0, 60*20+1, windowSize)
            rippleCounts = np.empty((numSessionsWithProbe, len(bins)-1))
            for si, sesh in enumerate(sessionsWithProbe):
                ripTs = np.array(sesh.btRipStartTimestampsProbeStats)
                ripTs = ripTs[np.logical_and(ripTs > sesh.bt_pos_ts[0], ripTs <
                                             sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                ripTs /= BTSession.TRODES_SAMPLING_RATE
                rippleCounts[si, :], _ = np.histogram(ripTs, bins=bins)

                for i in reversed(range(len(bins)-1)):
                    if rippleCounts[si, i] != 0:
                        break
                    rippleCounts[si, i] = np.nan

            rippleCounts = np.cumsum(rippleCounts, axis=1)

            axs.plot(bins[1:], rippleCounts.T, c="grey", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_rippleCountsProbeStats", outputDir=outputDir, statsFile=statsFile)

            rippleCountsSWR = rippleCounts[sessionWithProbeIsInterruption, :]
            rippleCountsCtrl = rippleCounts[np.logical_not(sessionWithProbeIsInterruption), :]
            axs.plot(bins[1:], rippleCountsSWR.T, c="orange", zorder=1)
            axs.plot(bins[1:], rippleCountsCtrl.T, c="cyan", zorder=1)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("Cumulative Ripple Count")
            axs = saveOrShow("lfp_rippleCountsProbeStats_by_condition",
                             outputDir=outputDir, statsFile=statsFile)

        if not SKIP_COMPARISON_PLOTS:
            # =============
            # stim rate vs persev measures
            curvature = []
            curvature90sec = []
            avgDwell = []
            avgDwell90sec = []
            wellCats = []
            wellSeshCats = []
            stimRateAtWell = []
            latencyToHome = []

            seshCats = []
            seshStimCount = []
            seshStimRate = []
            seshRippleCount = []
            seshRippleRate = []
            seshNumWellsFound = []

            wellHasProbeFlag = []
            seshHasProbeFlag = []

            for sesh in sessions:
                wellSeshCats.append(
                    "Interruption" if sesh.isRippleInterruption else "Control")
                seshCats.append(
                    "Interruption" if sesh.isRippleInterruption else "Control")
                wellCats.append("home")

                stimRateAtWell.append(sesh.numStimsAtWell(sesh.home_well) /
                                      sesh.total_dwell_time(False, sesh.home_well))

                if sesh.probe_performed:
                    curvature.append(sesh.avg_curvature_at_well(
                        True, sesh.home_well))
                    curvature90sec.append(sesh.avg_curvature_at_well(
                        True, sesh.home_well, timeInterval=[0, 90]))
                    avgDwell.append(sesh.avg_dwell_time(True, sesh.home_well))
                    avgDwell90sec.append(sesh.avg_dwell_time(
                        True, sesh.home_well, timeInterval=[0, 90]))
                    latencyToHome.append(sesh.getLatencyToWell(
                        True, sesh.home_well, returnSeconds=True, emptyVal=300))
                else:
                    curvature.append(np.nan)
                    curvature90sec.append(np.nan)
                    avgDwell.append(np.nan)
                    avgDwell90sec.append(np.nan)
                    latencyToHome.append(np.nan)

                seshStimCount.append(len(sesh.bt_interruption_pos_idxs))
                seshStimRate.append(len(sesh.bt_interruption_pos_idxs) /
                                    ((sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE))
                seshRippleCount.append(len(sesh.btRipStartTimestampsPreStats))
                seshRippleRate.append(len(sesh.btRipStartTimestampsPreStats) /
                                      ((sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE))
                seshNumWellsFound.append(sesh.num_home_found + sesh.num_away_found)

                wellHasProbeFlag.append(sesh.probe_performed)
                seshHasProbeFlag.append(sesh.probe_performed)

                for aw in sesh.visited_away_wells:
                    wellSeshCats.append(
                        "Interruption" if sesh.isRippleInterruption else "Control")
                    wellCats.append("away")
                    stimRateAtWell.append(sesh.numStimsAtWell(
                        aw) / sesh.total_dwell_time(False, aw))

                    if sesh.probe_performed:
                        curvature.append(sesh.avg_curvature_at_well(
                            True, aw))
                        curvature90sec.append(sesh.avg_curvature_at_well(
                            True, aw, timeInterval=[0, 90]))
                        avgDwell.append(sesh.avg_dwell_time(True, aw))
                        avgDwell90sec.append(sesh.avg_dwell_time(True, aw, timeInterval=[0, 90]))
                    else:
                        curvature.append(np.nan)
                        curvature90sec.append(np.nan)
                        avgDwell.append(np.nan)
                        avgDwell90sec.append(np.nan)

                    wellHasProbeFlag.append(sesh.probe_performed)

            homeIdx = np.array([w == "home" for w in wellCats])
            awayIdx = np.logical_not(homeIdx)
            swrIdx = np.array([c == "Interruption" for c in wellSeshCats])
            ctlIdx = np.logical_not(swrIdx)
            seshSwrIdx = np.array([c == "Interruption" for c in seshCats])
            seshCtlIdx = np.logical_not(seshSwrIdx)

            seshProbeIdx = np.array(seshHasProbeFlag)
            wellProbeIdx = np.array(wellHasProbeFlag)
            seshNoProbeIdx = np.logical_not(seshProbeIdx)
            wellNoProbeIdx = np.logical_not(wellProbeIdx)

            stimRateAtWell = np.array(stimRateAtWell)
            curvature = np.array(curvature)
            curvature90sec = np.array(curvature90sec)
            avgDwell = np.array(avgDwell)
            avgDwell90sec = np.array(avgDwell90sec)
            latencyToHome = np.array(latencyToHome)
            seshStimCount = np.array(seshStimCount)
            seshStimRate = np.array(seshStimRate)
            seshRippleCount = np.array(seshRippleCount)
            seshRippleRate = np.array(seshRippleRate)
            seshNumWellsFound = np.array(seshNumWellsFound)

            wellPersevMeasures = [curvature, curvature90sec, avgDwell, avgDwell90sec]
            wellPersevMeasureNames = ["Curvature", "Curvature, 90sec",
                                      "Average Dwell Time (s)", "Average Dwell Time (s), 90sec"]
            wellLFPMeasures = [stimRateAtWell]
            wellLFPMeasureNames = ["Stim Rate (Hz)"]
            seshPersevMeasures = [curvature[homeIdx], curvature90sec[homeIdx],
                                  avgDwell[homeIdx], avgDwell90sec[homeIdx], seshNumWellsFound, latencyToHome]
            seshPersevMeasureNames = ["Curvature at Home", "Curvature at Home, 90sec",
                                      "Average Dwell Time (s)", "Average Dwell Time (s), 90sec",
                                      "Number of Rewards Found", "Latency to home well (s)"]
            seshLFPMeasures = [seshStimCount, seshStimRate, seshRippleCount, seshRippleRate]
            seshLFPMeasureNames = ["Session Stim Count", "Session Stim Rate (Hz)",
                                   "Session Ripple Count", "Session Ripple Rate (Hz)"]

            fig.set_figheight(FIG_SCALE / 2)
            fig.set_figwidth(FIG_SCALE / 2)

            for wpmi, wellPersevMeasure in enumerate(wellPersevMeasures):
                for wlmi, wellLFPMeasure in enumerate(wellLFPMeasures):
                    noProbeVals = wellPersevMeasure[wellNoProbeIdx]
                    if not all(np.isnan(noProbeVals)):
                        # want to include sessions with no probe

                        axs.scatter(wellLFPMeasure[awayIdx & swrIdx],
                                    wellPersevMeasure[awayIdx & swrIdx], marker="s", color="orange")
                        axs.scatter(wellLFPMeasure[awayIdx & ctlIdx],
                                    wellPersevMeasure[awayIdx & ctlIdx], marker="s", color="cyan")
                        axs.scatter(wellLFPMeasure[homeIdx & swrIdx],
                                    wellPersevMeasure[homeIdx & swrIdx], marker="o", color="orange")
                        axs.scatter(wellLFPMeasure[homeIdx & ctlIdx],
                                    wellPersevMeasure[homeIdx & ctlIdx], marker="o", color="cyan")
                        axs.set_xlabel(wellLFPMeasureNames[wlmi])
                        axs.set_ylabel(wellPersevMeasureNames[wpmi])
                        axs = saveOrShow("compare_{}_{}_{}_{}".format(wellLFPMeasureNames[wlmi], wellPersevMeasureNames[wpmi], wlmi, wpmi),
                                         outputDir=outputDir, statsFile=statsFile)

                        axs.scatter(wellLFPMeasure[homeIdx & swrIdx],
                                    wellPersevMeasure[homeIdx & swrIdx], marker="o", color="orange")
                        axs.scatter(wellLFPMeasure[homeIdx & ctlIdx],
                                    wellPersevMeasure[homeIdx & ctlIdx], marker="o", color="cyan")
                        axs.set_xlabel(wellLFPMeasureNames[wlmi])
                        axs.set_ylabel(wellPersevMeasureNames[wpmi])
                        axs = saveOrShow("compare_{}_{}_{}_{}_justhomes".format(wellLFPMeasureNames[wlmi], wellPersevMeasureNames[wpmi], wlmi, wpmi),
                                         outputDir=outputDir, statsFile=statsFile)

                    axs.scatter(wellLFPMeasure[awayIdx & swrIdx & wellProbeIdx],
                                wellPersevMeasure[awayIdx & swrIdx & wellProbeIdx], marker="s", color="orange")
                    axs.scatter(wellLFPMeasure[awayIdx & ctlIdx & wellProbeIdx],
                                wellPersevMeasure[awayIdx & ctlIdx & wellProbeIdx], marker="s", color="cyan")
                    axs.scatter(wellLFPMeasure[homeIdx & swrIdx & wellProbeIdx],
                                wellPersevMeasure[homeIdx & swrIdx & wellProbeIdx], marker="o", color="orange")
                    axs.scatter(wellLFPMeasure[homeIdx & ctlIdx & wellProbeIdx],
                                wellPersevMeasure[homeIdx & ctlIdx & wellProbeIdx], marker="o", color="cyan")
                    axs.set_xlabel(wellLFPMeasureNames[wlmi])
                    axs.set_ylabel(wellPersevMeasureNames[wpmi])
                    axs = saveOrShow("compare_{}_{}_{}_{}_justprobe".format(wellLFPMeasureNames[wlmi], wellPersevMeasureNames[wpmi], wlmi, wpmi),
                                     outputDir=outputDir, statsFile=statsFile)

                    axs.scatter(wellLFPMeasure[homeIdx & swrIdx & wellProbeIdx],
                                wellPersevMeasure[homeIdx & swrIdx & wellProbeIdx], marker="o", color="orange")
                    axs.scatter(wellLFPMeasure[homeIdx & ctlIdx & wellProbeIdx],
                                wellPersevMeasure[homeIdx & ctlIdx & wellProbeIdx], marker="o", color="cyan")
                    axs.set_xlabel(wellLFPMeasureNames[wlmi])
                    axs.set_ylabel(wellPersevMeasureNames[wpmi])
                    axs = saveOrShow("compare_{}_{}_{}_{}_justhomes_justprobe".format(wellLFPMeasureNames[wlmi], wellPersevMeasureNames[wpmi], wlmi, wpmi),
                                     outputDir=outputDir, statsFile=statsFile)

            for spmi, seshPersevMeasure in enumerate(seshPersevMeasures):
                for slmi, seshLFPMeasure in enumerate(seshLFPMeasures):
                    noProbeVals = seshPersevMeasure[seshNoProbeIdx]
                    if not all(np.isnan(noProbeVals)):
                        # want to include sessions with no probe
                        axs.scatter(seshLFPMeasure[seshSwrIdx],
                                    seshPersevMeasure[seshSwrIdx], color="orange")
                        axs.scatter(seshLFPMeasure[seshCtlIdx],
                                    seshPersevMeasure[seshCtlIdx], color="cyan")
                        axs.set_xlabel(seshLFPMeasureNames[slmi])
                        axs.set_ylabel(seshPersevMeasureNames[spmi])

                        if seshPersevMeasureNames[spmi] == "Number of Rewards Found":
                            axs.set_ylim(-1, 20)
                            axs.set_yticks(np.arange(1, 20, 3))
                        elif seshPersevMeasureNames[spmi] == "Average Dwell Time (s)":
                            axs.set_ylim(1, 10)
                            # axs.set_yticks(np.arange(1, 20, 3))
                        elif "Curavture" in seshPersevMeasureNames[spmi]:
                            axs.set_ylim(0, 2.1)
                            # axs.set_yticks(np.arange(1, 20, 3))

                        # if seshLFPMeasureNames[slmi] == "Session Stim Rate (Hz)":
                            # axs.set_xlim(0, 2)
                        # elif seshLFPMeasureNames[slmi] == "Session Ripple Rate (Hz)":
                            # axs.set_xlim(0, 0.65)

                        axs = saveOrShow("compare_{}_{}_{}_{}".format(seshLFPMeasureNames[slmi], seshPersevMeasureNames[spmi], slmi, spmi),
                                         outputDir=outputDir, statsFile=statsFile)

                    axs.scatter(seshLFPMeasure[seshSwrIdx & seshProbeIdx],
                                seshPersevMeasure[seshSwrIdx & seshProbeIdx], color="orange")
                    axs.scatter(seshLFPMeasure[seshCtlIdx & seshProbeIdx],
                                seshPersevMeasure[seshCtlIdx & seshProbeIdx], color="cyan")
                    axs.set_xlabel(seshLFPMeasureNames[slmi])
                    axs.set_ylabel(seshPersevMeasureNames[spmi])

                    if seshPersevMeasureNames[spmi] == "Number of Rewards Found":
                        axs.set_ylim(0, 20)
                        axs.set_yticks(np.arange(1, 20, 3))
                    elif seshPersevMeasureNames[spmi] == "Average Dwell Time (s)":
                        axs.set_ylim(1, 10)
                        # axs.set_yticks(np.arange(1, 20, 3))
                    elif "Curavture" in seshPersevMeasureNames[spmi]:
                        axs.set_ylim(0, 2.1)
                        # axs.set_yticks(np.arange(1, 20, 3))

                    # if seshLFPMeasureNames[slmi] == "Session Stim Rate (Hz)":
                        # axs.set_xlim(0, 2)
                    # elif seshLFPMeasureNames[slmi] == "Session Ripple Rate (Hz)":
                        # axs.set_xlim(0, 0.65)

                    axs = saveOrShow("compare_{}_{}_{}_{}_justprobe".format(seshLFPMeasureNames[slmi], seshPersevMeasureNames[spmi], slmi, spmi),
                                     outputDir=outputDir, statsFile=statsFile)

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
            axs = saveOrShow("task_raw", outputDir=seshOutputDir, statsFile=statsFile)

            # =============
            # Raw behavioral trace (task) colored by curvature
            def lerp(v, c1, c2):
                return np.clip(c1 * (1. - v) + v * c2, 0, 1)

            def normalized(a):
                return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

            curve = normalized(sesh.bt_curvature)
            col1 = np.array([0, 1, 1, 1])
            col2 = np.array([1, 0, 0, 1])
            for i in range(len(sesh.bt_pos_xs)-1):
                axs.plot(sesh.bt_pos_xs[i:i+2], sesh.bt_pos_ys[i:i+2], c=lerp(curve[i], col1, col2))
            setupBehaviorTracePlot(axs, sesh)
            axs = saveOrShow("task_curvature_color", outputDir=seshOutputDir, statsFile=statsFile)

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
                axs = saveOrShow("task_all_trials", outputDir=seshOutputDir, statsFile=statsFile)
                fig.set_figheight(FIG_SCALE)
                fig.set_figwidth(FIG_SCALE)

            if False:
                for fti, ft in enumerate(sesh.home_well_find_pos_idxs):
                    i1 = 0 if fti == 0 else sesh.away_well_leave_pos_idxs[fti-1]
                    i2 = ft
                    axs.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], c="#deac7f")
                    setupBehaviorTracePlot(axs, sesh)
                    axs = saveOrShow("task_home_find_{}".format(
                        fti), outputDir=seshOutputDir, statsFile=statsFile)

                for fti, ft in enumerate(sesh.away_well_find_pos_idxs):
                    i1 = sesh.home_well_leave_pos_idxs[fti]
                    i2 = ft
                    axs.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], c="#deac7f")
                    setupBehaviorTracePlot(axs, sesh)
                    axs = saveOrShow("task_away_find_{}".format(
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
            axs = saveOrShow("probe_raw".format(sesh.name),
                             outputDir=seshOutputDir, statsFile=statsFile)

            # =============
            # Raw behavioral trace (probe) colored by curvature
            def lerp(v, c1, c2):
                return np.clip(c1 * (1. - v) + v * c2, 0, 1)

            def normalized(a):
                return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

            curve = normalized(sesh.probe_curvature)
            col1 = np.array([0, 1, 1, 1])
            col2 = np.array([1, 0, 0, 1])
            for i in range(len(sesh.probe_pos_xs)-1):
                axs.plot(sesh.probe_pos_xs[i:i+2],
                         sesh.probe_pos_ys[i:i+2], c=lerp(curve[i], col1, col2))
            setupBehaviorTracePlot(axs, sesh)
            axs = saveOrShow("probe_curvature_color", outputDir=seshOutputDir, statsFile=statsFile)

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
