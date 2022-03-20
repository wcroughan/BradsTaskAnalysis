import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from consts import allWellNames
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import warnings


class PlotCtx:
    def __init__(self, outputDir="./", priorityLevel=None):
        self.figSizeX, self.figSizeY = 5, 5
        self.fig = plt.figure(figsize=(self.figSizeX, self.figSizeY))
        self.fig.clf()
        self.axs = self.fig.subplots()
        # Save when exiting context?
        self.savePlot = True
        # Show when exiting context?
        self.showPlot = False

        self.timeStr = datetime.now().strftime("%Y%m%d_%H%M%S_info.txt")
        self.setOutputDir(outputDir)
        self.figName = ""

        self.priorityLevel = priorityLevel
        self.priority = None

        self.persistentCategories = {}
        self.savedYVals = {}
        self.savedCategories = {}

    def __enter__(self):
        if self.withStats:
            return (self.axs, self.yvals, self.categories)
        else:
            return self.axs

    def newFig(self, figName, subPlots=None, figScale=1.0, priority=None, withStats=False):
        self.clearFig()
        self.figName = figName
        self.priority = priority
        self.withStats = withStats
        self.yvals = {}
        self.categories = {}

        if subPlots is not None:
            assert len(subPlots) == 2
            self.axs.remove()
            self.axs = self.fig.subplots(*subPlots)

            self.fig.set_figheight(self.figSizeY * figScale * subPlots[0])
            self.fig.set_figwidth(self.figSizeX * figScale * subPlots[1])
        else:
            self.fig.set_figheight(self.figSizeY * figScale)
            self.fig.set_figwidth(self.figSizeX * figScale)

        return self

    def continueFig(self, figName, priority=None):
        self.figName = figName
        self.priority = priority
        self.yvals = {}
        self.categories = {}
        return self

    def setStatCategory(self, category, value):
        self.persistentCategories[category] = value

    def __exit__(self, *args):
        if self.withStats:
            statsName = self.figName.split("/")[-1]
            assert len(self.yvals) > 0
            assert len(self.persistentCategories) + len(self.categories) > 0

            l = None
            for k in self.yvals:
                if l is None:
                    l = len(self.yvals[k])
                else:
                    assert len(self.yvals[k]) == l
            for k in self.categories:
                assert len(self.categories[k]) == l

            for k in self.persistentCategories:
                if k not in self.categories:
                    self.categories[k] = [self.persistentCategories[k]] * l
                else:
                    print("warning: overlap between persistent category and this-plot category, both named {}".format(k))

            if statsName in self.savedYVals:
                savedYVals = self.savedYVals[statsName]
                for k in savedYVals:
                    savedYVals[k] = np.append(savedYVals[k], self.yvals[k])
                savedCategories = self.savedCategories[statsName]
                for k in savedCategories:
                    savedCategories[k] = np.append(savedCategories[k] , self.categories[k])
            else:
                self.savedYVals[statsName] = self.yvals
                self.savedCategories[statsName] = self.categories

        if self.priority is None or self.priorityLevel is None or self.priority <= self.priorityLevel:
            if self.showPlot:
                plt.show()

            if self.savePlot:
                self.saveFig()
    def setFigSize(self, width, height):
        pass

    def saveFig(self):
        if self.figName[0] != "/":
            fname = os.path.join(self.outputDir, self.figName)
        else:
            fname = self.figName
        plt.savefig(fname, bbox_inches="tight", dpi=200)
        self.writeToInfoFile("wrote file {}".format(fname))

    def clearFig(self):
        self.fig.clf()
        self.axs = self.fig.subplots(1,1)
        self.axs.cla()

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        self.txtOutputFName = os.path.join(
            outputDir, self.timeStr)

    def writeToInfoFile(self, txt, suffix="\n"):
        with open(self.txtOutputFName, "a") as f:
            f.write(txt + suffix)
        
    def runShuffles(self):
        for plotName in self.savedCategories:
            categories = self.savedCategories[plotName]
            yvals = self.savedYVals[plotName]
            print("========================================\nRunning shuffles from plot", plotName)
            print("Evaluating {} yvals:".format(len(yvals)))
            for yvalName in yvals:
                print("\t{}".format(yvalName))
            print("Shuffling {} categories:".format(len(categories)))
            for catName in categories:
                print("\t{}\t{}".format(catName, set(categories[catName])))

            for yvalName in yvals:
                yv = yvals[yvalName]
                r, rr = self._doShuffle(yv, yvalName, categories)
                for k in r:
                    print(k)
                    v = r[k]
                    for i in v:
                        print("\t", i, v[i])


            # For cats A1, A2,...Ak yvals Y:
            # If just A1:
                # for possible values v in A1:
                    # metric[v] = mean(data[A1 == v]) - mean(data[A1 != v])
                # Run shuffles on A1 label
                # return:
                    # metric: metric[v] is actual diff
                    # pctile: pctile[v] is frac of shufs metric was greater than
            # If A1, A2:
                # for possible values v in A1:
                    # globalmetrics[v] = mean(data[A1 == v]) - mean(data[A1 != v])
                # Run shuffles on A1 label
                # globalmetrics[v] is actual diff, globalpctile[v]
                # 
                # for possible values v in A1:
                    # withinmetrics[v,:], withinpctile[v,:] = run recursively on just data where A1==v
                # run shuffle on A1, get distribution of withinmetrics[v,:] for all v
                    # interactionmetrics[v,v2] = mean(withinmetrics[v,v2]) - mean(withinmetrics[!v,v2])
                    # get this for all shuffles of A1
            # if more, for A1, ... Ai:
                # run A1, A2 thing above, but v2 is vector representing all Ai>1 values 
        
    def _doShuffle(self, yvals, dataName, categories, valSet=None, numShuffles=4):
        if len(categories) == 1:
            # print(dataName, categories)
            # print(yvals, dataName, categories, valSet)
            catName = list(categories.keys())[0]
            catVals = categories[catName]
            if valSet is None:
                thisCatValSet = sorted(list(set(catVals)))
                print(yvals)
                print(categories)
            else:
                thisCatValSet = valSet[catName]
            df = pd.DataFrame(data={dataName: yvals, catName: catVals})

            meanDiff = np.empty((len(thisCatValSet),1))
            for vi, val in enumerate(thisCatValSet):
                groupMean = df[df[catName] == val][dataName].mean()
                otherMean = df[df[catName] != val][dataName].mean()
                meanDiff[vi,] = groupMean - otherMean

            sdf = df.copy()
            shuffleValues = np.empty((len(thisCatValSet), numShuffles))
            for si in range(numShuffles):
                sdf[catName] = sdf[catName].sample(frac=1, random_state=1).reset_index(drop=True)
                for vi, val in enumerate(thisCatValSet):
                    groupMean = sdf[sdf[catName] == val][dataName].mean()
                    otherMean = sdf[sdf[catName] != val][dataName].mean()
                    shuffleValues[vi,si] = groupMean - otherMean

            pctile = np.count_nonzero(shuffleValues < meanDiff, axis=1) / numShuffles
            # print(pctile)
            retMeanDiff = {catName: meanDiff}
            retPctile = {catName: pctile}
            return retMeanDiff, retPctile

        else:
            if valSet is None:
                valSet = {}
                for k in categories:
                    valSet[k] = sorted(list(set(categories[k])))
                print("generated valset,", valSet)
                print(yvals)
                print(categories)

            retMeanDiff = {}
            retPctile = {}
            for catName in categories:
                catVals = categories[catName]
                print("within", catName)
                catMeanDiff = {}
                catPctile = {}
                for v in valSet[catName]:
                    subYVals = yvals[catVals == v]
                    subCats = categories.copy()
                    del subCats[catName]
                    for subCatName in subCats:
                        subCats[subCatName] = subCats[subCatName][catVals == v]
                
                    meanDiff, pctile = self._doShuffle(subYVals, dataName, subCats, valSet=valSet, numShuffles=numShuffles)
                    catMeanDiff[v] = meanDiff
                    catPctile[v] = pctile

                    # print("\t", meanDiff, pctile)
                retMeanDiff[catName] = catMeanDiff
                retPctile[catName] = catPctile

            return retMeanDiff, retPctile





def setupBehaviorTracePlot(axs, sesh, showAllWells=True, showHome=True, showAways=True, zorder=2, outlineColors=None, wellSize=mpl.rcParams['lines.markersize']**2):
    if isinstance(axs, np.ndarray):
        axs = axs.flat
    elif not isinstance(axs, list):
        axs = [axs]

    if outlineColors is not None:
        if isinstance(outlineColors, np.ndarray):
            outlineColors = outlineColors.flat
        if not isinstance(outlineColors, list):
            outlineColors = [outlineColors]

        assert len(outlineColors) == len(axs)

    x1 = np.min(sesh.bt_pos_xs)
    x2 = np.max(sesh.bt_pos_xs)
    y1 = np.min(sesh.bt_pos_ys)
    y2 = np.max(sesh.bt_pos_ys)
    for axi, ax in enumerate(axs):
        if showAllWells:
            for w in allWellNames:
                wx, wy = sesh.well_coords_map[str(w)]
                ax.scatter(wx, wy, c="black", zorder=zorder, s=wellSize)
        if showAways:
            for w in sesh.visited_away_wells:
                wx, wy = sesh.well_coords_map[str(w)]
                ax.scatter(wx, wy, c="blue", zorder=zorder, s=wellSize)
        if showHome:
            wx, wy = sesh.well_coords_map[str(sesh.home_well)]
            ax.scatter(wx, wy, c="red", zorder=zorder, s=wellSize)

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.tick_params(axis="both", which="both", label1On=False,
                       label2On=False, tick1On=False, tick2On=False)

        if outlineColors is not None:
            color = outlineColors[axi]
            for v in ax.spines.values():
                v.set_color(color)
                v.set_linewidth(3)
            # ax.setp(ax.spines.values(), color=color)
            # ax.setp([ax.get_xticklines(), ax.get_yticklines()], color=color)


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
                    p1 = sns.violinplot(ax=ax, hue=axesNamesNoSpaces[0],
                                        y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, palette=pal, linewidth=0.2, cut=0, zorder=1)
                    p2 = sns.swarmplot(ax=ax, hue=axesNamesNoSpaces[0],
                                       y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, color="0.25", size=swarmDotSize, dodge=True, zorder=3)
                    # print("worked")
                    plotWorked = True
                except UserWarning as e:
                    swarmDotSize /= 2
                    p1.cla()

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


if __name__ == "__main__":
    # pp = PlotCtx("/media/WDC7/figures/test")
    pp = PlotCtx("/home/wcroughan/data/figures/test")

    # with pp.newFig("test") as axs:
    #     axs.plot(np.arange(5), np.arange(5))

    # with pp.continueFig("nothertest") as axs:
    #     axs.plot(np.arange(3, 6), np.arange(3, 6))

    # with pp.newFig("nothernothertest") as axs:
    #     axs.plot(np.arange(3, 6), np.arange(3, 6))

    for pcat in range(3):
        # pp.setStatCategory("pcat", pcat)
        with pp.newFig("testStats", withStats=True) as (ax, yvals, cats):
            yvals["v1"] = np.array([0, 0, 0, 1, 1, 1])
            # yvals["v2"] = np.linspace(0, 1, 6)
            cats["c1"] = np.array(["0", "0", "0", "1", "1", "1"])
            cats["c2"] = np.array(["0", "1", "2", "0", "1", "2"])

    pp.runShuffles()
