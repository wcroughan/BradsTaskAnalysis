import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from consts import allWellNames
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import warnings
import random
import time
from enum import IntEnum, auto


class ShufRes:
    def __init__(self, name, diff, pctile):
        self.name = name
        self.diff = diff
        self.pctile = pctile

    def __str__(self):
        return "{} ({}, {})".format(self.name, self.diff, self.pctile)

    def __repr__(self):
        return "{} ({}, {})".format(self.name, self.diff, self.pctile)

    def __lt__(self, other):
        if len(other.name) < len(self.name):
            return False
        elif len(other.name) > len(self.name):
            return True

        for i in range(len(other.name)):
            if other.name[i] == self.name[i]:
                continue
            return self.name < other.name
        return False


class ShuffSpec:
    class ShuffType(IntEnum):
        UNSPECIFIED = auto()
        GLOBAL = auto()
        WITHIN = auto()
        ACROSS = auto()
        INTERACTION = auto()

        def __repr__(self):
            return self.name

        def __str__(self):
            return self.name[0:3]

    def __init__(self, shuffType=ShuffType.UNSPECIFIED, categoryName="", value=None):
        self.shuffType = shuffType
        self.categoryName = categoryName
        self.value = value

    def __str__(self):
        return "{} {} ({})".format(self.shuffType, self.categoryName, self.value)

    def __repr__(self):
        return "{} {} ({})".format(self.shuffType, self.categoryName, self.value)

    def __lt__(self, other):
        if self.shuffType < other.shuffType:
            return True
        elif self.shuffType > other.shuffType:
            return False

        if self.categoryName < other.categoryName:
            return True
        elif self.categoryName > other.categoryName:
            return False

        if self.value is None or other.value is None:
            return False
        return self.value < other.value


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
                    print(
                        "warning: overlap between persistent category and this-plot category, both named {}".format(k))

            if statsName in self.savedYVals:
                savedYVals = self.savedYVals[statsName]
                for k in savedYVals:
                    savedYVals[k] = np.append(savedYVals[k], self.yvals[k])
                savedCategories = self.savedCategories[statsName]
                for k in savedCategories:
                    savedCategories[k] = np.append(savedCategories[k], self.categories[k])
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
        self.axs = self.fig.subplots(1, 1)
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

    def printShuffleResult(self, results):
        print("\n".join([str(v) for v in sorted(results)]))

    def runShuffles(self, numShuffles=4):
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
                if not isinstance(categories[catName], np.ndarray):
                    categories[catName] = np.array(categories[catName])

            for yvalName in yvals:
                assert yvalName not in categories

            cats = list(categories.keys())
            categories.update(yvals)
            df = pd.DataFrame(data=categories)
            specs = self.getAllShuffleSpecs(df, columnsToShuffle=cats)
            ss = [len(s) for s in specs]
            specs = [x for _, x in sorted(zip(ss, specs))]
            print("\n".join([str(s) for s in specs]))

            # for yvalName in yvals:
            #     yv = np.array(yvals[yvalName])
            #     r = self._doShuffle(yv, yvalName, categories, numShuffles=numShuffles)
            #     self.printShuffleResult(r)

    def _doShuffleOld(self, yvals, dataName, categories, valSet=None, numShuffles=4):
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
            # print(yvals, catVals)
            df = pd.DataFrame(data={dataName: yvals, catName: catVals})

            meanDiff = np.empty((len(thisCatValSet), 1))
            for vi, val in enumerate(thisCatValSet):
                groupMean = df[df[catName] == val][dataName].mean()
                otherMean = df[df[catName] != val][dataName].mean()
                meanDiff[vi, ] = groupMean - otherMean

            if numShuffles == 0:
                pctile = np.empty_like(meanDiff)
                pctile[:] = np.nan
            else:
                sdf = df.copy()
                shuffleValues = np.empty((len(thisCatValSet), numShuffles))
                for si in range(numShuffles):
                    sdf[catName] = sdf[catName].sample(
                        frac=1, random_state=1).reset_index(drop=True)
                    # note the below shuffle doesn't work because it throws a warning about possibly assigning to a copy
                    # random.shuffle(sdf[catName])
                    for vi, val in enumerate(thisCatValSet):
                        groupMean = sdf[sdf[catName] == val][dataName].mean()
                        otherMean = sdf[sdf[catName] != val][dataName].mean()
                        shuffleValues[vi, si] = groupMean - otherMean

                pctile = np.count_nonzero(shuffleValues < meanDiff, axis=1) / numShuffles

            ret = []
            for vi, val in enumerate(thisCatValSet):
                ret.append(ShufRes([("global", catName, val)], meanDiff[vi], pctile[vi]))

            # if any([np.isnan(v.diff) for v in ret]):
            #     raise Exception("found a nan")
            # print(ret)
            return ret

        else:
            if valSet is None:
                valSet = {}
                for k in categories:
                    valSet[k] = sorted(list(set(categories[k])))
                print("generated valset,", valSet)
                print(yvals)
                print(categories)

                dd = categories.copy()
                dd[dataName] = yvals
                print(pd.DataFrame(data=dd))

            recInfo = ", ".join([k for k in categories])

            ret = []
            for catName in categories:
                catVals = categories[catName]

                # print("{}: global {}".format(recInfo, catName))
                gcats = {catName: categories[catName]}
                globalRet = self._doShuffle(yvals, dataName, gcats,
                                            valSet=valSet, numShuffles=numShuffles)

                # print("{}: within {}".format(recInfo, catName))
                withinRet = []
                withoutRet = []
                for vi, val in enumerate(valSet[catName]):
                    # print("\t{}: {}={}".format(recInfo, catName, val))
                    withinIdx = catVals == val
                    # print("\twithinIDX:", withinIdx)
                    # print("\tcatVals:", catVals)
                    # print("\tval:", val)

                    subCats = categories.copy()
                    del subCats[catName]
                    for subCatName in subCats:
                        subCats[subCatName] = subCats[subCatName][withinIdx]
                    withinRet.append(self._doShuffle(
                        yvals[withinIdx], dataName, subCats, valSet=valSet, numShuffles=numShuffles))

                    subCats = categories.copy()
                    del subCats[catName]
                    for subCatName in subCats:
                        subCats[subCatName] = subCats[subCatName][np.logical_not(withinIdx)]
                    withoutRet.append(self._doShuffle(
                        yvals[np.logical_not(withinIdx)], dataName, subCats, valSet=valSet, numShuffles=0))

                interactions = []
                dprime = np.empty((len(valSet[catName]),))
                for vi, val in enumerate(valSet[catName]):
                    wrj = withinRet[vi]
                    worj = withoutRet[vi]
                    assert len(wrj) == len(worj)
                    for wi in range(len(wrj)):
                        wr = wrj[wi]
                        wor = worj[wi]
                        assert wr.name == wor.name
                        dj = wr.diff
                        dnotj = wor.diff
                        interactions.append(ShufRes(
                            [("interaction", catName, val)] + wr.name, dj - dnotj, np.nan))

                darray = np.array([v.diff for v in interactions])
                if numShuffles == 0:
                    pctiles = np.empty_like(darray)
                else:
                    # print("{}: shuffles {}".format(recInfo, catName))
                    catcopy = categories.copy()
                    catcopy[catName] = catcopy[catName].copy()
                    shufRes = np.empty((len(interactions), numShuffles))
                    for si in range(numShuffles):
                        raise Exception("unimplemented")
                        # TODO if interacting with effect in category C, value c, need to make sure
                        # the counts of values of catName are preserved within and without of c
                        # i.e. catName is condition, category C is well type, c is home well
                        # shuffle condition but ensure that there are as many SWR and delays that coincide with home
                        # well as before, otherwise will end up with 0 home well SWR data points for some shuffles

                        random.shuffle(catcopy[catName])
                        inti = 0
                        for vi, val in enumerate(valSet[catName]):
                            withinIdx = catcopy[catName] == val

                            subCats = catcopy.copy()
                            del subCats[catName]
                            for subCatName in subCats:
                                subCats[subCatName] = subCats[subCatName][withinIdx]
                            rin = self._doShuffle(
                                yvals[withinIdx], dataName, subCats, valSet=valSet, numShuffles=0)

                            subCats = catcopy.copy()
                            del subCats[catName]
                            for subCatName in subCats:
                                subCats[subCatName] = subCats[subCatName][np.logical_not(withinIdx)]
                            rout = self._doShuffle(
                                yvals[np.logical_not(withinIdx)], dataName, subCats, valSet=valSet, numShuffles=0)

                            assert len(rin) == len(rout)
                            for ri in range(len(rin)):
                                assert rin[ri].name == rout[ri].name
                                dj = rin[ri].diff
                                dnotj = rout[ri].diff
                                shufRes[inti, si] = dj - dnotj
                                inti += 1
                        assert inti == len(darray)

                    # print(darray, shufRes)
                    pctiles = np.count_nonzero(darray > shufRes,  axis=1) / numShuffles
                    # print(pctiles)

                for ii in range(len(interactions)):
                    # print(interactions)
                    interactions[ii].pctile = pctiles[ii]

                for wi in range(len(withinRet)):
                    wr = withinRet[wi]
                    for wri in range(len(wr)):
                        f = wr[wri]
                        # print(f)
                        withinRet[wi][wri].name = [
                            ("within", catName, valSet[catName][wi])] + f.name
                withinRet = [f for wr in withinRet for f in wr]

                ret += interactions + withinRet + globalRet
                # print("{} ret: {}".format(catName, "\n\t".join([str(r) for r in ret])))
            return ret

    def getAllShuffleSpecs(self, df, columnsToShuffle=None):
        if columnsToShuffle is None:
            columnsToShuffle = list(df.columns)

        valSet = {}
        for col in columnsToShuffle:
            valSet[col] = sorted(list(set(df[col])))

        ret = []
        for col in columnsToShuffle:
            for val in valSet[col]:
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName=col, value=val)])

            otherCols = [c for c in columnsToShuffle if c != col]
            rec = self.getAllShuffleSpecs(df, otherCols)
            for r in rec:
                for val in valSet[col]:
                    ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.WITHIN,
                               categoryName=col, value=val)] + r)
                    ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.INTERACTION,
                               categoryName=col, value=val)] + r)
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.ACROSS,
                           categoryName=col, value=None)] + r)

        return ret

    def _doShuffleSpec(self, df, spec):
        assert isinstance(spec, list)
        for s in spec:
            assert isinstance(s, ShuffSpec)
            assert s.shuffType != ShuffSpec.ShuffType.UNSPECIFIED
        assert spec[-1].shuffType == ShuffSpec.ShuffType.GLOBAL

    def _doShuffle(self, df, specs):
        # wait no ... think about how to optimize calls so across and withins use same data, and interactions need that too!
        for spec in specs:
            self._doShuffleSpec(df, spec)


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


def testShuffles():
    pp = PlotCtx("/home/wcroughan/data/figures/test")

    # for pcat in range(3):
    # pp.setStatCategory("pcat", pcat)
    with pp.newFig("testStats", withStats=True) as (ax, yvals, cats):
        cats["c1"] = []
        cats["c2"] = []
        cats["c3"] = []
        yvals["v"] = []
        for reps in range(2):
            for c1 in range(2):
                for c2 in range(3):
                    for c3 in range(3):
                        cats["c1"].append("asdf" if c1 == 0 else "poiu")
                        cats["c2"].append(str(c2))
                        cats["c3"].append(c3)
                        val = c1 * c2 + 4 * c1 + 0 * c3 + np.random.uniform() * 0.2
                        yvals["v"].append(val)

        print(cats, yvals)

    pp.runShuffles(numShuffles=4)
    # sizes = np.linspace(15, 500, 5).astype(int)
    # res = np.empty((len(sizes), 2))
    # res[:, 0] = sizes
    # ts = [time.perf_counter()]

    # for ns in sizes:
    #     pp.runShuffles(numShuffles=ns)
    #     ts.append(time.perf_counter())

    # res[:, 1] = np.diff(np.array(ts))
    # print(res)


if __name__ == "__main__":
    testShuffles()
