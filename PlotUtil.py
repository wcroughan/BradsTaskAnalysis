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
from enum import IntEnum, auto
import matplotlib.image as mpimg


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
        RECURSIVE_ALL = auto()  # includes interaction, across, and within

        def __repr__(self):
            return self.name

        def __str__(self):
            return self.name[0:3]

    def __init__(self, shuffType=ShuffType.UNSPECIFIED, categoryName="", value=None):
        """
        if type is WITHIN, GLOBAL, or INTERACTION, value is a single value
        if type is ACROSS or RECURSIVE_ALL, value is None
        """
        self.shuffType = shuffType
        self.categoryName = categoryName
        self.value = value

    def copy(self):
        ret = ShuffSpec()
        ret.shuffType = self.shuffType
        ret.categoryName = self.categoryName
        ret.value = self.value
        return ret

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


class ShuffleResult:
    def __init__(self, specs=[], diff=np.nan, shuffleDiffs=None, dataNames=[]):
        self.specs = specs
        self.diff = diff
        self.shuffleDiffs = shuffleDiffs
        self.dataNames = dataNames

    def copy(self):
        ret = ShuffleResult()
        ret.diff = self.diff
        if self.shuffleDiffs is not None:
            ret.shuffleDiffs = np.copy(self.shuffleDiffs)
        ret.specs = [s.copy() for s in self.specs]
        ret.dataNames = self.dataNames.copy()
        return ret

    def getFullInfoString(self, linePfx=""):
        sdmin = np.min(self.shuffleDiffs, axis=0)
        sdmax = np.max(self.shuffleDiffs, axis=0)
        pvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        pvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        ret = [f"{self.specs}:"] + [linePfx + f"{self.dataNames[i]}: {float(self.diff[i])} ({sdmin[i]}, {sdmax[i]}) "
                                    f"p1 = {pvals1[i]}\tp2 = {pvals2[i]}" for i in range(len(self.diff))]
        return "\n".join(ret)

    def __str__(self):
        if self.shuffleDiffs is None:
            return "{}: {}".format(self.specs, self.diff)
        else:
            return "{}: {} ({}, {})".format(self.specs, self.diff, np.min(self.shuffleDiffs), np.max(self.shuffleDiffs))

    def __repr__(self):
        if self.shuffleDiffs is None:
            return "{}: {}".format(self.specs, self.diff)
        else:
            return "{}: {} ({}-{})".format(self.specs, self.diff, np.min(self.shuffleDiffs), np.max(self.shuffleDiffs))


class PlotCtx:
    def __init__(self, outputDir="./", priorityLevel=None, randomSeed=None, verbosity=3, infoFileName=None):
        self.figSizeX, self.figSizeY = 5, 5
        self.fig = plt.figure(figsize=(self.figSizeX, self.figSizeY))
        self.fig.clf()
        self.axs = self.fig.subplots()
        # Save when exiting context?
        self.savePlot = True
        # Show when exiting context?
        self.showPlot = False
        self.verbosity = verbosity
        self.showedLastFig = False

        if infoFileName is None:
            self.infoFileName = datetime.now().strftime("%Y%m%d_%H%M%S_info.txt")
        else:
            self.infoFileName = infoFileName
        self.outputSubDir = ""
        self.setOutputDir(outputDir)
        self.figName = ""
        self.createdPlots = set()
        self.savedFigsByName = {}

        self.priorityLevel = priorityLevel
        self.priority = None

        self.persistentCategories = {}
        self.persistentInfoValues = {}
        self.savedYVals = {}
        self.savedCategories = {}
        self.savedInfoVals = {}
        self.numShuffles = 0

        self.rng = np.random.default_rng(seed=randomSeed)
        self.customShuffleFunctions = {}
        self.uniqueInfoValue = -1

    def __enter__(self):
        if self.withStats:
            return (self.axs, self.yvals, self.categories, self.infoVals)
        else:
            return self.axs

    def newFig(self, figName, subPlots=None, figScale=1.0, priority=None, withStats=False,
               showPlot=None, savePlot=None, enableOverwriteSameName=False):

        fname = os.path.join(self.outputDir, self.outputSubDir, figName)
        if fname in self.createdPlots and not enableOverwriteSameName:
            raise Exception("Would overwrite file {} that was just made!".format(fname))

        self.clearFig()
        self.figName = figName
        self.priority = priority
        self.withStats = withStats
        self.yvals = {}
        self.categories = {}
        self.infoVals = {}

        self.temporaryShowPlot = showPlot
        self.temporarySavePlot = savePlot

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

    def continueFig(self, figName, priority=None, showPlot=None, savePlot=None, enableOverwriteSameName=False):
        if self.showedLastFig:
            raise Exception("currently unable to show a figure and then continue it")

        self.temporaryShowPlot = showPlot
        self.temporarySavePlot = savePlot
        self.figName = figName
        self.priority = priority
        self.yvals = {}
        self.categories = {}
        self.infoVals = {}
        return self

    def setStatCategory(self, category, value):
        self.persistentCategories[category] = value

    def setStatInfo(self, infoCategory, value):
        self.persistentInfoValues[infoCategory] = value

    def setCustomShuffleFunction(self, category, func):
        self.customShuffleFunctions[category] = func

    def setPriorityLevel(self, priorityLevel):
        self.priorityLevel = priorityLevel

    def __exit__(self, *args):
        if self.priority is None or self.priorityLevel is None or self.priority <= self.priorityLevel:
            if self.withStats and len(self.yvals) > 0:
                statsName = self.figName.split("/")[-1]
                assert len(self.yvals) > 0
                assert len(self.persistentCategories) + len(self.categories) > 0

                savedLen = None
                for k in self.yvals:
                    if savedLen is None:
                        savedLen = len(self.yvals[k])
                    else:
                        assert len(self.yvals[k]) == savedLen
                for k in self.categories:
                    assert len(self.categories[k]) == savedLen
                for k in self.infoVals:
                    assert len(self.infoVals[k]) == savedLen

                for k in self.persistentCategories:
                    if k not in self.categories:
                        self.categories[k] = [self.persistentCategories[k]] * savedLen
                    else:
                        print(
                            "WARNING: overlap between persistent category and this-plot category, "
                            "both named {}".format(k))

                for k in self.persistentInfoValues:
                    if k not in self.infoVals:
                        self.infoVals[k] = [self.persistentInfoValues[k]] * savedLen
                    else:
                        print(
                            "WARNING: overlap between persistent info category and this-plot category, "
                            "both named {}".format(k))

                if statsName in self.savedYVals:
                    savedYVals = self.savedYVals[statsName]
                    for k in savedYVals:
                        savedYVals[k] = np.append(savedYVals[k], self.yvals[k])
                    savedCategories = self.savedCategories[statsName]
                    for k in savedCategories:
                        savedCategories[k] = np.append(savedCategories[k], self.categories[k])
                    savedInfoVals = self.savedInfoVals[statsName]
                    for k in savedInfoVals:
                        savedInfoVals[k] = np.append(savedInfoVals[k], self.infoVals[k])
                else:
                    self.savedYVals[statsName] = self.yvals
                    self.savedCategories[statsName] = self.categories
                    self.savedInfoVals[statsName] = self.infoVals

            if (self.temporarySavePlot is not None and self.temporarySavePlot) or \
                    (self.temporarySavePlot is None and self.savePlot):
                self.saveFig()

            if (self.temporaryShowPlot is not None and self.temporaryShowPlot) or \
                    (self.temporaryShowPlot is None and self.showPlot):
                self.showedLastFig = True
                plt.show()
                self.fig = plt.figure(figsize=(self.figSizeX, self.figSizeY))
            else:
                self.showedLastFig = False

    def setFigSize(self, width, height):
        pass

    def saveFig(self):
        if self.figName[0] != "/":
            fname = os.path.join(self.outputDir, self.outputSubDir, self.figName)
        else:
            fname = self.figName
        plt.savefig(fname, bbox_inches="tight", dpi=200)
        self.writeToInfoFile("wrote file {}".format(fname))
        if self.verbosity >= 3:
            print("wrote file {}".format(fname))
        self.createdPlots.add(fname)

        figFileName = fname.split('/')[-1]
        if figFileName in self.savedFigsByName:
            self.savedFigsByName[figFileName].append(self.outputSubDir)
        else:
            self.savedFigsByName[figFileName] = [self.outputSubDir]

    def clearFig(self):
        self.fig.clf()
        self.axs = self.fig.subplots(1, 1)
        self.axs.cla()

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.path.exists(os.path.join(outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(outputDir, self.outputSubDir))
        self.infoFileFullName = os.path.join(
            outputDir, self.infoFileName)

    def setOutputSubDir(self, outputSubDir):
        self.outputSubDir = outputSubDir
        if not os.path.exists(os.path.join(self.outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, self.outputSubDir))

    def writeToInfoFile(self, txt, suffix="\n"):
        with open(self.infoFileFullName, "a") as f:
            f.write(txt + suffix)

    def makeCombinedFigs(self, outputSubDir="combined"):
        if not os.path.exists(os.path.join(self.outputDir, outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, outputSubDir))

        for figFileName in self.savedFigsByName:
            figSubDirs = self.savedFigsByName[figFileName]
            if len(figSubDirs) <= 1:
                continue

            self.clearFig()

            self.axs.remove()
            self.axs = self.fig.subplots(1, len(figSubDirs))
            self.fig.set_figheight(self.figSizeY)
            self.fig.set_figwidth(self.figSizeX * len(figSubDirs))

            for sdi, sd in enumerate(figSubDirs):
                im = mpimg.imread(os.path.join(self.outputDir, sd, figFileName + ".png"))
                # self.axs[sdi].set_xticks([])
                # self.axs[sdi].set_yticks([])
                self.axs[sdi].axis('off')
                self.axs[sdi].set_title(sd)
                self.axs[sdi].imshow(im)

            fname = os.path.join(self.outputDir, outputSubDir, figFileName)
            plt.savefig(fname, bbox_inches="tight", dpi=200)

            self.writeToInfoFile("wrote file {}".format(fname))
            if self.verbosity >= 3:
                print("wrote file {}".format(fname))

    def printShuffleResult(self, results):
        print("\n".join([str(v) for v in sorted(results)]))

    def runShuffles(self, numShuffles=100):
        self.numShuffles = numShuffles
        for plotName in self.savedCategories:
            categories = self.savedCategories[plotName]
            yvals = self.savedYVals[plotName]
            infoVals = self.savedInfoVals[plotName]
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

            catsToShuffle = list(categories.keys())
            todel = set()
            for cat in catsToShuffle:
                vals = set(categories[cat])
                if len(vals) <= 1:
                    todel.add(cat)
                    print("Category {} has only one val ({}). Not including in shuffle for plot {}".format(
                        cat, vals, plotName))

            for td in todel:
                catsToShuffle.remove(td)

            categories.update(yvals)
            categories.update(infoVals)
            df = pd.DataFrame(data=categories)
            specs = self.getAllShuffleSpecs(df, columnsToShuffle=catsToShuffle)
            ss = [len(s) for s in specs]
            specs = [x for _, x in sorted(zip(ss, specs))]
            print("\n".join([str(s) for s in specs]))
            self._doShuffles(df, specs, list(yvals.keys()))

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

            # recInfo = ", ".join([k for k in categories])

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
                # dprime = np.empty((len(valSet[catName]),))
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
                    pctiles = np.count_nonzero(darray > shufRes, axis=1) / numShuffles
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
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL,
                           categoryName=col, value=None)] + r)

        return ret

    def _doOneWayShuffle(self, df, spec, dataNames):
        assert isinstance(spec, ShuffSpec) and \
            spec.shuffType == ShuffSpec.ShuffType.GLOBAL

        if spec.categoryName in self.customShuffleFunctions:
            shufFunc = self.customShuffleFunctions[spec.categoryName]
        else:
            def shufFunc(dataframe, colName, rng):
                return dataframe[colName].sample(frac=1, random_state=rng).reset_index(drop=True)

        ret = ShuffleResult([spec], dataNames=dataNames)

        withinIdx = df[spec.categoryName] == spec.value
        d = np.array(df[withinIdx][dataNames].mean() - df[~withinIdx][dataNames].mean())
        ret.diff = np.reshape(d, (-1, 1))

        ret.shuffleDiffs = np.empty((self.numShuffles, len(dataNames)))
        sdf = df.copy().reset_index(drop=True)
        for si in range(self.numShuffles):
            sdf[spec.categoryName] = shufFunc(sdf, spec.categoryName, self.rng)
            withinIdx = sdf[spec.categoryName] == spec.value
            ret.shuffleDiffs[si, :] = sdf[withinIdx][dataNames].mean() - \
                sdf[~withinIdx][dataNames].mean()

        return ret

    def _doShuffleSpec(self, df, spec, valSet, dataNames):
        assert isinstance(spec, list)
        for si, s in enumerate(spec):
            assert isinstance(s, ShuffSpec)
            assert s.shuffType != ShuffSpec.ShuffType.UNSPECIFIED
            assert s.shuffType != ShuffSpec.ShuffType.GLOBAL or si == len(spec) - 1

        # print(df)
        # print("\t", spec)

        s = spec[0]
        if s.shuffType == ShuffSpec.ShuffType.GLOBAL:
            return [self._doOneWayShuffle(df, s, dataNames)]

        if s.shuffType == ShuffSpec.ShuffType.WITHIN:
            withinIdx = df[s.categoryName] == s.value
            recres = self._doShuffleSpec(df.loc[withinIdx], spec[1:], valSet, dataNames)
            for r in recres:
                r.specs = [s] + r.specs.copy()
            return recres

        if s.shuffType == ShuffSpec.ShuffType.INTERACTION:
            withinIdx = df[s.categoryName] == s.value
            withoutIdx = ~ withinIdx
            rin = self._doShuffleSpec(df.loc[withinIdx], spec[1:], valSet, dataNames)
            rout = self._doShuffleSpec(df.loc[withoutIdx], spec[1:], valSet, dataNames)
            ret = []
            for reti, (i, o) in enumerate(zip(rin, rout)):
                r = i.copy()
                r.specs = [s] + i.specs.copy()
                r.diff = i.diff - o.diff
                r.shuffleDiffs = i.shuffleDiffs - o.shuffleDiffs
                ret.append(r)
            return ret

        if s.shuffType == ShuffSpec.ShuffType.ACROSS:
            vals = valSet[s.categoryName]
            withinRes = []
            for val in vals:
                withinIdx = df[s.categoryName] == val
                withinRes.append(self._doShuffleSpec(
                    df.loc[withinIdx], spec[1:], valSet, dataNames))

            numEffects = len(withinRes[0])
            ret = []
            for ei in range(numEffects):
                r = withinRes[0][ei].copy()
                r.specs = [s] + r.specs.copy()
                r.diff = np.zeros((len(dataNames), 1))
                r.shuffleDiffs = np.zeros((self.numShuffles, len(dataNames)))
                for vi in range(len(vals)):
                    r.diff += withinRes[vi][ei].diff
                    r.shuffleDiffs += withinRes[vi][ei].shuffleDiffs
                ret.append(r)
            return ret

        if s.shuffType == ShuffSpec.ShuffType.RECURSIVE_ALL:
            # all the recursive calls needed
            vals = valSet[s.categoryName]
            withinRes = []
            withoutRes = []
            for val in vals:
                withinIdx = df[s.categoryName] == val
                withinRes.append(self._doShuffleSpec(
                    df.loc[withinIdx], spec[1:], valSet, dataNames))
                withoutIdx = ~ withinIdx
                withoutRes.append(self._doShuffleSpec(
                    df.loc[withoutIdx], spec[1:], valSet, dataNames))

            ret = []
            # within effects
            for vi, val in enumerate(vals):
                wr = withinRes[vi]
                for ei, effect in enumerate(wr):
                    r = effect.copy()
                    ss = s.copy()
                    ss.shuffType = ShuffSpec.ShuffType.WITHIN
                    ss.value = val
                    r.specs = [ss] + r.specs.copy()
                    ret.append(r)

            # across effect
            numEffects = len(withinRes[0])
            for ei in range(numEffects):
                r = withinRes[0][ei].copy()
                ss = s.copy()
                ss.shuffType = ShuffSpec.ShuffType.ACROSS
                r.specs = [ss] + r.specs.copy()
                r.diff = np.zeros((len(dataNames), 1))
                r.shuffleDiffs = np.zeros((self.numShuffles, len(dataNames)))
                for vi in range(len(vals)):
                    r.diff += withinRes[vi][ei].diff
                    r.shuffleDiffs += withinRes[vi][ei].shuffleDiffs
                ret.append(r)

            # interaction effects
            for vi, val in enumerate(vals):
                wr = withinRes[vi]
                wor = withoutRes[vi]
                for ei, (rin, rout) in enumerate(zip(wr, wor)):
                    ss = s.copy()
                    ss.shuffType = ShuffSpec.ShuffType.INTERACTION
                    ss.value = val
                    r = ShuffleResult([ss] + rin.specs.copy())
                    r.dataNames = dataNames
                    r.diff = rin.diff - rout.diff
                    r.shuffleDiffs = rin.shuffleDiffs - rout.shuffleDiffs
                    ret.append(r)

            return ret

    def _doShuffles(self, df, specs, dataNames):
        columnsToShuffle = set()
        for spec in specs:
            for s in spec:
                columnsToShuffle.add(s.categoryName)

        print(df)
        valSet = {}
        for col in columnsToShuffle:
            vs = set(df[col])
            if len(vs) <= 1:
                raise Exception("Only one value {} for category {}".format(vs, col))
            valSet[col] = vs
        print("created valset:", valSet)

        with open(self.infoFileFullName, "a") as f:
            for spec in specs:
                print(f'\r{spec}                 ', end='')
                res = self._doShuffleSpec(df, spec, valSet, dataNames)
                for r in res:
                    # print(r.getFullInfoString(linePfx="\t"))
                    f.write(r.getFullInfoString(linePfx="\t") + "\n")
        print("")

    def getUniqueInfoValue(self):
        self.uniqueInfoValue += 1
        return self.uniqueInfoValue


def conditionShuffle(dataframe, colName, rng):
    def swapFunc(val): return "SWR" if val == "Ctrl" else "Ctrl"
    numGroups = dataframe["conditionGroup"].max() + 1
    swapBool = (np.random.uniform(size=(numGroups,)) < 0.5).astype(bool)
    return [swapFunc(val) if swapBool[cg] else val for cg, val in zip(dataframe["conditionGroup"], dataframe[colName])]


def setupBehaviorTracePlot(axs, sesh, showAllWells=True, showHome=True, showAways=True, zorder=2, outlineColors=None,
                           wellSize=mpl.rcParams['lines.markersize']**2):
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


def plotIndividualAndAverage(ax, dataPoints, xvals, individualColor="grey", avgColor="blue", spread="std",
                             individualZOrder=1, averageZOrder=2):
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


def boxPlot(ax, yvals, categories, categories2=None, dotColors=None, axesNames=None, violin=False):
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
        elif len(set(categories2)) == 2 and "same" in categories2 and "other" in categories2:
            sortingCategories2 = ["aaasame" if c == "same" else "other" for c in categories2]
        else:
            sortingCategories2 = categories2

    if dotColors is None:
        swarmPallete = sns.color_palette(palette=["black"])
        dotColors = np.array([0.0 for _ in yvals])
    else:
        nColors = len(np.unique(dotColors))
        swarmPallete = sns.color_palette("coolwarm", n_colors=nColors)

    # Same sorting here as in perseveration plot function so colors are always the same
    sortList = ["{}__{}__{}".format(x, y, xi)
                for xi, (x, y) in enumerate(zip(categories, sortingCategories2))]
    categories = [x for _, x in sorted(zip(sortList, categories))]
    yvals = [x for _, x in sorted(zip(sortList, yvals))]
    categories2 = [x for _, x in sorted(zip(sortList, categories2))]
    dotColors = [x for _, x in sorted(zip(sortList, dotColors))]

    if axesNames is None:
        hideAxes = True
        axesNames = ["X", "Y", "Z"]
    else:
        hideAxes = False

    axesNamesNoSpaces = [a.replace(" ", "_") for a in axesNames]
    axesNamesNoSpaces.append("dotcoloraxisname")
    s = pd.Series([categories, yvals, categories2, dotColors], index=axesNamesNoSpaces)

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
                    # p1 = sns.violinplot(ax=ax, hue=axesNamesNoSpaces[0],
                    # y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, palette=pal, linewidth=0.2, cut=0,
                    # zorder=1)
                    # sns.swarmplot(ax=ax, hue=axesNamesNoSpaces[0],
                    #   y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, color="0.25", size=swarmDotSize,
                    # dodge=True, zorder=3)

                    sx = np.array(s[axesNamesNoSpaces[2]])
                    sy = np.array(s[axesNamesNoSpaces[1]]).astype(float)
                    sh = np.array(s[axesNamesNoSpaces[0]])
                    swarmx = np.array([a + b for a, b in zip(sx, sh)])
                    # print(swarmx)
                    # print(sx)
                    # print(sy)
                    # print(sy.dtype)

                    p1 = sns.violinplot(ax=ax, hue=sh,
                                        y=sy, x=swarmx, data=s, palette=pal, linewidth=0.2, cut=0, zorder=1)
                    sns.swarmplot(ax=ax, x=swarmx, y=sy, hue=dotColors, data=s,
                                  size=swarmDotSize, zorder=3, dodge=False, palette=swarmPallete)
                    # print("worked")
                    plotWorked = True
                except UserWarning:
                    swarmDotSize /= 2
                    p1.cla()

                    # if swarmDotSize < 0.1:
                    #     raise e
    else:
        p1 = sns.boxplot(
            ax=ax, hue=axesNamesNoSpaces[0], y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s,
            palette=pal, zorder=1)
        sns.swarmplot(ax=ax, hue=axesNamesNoSpaces[0],
                      y=axesNamesNoSpaces[1], x=axesNamesNoSpaces[2], data=s, color=s[axesNamesNoSpaces[3]],
                      dodge=True, zorder=3)

    if cat2IsFake:
        p1.set(xticklabels=[])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], fontsize=6).set_zorder(2)

    if not hideAxes:
        if not cat2IsFake:
            ax.set_xlabel(axesNames[2])
        ax.set_ylabel(axesNames[1])


def testShuffles():
    pp = PlotCtx("/home/wcroughan/data/figures/test")

    # for pcat in range(3):
    # pp.setStatCategory("pcat", pcat)
    with pp.newFig("testStats", withStats=True) as (ax, yvals, cats, infoVals):
        cats["c1"] = []
        cats["c2"] = []
        cats["c3"] = []
        yvals["v1"] = []
        yvals["v2"] = []
        infoVals["i1"] = []
        infoVals["i2"] = []
        for reps in range(2):
            for c1 in range(2):
                for c2 in range(3):
                    for c3 in range(3):
                        cats["c1"].append("asdf" if c1 == 0 else "poiu")
                        cats["c2"].append(str(c2))
                        cats["c3"].append(c3)
                        val = c1 * c2 + 4 * c1 + 0 * c3 + np.random.uniform() * 0.2
                        yvals["v1"].append(val)
                        yvals["v2"].append(np.random.uniform())
                        infoVals["i1"].append(str(np.random.uniform()))
                        infoVals["i2"].append(np.random.uniform())

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


def testIndividualShuffleSpecs():
    pp = PlotCtx("/home/wcroughan/data/figures/test")
    pp.numShuffles = 5

    df = pd.DataFrame(data={"c1": [0, 0, 0, 0, 1, 1, 1, 1],
                            "c2": [10, 10, 20, 20, 10, 10, 20, 20],
                            "y": np.linspace(0, 1, 8)})

    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    print("===============================\n")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.WITHIN, categoryName="c1", value=1),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=10), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.WITHIN, categoryName="c2", value=10),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=1), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.WITHIN, categoryName="c1", value=0),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.ACROSS, categoryName="c1", value=None),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.INTERACTION, categoryName="c1", value=0),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL, categoryName="c1", value=None),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")


def testAcross():
    pp = PlotCtx("/home/wcroughan/data/figures/test")
    pp.numShuffles = 500
    c1 = np.array([0] * 8 + [1] * 8)
    c2 = np.array(([10] * 2 + [20] * 2) * 4)
    y = c1 + 50 * c2 + np.random.uniform() * 0.2
    df = pd.DataFrame(data={"c1": c1, "c2": c2, "y": y})
    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    print("testing where c1 has small effect, c2 has big effect")
    print("===============================\n")
    print("First just global c1")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=0)]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))

    print("===============================\n")
    print("global c2")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=10)]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))

    print("===============================\n")
    print("Now across c2, global c1")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.ACROSS, categoryName="c2", value=None),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=0), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    r = res[0]
    print("p =", np.count_nonzero(r.shuffleDiffs <= r.diff) / len(r.shuffleDiffs))


def testCustomShuffleFunction():
    pp = PlotCtx("/home/wcroughan/data/figures/test", randomSeed=1)
    pp.numShuffles = 5
    c1 = np.array([0] * 8 + [1] * 8)
    c2 = np.array(([10] * 2 + [20] * 2) * 4)
    y = c1 + 50 * c2 + np.random.uniform() * 0.2
    info = (np.random.uniform(size=y.shape) > 0.5).astype(int)
    df = pd.DataFrame(data={"c1": c1, "c2": c2, "y": y, "i": info})
    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    def shufFunc(dataframe, colName, rng):
        # r = dataframe["c1"].sample(frac=1, random_state=rng).reset_index(drop=True)
        r = dataframe["i"]
        print(r)
        return r
    # def shufFunc(df, rng):
    #     print("hi")
    #     return [0] * len(c1)
    pp.customShuffleFunctions["c1"] = shufFunc

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=0)]
    res = pp._doShuffleSpec(df, spec, valSet, ["y"])
    for r in res:
        print(r.getFullInfoString(linePfx="\t"))


def testConditionShuffle():
    pp = PlotCtx("/home/wcroughan/data/figures/test", randomSeed=1)
    pp.numShuffles = 5
    conditionGroup = np.array([0] * 8 + [1] * 8)
    condition = ["SWR", "Ctrl"] * 8
    y = np.linspace(0, 1, len(condition))
    df = pd.DataFrame(data={"conditionGroup": conditionGroup, "condition": condition, "y": y})
    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    pp.customShuffleFunctions["condition"] = conditionShuffle

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")]
    res = pp._doShuffleSpec(df, spec, valSet, ["y"])
    for r in res:
        print(r.getFullInfoString(linePfx="\t"))


if __name__ == "__main__":
    testConditionShuffle()
