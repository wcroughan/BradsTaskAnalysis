from __future__ import annotations
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
from matplotlib.offsetbox import AnchoredText
import pickle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from UtilFunctions import getWellPosCoordinates
from dataclasses import dataclass, field
from typing import Optional, Tuple
from numpy.typing import ArrayLike


class ShuffSpec:
    """
    Specifies one shuffle to run
    if type is WITHIN, GLOBAL, or INTERACTION, value is a single value
    if type is ACROSS or RECURSIVE_ALL, value is None
    """
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

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ShuffSpec):
            return False
        return self.shuffType == __o.shuffType and self.categoryName == __o.categoryName and \
            (self.value is None) == (__o.value is None) and (
                self.value is None or self.value == __o.value)

    def __ne__(self, __o: object) -> bool:
        return not self == __o


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

    def getPVals(self):
        pvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        pvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        return (pvals1 + pvals2) / 2

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


@dataclass
class PlotContext:
    figName: str
    axs: ArrayLike[Axes]
    yvals: dict = field(default_factory=dict)
    immediateShuffles: list = field(default_factory=list)
    categories: dict = field(default_factory=dict)
    infoVals: dict = field(default_factory=dict)
    showPlot: Optional[bool] = None
    savePlot: Optional[bool] = None

    @property
    def ax(self) -> Axes:
        return self.axs.flat[0]

    @property
    def plotShape(self) -> Tuple[int, int]:
        s = self.axs.shape
        if len(s) == 2:
            return self.axs.shape
        else:
            return (1, s[0])

    @property
    def isSinglePlot(self) -> bool:
        return self.axs.size == 1

    @property
    def figure(self) -> Figure:
        return self.ax.figure


class PlotManager:
    def __init__(self, outputDir="./", priorityLevel=None, randomSeed=None, verbosity=3, infoFileName=None):
        self.figSizeX, self.figSizeY = 5, 5
        self.fig = plt.figure(figsize=(self.figSizeX, self.figSizeY))
        self.fig.clf()
        self.axs = self.fig.subplots()
        if isinstance(self.axs, Axes):
            self.axs = np.array([self.axs])
        # Save when exiting context?
        self.defaultSavePlot = True
        # Show when exiting context?
        self.defaultShowPlot = False
        self.verbosity = verbosity
        self.showedLastFig = False

        if infoFileName is None:
            self.infoFileName = datetime.now().strftime("%Y%m%d_%H%M%S_info.txt")
        else:
            self.infoFileName = infoFileName
        self.outputSubDir = ""
        self.outputSubDirStack = []
        self.setOutputDir(outputDir)
        self.createdPlots = set()
        self.savedFigsByName = {}

        self.priorityLevel = priorityLevel
        self.priority = None

        self.persistentCategories = {}
        self.persistentInfoValues = {}
        self.savedYVals = {}
        self.savedCategories = {}
        self.savedPersistentCategories = {}
        self.savedInfoVals = {}
        self.savedImmediateShuffles = {}
        self.numShuffles = 0

        self.rng = np.random.default_rng(seed=randomSeed)
        self.customShuffleFunctions = {}

        self.plotContext = None

    def __enter__(self) -> PlotContext:
        return self.plotContext

    def newFig(self, figName, subPlots=None, figScale=1.0, priority=None,
               showPlot=None, savePlot=None, enableOverwriteSameName=False) -> PlotManager:
        # print(self.savedPersistentCategories)
        if figName[0] != "/":
            fname = os.path.join(self.outputDir, self.outputSubDir, figName)
        else:
            fname = self.plotContext.figName
        if fname in self.createdPlots and not enableOverwriteSameName:
            raise Exception("Would overwrite file {} that was just made!".format(fname))

        self.clearFig()
        self.priority = priority

        if subPlots is not None:
            assert len(subPlots) == 2
            for a in self.axs:
                a.remove()
            self.axs = self.fig.subplots(*subPlots)
            if isinstance(self.axs, Axes):
                self.axs = np.array([self.axs])

            self.fig.set_figheight(self.figSizeY * figScale * subPlots[0])
            self.fig.set_figwidth(self.figSizeX * figScale * subPlots[1])
        else:
            self.fig.set_figheight(self.figSizeY * figScale)
            self.fig.set_figwidth(self.figSizeX * figScale)

        self.plotContext = PlotContext(fname, self.axs, showPlot=showPlot, savePlot=savePlot)

        # print(self.savedPersistentCategories)
        return self

    def continueFig(self, figName, priority=None, showPlot=None, savePlot=None, enableOverwriteSameName=False):
        if self.showedLastFig:
            raise Exception("currently unable to show a figure and then continue it")

        self.plotContext.showPlot = showPlot
        self.plotContext.savePlot = savePlot
        if figName[0] != "/":
            fname = os.path.join(self.outputDir, self.outputSubDir, figName)
        else:
            fname = self.plotContext.figName
        if fname in self.createdPlots and not enableOverwriteSameName:
            raise Exception("Would overwrite file {} that was just made!".format(fname))
        self.plotContext.figName = fname
        self.priority = priority
        self.plotContext.yvals = {}
        self.plotContext.immediateShuffles = []
        self.plotContext.categories = {}
        self.plotContext.infoVals = {}
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
        if self.priority is not None and self.priorityLevel is not None and self.priority > self.priorityLevel:
            return

        if len(self.plotContext.yvals) > 0:
            # print(self.savedPersistentCategories)
            statsName = self.plotContext.figName.split("/")[-1]
            assert len(self.plotContext.yvals) > 0
            assert len(self.persistentCategories) + len(self.plotContext.categories) > 0

            savedLen = None
            for k in self.plotContext.yvals:
                if savedLen is None:
                    savedLen = len(self.plotContext.yvals[k])
                else:
                    assert len(self.plotContext.yvals[k]) == savedLen
            for k in self.plotContext.categories:
                assert len(self.plotContext.categories[k]) == savedLen
            for k in self.plotContext.infoVals:
                assert len(self.plotContext.infoVals[k]) == savedLen

            for k in self.persistentCategories:
                if k not in self.plotContext.categories:
                    # self.plotContext.categories[k] = [self.persistentCategories[k]] * savedLen
                    # Now gonna save these separately
                    pass
                else:
                    print(
                        "WARNING: overlap between persistent category and this-plot category, "
                        "both named {}".format(k))

            for k in self.persistentInfoValues:
                if k not in self.plotContext.infoVals:
                    self.plotContext.infoVals[k] = [self.persistentInfoValues[k]] * savedLen
                else:
                    print(
                        "WARNING: overlap between persistent info category and this-plot category, "
                        "both named {}".format(k))

            if statsName in self.savedYVals:
                savedYVals = self.savedYVals[statsName]
                for k in savedYVals:
                    savedYVals[k] = np.append(savedYVals[k], self.plotContext.yvals[k])
                savedCategories = self.savedCategories[statsName]
                for k in savedCategories:
                    savedCategories[k] = np.append(
                        savedCategories[k], self.plotContext.categories[k])
                savedPersistentCategories = self.savedPersistentCategories[statsName]
                for k in savedPersistentCategories:
                    savedPersistentCategories[k] = np.append(savedPersistentCategories[k], [
                                                             self.persistentCategories[k]] * savedLen)
                savedInfoVals = self.savedInfoVals[statsName]
                for k in savedInfoVals:
                    savedInfoVals[k] = np.append(savedInfoVals[k], self.plotContext.infoVals[k])
                savedImmediateShuffles = self.savedImmediateShuffles[statsName]
                if savedImmediateShuffles is not None:
                    print(savedImmediateShuffles)
                    print(self.plotContext.immediateShuffles)
                    if len(self.plotContext.immediateShuffles) == 0 or savedImmediateShuffles != self.plotContext.immediateShuffles[0][0]:
                        raise Exception("Repeated figure must have identical immediate shuffle")
                elif len(self.plotContext.immediateShuffles) != 0:
                    raise Exception("Repeated figure must have identical immediate shuffle")
            else:
                self.savedYVals[statsName] = self.plotContext.yvals.copy()
                self.savedCategories[statsName] = self.plotContext.categories
                self.savedPersistentCategories[statsName] = self.persistentCategories.copy()
                for k in self.savedPersistentCategories[statsName]:
                    self.savedPersistentCategories[statsName][k] = [
                        self.savedPersistentCategories[statsName][k]] * savedLen
                self.savedInfoVals[statsName] = self.plotContext.infoVals
                self.savedImmediateShuffles[statsName] = None if len(
                    self.plotContext.immediateShuffles) == 0 else self.plotContext.immediateShuffles[0][0].copy()

            if len(self.plotContext.immediateShuffles) > 0:
                ss = [len(s) for s, _ in self.plotContext.immediateShuffles]
                self.plotContext.immediateShuffles = [x for _, x in sorted(
                    zip(ss, self.plotContext.immediateShuffles))]
                shufSpecs = [x for x, _ in self.plotContext.immediateShuffles]
                numShuffles = [x for _, x in self.plotContext.immediateShuffles]
                shufCats = self.plotContext.categories.copy()
                shufCats.update(self.plotContext.yvals)
                shufCats.update(self.plotContext.infoVals)
                df = pd.DataFrame(data=shufCats)
                # print("\n".join([str(s) for s in shufSpecs]))
                immediateRes = self._doShuffles(df, shufSpecs, list(
                    self.plotContext.yvals.keys()), numShuffles=numShuffles)
                # print(immediateRes)
                pval = None
                for rr in immediateRes:
                    # print(rr)
                    for r in rr:
                        # print(r.getFullInfoString(linePfx="\t"))
                        if pval is not None:
                            raise Exception("Should only be doing one shuffle here!")
                        pval = r.getPVals()[0]

                if not self.plotContext.isSinglePlot:
                    raise Exception("Can't have multiple plots and do shuffle")

                self.plotContext.ax.add_artist(AnchoredText(f"p={pval}", "upper center"))

        if (self.plotContext.savePlot is not None and self.plotContext.savePlot) or \
                (self.plotContext.savePlot is None and self.defaultSavePlot):
            self.saveFig()

        if (self.plotContext.showPlot is not None and self.plotContext.showPlot) or \
                (self.plotContext.showPlot is None and self.defaultShowPlot):
            self.showedLastFig = True
            plt.show()
            self.fig = plt.figure(figsize=(self.figSizeX, self.figSizeY))
        else:
            self.showedLastFig = False

        # print(self.savedPersistentCategories)

    def saveFig(self):
        if self.plotContext.figName[0] != "/":
            fname = os.path.join(self.outputDir, self.outputSubDir, self.plotContext.figName)
        else:
            fname = self.plotContext.figName

        if fname[-4:] != ".png":
            fname += ".png"

        if self.plotContext.isSinglePlot:
            for _ in range(3):
                try:
                    with open(fname + ".pkl", "wb") as fid:
                        pickle.dump(self.plotContext.ax, fid)
                    break
                except RuntimeError:
                    pass

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
        self.axs = np.array([self.fig.subplots(1, 1)])
        self.axs[0].cla()

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.path.exists(os.path.join(outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(outputDir, self.outputSubDir))
        self.infoFileFullName = os.path.join(
            outputDir, self.infoFileName)

    def setOutputSubDir(self, outputSubDir):
        self.outputSubDirStack = [outputSubDir]
        self.outputSubDir = outputSubDir
        if not os.path.exists(os.path.join(self.outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, self.outputSubDir))

    def pushOutputSubDir(self, outputSubDir):
        self.outputSubDirStack.append(outputSubDir)
        self.outputSubDir = os.path.join(*self.outputSubDirStack)
        if not os.path.exists(os.path.join(self.outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, self.outputSubDir))

    def popOutputSubDir(self):
        self.outputSubDirStack.pop()
        if len(self.outputSubDirStack) == 0:
            self.outputSubDir = ""
        else:
            self.outputSubDir = os.path.join(*self.outputSubDirStack)
        if not os.path.exists(os.path.join(self.outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, self.outputSubDir))

    def writeToInfoFile(self, txt, suffix="\n"):
        with open(self.infoFileFullName, "a") as f:
            f.write(txt + suffix)

    def makeCombinedFigs(self, outputSubDir="combined", suggestedSubPlotLayout=None):
        if not os.path.exists(os.path.join(self.outputDir, outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, outputSubDir))

        if suggestedSubPlotLayout is not None:
            assert len(suggestedSubPlotLayout) == 2

        for figFileName in self.savedFigsByName:
            figSubDirs = self.savedFigsByName[figFileName]
            if len(figSubDirs) <= 1:
                continue

            if suggestedSubPlotLayout is not None and \
                    len(figSubDirs) == suggestedSubPlotLayout[0] * suggestedSubPlotLayout[1]:
                subPlotLayout = suggestedSubPlotLayout
            else:
                subPlotLayout = (1, len(figSubDirs))

            self.clearFig()
            self.axs[0].remove()
            self.axs = self.fig.subplots(subPlotLayout[0], subPlotLayout[1])
            self.fig.set_figheight(self.figSizeY * subPlotLayout[0])
            self.fig.set_figwidth(self.figSizeX * subPlotLayout[1])

            for sdi, sd in enumerate(figSubDirs):
                im = mpimg.imread(os.path.join(self.outputDir, sd, figFileName + ".png"))
                # self.axs[sdi].set_xticks([])
                # self.axs[sdi].set_yticks([])
                if len(self.axs.shape) > 1:
                    axc = sdi % subPlotLayout[1]
                    ayc = sdi // subPlotLayout[1]
                    self.axs[ayc, axc].axis('off')
                    self.axs[ayc, axc].set_title(sd)
                    self.axs[ayc, axc].imshow(im)
                else:
                    self.axs[sdi].axis('off')
                    self.axs[sdi].set_title(sd)
                    self.axs[sdi].imshow(im)

            fname = os.path.join(self.outputDir, outputSubDir, figFileName)
            plt.savefig(fname, bbox_inches="tight", dpi=200)

            self.writeToInfoFile("wrote file {}".format(fname))
            if self.verbosity >= 3:
                print("wrote file {}".format(fname))

            firstPickleFName = os.path.join(self.outputDir, figSubDirs[0], figFileName + ".pkl")
            if not os.path.exists(firstPickleFName):
                continue

            self.clearFig()
            self.axs[0].remove()
            self.axs = self.fig.subplots(subPlotLayout[0], subPlotLayout[1])
            self.fig.set_figheight(self.figSizeY * subPlotLayout[0])
            self.fig.set_figwidth(self.figSizeX * subPlotLayout[1])

            ymin = np.inf
            ymax = -np.inf

            loaded_axs = []
            for sdi, sd in enumerate(figSubDirs):
                pickleFName = os.path.join(self.outputDir, sd, figFileName + ".pkl")
                with open(pickleFName, "rb") as fid:
                    ax = pickle.load(fid)
                    ax.set_title(sd)
                    loaded_axs.append(ax)
                y1, y2 = ax.get_ylim()
                if y1 < ymin:
                    ymin = y1
                if y2 > ymax:
                    ymax = y2

            # print(ymin, ymax)
            ax_xbuf = 0.2
            ax_xsz = 1.7
            ax_ysz = 1.7
            ax_xstep = 2
            ax_ystep = 2
            ax_ybuf = 0.1
            for ai, ax in enumerate(loaded_axs):
                axc = ai % subPlotLayout[1]
                ayc = ai // subPlotLayout[1]
                assert isinstance(ax, Axes)
                ax.figure = self.fig
                self.fig.add_axes(ax)
                ax.set_ylim(ymin, ymax)
                if len(self.axs.shape) > 1:
                    self.axs[ayc, axc].remove()
                else:
                    self.axs[ai].remove()

                p = (axc * ax_xstep + ax_xbuf, ax_ybuf +
                     (subPlotLayout[0] - ayc - 1) * ax_ystep,  ax_xsz, ax_ysz)
                # print(ax.get_title(), p)
                ax.set_position(p)
                if axc > 0:
                    # ax.tick_params(axis="y", which="both", label1On=False,
                    #                label2On=False)
                    ax.set_ylabel("")
                if ayc < subPlotLayout[1] - 1:
                    ax.set_xlabel("")

            plt.figure(self.fig)

            fname = os.path.join(self.outputDir, outputSubDir, figFileName + "_aligned")
            plt.savefig(fname, bbox_inches="tight", dpi=200)

            self.writeToInfoFile("wrote file {}".format(fname))
            if self.verbosity >= 3:
                print("wrote file {}".format(fname))

    def printShuffleResult(self, results):
        print("\n".join([str(v) for v in sorted(results)]))

    def runImmidateShufflesAcrossPersistentCategories(self, numShuffles=100):
        self.numShuffles = numShuffles
        for plotName in self.savedCategories:
            categories = self.savedCategories[plotName]
            persistentCategories = self.savedPersistentCategories[plotName]
            yvals = self.savedYVals[plotName]
            infoVals = self.savedInfoVals[plotName]
            immediateShuffles = self.savedImmediateShuffles[plotName]
            if immediateShuffles is None:
                print(f"No immediate shuffles for {plotName}, skipping")
                continue
            print("========================================\nRunning earlier shuffles from "
                  f"{plotName} across persistent categories")
            print("Evaluating {} yvals:".format(len(yvals)))
            for yvalName in yvals:
                print("\t{}".format(yvalName))
            print(f"Original spec: {immediateShuffles}")
            print("Shuffling {} categories:".format(len(persistentCategories)))
            for catName in persistentCategories:
                print("\t{}\t{}".format(catName, set(persistentCategories[catName])))
                if not isinstance(persistentCategories[catName], np.ndarray):
                    persistentCategories[catName] = np.array(persistentCategories[catName])

            for yvalName in yvals:
                assert yvalName not in categories
                assert yvalName not in persistentCategories

            catsToShuffle = list(persistentCategories.keys())
            todel = set()
            for cat in catsToShuffle:
                vals = set(persistentCategories[cat])
                if len(vals) <= 1:
                    todel.add(cat)
                    print("Category {} has only one val ({}). Not including in shuffle for plot {}".format(
                        cat, vals, plotName))

            for td in todel:
                catsToShuffle.remove(td)

            categories.update(yvals)
            categories.update(infoVals)
            categories.update(persistentCategories)
            df = pd.DataFrame(data=categories)
            specs = self.getAllShuffleSpecsWithLeaf(
                df, leaf=immediateShuffles, columnsToShuffle=catsToShuffle)
            ss = [len(s) for s in specs]
            specs = [x for _, x in sorted(zip(ss, specs))]
            print("\n".join([str(s) for s in specs]))
            self._doShuffles(df, specs, list(yvals.keys()))
            # print(sr)

    def runShuffles(self, numShuffles=100):
        self.numShuffles = numShuffles
        for plotName in self.savedCategories:
            categories = self.savedCategories[plotName]
            persistentCategories = self.savedPersistentCategories[plotName]
            categories.update(persistentCategories)
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

    def getAllShuffleSpecsWithLeaf(self, df: pd.DataFrame, leaf: ShuffSpec, columnsToShuffle=None):
        if columnsToShuffle is None:
            columnsToShuffle = set(df.columns)
            columnsToShuffle.remove(leaf.categoryName)
            columnsToShuffle = list(columnsToShuffle)

        ret = [leaf]
        for col in columnsToShuffle:
            otherCols = [c for c in columnsToShuffle if c != col]
            rec = self.getAllShuffleSpecsWithLeaf(df, leaf, otherCols)
            for r in rec:
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL,
                                      categoryName=col, value=None)] + r)

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

    def _doShuffles(self, df, specs, dataNames, numShuffles=None):
        if numShuffles is not None:
            assert len(numShuffles) == len(specs)

        columnsToShuffle = set()
        for spec in specs:
            for s in spec:
                columnsToShuffle.add(s.categoryName)

        # print(df)
        valSet = {}
        for col in columnsToShuffle:
            vs = set(df[col])
            if len(vs) <= 1:
                raise Exception("Only one value {} for category {}".format(vs, col))
            valSet[col] = vs
        # print("created valset:", valSet)

        ret = []
        with open(self.infoFileFullName, "a") as f:
            for si, spec in enumerate(specs):
                # print(f'\r{spec}                 ', end='')
                if numShuffles is not None:
                    self.numShuffles = numShuffles[si]
                res = self._doShuffleSpec(df, spec, valSet, dataNames)
                ret.append(res)
                for r in res:
                    # print(r.getFullInfoString(linePfx="\t"))
                    f.write(r.getFullInfoString(linePfx="\t") + "\n")
        # print("")
        return ret


def setupBehaviorTracePlot(axs, sesh, showWells: str = "HAO", wellZOrder=2, outlineColors=None,
                           wellSize=mpl.rcParams['lines.markersize']**2, extent=None):
    # showWells, string in any order containing or not contianing these letters as flags:
    #   H: home
    #   A: aways
    #   O: others
    #
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

    for axi, ax in enumerate(axs):
        if "O" in showWells:
            for w in allWellNames:
                wx, wy = getWellPosCoordinates(w)
                ax.scatter(wx, wy, c="black", zorder=wellZOrder, s=wellSize)
        if "A" in showWells:
            for w in sesh.visitedAwayWells:
                wx, wy = getWellPosCoordinates(w)
                ax.scatter(wx, wy, c="blue", zorder=wellZOrder, s=wellSize)
        if "H" in showWells:
            wx, wy = getWellPosCoordinates(sesh.homeWell)
            ax.scatter(wx, wy, c="red", zorder=wellZOrder, s=wellSize)

        if extent is None:
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.tick_params(axis="both", which="both", label1On=False,
                       label2On=False, tick1On=False, tick2On=False)
        ax.set_axisbelow(True)

        if outlineColors is not None:
            color = outlineColors[axi]
            for v in ax.spines.values():
                v.set_color(color)
                v.set_linewidth(3)


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
        return 3
    return 4


def violinPlot(ax, yvals, categories, categories2=None, dotColors=None, axesNames=None):
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
    sx = np.array(s[axesNamesNoSpaces[2]])
    sy = np.array(s[axesNamesNoSpaces[1]]).astype(float)
    sh = np.array(s[axesNamesNoSpaces[0]])
    swarmx = np.array([a + b for a, b in zip(sx, sh)])

    p1 = sns.violinplot(ax=ax, hue=sh,
                        y=sy, x=swarmx, data=s, palette=pal, linewidth=0.2, cut=0, zorder=1)
    sns.stripplot(ax=ax, x=swarmx, y=sy, hue=dotColors, data=s,
                  zorder=3, dodge=False, palette=swarmPallete)

    if cat2IsFake:
        p1.set(xticklabels=[])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], fontsize=6).set_zorder(2)

    if not hideAxes:
        if not cat2IsFake:
            ax.set_xlabel(axesNames[2])
        ax.set_ylabel(axesNames[1])
