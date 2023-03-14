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
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnchoredText
import pickle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from UtilFunctions import getWellPosCoordinates, getPreferredCategoryOrder
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable
from numpy.typing import ArrayLike
import multiprocessing
from Shuffler import ShuffSpec, Shuffler, ShuffleResult, notNanPvalFilter
from itertools import product


@dataclass
class PlotContext:
    figName: str
    axs: ArrayLike[Axes]
    yvals: Dict[str, ArrayLike] = field(default_factory=dict)
    immediateShuffles: List[Tuple[List[ShuffSpec], int]] = field(default_factory=list)
    xvals: Dict[str, ArrayLike] = field(default_factory=dict)
    immediateCorrelations: List[Tuple[str, str]] = field(default_factory=list)
    categories: Dict[str, ArrayLike] = field(default_factory=dict)
    infoVals: Dict[str, ArrayLike] = field(default_factory=dict)
    showPlot: Optional[bool] = None
    savePlot: Optional[bool] = None
    excludeFromCombo: bool = False
    immediateShufflePvalThreshold: float = 0.15
    transparent: bool = False

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
    def __init__(self, outputDir: Optional[str] = None,
                 randomSeed=None, verbosity: int = 3, infoFileName: Optional[str] = None) -> None:
        self.figSizeX: int = 5
        self.figSizeY: int = 5
        self.fig = plt.figure(figsize=(self.figSizeX, self.figSizeY))
        self.fig.clf()
        self.axs = self.fig.subplots()
        if isinstance(self.axs, Axes):
            self.axs = np.array([self.axs])
        # Save when exiting context?
        self.defaultSavePlot: bool = True
        # Show when exiting context?
        self.defaultShowPlot: bool = False
        self.verbosity: int = verbosity
        self.showedLastFig: bool = False

        if infoFileName is None:
            self.infoFileName = datetime.now().strftime("%Y%m%d_%H%M%S_info.txt")
        else:
            self.infoFileName = infoFileName
        self.outputSubDir = ""
        self.outputSubDirStack: List[str] = []
        if outputDir is None:
            outputDir = os.curdir
        self.setOutputDir(outputDir)
        self.createdPlots = set()
        self.savedFigsByName: Dict[str, List[str]] = {}

        self.persistentCategories: Dict[str, Any] = {}
        self.persistentInfoValues: Dict[str, Any] = {}

        self.rng = np.random.default_rng(seed=randomSeed)

        self.plotContext = None

        self.infoFileLock = multiprocessing.Lock()
        self.globalLock = multiprocessing.Lock()

        self.shuffler = Shuffler(self.rng)

    def __enter__(self) -> PlotContext:
        return self.plotContext

    def newFig(self, figName: str, subPlots: Optional[Tuple[int, int] | List[int, int]] = None,
               figScale: float = 1.0, excludeFromCombo: bool = False,
               showPlot: bool = None, savePlot: bool = None, enableOverwriteSameName: bool = False,
               transparent=False) -> PlotManager:
        fname = os.path.join(self.outputDir, self.outputSubDir, figName)
        if fname in self.createdPlots and not enableOverwriteSameName:
            raise Exception("Would overwrite file {} that was just made!".format(fname))

        self.clearFig()

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

        self.plotContext = PlotContext(fname, self.axs, showPlot=showPlot,
                                       savePlot=savePlot, excludeFromCombo=excludeFromCombo,
                                       transparent=transparent)

        return self

    def continueFig(self, figName, showPlot=None, savePlot=None, excludeFromCombo=False,
                    enableOverwriteSameName=False, transparent=False):
        if self.showedLastFig:
            raise Exception("currently unable to show a figure and then continue it")

        self.plotContext.showPlot = showPlot
        self.plotContext.savePlot = savePlot
        self.plotContext.transparent = transparent
        self.plotContext.excludeFromCombo = excludeFromCombo
        fname = os.path.join(self.outputDir, self.outputSubDir, figName)
        if fname in self.createdPlots and not enableOverwriteSameName:
            raise Exception("Would overwrite file {} that was just made!".format(fname))
        self.plotContext.figName = fname
        self.plotContext.yvals = {}
        self.plotContext.immediateShuffles = []
        self.plotContext.categories = {}
        self.plotContext.infoVals = {}
        return self

    def setStatCategory(self, category: str, value: Any):
        self.persistentCategories[category] = value

    def setStatInfo(self, infoCategory: str, value: Any):
        self.persistentInfoValues[infoCategory] = value

    def setCustomShuffleFunction(self, category: str, func: Callable[[pd.DataFrame, str, np.random.Generator], pd.Series]):
        self.shuffler.setCustomShuffleFunction(category, func)

    def __exit__(self, *args):

        if (self.plotContext.savePlot is not None and self.plotContext.savePlot) or \
                (self.plotContext.savePlot is None and self.defaultSavePlot):
            willSaveFig = True
        else:
            willSaveFig = False
        if (self.plotContext.showPlot is not None and self.plotContext.showPlot) or \
                (self.plotContext.showPlot is None and self.defaultShowPlot):
            willShowFig = True
        else:
            willShowFig = False

        if len(self.plotContext.yvals) > 0:
            # Everything is referenced according to this name
            figureName = os.path.basename(self.plotContext.figName)

            # There needs to be at least one category over which to shuffle
            assert len(self.persistentCategories) + len(self.plotContext.categories) + \
                len(self.plotContext.xvals) > 0

            # Check if there's any overlap between the keys of yvals, xvals, categories, infoVals, persistentCategories, and persistentInfoValues
            dicts = [self.plotContext.yvals, self.plotContext.xvals, self.plotContext.categories,
                     self.plotContext.infoVals, self.persistentCategories, self.persistentInfoValues]
            dictPairs = product(dicts, dicts)
            for (d1, d2) in dictPairs:
                if d1 is d2:
                    continue
                for k1 in d1.keys():
                    for k2 in d2.keys():
                        if k1 == k2:
                            raise Exception(
                                "Key {} is used in multiple dictionaries!".format(k1))

            # Build a pandas dataframe with all the data
            allData = self.plotContext.categories.copy()
            allData.update(self.plotContext.yvals)
            allData.update(self.plotContext.xvals)
            allData.update(self.plotContext.infoVals)
            allData.update(self.persistentCategories)
            allData.update(self.persistentInfoValues)
            df = pd.DataFrame(data=allData)
            yvalNames = pd.Series(list(self.plotContext.yvals.keys()), dtype=object)
            xvalNames = pd.Series(list(self.plotContext.xvals.keys()), dtype=object)
            categoryNames = pd.Series(list(self.plotContext.categories.keys()), dtype=object)
            persistentCategoryNames = pd.Series(
                list(self.persistentCategories.keys()), dtype=object)
            infoNames = pd.Series(list(self.plotContext.infoVals.keys()), dtype=object)
            persistentInfoNames = pd.Series(list(self.persistentInfoValues.keys()), dtype=object)

            # Save the stats to a file
            statsDir = os.path.join(self.outputDir, self.outputSubDir, "stats")
            uniqueID = f"{self.outputSubDirStack[-1]}_{figureName}"
            if not os.path.exists(statsDir):
                os.makedirs(statsDir)
            statsFile = os.path.join(statsDir, figureName + ".h5")
            df.to_hdf(statsFile, key="stats", mode="w")
            yvalNames.to_hdf(statsFile, key="yvalNames", mode="a")
            xvalNames.to_hdf(statsFile, key="xvalNames", mode="a")
            categoryNames.to_hdf(statsFile, key="categoryNames", mode="a")
            persistentCategoryNames.to_hdf(statsFile, key="persistentCategoryNames", mode="a")
            infoNames.to_hdf(statsFile, key="infoNames", mode="a")
            persistentInfoNames.to_hdf(statsFile, key="persistentInfoNames", mode="a")
            self.writeToInfoFile(f"statsFile:__!__{uniqueID}__!__{statsFile}")

            # If neither showing nor saving the figure, can safely skip this part since
            # the only output is a graphic on the figure itself
            # NOTE: this isn't actually true if I ever call continueFig after this,
            # and ideally I'd also pickle the immediateshuffles and immediatecorrelations,
            # but for now just running the big screen I don't need either of those
            runImmediate = willSaveFig or willShowFig

            # Now if there are any immediate shuffles, do them
            if len(self.plotContext.immediateShuffles) > 0 and runImmediate:
                if not self.plotContext.isSinglePlot:
                    raise Exception("Can't have multiple plots and do shuffle")
                if len(self.plotContext.immediateShuffles) > 1:
                    raise Exception("Can't have multiple yvals and do shuffle")

                with open(os.path.join(statsDir, figureName + "_immediateShuffles.pkl"), "wb") as fid:
                    pickle.dump(self.plotContext.immediateShuffles, fid)

                ss = [len(s) for s, _ in self.plotContext.immediateShuffles]
                self.plotContext.immediateShuffles = [x for _, x in sorted(
                    zip(ss, self.plotContext.immediateShuffles))]
                shufSpecs = [x for x, _ in self.plotContext.immediateShuffles]
                numShuffles = [x for _, x in self.plotContext.immediateShuffles]
                df.drop(columns=list(self.persistentCategories.keys()) +
                        list(self.persistentInfoValues.keys()), inplace=True)
                immediateRes = self.shuffler._doShuffles(df, shufSpecs, list(
                    self.plotContext.yvals.keys()), numShuffles=numShuffles)

                # print("Immediate shuffles:")
                # for rr in immediateRes:
                #     print("----")
                #     for r in rr:
                #         print(r)

                if len(self.plotContext.yvals) == 1:
                    pval = None
                    for rr in immediateRes:
                        for r in rr:
                            if pval is not None:
                                raise Exception("Should only be doing one shuffle here!")
                            pval = r.getPVals().item()
                            shufDiffs = r.shuffleDiffs
                            dataDiff = r.diff

                    if not np.isnan(shufDiffs).all():
                        shuffledCategories = df[shufSpecs[0][0].categoryName].unique()
                        cat1 = shufSpecs[0][0].value
                        cat2 = shuffledCategories[shuffledCategories != cat1].item()
                        direction = pval > 0.5
                        if direction:
                            pval = 1 - pval
                        if pval > 0:
                            pvalText = f"p={round(pval, 3)}\n{cat1} {'<' if direction else '>'} {cat2}"
                        else:
                            pvalText = f"p<{round(1/numShuffles[0], 3)}\n{cat1} {'<' if direction else '>'} {cat2}"
                        self.plotContext.ax.add_artist(AnchoredText(pvalText, "upper center"))

                        # Now create an inset axis in self.plotContext.ax in the bottom middle that shows the
                        # histogram of the shuffled differences and a red vertical line at the data difference
                        insetAx = self.plotContext.ax.inset_axes([0.4, 0.15, 0.2, 0.1])
                        insetAx.hist(shufDiffs, bins=20, color="black")
                        insetAx.axvline(dataDiff, color="red")
                        insetAx.set_xlabel(f"{cat1} - {cat2}")
                        # turn off y axis
                        insetAx.get_yaxis().set_visible(False)
                else:
                    try:
                        xvals = [float(v.split("_")[-1])
                                 for v in list(self.plotContext.yvals.keys())]
                    except:
                        raise Exception("Can't do multiple shuffles like this. If you want multiple comparisons, "
                                        "specify yvals as measureName_x, where x is the x value for that measure")

                    xvals = np.array(xvals)
                    pvals = immediateRes[0][0].getPVals()
                    pvals = np.array([r if r < 0.5 else 1 - r for r in pvals])
                    # pvals = np.array([r.getPVals().item() for r in immediateRes[0]])
                    pValSig = pvals < self.plotContext.immediateShufflePvalThreshold
                    # Now add an aterisk to self.plotContext.ax 95% up the y axis where the pval is significant
                    ymin, ymax = self.plotContext.ax.get_ylim()
                    ycoord = ymax - 0.05 * (ymax - ymin)
                    self.plotContext.ax.scatter(xvals[pValSig], [ycoord] * len(xvals[pValSig]), marker="*",
                                                color="black", s=100)

            elif len(self.plotContext.immediateCorrelations) > 0 and runImmediate:
                # Working with continuous x values here
                # find the correlation coefficient and pvalue between the x and y values
                # and indicate it on the plot. If there are multiple categories, do this for each
                # category separately
                if not self.plotContext.isSinglePlot:
                    raise Exception("Can't have multiple plots and do correlation")
                if len(self.plotContext.immediateCorrelations) > 1:
                    raise Exception("Can't have multiple immediate correlations")

                ic = self.plotContext.immediateCorrelations[0]
                xvar = ic[0]
                yvar = ic[1]
                if xvar not in df.columns or yvar not in df.columns:
                    raise Exception(
                        f"Can't do immediate correlation because {xvar} or {yvar} not in data frame")

                # Now get all combination of categories specified in plotContext.categories
                # and do the correlation for each
                catNames = list(self.plotContext.categories.keys())
                results = self.shuffler.runCorrelations(df, xvar, yvar, catNames)
                resTest = []
                for catlist, r, p in results:
                    if len(catlist) > 0:
                        catStr = "_".join([str(c) for c in catlist]) + ":  "
                    else:
                        catStr = ""
                    resTest.append(f"{catStr}r={round(r, 3)}, p={round(p, 3)}")
                self.plotContext.ax.add_artist(AnchoredText("\n".join(resTest), "upper center"))

        # Finally, save or show the figure
        if willSaveFig:
            self.saveFig()
        if willShowFig:
            self.showedLastFig = True
            plt.show()
            self.fig = plt.figure(figsize=(self.figSizeX, self.figSizeY))
        else:
            self.showedLastFig = False

    @property
    def fullOutputDir(self):
        return os.path.join(self.outputDir, self.outputSubDir)

    def saveFig(self):
        fname = os.path.join(self.outputDir, self.outputSubDir, self.plotContext.figName)

        if fname[-4:] != ".png" and fname[-4:] != ".pdf":
            fname += ".png"

        if self.plotContext.isSinglePlot and not self.plotContext.excludeFromCombo:
            for _ in range(3):
                try:
                    with open(fname + ".pkl", "wb") as fid:
                        pickle.dump(self.plotContext.ax, fid)
                    break
                except RuntimeError:
                    pass

        plt.savefig(fname, bbox_inches="tight", dpi=100, transparent=self.plotContext.transparent)
        self.writeToInfoFile("wrote file {}".format(fname))
        if self.verbosity >= 3:
            print("wrote file {}".format(fname))
        self.createdPlots.add(fname)

        figFileName = os.path.basename(fname)
        self.globalLock.acquire()
        if figFileName not in self.savedFigsByName:
            self.savedFigsByName[figFileName] = [self.outputSubDir]
            self.globalLock.release()
        else:
            thisFigList = self.savedFigsByName[figFileName]
            self.globalLock.release()
            thisFigList.append(self.outputSubDir)

    def clearFig(self) -> None:
        self.fig.clf()
        self.axs = np.array([self.fig.subplots(1, 1)])
        self.axs[0].cla()

    def setOutputDir(self, outputDir: str) -> None:
        self.outputDir = outputDir
        if not os.path.exists(os.path.join(outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(outputDir, self.outputSubDir))
        self.infoFileFullName = os.path.join(
            outputDir, self.infoFileName)

    def setOutputSubDir(self, outputSubDir: str) -> None:
        self.outputSubDirStack = [outputSubDir]
        self.outputSubDir = outputSubDir
        if not os.path.exists(os.path.join(self.outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, self.outputSubDir))

    def pushOutputSubDir(self, outputSubDir: str) -> None:
        self.outputSubDirStack.append(outputSubDir)
        self.outputSubDir = os.path.join(*self.outputSubDirStack)
        if not os.path.exists(os.path.join(self.outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, self.outputSubDir))

    def popOutputSubDir(self) -> None:
        self.outputSubDirStack.pop()
        if len(self.outputSubDirStack) == 0:
            self.outputSubDir = ""
        else:
            self.outputSubDir = os.path.join(*self.outputSubDirStack)
        if not os.path.exists(os.path.join(self.outputDir, self.outputSubDir)):
            os.makedirs(os.path.join(self.outputDir, self.outputSubDir))

    def writeToInfoFile(self, txt: str, suffix="\n") -> None:
        with self.infoFileLock:
            with open(self.infoFileFullName, "a") as f:
                f.write(txt + suffix)

    def makeCombinedFigs(self, outputSubDir: str = "combined",
                         suggestedSubPlotLayout: Optional[Tuple[int, int] | List[int, int]] = None) -> None:
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

    def runImmediateShufflesAcrossPersistentCategories(self, numShuffles=100) -> None:
        outFile = self.shuffler.runImmediateShufflesAcrossPersistentCategories(
            [self.infoFileFullName], numShuffles)
        self.shuffler.summarizeShuffleResults(outFile)

    def runShuffles(self, numShuffles=100, significantThreshold: Optional[float] = 0.15,
                    resultsFilter: Callable[[str, ShuffleResult, float], bool] = notNanPvalFilter) -> None:
        outFile = self.shuffler.runAllShuffles([self.infoFileFullName], numShuffles,
                                               significantThreshold, resultsFilter)
        self.shuffler.summarizeShuffleResults(outFile)


def setupBehaviorTracePlot(axs, sesh, showWells: str = "HAO", wellZOrder=2, outlineColors=-1,
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

    if outlineColors == -1:
        outlineColors = "orange" if sesh.isRippleInterruption else "cyan"
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


def plotIndividualAndAverage(ax: Axes, dataPoints, xvals, individualColor="grey", avgColor="blue", spread="std",
                             individualZOrder=1, averageZOrder=2, label=None, individualAmt=None, color=None,
                             interpolateOverNans=False):
    """
    values of individualAmt on range [0, 1] are interpreted as a fraction. Larger numbers as the count of traces to plot
    """
    if color is not None:
        avgColor = color
        individualColor = color

    # if xvals is a vector, make it a matrix where each column is the same vector
    xvals = np.array(xvals)
    if xvals.ndim == 1:
        xvals2d = np.tile(xvals, (dataPoints.shape[0], 1)).T
        xValsMatch = True
    else:
        xvals2d = xvals.T
        if np.all(np.isclose(xvals, xvals[0, :]) | np.isnan(xvals)):
            xvals = xvals[0, :]
            xValsMatch = True
        else:
            xValsMatch = False

    if individualAmt is None:
        yvals = dataPoints.T
    else:
        n = dataPoints.shape[0]
        if individualAmt <= 1:
            numPlot = int(individualAmt * n)
        else:
            numPlot = individualAmt
            if numPlot > n:
                numPlot = n
        indIdxs = np.random.choice(n, size=numPlot, replace=False)
        yvals = dataPoints[indIdxs, :].T

    if interpolateOverNans:
        for col in range(yvals.shape[1]):
            yvals[:, col] = np.interp(xvals2d[:, col], xvals2d[~np.isnan(
                yvals[:, col]), col], yvals[~np.isnan(yvals[:, col]), col],
                left=np.nan, right=np.nan)

    ax.plot(xvals2d, yvals, c=individualColor, lw=0.5, zorder=individualZOrder)

    if xValsMatch:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
            warnings.filterwarnings("ignore", r"Mean of empty slice")
            warnings.filterwarnings("ignore", r"invalid value encountered in subtract")
            hm = np.nanmean(dataPoints, axis=0)
            try:
                hs = np.nanstd(dataPoints, axis=0)
                hs[np.isinf(hs)] = np.nan
            except RuntimeWarning as re:
                print(f"{ dataPoints.shape=  }")
                print(f"{ np.isnan(dataPoints).sum(axis=0)=  }")
                print(f"{ np.isfinite(dataPoints).sum(axis=0)=  }")
                print(f"{ np.isinf(dataPoints).sum(axis=0)=  }")
                raise re
        if spread == "sem":
            n = dataPoints.shape[0] - np.count_nonzero(np.isnan(dataPoints), axis=0)
            hs = hs / np.sqrt(n)
        h1 = hm - hs
        h2 = hm + hs

        if interpolateOverNans:
            hm = np.interp(xvals, xvals[~np.isnan(hm)], hm[~np.isnan(hm)],
                           left=np.nan, right=np.nan)
            h1 = np.interp(xvals, xvals[~np.isnan(h1)], h1[~np.isnan(h1)],
                           left=np.nan, right=np.nan)
            h2 = np.interp(xvals, xvals[~np.isnan(h2)], h2[~np.isnan(h2)],
                           left=np.nan, right=np.nan)

        # hasSpread = ~np.isnan(h1) & ~np.isnan(h2)

        if label is None:
            labelKwarg = {}
        else:
            labelKwarg = {"label": label}
        ax.plot(xvals, hm, color=avgColor, zorder=averageZOrder, **labelKwarg)
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


def violinPlot(ax: Axes, yvals: pd.Series | ArrayLike, categories: pd.Series | ArrayLike,
               categories2: Optional[pd.Series | ArrayLike] = None,
               dotColors: Optional[pd.Series | ArrayLike] = None,
               axesNames: Optional[pd.Series | ArrayLike] = None,
               categoryOrder: Optional[List] = None, category2Order: Optional[List] = None,
               categoryColors: Optional[List] = None, dotColorLabels: Optional[Dict[Any, str]] = None):
    if isinstance(yvals, pd.Series):
        yvals = yvals.to_numpy()
    if isinstance(categories, pd.Series):
        categories = categories.to_numpy()
    if dotColors is not None and isinstance(dotColors, pd.Series):
        dotColors = dotColors.to_numpy()
    if categories2 is not None and isinstance(categories2, pd.Series):
        categories2 = categories2.to_numpy()

    if categoryOrder is None:
        categoryOrder = getPreferredCategoryOrder(categories)
    sortingCategories = [categoryOrder.index(c) for c in categories]

    if categories2 is None:
        categories2 = ["a" for _ in categories]
        category2Order = ["a"]
        sortingCategories2 = categories2
        cat2IsFake = True
        if axesNames is not None:
            axesNames.append("Z")
    else:
        cat2IsFake = False

    if category2Order is None:
        category2Order = getPreferredCategoryOrder(categories2)
    sortingCategories2 = [category2Order.index(c) for c in categories2]

    if dotColors is None:
        swarmPallete = sns.color_palette(palette=["black"])
        dotColors = np.array([0.0 for _ in yvals])
    elif dotColorLabels is not None:
        dotColors = [dotColorLabels[c] for c in dotColors]
        swarmPallete = sns.color_palette(palette=dotColorLabels.keys())
    else:
        uniqueDotColors = list(np.unique(dotColors))
        nColors = len(uniqueDotColors)
        if isinstance(dotColors[0], str):
            dotColors = [uniqueDotColors.index(c) for c in dotColors]
            swarmPallete = sns.color_palette(palette=uniqueDotColors, n_colors=nColors)
        else:
            swarmPallete = sns.color_palette("coolwarm", n_colors=nColors)

    # Same sorting here as in perseveration plot function so colors are always the same
    # sortList = ["{}__{}__{}".format(x, y, xi)
    #             for xi, (x, y) in enumerate(zip(sortingCategories, sortingCategories2))]
    sortList = [(x, y, xi) for xi, (x, y) in enumerate(zip(sortingCategories, sortingCategories2))]
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

    if categoryColors is None:
        if set(categories) == {"Ctrl", "SWR"}:
            categoryColors = ["orange", "cyan"]
        else:
            categoryColors = ["red", "blue"]
    pal = sns.color_palette(palette=categoryColors)
    sx = np.array(s[axesNamesNoSpaces[2]])
    # sy = np.array(s[axesNamesNoSpaces[1]]).astype(float)
    sh = np.array(s[axesNamesNoSpaces[0]])
    swarmx = np.array([a + b for a, b in zip(sx, sh)])

    swarmCategories = [(a, b) for a, b in zip(categories, categories2)]
    swarmxSortList = [(x, y) for x, y in product(categoryOrder, category2Order)]
    swarmxValue = np.array([swarmxSortList.index(v) for v in swarmCategories])

    # p1 = sns.violinplot(ax=ax, hue=sh,
    #                     y=sy, x=swarmx, data=s, palette=pal, linewidth=0.2, cut=0, zorder=1)
    # sns.stripplot(ax=ax, x=swarmx, y=sy, hue=dotColors, data=s,
    #               zorder=3, dodge=False, palette=swarmPallete)
    p1 = sns.violinplot(ax=ax, hue=categories,
                        y=yvals, x=swarmx, data=s, palette=pal, linewidth=0.2, cut=0, zorder=1)
    sns.stripplot(ax=ax, x=swarmxValue, y=yvals, hue="dotcoloraxisname", data=s,
                  zorder=3, dodge=False, palette=swarmPallete, linewidth=1)

    if cat2IsFake:
        p1.set(xticklabels=[])
    else:
        # Override the default x-axis labels
        p1.set(xticklabels=category2Order * len(categoryOrder))
    handles, labels = ax.get_legend_handles_labels()
    if dotColorLabels is not None:
        ax.legend(handles, labels, fontsize=6).set_zorder(2)
    else:
        ax.legend(handles[:2], labels[:2], fontsize=6).set_zorder(2)

    if not hideAxes:
        if not cat2IsFake:
            ax.set_xlabel(axesNames[2])
        ax.set_ylabel(axesNames[1])


def blankPlot(ax):
    ax.cla()
    ax.tick_params(axis="both", which="both", label1On=False,
                   label2On=False, tick1On=False, tick2On=False)
