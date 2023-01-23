import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Callable

from UtilFunctions import offWall
from PlotUtil import violinPlot, PlotManager, ShuffSpec, setupBehaviorTracePlot
from consts import allWellNames, TRODES_SAMPLING_RATE
from BTSession import BTSession


class TrialMeasure():
    def __init__(self, name="", measureFunc=None, sessionList=None, trialFilter=None, runStats=True):
        self.measure = []
        self.trialCategory = []
        self.conditionCategoryByTrial = []
        self.dotColors = []
        self.name = name
        self.runStats = runStats

        if measureFunc is not None:
            assert sessionList is not None
            for si, sesh in enumerate(sessionList):
                assert isinstance(sesh, BTSession)
                # home trials
                t1 = np.array(sesh.homeRewardEnter_posIdx)
                t0 = np.array(np.hstack(([0], sesh.awayRewardExit_posIdx)))
                if not sesh.endedOnHome:
                    t0 = t0[0:-1]
                # print(t0)
                # print(t1)
                # print(sesh.endedOnHome)
                # print(sesh.name)
                assert len(t1) == len(t0)

                for ii, (i0, i1) in enumerate(zip(t0, t1)):
                    if trialFilter is not None and not trialFilter("home", ii, i0, i1, sesh.homeWell):
                        continue

                    val = measureFunc(sesh, i0, i1, "home")
                    self.measure.append(val)
                    self.trialCategory.append("home")
                    self.conditionCategoryByTrial.append(
                        "SWR" if sesh.isRippleInterruption else "Ctrl")
                    self.dotColors.append(si)

                # away trials
                t1 = np.array(sesh.awayRewardEnter_posIdx)
                t0 = np.array(sesh.homeRewardExit_posIdx)
                if sesh.endedOnHome:
                    t0 = t0[0:-1]
                assert len(t1) == len(t0)

                for ii, (i0, i1) in enumerate(zip(t0, t1)):
                    if trialFilter is not None and not trialFilter("away", ii, i0, i1, sesh.visitedAwayWells[ii]):
                        continue

                    val = measureFunc(sesh, i0, i1, "away")
                    self.measure.append(val)
                    self.trialCategory.append("away")
                    self.conditionCategoryByTrial.append(
                        "SWR" if sesh.isRippleInterruption else "Ctrl")
                    self.dotColors.append(si)

        self.measure = np.array(self.measure)
        self.dotColors = np.array(self.dotColors)
        self.trialCategory = np.array(self.trialCategory)
        self.conditionCategoryByTrial = np.array(self.conditionCategoryByTrial)

    def makeFigures(self, plotManager: PlotManager):
        figName = self.name.replace(" ", "_")
        with plotManager.newFig(figName) as pc:
            violinPlot(pc.ax, self.measure, categories2=self.trialCategory, categories=self.conditionCategoryByTrial,
                       axesNames=["Condition", self.name, "Trial type"])

            if self.runStats:
                pc.yvals[figName] = self.measure
                pc.categories["trial"] = self.trialCategory
                pc.categories["condition"] = self.conditionCategoryByTrial


class WellMeasure():
    def __init__(self, name: str = "",
                 measureFunc: None | Callable[[BTSession, int], float] = None,
                 sessionList: list = [],
                 wellFilter: Callable[[int, int], bool] = lambda ai, aw: offWall(aw),
                 runStats: bool = True):
        self.measure = []
        self.wellCategory = []
        self.conditionCategoryByWell = []
        self.conditionCategoryBySession = []
        self.name = name
        self.withinSessionMeasureDifference = []
        self.acrossSessionMeasureDifference = []
        self.dotColors = []
        self.dotColorsBySession = []
        self.runStats = runStats
        self.numSessions = len(sessionList)

        self.allMeasureValsBySession = []
        self.measureMin = np.inf
        self.measureMax = -np.inf

        self.sessionList = sessionList

        if measureFunc is not None:
            assert sessionList is not None
            # print(sessionList)
            for si, sesh in enumerate(sessionList):
                measureDict = {}
                for well in allWellNames:
                    v = measureFunc(sesh, well)
                    measureDict[well] = v
                    if v > self.measureMax:
                        self.measureMax = v
                    if v < self.measureMin:
                        self.measureMin = v
                self.allMeasureValsBySession.append(measureDict)

            for si, sesh in enumerate(sessionList):
                homeval = self.allMeasureValsBySession[si][sesh.homeWell]
                self.measure.append(homeval)
                self.wellCategory.append("home")
                self.dotColors.append(si)
                self.dotColorsBySession.append(si)
                self.conditionCategoryByWell.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                self.conditionCategoryBySession.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")

                awayVals = []
                aways = sesh.visitedAwayWells
                if wellFilter is not None:
                    aways = [aw for ai, aw in enumerate(aways) if wellFilter(ai, aw)]
                if len(aways) == 0:
                    print("warning: no off wall aways for session {}".format(sesh.name))
                    self.withinSessionMeasureDifference.append(np.nan)
                else:
                    for ai, aw in enumerate(aways):
                        # av = measureFunc(sesh, aw)
                        av = self.allMeasureValsBySession[si][aw]
                        awayVals.append(av)
                        self.measure.append(av)
                        self.wellCategory.append("away")
                        self.dotColors.append(si)
                        self.conditionCategoryByWell.append(self.conditionCategoryByWell[-1])

                    awayVals = np.array(awayVals)
                    if len(awayVals) == 0:
                        assert False
                    elif all(np.isnan(awayVals)):
                        awayMean = np.nan
                    else:
                        awayMean = np.nanmean(awayVals)
                    self.withinSessionMeasureDifference.append(homeval - awayMean)

                otherSeshVals = []
                for isi, isesh in enumerate(sessionList):
                    if isi == si:
                        continue
                    osv = self.allMeasureValsBySession[isi][sesh.homeWell]
                    otherSeshVals.append(osv)
                    self.measure.append(osv)
                    self.wellCategory.append("othersesh")
                    self.dotColors.append(si)
                    self.conditionCategoryByWell.append(
                        "SWR" if sesh.isRippleInterruption else "Ctrl")
                self.acrossSessionMeasureDifference.append(homeval - np.nanmean(otherSeshVals))

        self.measure = np.array(self.measure)
        self.wellCategory = np.array(self.wellCategory)
        self.dotColors = np.array(self.dotColors)
        self.dotColorsBySession = np.array(self.dotColorsBySession)
        self.conditionCategoryByWell = np.array(self.conditionCategoryByWell)
        self.conditionCategoryBySession = np.array(self.conditionCategoryBySession)
        self.withinSessionMeasureDifference = np.array(self.withinSessionMeasureDifference)

    def makeFigures(self,
                    plotManager: PlotManager,
                    makeMeasureBoxPlot=True, makeDiffBoxPlot=True, makeOtherSeshBoxPlot=True,
                    makeOtherSeshDiffBoxPlot=True, makeEverySessionPlot=True,
                    everySessionTraceType: None | str = None,
                    everySessionTraceTimeInterval: None | Callable[[
                        BTSession], tuple | list] = None,
                    priority=None):
        figName = self.name.replace(" ", "_")

        if makeMeasureBoxPlot:
            # print("Making " + figName)
            with plotManager.newFig(figName) as pc:
                midx = [v in ["home", "away"] for v in self.wellCategory]
                fmeasure = self.measure[midx]
                fwellCat = self.wellCategory[midx]
                fcondCat = self.conditionCategoryByWell[midx]
                fdotColors = self.dotColors[midx]

                violinPlot(pc.ax, fmeasure, categories2=fwellCat, categories=fcondCat,
                           axesNames=["Condition", self.name, "Well type"],
                           dotColors=fdotColors)

                if self.runStats:
                    pc.yvals[figName] = fmeasure
                    pc.categories["well"] = fwellCat
                    pc.categories["condition"] = fcondCat

        if makeDiffBoxPlot:
            # print("Making diff, " + figName)
            with plotManager.newFig(figName + "_diff") as pc:
                violinPlot(pc.ax, self.withinSessionMeasureDifference, self.conditionCategoryBySession,
                           axesNames=["Contidion", self.name + " within-session difference"],
                           dotColors=self.dotColorsBySession)

                if self.runStats:
                    pc.yvals[figName + "_diff"] = self.withinSessionMeasureDifference
                    pc.categories["condition"] = self.conditionCategoryBySession
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="Ctrl")], 100))

        if makeOtherSeshBoxPlot:
            with plotManager.newFig(figName + "_othersesh") as pc:
                midx = [v in ["home", "othersesh"] for v in self.wellCategory]
                fmeasure = self.measure[midx]
                fwellCat = self.wellCategory[midx]
                fwellCat[fwellCat == "home"] = "same"
                fwellCat[fwellCat == "othersesh"] = "other"
                fcondCat = self.conditionCategoryByWell[midx]
                fdotColors = self.dotColors[midx]

                violinPlot(pc.ax, fmeasure, categories2=fwellCat, categories=fcondCat,
                           axesNames=["Condition", self.name, "Session"],
                           dotColors=fdotColors)

                if self.runStats:
                    pc.yvals[figName] = fmeasure
                    pc.categories["session"] = fwellCat
                    pc.categories["condition"] = fcondCat

        if makeOtherSeshDiffBoxPlot:
            with plotManager.newFig(figName + "_othersesh_diff") as pc:
                violinPlot(pc.ax, self.acrossSessionMeasureDifference, self.conditionCategoryBySession,
                           axesNames=["Contidion", self.name + " across-session difference"],
                           dotColors=self.dotColorsBySession)

                if self.runStats:
                    pc.yvals[figName + "_othersesh_diff"] = self.acrossSessionMeasureDifference
                    pc.categories["condition"] = self.conditionCategoryBySession
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="Ctrl")], 100))

        if makeEverySessionPlot:
            assert everySessionTraceType is None or everySessionTraceType in [
                "task", "probe", "task_bouts", "probe_bouts", "test"
            ]

            numCols = math.ceil(math.sqrt(self.numSessions))
            wellSize = mpl.rcParams['lines.markersize']**2 / 4
            zorder = 2

            with plotManager.newFig(figName + "_every_session", subPlots=(numCols, numCols), figScale=0.3) as pc:
                # for si, (sk, cond) in enumerate(zip(self.allMeasureValsBySession, self.conditionCategoryBySession)):
                for si, sesh in enumerate(self.sessionList):
                    assert isinstance(sesh, BTSession)

                    sk = self.allMeasureValsBySession[si]
                    cond = self.conditionCategoryBySession[si]
                    ax = pc.axs[si // numCols, si % numCols]
                    assert isinstance(ax, Axes)

                    # ax.invert_yaxis()
                    # ax.tick_params(axis="both", which="both", label1On=False,
                    #                label2On=False, tick1On=False, tick2On=False)

                    if everySessionTraceType is not None:
                        if "task" in everySessionTraceType:
                            xs = np.array(sesh.btPosXs)
                            ys = np.array(sesh.btPosYs)
                            mv = np.array(sesh.btBoutCategory == BTSession.BOUT_STATE_EXPLORE)
                            ts = np.array(sesh.btPos_ts)
                        elif "probe" in everySessionTraceType:
                            xs = np.array(sesh.probePosXs)
                            ys = np.array(sesh.probePosYs)
                            mv = np.array(sesh.probeBoutCategory == BTSession.BOUT_STATE_EXPLORE)
                            ts = np.array(sesh.probePos_ts)
                        elif "test" in everySessionTraceType:
                            xs = np.linspace(34, 1100, 1000)
                            ys = np.linspace(-40, 960, 1000)
                            mv = np.ones_like(xs).astype(bool)
                            ts = np.arange(len(xs)) * TRODES_SAMPLING_RATE / 15

                        if everySessionTraceTimeInterval is not None:
                            if callable(everySessionTraceTimeInterval):
                                timeInterval = everySessionTraceTimeInterval(sesh)
                            else:
                                timeInterval = everySessionTraceTimeInterval
                            assert len(xs) == len(ts)
                            durIdx = np.searchsorted(ts, np.array(
                                [ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE, ts[0] +
                                 timeInterval[1] * TRODES_SAMPLING_RATE]))
                            xs = xs[durIdx[0]:durIdx[1]]
                            ys = ys[durIdx[0]:durIdx[1]]
                            mv = mv[durIdx[0]:durIdx[1]]

                        if "_bouts" in everySessionTraceType:
                            xs = xs[mv]
                            ys = ys[mv]

                        ax.plot(xs, ys, c="#deac7f")
                        ax.scatter(xs[-1], ys[-1], marker="*")
                        # print(np.min(xs), np.max(xs), np.min(ys), np.max(ys))

                    c = "orange" if cond == "SWR" else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c,
                                           wellSize=wellSize, showWells="HA")

                    # TODO might not need this?
                    for v in ax.spines.values():
                        v.set_zorder(0)
                    ax.set_title(sesh.name, fontdict={'fontsize': 6})

                    valImg = np.empty((6, 6))
                    for wr in range(6):
                        for wc in range(6):
                            wname = 8*wr + wc + 2
                            valImg[wr, wc] = sk[wname]

                    im = ax.imshow(valImg, cmap=mpl.colormaps["coolwarm"],
                                   vmin=self.measureMin, vmax=self.measureMax,
                                   interpolation="nearest", extent=(0, 6, 0, 6),
                                   origin="lower")

                    # some stuff to stretch the image out to the edges
                    z = im.get_zorder()
                    ax.imshow(valImg, cmap=mpl.colormaps["coolwarm"],
                              vmin=self.measureMin, vmax=self.measureMax,
                              interpolation="nearest", extent=(-0.5, 6.5, -0.5, 6.5),
                              origin="lower", zorder=z-0.03)
                    ax.imshow(valImg, cmap=mpl.colormaps["coolwarm"],
                              vmin=self.measureMin, vmax=self.measureMax,
                              interpolation="nearest", extent=(-0, 6, -0.5, 6.5),
                              origin="lower", zorder=z-0.02)
                    ax.imshow(valImg, cmap=mpl.colormaps["coolwarm"],
                              vmin=self.measureMin, vmax=self.measureMax,
                              interpolation="nearest", extent=(-0.5, 6.5, 0, 6),
                              origin="lower", zorder=z-0.02)

                for si in range(self.numSessions, numCols * numCols):
                    ax = pc.axs[si // numCols, si % numCols]
                    ax.cla()
                    ax.tick_params(axis="both", which="both", label1On=False,
                                   label2On=False, tick1On=False, tick2On=False)

                plt.colorbar(im, ax=ax)
