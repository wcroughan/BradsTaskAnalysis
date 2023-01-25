import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Callable, List
import pandas as pd

from UtilFunctions import offWall
from PlotUtil import violinPlot, PlotManager, ShuffSpec, setupBehaviorTracePlot
from consts import allWellNames, TRODES_SAMPLING_RATE
from BTSession import BTSession


class TrialMeasure():
    """
    A measure that has a value for every trial during a task, such as trial duration.

    Callbacks:
    measureFunc(session, trialStart_posIdx, trialEnd_posIdx, trial type ("home" | "away")) -> measure value

    optional trial filter
    trialFilter(session, trial type ("home" | "away"), trial index, trialStart_posIdx, trialEnd_posIdx, well number)
        -> True if trial should be included, False if if should be skipped
    trial index is independently tracked for home and away. So the first home and first away trials
        will both have index 0
    These functions are not necessarily called in the order in which they occur during the task
    """

    def __init__(self, name: str = "",
                 measureFunc: Callable[[BTSession, int, int, str], float] = lambda _: np.nan,
                 sessionList: List[BTSession] = [],
                 trialFilter: None | Callable[[BTSession, str, int, int, int, int], bool] = None,
                 runStats=True):
        self.name = name
        self.runStats = runStats

        measure = []
        trialType = []
        conditionColumn = []
        dotColors = []
        sessionColumn = []

        # coffee
        # pomodoro:
        # calc within sesh diff
        # and across sesh??
        # plot so can see output for every single trial along with at least path
        # plot each session so can see all trials, within sesh diff
        # across session diff somehow
        # individual and average plot across sessions by trial number

        for si, sesh in enumerate(sessionList):
            t1 = np.array(sesh.homeRewardEnter_posIdx)
            t0 = np.array(np.hstack(([0], sesh.awayRewardExit_posIdx)))
            if not sesh.endedOnHome:
                t0 = t0[0:-1]
            assert len(t1) == len(t0)

            for ii, (i0, i1) in enumerate(zip(t0, t1)):
                if trialFilter is not None and not trialFilter(sesh, "home", ii, i0, i1, sesh.homeWell):
                    continue

                val = measureFunc(sesh, i0, i1, "home")
                measure.append(val)
                trialType.append("home")
                conditionColumn.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")
                dotColors.append(si)
                sessionColumn.append(sesh)

            # away trials
            t1 = np.array(sesh.awayRewardEnter_posIdx)
            t0 = np.array(sesh.homeRewardExit_posIdx)
            if sesh.endedOnHome:
                t0 = t0[0:-1]
            assert len(t1) == len(t0)

            for ii, (i0, i1) in enumerate(zip(t0, t1)):
                if trialFilter is not None and not trialFilter(sesh, "away", ii, i0, i1, sesh.visitedAwayWells[ii]):
                    continue

                val = measureFunc(sesh, i0, i1, "away")
                measure.append(val)
                trialType.append("away")
                conditionColumn.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")
                dotColors.append(si)
                sessionColumn.append(sesh)

        self.trialDf = pd.DataFrame({"session": sessionColumn,
                                     "val": measure,
                                     "trialType": trialType,
                                     "condition": conditionColumn,
                                     "dotColor": dotColors})

    def makeFigures(self, plotManager: PlotManager):
        # TODO:
        # flags for what to make
        # every_trial plot. Maybe whole background is coolwarm, and trace in black?
        # maybe individual and average for i.e. trial duration plot
        figName = self.name.replace(" ", "_")
        with plotManager.newFig(figName) as pc:
            violinPlot(pc.ax, self.trialDf["val"], categories2=self.trialDf["trialType"],
                       categories=self.trialDf["condition"], dotColors=self.trialDf["dotColor"],
                       axesNames=["Condition", self.name, "Trial type"])

            if self.runStats:
                pc.yvals[figName] = self.trialDf["val"].to_numpy()
                pc.categories["trial"] = self.trialDf["trialType"].to_numpy()
                pc.categories["condition"] = self.trialDf["condition"].to_numpy()


class WellMeasure():
    """
    A measure that has a value for every well, such as probe well time

    Callbacks:
    measureFunc(session, wellname) -> measure value

    optional well filter for away wells
    wellFilter(away trial index, away well name)
        -> True if away well should be included, False if if should be skipped
    """

    def __init__(self, name: str = "",
                 measureFunc: Callable[[BTSession, int], float] = lambda _: np.nan,
                 sessionList: List[BTSession] = [],
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
                "task", "probe", "task_bouts", "probe_bouts"
            ]

            numCols = math.ceil(math.sqrt(self.numSessions))
            wellSize = mpl.rcParams['lines.markersize']**2 / 4

            with plotManager.newFig(figName + "_every_session", subPlots=(numCols, numCols), figScale=0.3) as pc:
                # for si, (sk, cond) in enumerate(zip(self.allMeasureValsBySession, self.conditionCategoryBySession)):
                for si, sesh in enumerate(self.sessionList):
                    sk = self.allMeasureValsBySession[si]
                    cond = self.conditionCategoryBySession[si]
                    ax = pc.axs[si // numCols, si % numCols]
                    assert isinstance(ax, Axes)

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
                        if len(xs) > 0:
                            ax.scatter(xs[-1], ys[-1], marker="*")

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
