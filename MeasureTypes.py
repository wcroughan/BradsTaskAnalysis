import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Callable, List, Tuple, Optional, Iterable
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike

from UtilFunctions import offWall, getWellPosCoordinates
from PlotUtil import violinPlot, PlotManager, ShuffSpec, setupBehaviorTracePlot, blankPlot, \
    plotIndividualAndAverage
from consts import allWellNames, TRODES_SAMPLING_RATE
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP


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
        self.measure2d = np.empty((len(sessionList), 25))
        self.measure2d[:, :] = np.nan
        trialType = []
        conditionColumn = []
        dotColors = []
        sessionIdxColumn = []
        wasExcluded = []
        trial_posIdx = []
        self.sessionList = sessionList

        for si, sesh in enumerate(sessionList):
            t1 = np.array(sesh.homeRewardEnter_posIdx)
            t0 = np.array(np.hstack(([0], sesh.awayRewardExit_posIdx)))
            if not sesh.endedOnHome:
                t0 = t0[0:-1]
            assert len(t1) == len(t0)

            for ii, (i0, i1) in enumerate(zip(t0, t1)):
                trialType.append("home")
                conditionColumn.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")
                dotColors.append(si)
                sessionIdxColumn.append(si)
                trial_posIdx.append((i0, i1))

                if trialFilter is not None and not trialFilter(sesh, "home", ii, i0, i1, sesh.homeWell):
                    wasExcluded.append(1)
                    measure.append(np.nan)
                else:
                    val = measureFunc(sesh, i0, i1, "home")
                    measure.append(val)
                    self.measure2d[si, ii*2] = val
                    wasExcluded.append(0)

            # away trials
            t1 = np.array(sesh.awayRewardEnter_posIdx)
            t0 = np.array(sesh.homeRewardExit_posIdx)
            if sesh.endedOnHome:
                t0 = t0[0:-1]
            assert len(t1) == len(t0)

            for ii, (i0, i1) in enumerate(zip(t0, t1)):
                trialType.append("away")
                conditionColumn.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")
                dotColors.append(si)
                sessionIdxColumn.append(si)
                trial_posIdx.append((i0, i1))

                if trialFilter is not None and not trialFilter(sesh, "away", ii, i0, i1, sesh.visitedAwayWells[ii]):
                    wasExcluded.append(1)
                    measure.append(np.nan)
                else:
                    val = measureFunc(sesh, i0, i1, "away")
                    measure.append(val)
                    self.measure2d[si, ii*2+1] = val
                    wasExcluded.append(0)

        self.valMin = np.nanmin(measure)
        self.valMax = np.nanmax(measure)
        self.trialDf = pd.DataFrame({"sessionIdx": sessionIdxColumn,
                                     "val": measure,
                                     "trialType": trialType,
                                     "condition": conditionColumn,
                                     "wasExcluded": wasExcluded,
                                     "trial_posIdx": trial_posIdx,
                                     "dotColor": dotColors})
        self.sessionDf = self.trialDf.groupby("sessionIdx")[["condition", "dotColor"]].nth(0)

        # Groups home and away trials within a session, takes nanmean of each
        # Then takes difference of home and away values within each sesion
        # Finally, lines them up with sessionDf info
        self.withinSessionDiffs = self.trialDf.groupby(["sessionIdx", "trialType"]).agg(
            withinSessionDiff=("val", "mean")).diff().xs("home", level="trialType").join(self.sessionDf)
        diffs = self.withinSessionDiffs["withinSessionDiff"].to_numpy()
        self.diffMin = np.nanmin(diffs)
        self.diffMax = np.nanmax(diffs)

        # print(self.trialDf)
        # print(self.sessionDf)

    def makeFigures(self,
                    plotManager: PlotManager,
                    plotFlags: str | List[str] = "all"):

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = ["measure", "diff", "everytrial", "everysession", "averages"]
            else:
                plotFlags = [plotFlags]
        else:
            # So passed in list isn't modified by remove
            plotFlags = [v for v in plotFlags]

        figName = self.name.replace(" ", "_")
        if "measure" in plotFlags:
            plotFlags.remove("measure")
            with plotManager.newFig(figName) as pc:
                violinPlot(pc.ax, self.trialDf["val"], categories2=self.trialDf["trialType"],
                           categories=self.trialDf["condition"], dotColors=self.trialDf["dotColor"],
                           axesNames=["Condition", self.name, "Trial type"])

                if self.runStats:
                    pc.yvals[figName] = self.trialDf["val"].to_numpy()
                    pc.categories["trial"] = self.trialDf["trialType"].to_numpy()
                    pc.categories["condition"] = self.trialDf["condition"].to_numpy()

        if "diff" in plotFlags:
            plotFlags.remove("diff")
            with plotManager.newFig(figName + "_diff") as pc:
                violinPlot(pc.ax, self.withinSessionDiffs["withinSessionDiff"],
                           categories=self.withinSessionDiffs["condition"],
                           dotColors=self.withinSessionDiffs["dotColor"],
                           axesNames=["Contidion", self.name + " within-session difference"])

        if "averages" in plotFlags:
            plotFlags.remove("averages")
            xvalsAll = np.arange(self.measure2d.shape[1]) + 1
            xvalsHalf = np.arange(math.ceil(self.measure2d.shape[1] / 2)) + 1
            swrIdx = np.array([sesh.isRippleInterruption for sesh in self.sessionList])
            ctrlIdx = np.array([not v for v in swrIdx])

            with plotManager.newFig(figName + "_byTrialAvgs_all") as pc:
                plotIndividualAndAverage(pc.ax, self.measure2d, xvalsAll, avgColor="grey")
                pc.ax.set_xlim(1, len(xvalsAll))
                pc.ax.set_xticks(np.arange(0, len(xvalsAll), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_all_byCond") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, :], xvalsAll, avgColor="orange", label="SWR")
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, :], xvalsAll, avgColor="cyan", label="Ctrl")
                pc.ax.set_xlim(1, len(xvalsAll))
                pc.ax.set_xticks(np.arange(0, len(xvalsAll), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_all_byTrialType") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[:, ::2], xvalsAll, avgColor="red", label="home")
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[:, 1::2], xvalsAll, avgColor="blue", label="away")
                pc.ax.set_xlim(1, len(xvalsAll))
                pc.ax.set_xticks(np.arange(0, len(xvalsAll), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_home") as pc:
                plotIndividualAndAverage(pc.ax, self.measure2d[:, ::2], xvalsHalf, avgColor="grey")
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_away") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[:, 1::2], xvalsHalf[:-1], avgColor="grey")
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_home_byCond") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, ::2], xvalsHalf, avgColor="orange", label="SWR")
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, ::2], xvalsHalf, avgColor="cyan", label="Ctrl")
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_away_byCond") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, 1::2], xvalsHalf[:-1], avgColor="orange", label="SWR")
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, 1::2], xvalsHalf[:-1], avgColor="cyan", label="Ctrl")
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_ctrl_byTrialType") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, ::2], xvalsHalf, avgColor="red", label="home")
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[ctrlIdx, 1::2], xvalsHalf[:-1], avgColor="blue", label="away")
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

            with plotManager.newFig(figName + "_byTrialAvgs_SWR_byTrialType") as pc:
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, ::2], xvalsHalf, avgColor="red", label="home")
                plotIndividualAndAverage(
                    pc.ax, self.measure2d[swrIdx, 1::2], xvalsHalf[:-1], avgColor="blue", label="away")
                pc.ax.set_xlim(1, len(xvalsHalf))
                pc.ax.set_xticks(np.arange(0, len(xvalsHalf), 2) + 1)

        if "everytrial" in plotFlags:
            plotFlags.remove("everytrial")
            cmap = mpl.colormaps["coolwarm"]
            for si, sesh in enumerate(self.sessionList):
                thisSession = self.trialDf[self.trialDf["sessionIdx"]
                                           == si].sort_values("trial_posIdx")
                tpis = np.array([list(v) for v in thisSession["trial_posIdx"]])
                vals = thisSession["val"].to_numpy()
                normvals = (vals - self.valMin) / (self.valMax - self.valMin)
                plotManager.pushOutputSubDir(sesh.name)

                ncols = len(sesh.visitedAwayWells) + 1
                with plotManager.newFig(figName + "_allTrials", subPlots=(2, ncols)) as pc:
                    for ti in range(2*ncols):
                        ai0 = ti % 2
                        ai1 = ti // 2
                        ax = pc.axs[ai0, ai1]

                        if ti >= tpis.shape[0]:
                            blankPlot(ax)
                            continue

                        assert isinstance(ax, Axes)
                        c = "orange" if self.withinSessionDiffs.loc[si,
                                                                    "condition"] == "SWR" else "cyan"
                        setupBehaviorTracePlot(ax, sesh, outlineColors=c)
                        t0 = tpis[ti, 0]
                        t1 = tpis[ti, 1]
                        ax.plot(sesh.btPosXs[t0:t1], sesh.btPosYs[t0:t1], c="black")
                        ax.set_facecolor(cmap(normvals[ti]))
                        ax.set_title(str(vals[ti]))

                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.valMin, self.valMax), cmap=cmap), ax=ax)

                    ax = pc.axs[1, -1]
                    assert isinstance(ax, Axes)
                    v = self.withinSessionDiffs.loc[si, "withinSessionDiff"]
                    ax.set_facecolor(cmap((v - self.diffMin) / (self.diffMax - self.diffMin)))
                    c = "orange" if self.withinSessionDiffs.loc[si,
                                                                "condition"] == "SWR" else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, showWells="")
                    ax.set_title(f"avg diff = {v}")
                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.diffMin, self.diffMax), cmap=cmap), ax=ax)

                plotManager.popOutputSubDir()

        if "everysession" in plotFlags:
            plotFlags.remove("everysession")
            cmap = mpl.colormaps["coolwarm"]
            ncols = 14
            wellSize = mpl.rcParams['lines.markersize']**2 / 4
            with plotManager.newFig(figName + "_allTrials_allSessions",
                                    subPlots=(2*len(self.sessionList), ncols), figScale=0.3) as pc:
                for si, sesh in enumerate(self.sessionList):
                    print(si)
                    thisSession = self.trialDf[self.trialDf["sessionIdx"]
                                               == si].sort_values("trial_posIdx")
                    tpis = np.array([list(v) for v in thisSession["trial_posIdx"]])
                    vals = thisSession["val"].to_numpy()
                    normvals = (vals - self.valMin) / (self.valMax - self.valMin)

                    blankPlot(pc.axs[2*si, 0])
                    blankPlot(pc.axs[2*si+1, 0])

                    for ti in range(2 * ncols - 2):
                        ai0 = (ti % 2) + 2 * si
                        ai1 = (ti // 2) + 1
                        ax = pc.axs[ai0, ai1]
                        if ti >= tpis.shape[0]:
                            blankPlot(ax)
                            continue

                        assert isinstance(ax, Axes)
                        c = "orange" if self.withinSessionDiffs.loc[si,
                                                                    "condition"] == "SWR" else "cyan"
                        setupBehaviorTracePlot(ax, sesh, outlineColors=c, wellSize=wellSize)
                        t0 = tpis[ti, 0]
                        t1 = tpis[ti, 1]
                        ax.plot(sesh.btPosXs[t0:t1], sesh.btPosYs[t0:t1], c="black")
                        ax.set_facecolor(cmap(normvals[ti]))
                        ax.set_title(str(vals[ti]))

                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.valMin, self.valMax), cmap=cmap), ax=pc.axs[2*si+1, -1])

                    ax = pc.axs[2*si+1, 0]
                    assert isinstance(ax, Axes)
                    v = self.withinSessionDiffs.loc[si, "withinSessionDiff"]
                    ax.set_facecolor(cmap((v - self.diffMin) / (self.diffMax - self.diffMin)))
                    c = "orange" if self.withinSessionDiffs.loc[si,
                                                                "condition"] == "SWR" else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c, showWells="")
                    ax.set_title(f"avg diff = {v}")
                    pc.figure.colorbar(mappable=ScalarMappable(
                        norm=Normalize(self.diffMin, self.diffMax), cmap=cmap), ax=ax)

                    pc.axs[2*si, 0].set_title(sesh.name)

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")


class WellMeasure():
    """
    A measure that has a value for every well, such as probe well time

    Callbacks:
    measureFunc(session, wellname) -> measure value

    optional well filter for away wells
    wellFilter(away trial index, away well name)
        -> True if away well should be included in within-session diff, False if if should be skipped
        Note: measureFunc will be called on all wells regardless of this return value

    optional display function:
    displayFunc(session, wellname) -> array-like of values to display in the well as a circular histogram
    in makeFigures, this will be used to populate the radial plot
    """

    def __init__(self, name: str = "",
                 measureFunc: Callable[[BTSession, int], float] = lambda _: np.nan,
                 sessionList: List[BTSession] = [],
                 wellFilter: Callable[[int, int], bool] = lambda ai, aw: offWall(aw),
                 displayFunc: Optional[Callable[[BTSession, int], ArrayLike]] = None,
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

        if displayFunc is None:
            self.hasRadialDisplayValues = False
        else:
            self.hasRadialDisplayValues = True
            self.allDisplayValsBySession = []
            self.displayColumns = 0
            self.radialMin = np.inf
            self.radialMax = -np.inf

        self.sessionList = sessionList

        for si, sesh in enumerate(sessionList):
            measureDict = {}
            displayDict = {}
            for well in allWellNames:
                v = measureFunc(sesh, well)
                measureDict[well] = v
                if v > self.measureMax:
                    self.measureMax = v
                if v < self.measureMin:
                    self.measureMin = v

                if self.hasRadialDisplayValues:
                    v = displayFunc(sesh, well)
                    if v.shape[0] > self.displayColumns:
                        self.displayColumns = v.shape[0]
                    displayDict[well] = v
                    if np.nanmax(v) > self.radialMax:
                        self.radialMax = np.nanmax(v)
                    if np.nanmin(v) < self.radialMin:
                        self.radialMin = np.nanmin(v)

            self.allMeasureValsBySession.append(measureDict)
            if self.hasRadialDisplayValues:
                self.allDisplayValsBySession.append(displayDict)

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
        self.acrossSessionMeasureDifference = np.array(self.acrossSessionMeasureDifference)

        # measure = []
        # sessionIdxColumn = []
        # condition = []
        # conditionStr = []
        # excludeFromDiff = []
        # dotColors = []
        # wellName = []
        # wellType = []
        # for si, sesh in enumerate(sessionList):
        #     for well in allWellNames:
        #         measure.append(measureFunc(sesh, well))
        #         sessionIdxColumn.append(si)
        #         condition.append(sesh.condition)
        #         conditionStr.append("SWR" if sesh.isRippleInterruption else "Ctrl")
        #         dotColors.append("orange" if sesh.isRippleInterruption else "cyan")
        #         wellName.append(well)
        #         if well == sesh.homeWell:
        #             wellType.append("home")
        #             excludeFromDiff.append(False)
        #         elif well in sesh.visitedAwayWells:
        #             ai = sesh.visitedAwayWells.index(well)
        #             wellType.append("away")
        #             excludeFromDiff.append(not wellFilter(ai, well))
        #         else:
        #             wellType.append("other")
        #             excludeFromDiff.append(False)

        # self.measureDf = pd.DataFrame({
        #     "sessionIdx": sessionIdxColumn,
        #     "wellName": wellName,
        #     "val": measure,
        #     "wellType": wellType,
        #     "condition": condition,
        #     "conditionStr": conditionStr,
        #     "excludeFromDiff": excludeFromDiff,
        #     "dotColor": dotColors,
        # })
        # print(self.measureDf.to_string())

        # # Groups home and away trials within a session, takes nanmean of each
        # # Then takes difference of home and away values within each sesion
        # # Finally, lines them up with sessionDf info
        # k = self.measureDf.groupby(["sessionIdx", "wellType"]).agg(
        #     withinSessionAvg=("val", "mean")).loc[(slice(None), ("home", "away")), :].groupby(
        #     "sessionIdx").agg(withinSessionDiff=("withinSessionAvg", "diff")).xs("away", level="wellType").copy()
        # k["withinSessionDiff"] = -k["withinSessionDiff"]

        # sd = self.measureDf.groupby(
        #     "sessionIdx")[["condition", "dotColor", "conditionStr"]].nth(0)
        # self.sessionDf = k.join(sd)
        # print(self.sessionDf.to_string())

        # This all seems like it's working. Just might come back to it later, not necessary for now

    def makeFigures(self,
                    plotManager: PlotManager,
                    plotFlags: str | List[str] = "all",
                    everySessionTraceType: None | str = None,
                    everySessionTraceTimeInterval: None | Callable[[
                        BTSession], tuple | list] = None,
                    radialTraceType: None | str | Iterable[str] = None,
                    radialTraceTimeInterval: None |
                    Callable[[BTSession, int], tuple | list] |
                    Iterable[Callable[[BTSession], tuple | list]] = None,
                    priority=None):
        figName = self.name.replace(" ", "_")

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = ["measure", "diff", "othersesh", "otherseshdiff", "everysession"]
                if self.hasRadialDisplayValues:
                    plotFlags.append("radial")
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if "measure" in plotFlags:
            plotFlags.remove("measure")
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

        if "diff" in plotFlags:
            plotFlags.remove("diff")
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

        if "othersesh" in plotFlags:
            plotFlags.remove("othersesh")
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

        if "otherseshdiff" in plotFlags:
            plotFlags.remove("otherseshdiff")
            with plotManager.newFig(figName + "_othersesh_diff") as pc:
                violinPlot(pc.ax, self.acrossSessionMeasureDifference, self.conditionCategoryBySession,
                           axesNames=["Contidion", self.name + " across-session difference"],
                           dotColors=self.dotColorsBySession)

                if self.runStats:
                    pc.yvals[figName + "_othersesh_diff"] = self.acrossSessionMeasureDifference
                    pc.categories["condition"] = self.conditionCategoryBySession
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="Ctrl")], 100))

        if "radial" in plotFlags:
            plotFlags.remove("radial")

            traceTypes = ["task", "probe"]
            traceMods = ["", "_bouts", "_mv", "_mv_bouts", "_bouts_mv"]
            validTraceStrings = [t + m for t in traceTypes for m in traceMods]
            if not isinstance(radialTraceType, (tuple, list)):
                radialTraceType = [radialTraceType] * self.displayColumns
            for estt in radialTraceType:
                assert estt is None or estt in validTraceStrings

            wellSize = mpl.rcParams['lines.markersize']**2 / 4

            with plotManager.newFig(figName + "_radial", subPlots=(len(self.sessionList), self.displayColumns), figScale=0.6) as pc:
                for si, sesh in enumerate(self.sessionList):
                    rvals = self.allDisplayValsBySession[si]
                    ncols = rvals[2].shape[0]
                    cond = self.conditionCategoryBySession[si]
                    for ri in range(ncols):
                        if self.displayColumns == 1:
                            ax = pc.axs[si]
                        elif len(self.sessionList) == 1:
                            ax = pc.axs[ri]
                        else:
                            ax = pc.axs[si, ri]
                        assert isinstance(ax, Axes)

                        if radialTraceType[ri] is not None:
                            rtt = radialTraceType[ri]
                            if "task" in rtt:
                                xs = sesh.btPosXs
                                ys = sesh.btPosYs
                                bout = sesh.btBoutCategory == BTSession.BOUT_STATE_EXPLORE
                                mv = sesh.btIsMv
                                ts = sesh.btPos_ts
                            elif "probe" in rtt:
                                xs = sesh.probePosXs
                                ys = sesh.probePosYs
                                bout = sesh.probeBoutCategory == BTSession.BOUT_STATE_EXPLORE
                                mv = sesh.probeIsMv
                                ts = sesh.probePos_ts

                            if radialTraceTimeInterval is not None:
                                if callable(radialTraceTimeInterval):
                                    timeInterval = radialTraceTimeInterval(sesh)
                                elif isinstance(radialTraceTimeInterval[0], (tuple, list)):
                                    timeInterval = radialTraceTimeInterval[ri]
                                else:
                                    timeInterval = radialTraceTimeInterval

                                durIdx = sesh.timeIntervalToPosIdx(ts, timeInterval)
                                xs = xs[durIdx[0]:durIdx[1]]
                                ys = ys[durIdx[0]:durIdx[1]]
                                mv = mv[durIdx[0]:durIdx[1]]
                                bout = bout[durIdx[0]:durIdx[1]]

                            sectionFlag = np.ones_like(xs, dtype=bool)
                            if "_bouts" in rtt:
                                sectionFlag[~ bout] = False
                            if "_mv" in rtt:
                                sectionFlag[~ mv] = False
                            xs[~ sectionFlag] = np.nan

                            ax.plot(xs, ys, c="#deac7f", lw=0.5)
                            if len(xs) > 0:
                                ax.scatter(xs[-1], ys[-1], marker="*")

                        c = "orange" if cond == "SWR" else "cyan"
                        setupBehaviorTracePlot(ax, sesh, outlineColors=c,
                                               wellSize=wellSize, showWells="HA")

                        for v in ax.spines.values():
                            v.set_zorder(0)
                        ax.set_title(sesh.name, fontdict={'fontsize': 6})

                        for wr in range(6):
                            for wc in range(6):
                                wname = 8*wr + wc + 2
                                vals = rvals[wname][ri, :]
                                # if wname == 10:
                                #     print(f"vals: {vals}")
                                normVals = (vals - self.radialMin) / \
                                    (self.radialMax - self.radialMin)
                                for vi in range(len(normVals)):
                                    v = normVals[vi]
                                    vrad = vi / len(normVals) * 2 * np.pi - np.pi
                                    vx = v * np.cos(vrad) * 0.5
                                    vy = v * np.sin(vrad) * 0.5
                                    ax.plot([wc + 0.5, wc + vx + 0.5], [wr + 0.5,
                                            wr + vy + 0.5], c="k", lw=0.5, zorder=3)
                                    v2 = normVals[(vi + 1) % len(normVals)]
                                    # v2rad = (vi + 1) / len(normVals)
                                    v2rad = (vi + 1) / len(normVals) * 2 * np.pi - np.pi
                                    v2x = v2 * np.cos(v2rad) * 0.5
                                    v2y = v2 * np.sin(v2rad) * 0.5
                                    ax.plot([wc + vx + 0.5, wc + v2x + 0.5], [wr + vy +
                                            0.5, wr + v2y + 0.5], c="k", lw=0.5, zorder=3)

                    for ri in range(ncols, self.displayColumns):
                        if self.displayColumns == 1:
                            ax = pc.axs[si]
                        elif len(self.sessionList) == 1:
                            ax = pc.axs[ri]
                        else:
                            ax = pc.axs[si, ri]
                        blankPlot(ax)

        if "everysession" in plotFlags:
            plotFlags.remove("everysession")

            traceTypes = ["task", "probe"]
            traceMods = ["", "_bouts", "_mv", "_mv_bouts", "_bouts_mv"]
            validTraceStrings = [t + m for t in traceTypes for m in traceMods]
            assert everySessionTraceType is None or everySessionTraceType in validTraceStrings

            ncols = math.ceil(np.sqrt(len(self.sessionList)))
            wellSize = mpl.rcParams['lines.markersize']**2 / 4

            with plotManager.newFig(figName + "_every_session", subPlots=(ncols, ncols), figScale=0.6) as pc:
                for si, sesh in enumerate(self.sessionList):
                    sk = self.allMeasureValsBySession[si]
                    cond = self.conditionCategoryBySession[si]
                    ax = pc.axs[si // ncols, si % ncols]
                    assert isinstance(ax, Axes)

                    if everySessionTraceType is not None:
                        if "task" in everySessionTraceType:
                            xs = sesh.btPosXs
                            ys = sesh.btPosYs
                            bout = sesh.btBoutCategory == BTSession.BOUT_STATE_EXPLORE
                            mv = sesh.btIsMv
                            ts = sesh.btPos_ts
                        elif "probe" in everySessionTraceType:
                            xs = sesh.probePosXs
                            ys = sesh.probePosYs
                            bout = sesh.probeBoutCategory == BTSession.BOUT_STATE_EXPLORE
                            mv = sesh.probeIsMv
                            ts = sesh.probePos_ts

                        if everySessionTraceTimeInterval is not None:
                            if callable(everySessionTraceTimeInterval):
                                timeInterval = everySessionTraceTimeInterval(sesh)
                            else:
                                timeInterval = everySessionTraceTimeInterval
                            durIdx = sesh.timeIntervalToPosIdx(ts, timeInterval)
                            assert len(xs) == len(ts)
                            xs = xs[durIdx[0]:durIdx[1]]
                            ys = ys[durIdx[0]:durIdx[1]]
                            mv = mv[durIdx[0]:durIdx[1]]
                            bout = bout[durIdx[0]:durIdx[1]]

                        sectionFlag = np.ones_like(xs, dtype=bool)
                        if "_bouts" in everySessionTraceType:
                            sectionFlag[~ bout] = False
                        if "_mv" in everySessionTraceType:
                            sectionFlag[~ mv] = False
                        xs[~ sectionFlag] = np.nan

                        ax.plot(xs, ys, c="#deac7f")
                        if len(xs) > 0:
                            ax.scatter(xs[-1], ys[-1], marker="*")

                    c = "orange" if cond == "SWR" else "cyan"
                    setupBehaviorTracePlot(ax, sesh, outlineColors=c,
                                           wellSize=wellSize, showWells="HA")

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

                for si in range(self.numSessions, ncols * ncols):
                    ax = pc.axs[si // ncols, si % ncols]
                    blankPlot(ax)

                plt.colorbar(im, ax=ax)

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")


class SessionMeasure():
    @staticmethod
    def measureAtWell(vals: ArrayLike, well: int) -> ArrayLike:
        v = np.array(vals)
        wx, wy = getWellPosCoordinates(well)
        wxc = np.digitize(wx, np.linspace(-0.5, 6.5, v.shape[0]))
        wyc = np.digitize(wy, np.linspace(-0.5, 6.5, v.shape[1]))
        return v[wxc, wyc]

    @staticmethod
    def measureValueAtHome(sesh: BTSession, vals: ArrayLike) -> ArrayLike:
        return SessionMeasure.measureAtWell(vals, sesh.homeWell)

    @staticmethod
    def measureValueAtAwayWells(sesh: BTSession, vals: ArrayLike, allVals: ArrayLike, seshIdx: int, offWallOnly=True) -> ArrayLike:
        awVals = []
        for aw in sesh.visitedAwayWells:
            if offWallOnly and not offWall(aw):
                continue
            awVals.append(SessionMeasure.measureAtWell(vals, aw))
        return np.nanmean(awVals)

    @staticmethod
    def measureValueAtHomeOtherSeshs(sesh: BTSession, vals: ArrayLike, allVals: ArrayLike,  seshIdx: int, offWallOnly=True) -> ArrayLike:
        osVals = []
        for i in range(allVals.shape[0]):
            if i == seshIdx:
                continue
            osVals.append(SessionMeasure.measureAtWell(allVals[i], sesh.homeWell))
        return np.nanmean(osVals)

    def __init__(self, name: str, measureFunc: Callable[[BTSession], ArrayLike],
                 sessionList: List[BTSession],
                 sessionValFunc: Callable[[BTSession, ArrayLike],
                                          ArrayLike] = measureValueAtHome,
                 sessionCtrlValFunc: Callable[[BTSession, ArrayLike, ArrayLike, int],
                                              ArrayLike] = measureValueAtAwayWells,
                 runStats: bool = True) -> None:
        """
        for the sake of using measure as a background image, it should be indexed x-first
        """
        self.name = name
        self.sessionList = sessionList
        self.runStats = runStats
        self.measureValsBySession = [None] * len(sessionList)
        self.sessionValsBySession = [None] * len(sessionList)
        self.sessionCtrlValsBySession = [None] * len(sessionList)
        self.dotColorsBySession = [None] * len(sessionList)
        self.conditionBySession = [None] * len(sessionList)

        for si, sesh in enumerate(sessionList):
            v = measureFunc(sesh)
            self.measureValsBySession[si] = v
            self.sessionValsBySession[si] = sessionValFunc(sesh, v)
            if sesh.isRippleInterruption:
                self.conditionBySession[si] = "SWR"
                self.dotColorsBySession[si] = "orange"
            else:
                self.conditionBySession[si] = "Ctrl"
                self.dotColorsBySession[si] = "cyan"

        self.measureValsBySession = np.array(self.measureValsBySession)
        self.sessionValsBySession = np.array(self.sessionValsBySession)
        for si, sesh in enumerate(sessionList):
            self.sessionCtrlValsBySession[si] = sessionCtrlValFunc(
                sesh, self.measureValsBySession[si], self.measureValsBySession, si)

        self.valMin = np.nanmin(self.measureValsBySession)
        self.valMax = np.nanmax(self.measureValsBySession)
        self.sessionValMin = np.nanmin((self.sessionValsBySession, self.sessionCtrlValsBySession))
        self.sessionValMax = np.nanmax((self.sessionValsBySession, self.sessionCtrlValsBySession))

    def makeFigures(self,
                    plotManager: PlotManager,
                    plotFlags: str | List[str] = "all",
                    everySessionBehaviorPeriod: Optional[BP | Callable[[BTSession], BP]] = None,
                    priority=None):
        figName = self.name.replace(" ", "_")

        if isinstance(plotFlags, str):
            if plotFlags == "all":
                plotFlags = ["measure", "diff", "everysession"]
            else:
                plotFlags = [plotFlags]
        else:
            # so list passed in isn't modified
            plotFlags = [v for v in plotFlags]

        if "measure" in plotFlags:
            plotFlags.remove("measure")

            with plotManager.newFig(figName) as pc:
                violinPlot(pc.ax, self.sessionValsBySession, self.conditionBySession,
                           dotColors=self.dotColorsBySession, axesNames=["Condition", self.name])
                pc.ax.set_title(self.name, fontdict={'fontsize': 6})

                if self.runStats:
                    pc.yvals[figName] = self.sessionValsBySession
                    pc.categories["condition"] = self.conditionBySession
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="Ctrl")], 100))

        if "diff" in plotFlags:
            plotFlags.remove("diff")

            with plotManager.newFig(figName + "_diff") as pc:
                violinPlot(pc.ax, self.sessionValsBySession - self.sessionCtrlValsBySession, self.conditionBySession,
                           dotColors=self.dotColorsBySession, axesNames=["Condition", self.name])
                pc.ax.set_title(self.name, fontdict={'fontsize': 6})

                if self.runStats:
                    pc.yvals[figName] = self.sessionValsBySession - self.sessionCtrlValsBySession
                    pc.categories["condition"] = self.conditionBySession
                    pc.immediateShuffles.append((
                        [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="Ctrl")], 100))

        if "everysession" in plotFlags:
            plotFlags.remove("everysession")

            wellSize = mpl.rcParams['lines.markersize']**2 / 4
            ncols = int(np.ceil(np.sqrt(len(self.sessionList))))
            with plotManager.newFig(figName + "_every_session", subPlots=(ncols, ncols), figScale=0.6) as pc:
                for si, sesh in enumerate(self.sessionList):
                    sk = np.array(self.measureValsBySession[si])
                    cond = self.conditionBySession[si]
                    ax = pc.axs[si // ncols, si % ncols]
                    assert isinstance(ax, Axes)

                    if everySessionBehaviorPeriod is not None:
                        if callable(everySessionBehaviorPeriod):
                            bp = everySessionBehaviorPeriod(sesh)
                        else:
                            bp = everySessionBehaviorPeriod

                        _, xs, ys = sesh.posDuringBehaviorPeriod(bp)
                        ax.plot(xs, ys, c="#deac7f", lw=0.5)
                        if len(xs) > 0:
                            ax.scatter(xs[-1], ys[-1], marker="*")

                    setupBehaviorTracePlot(ax, sesh, wellSize=wellSize, showWells="HA")

                    for v in ax.spines.values():
                        v.set_zorder(0)
                    if np.isscalar(sk):
                        ax.set_title(f"{sesh.name} - {sk}", fontdict={'fontsize': 6})
                    else:
                        ax.set_title(f"{sesh.name}", fontdict={'fontsize': 6})

                    im = ax.imshow(self.measureValsBySession[si].T, cmap=mpl.colormaps["coolwarm"],
                                   vmin=self.valMin, vmax=self.valMax,
                                   interpolation="nearest", extent=(-0.5, 6.5, -0.5, 6.5),
                                   origin="lower")

                for si in range(len(self.sessionList), ncols * ncols):
                    ax = pc.axs[si // ncols, si % ncols]
                    blankPlot(ax)

                plt.colorbar(im, ax=ax)

        if len(plotFlags) > 0:
            print(f"Warning: unused plot flags: {plotFlags}")
