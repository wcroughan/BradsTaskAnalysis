import numpy as np
from UtilFunctions import offWall
from PlotUtil import boxPlot, PlotCtx


class TrialMeasure():
    def __init__(self, name="", measureFunc=None, sessionList=None, forceRemake=False, trialFilter=None, runStats=True):
        self.measure = []
        self.trialCategory = []
        self.conditionCategoryByTrial = []
        self.dotColors = []
        self.name = name
        self.runStats = runStats

        if measureFunc is not None:
            assert sessionList is not None
            for si, sesh in enumerate(sessionList):
                # home trials
                t1 = np.array(sesh.home_well_find_pos_idxs)
                t0 = np.array(np.hstack(([0], sesh.away_well_leave_pos_idxs)))
                if not sesh.ended_on_home:
                    t0 = t0[0:-1]
                # print(t0)
                # print(t1)
                # print(sesh.ended_on_home)
                # print(sesh.name)
                assert len(t1) == len(t0)

                for ii, (i0, i1) in enumerate(zip(t0, t1)):
                    if trialFilter is not None and not trialFilter("home", ii, i0, i1, sesh.home_well):
                        continue

                    val = measureFunc(sesh, i0, i1, "home")
                    self.measure.append(val)
                    self.trialCategory.append("home")
                    self.conditionCategoryByTrial.append(
                        "SWR" if sesh.isRippleInterruption else "Ctrl")
                    self.dotColors.append(si)

                # away trials
                t1 = np.array(sesh.away_well_find_pos_idxs)
                t0 = np.array(sesh.home_well_leave_pos_idxs)
                if sesh.ended_on_home:
                    t0 = t0[0:-1]
                assert len(t1) == len(t0)

                for ii, (i0, i1) in enumerate(zip(t0, t1)):
                    if trialFilter is not None and not trialFilter("away", ii, i0, i1, sesh.visited_away_wells[ii]):
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

    def makeFigures(self, plotCtx: PlotCtx):
        figName = self.name.replace(" ", "_")
        with plotCtx.newFig(figName, withStats=self.runStats) as nf:
            if self.runStats:
                ax, yvals, cats, info = nf
            else:
                ax = nf

            boxPlot(ax, self.measure, categories2=self.trialCategory, categories=self.conditionCategoryByTrial,
                    axesNames=["Condition", self.name, "Trial type"], violin=True, doStats=False)

            if self.runStats:
                yvals[figName] = self.measure
                cats["trial"] = self.trialCategory
                cats["condition"] = self.conditionCategoryByTrial


class WellMeasure():
    def __init__(self, name="", measureFunc=None, sessionList=None, forceRemake=False, wellFilter=lambda ai, aw: offWall(aw), runStats=True):
        self.measure = []
        self.wellCategory = []
        self.conditionCategoryByWell = []
        self.conditionCategoryBySession = []
        self.name = name
        self.withinSessionMeasureDifference = []
        self.dotColors = []
        self.dotColorsBySession = []
        self.runStats = runStats

        if measureFunc is not None:
            assert sessionList is not None
            # print(sessionList)
            for si, sesh in enumerate(sessionList):
                # print(sesh.home_well_find_times)
                homeval = measureFunc(sesh, sesh.home_well)
                self.measure.append(homeval)
                self.wellCategory.append("home")
                self.dotColors.append(si)
                self.dotColorsBySession.append(si)
                self.conditionCategoryByWell.append("SWR" if sesh.isRippleInterruption else "Ctrl")
                self.conditionCategoryBySession.append(
                    "SWR" if sesh.isRippleInterruption else "Ctrl")

                awayVals = []
                aways = sesh.visited_away_wells
                if wellFilter is not None:
                    aways = [aw for ai, aw in enumerate(aways) if wellFilter(ai, aw)]
                if len(aways) == 0:
                    print("warning: no off wall aways for session {}".format(sesh.name))
                    self.withinSessionMeasureDifference.append(np.nan)
                else:
                    for ai, aw in enumerate(aways):
                        av = measureFunc(sesh, aw)
                        awayVals.append(av)
                        self.measure.append(av)
                        self.wellCategory.append("away")
                        self.dotColors.append(si)
                        self.conditionCategoryByWell.append(self.conditionCategoryByWell[-1])

                    awayVals = np.array(awayVals)
                    self.withinSessionMeasureDifference.append(homeval - np.nanmean(awayVals))

        self.measure = np.array(self.measure)
        self.wellCategory = np.array(self.wellCategory)
        self.dotColors = np.array(self.dotColors)
        self.dotColorsBySession = np.array(self.dotColorsBySession)
        self.conditionCategoryByWell = np.array(self.conditionCategoryByWell)
        self.conditionCategoryBySession = np.array(self.conditionCategoryBySession)
        self.withinSessionMeasureDifference = np.array(self.withinSessionMeasureDifference)

    def makeFigures(self, plotCtx: PlotCtx, makeMeasureBoxPlot=True, makeDiffBoxPlot=True):
        figName = self.name.replace(" ", "_")

        if makeMeasureBoxPlot:
            # print("Making " + figName)
            with plotCtx.newFig(figName, withStats=self.runStats) as nf:
                if self.runStats:
                    ax, yvals, cats, info = nf
                else:
                    ax = nf

                boxPlot(ax, self.measure, categories2=self.wellCategory, categories=self.conditionCategoryByWell,
                        axesNames=["Condition", self.name, "Well type"], violin=True, doStats=False,
                        dotColors=self.dotColors)

                if self.runStats:
                    yvals[figName] = self.measure
                    cats["well"] = self.wellCategory
                    cats["condition"] = self.conditionCategoryByWell

        if makeDiffBoxPlot:
            # print("Making diff, " + figName)
            with plotCtx.newFig(figName + "_diff", withStats=self.runStats) as nf:
                if self.runStats:
                    ax, yvals, cats, info = nf
                else:
                    ax = nf

                boxPlot(ax, self.withinSessionMeasureDifference, self.conditionCategoryBySession,
                        axesNames=["Contidion", self.name + " with-session difference"], violin=True, doStats=False,
                        dotColors=self.dotColorsBySession)

                if self.runStats:
                    yvals[figName + "_diff"] = self.withinSessionMeasureDifference
                    cats["condition"] = self.conditionCategoryBySession
