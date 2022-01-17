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


class MyPlottingFunctions:
    def __init__(self, allData, output_dir, savePlots=True, showPlots=False, skipBoxPlots=False, skipScatterPlots=False, skipSwarmPlots=False, skipHistograms=False):
        self.SAVE_OUTPUT_PLOTS = savePlots
        self.SHOW_OUTPUT_PLOTS = showPlots
        self.SKIP_BOX_PLOTS = skipBoxPlots
        self.SKIP_SCATTER_PLOTS = skipScatterPlots
        self.SKIP_SWARM_PLOTS = skipSwarmPlots
        self.SKIP_HISTOGRAMS = skipHistograms
        self.output_dir = output_dir
        self.all_well_names = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])
        self.all_sessions = allData.getSessions()
        self.all_sessions_with_probe = [s for s in self.all_sessions if s.probe_performed]
        self.tlbls = [self.trial_label(sesh) for sesh in self.all_sessions]
        self.tlbls_with_probe = [self.trial_label(sesh) for sesh in self.all_sessions_with_probe]
        statsFileName = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S_stats.txt"))
        self.statsFile = open(statsFileName, "w")
        self.all_rest_sessions = allData.getRestSessions()

    def saveOrShow(self, fname):
        if self.SAVE_OUTPUT_PLOTS:
            plt.savefig(os.path.join(self.output_dir, fname), dpi=800)
        if self.SHOW_OUTPUT_PLOTS:
            plt.show()

    def makeALinePlot(self, valsFunc, title,
                      colorFunc=None, individualSessions=True, holdLastPlot=False,
                      saveAllValuePairsSeparately=False, plotAverage=False, avgError="std",
                      xlabel="", ylabel="", axisLims=None, includeNoProbeSessions=False,
                      restSessions=False, linewidth=None, errorFunc=None):
        """
        valsFunc : session => [(xvals, yvals), (xvals, yvals), ...]
        """
        print("Line plot: {}".format(title))

        if restSessions:
            seshs = self.all_rest_sessions
        else:
            seshs = self.all_sessions if includeNoProbeSessions else self.all_sessions_with_probe

        if plotAverage:
            avgs = {}

        if not holdLastPlot:
            plt.clf()

        for sesh in seshs:
            if axisLims == "environment":
                xlim1 = np.min(sesh.bt_pos_xs) - 20
                xlim2 = np.max(sesh.bt_pos_xs) + 20
                ylim1 = np.min(sesh.bt_pos_ys) - 20
                ylim2 = np.max(sesh.bt_pos_ys) + 20
            elif type(axisLims) == list:
                xlim1 = axisLims[0][0]
                xlim2 = axisLims[0][1]
                ylim1 = axisLims[1][0]
                ylim2 = axisLims[1][1]

            vals = valsFunc(sesh)
            # print(vals)
            if colorFunc is not None:
                colors = colorFunc(sesh)
            # print(vals[0][0])
            if errorFunc is not None:
                errorBars = errorFunc(sesh)
            if not individualSessions and saveAllValuePairsSeparately and len(vals) > 1:
                print("Warning: won't be saving all pairs separately since combining across sessions")

            for vi, (xvs, yvs) in enumerate(vals):
                if plotAverage:
                    for xyi in range(len(xvs)):
                        if xvs[xyi] in avgs:
                            avgs[xvs[xyi]].append(yvs[xyi])
                        else:
                            avgs[xvs[xyi]] = [yvs[xyi]]
                    if linewidth is None:
                        linewidth = 0.5
                else:
                    if linewidth is None:
                        linewidth = 1.0

                if errorFunc is not None:
                    ers = errorBars[vi]

                if colorFunc is None:
                    if errorFunc is not None:
                        plt.errorbar(xvs, yvs, ers, linewidth=linewidth)
                    else:
                        plt.plot(xvs, yvs, linewidth=linewidth)
                elif not isinstance(colors[vi], list):
                    if errorFunc is not None:
                        plt.errorbar(xvs, yvs, ers, linewidth=linewidth, color=colors[vi])
                    else:
                        plt.plot(xvs, yvs, linewidth=linewidth, color=colors[vi])
                else:
                    for i in range(len(xvs)-1):
                        if errorFunc is not None:
                            plt.errorbar(xvs[i:i+2], yvs[i:i+2], ers[vi][i], color=colors[vi]
                                         [i], linewidth=linewidth)
                        else:
                            plt.plot(xvs[i:i+2], yvs[i:i+2], color=colors[vi]
                                     [i], linewidth=linewidth)
                    if errorFunc is not None:
                        plt.errorbar([xvs[-1]], [yvs[-1]], [ers[vi][-1]], color=colors[vi]
                                     [-1], linewidth=linewidth)

                if individualSessions and saveAllValuePairsSeparately:
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    if axisLims is not None:
                        plt.xlim(xlim1, xlim2)
                        plt.ylim(ylim1, ylim2)
                    self.saveOrShow("{}_{}_{}".format(title, sesh.name, vi))
                    if not holdLastPlot:
                        plt.clf()
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    if axisLims is not None:
                        plt.xlim(xlim1, xlim2)
                        plt.ylim(ylim1, ylim2)

            if individualSessions and not saveAllValuePairsSeparately:
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                if axisLims is not None:
                    plt.xlim(xlim1, xlim2)
                    plt.ylim(ylim1, ylim2)
                self.saveOrShow("{}_{}".format(title, sesh.name))
                if not holdLastPlot:
                    plt.clf()

        if not individualSessions:
            if plotAverage:
                xs = sorted(avgs.keys())
                ys = [np.nanmean(avgs[x]) for x in xs]
                ers = [np.nanstd(avgs[x]) for x in xs]
                if avgError == "std":
                    pass
                elif avgError == "sem":
                    ers = np.array(ers) / np.sqrt(np.array([len(avgs[x]) for x in xs]))
                else:
                    raise Exception("Unknown avg error type")
                if colorFunc is None:
                    c = "gray"
                else:
                    if not isinstance(colors[vi], list):
                        c = colors[0]
                    else:
                        c = colors[0][0]
                plt.errorbar(xs, ys, ers, linewidth=2, color=c)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if axisLims is not None:
                plt.xlim(xlim1, xlim2)
                plt.ylim(ylim1, ylim2)
            self.saveOrShow(title)

    def makeASimpleBoxPlot(self, valFunc, title, yAxisName=None, alsoMakeSessionOrderScatterPlot=True, includeNoProbeSessions=False, restSessions=False):
        if yAxisName is None:
            yAxisName = title

        if restSessions:
            seshs = self.all_rest_sessions
            L = [self.trial_label(rs.btwpSession) for rs in seshs]
        else:
            seshs = self.all_sessions if includeNoProbeSessions else self.all_sessions_with_probe
            L = self.tlbls if includeNoProbeSessions else self.tlbls_with_probe
        ys = [valFunc(sesh) for sesh in seshs]
        self.makeABoxPlot(ys, L, ["Condition", yAxisName], title=title)
        if alsoMakeSessionOrderScatterPlot:
            self.makeAScatterPlot(np.arange(len(seshs)), ys, [
                                  "Session idx", yAxisName], categories=L, title=title)

    def makeABoxPlot(self, yvals, categories, axesNames, output_filename="", title="", doStats=True, scaleValue=None):
        if self.SKIP_BOX_PLOTS:
            print("Warning, skipping box plots!")
            return

        print("Box plot: {}".format(title))

        axesNamesNoSpaces = [a.replace(" ", "_") for a in axesNames]

        # Same sorting here as in perseveration plot function so colors are always the same
        sortList = [x + str(xi) for xi, x in enumerate(categories)]
        categories = [x for _, x in sorted(zip(sortList, categories))]
        yvals = [x for _, x in sorted(zip(sortList, yvals))]

        s = pd.Series([categories, yvals], index=axesNamesNoSpaces)
        plt.clf()
        # print(s)
        sns.boxplot(x=axesNamesNoSpaces[0], y=axesNamesNoSpaces[1], data=s, palette="Set3")
        sns.swarmplot(x=axesNamesNoSpaces[0], y=axesNamesNoSpaces[1], data=s, color="0.25")
        plt.title(title)
        plt.xlabel(axesNames[0])
        plt.ylabel(axesNames[1])
        ucats = set(categories)

        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, "_".join(
                    axesNamesNoSpaces[::-1]) + "_" + "_".join(sorted(list(ucats))))
            else:
                output_filename = os.path.join(self.output_dir, output_filename)

        if doStats:
            if scaleValue is not None:
                for i in range(len(yvals)):
                    yvals[i] *= scaleValue

            s = pd.Series([categories, yvals], index=axesNamesNoSpaces)

            self.statsFile.write("============================\n" +
                                 output_filename + " ANOVA:\n")
            if s.size == 0 or np.count_nonzero(np.logical_not(np.isnan(np.array(yvals)))) == 0:
                self.statsFile.write("No data ... can't do stats")
            else:
                # print(s)
                anovaModel = ols(
                    axesNamesNoSpaces[1] + " ~ C(" + axesNamesNoSpaces[0] + ")", data=s).fit()
                anova_table = anova_lm(anovaModel, typ=1)
                self.statsFile.write(str(anova_table))
            for cat in ucats:
                self.statsFile.write(str(cat) + " n = " + str(sum([i == cat for i in categories])))
            self.statsFile.write("\n\n")

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            plt.savefig(output_filename, dpi=800)

    def makeAScatterPlotWithFunc(self, valsFunc, title, colorFunc=None, individualSessions=True,
                                 saveAllValuePairsSeparately=False, plotAverage=False, xlabel="", ylabel="",
                                 axisLims=None, includeNoProbeSessions=False, restSessions=False):
        print("scatter plot:", title)
        if restSessions:
            seshs = self.all_rest_sessions
        else:
            seshs = self.all_sessions if includeNoProbeSessions else self.all_sessions_with_probe

        if axisLims is not None:
            xlim1 = axisLims[0][0]
            xlim2 = axisLims[0][1]
            ylim1 = axisLims[1][0]
            ylim2 = axisLims[1][1]

        plt.clf()
        for sesh in seshs:
            vals = valsFunc(sesh)
            if colorFunc is not None:
                colors = colorFunc(sesh)
            # print(vals[0][0])
            if not individualSessions and saveAllValuePairsSeparately and len(vals) > 1:
                print("Warning: won't be saving all pairs separately since combining across sessions")

            for vi, (xvs, yvs) in enumerate(vals):
                if colorFunc is None:
                    plt.scatter(xvs, yvs)
                else:
                    plt.scatter(xvs, yvs, c=colors[vi])

                if individualSessions and saveAllValuePairsSeparately:
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    if axisLims is not None:
                        plt.xlim(xlim1, xlim2)
                        plt.ylim(ylim1, ylim2)
                    self.saveOrShow("{}_{}_{}".format(title, sesh.name, vi))
                    plt.clf()
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)

            if individualSessions and not saveAllValuePairsSeparately:
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                if axisLims is not None:
                    plt.xlim(xlim1, xlim2)
                    plt.ylim(ylim1, ylim2)
                self.saveOrShow("{}_{}".format(title, sesh.name))
                plt.clf()

        if not individualSessions:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if axisLims is not None:
                plt.xlim(xlim1, xlim2)
                plt.ylim(ylim1, ylim2)
            self.saveOrShow(title)

    def makeAScatterPlot(self, xvals, yvals, axesNames, categories=list(), output_filename="", title="", midline=False, makeLegend=False, ax=plt, bigDots=True):
        if self.SKIP_SCATTER_PLOTS:
            print("Warning, skipping scatter plots!")
            return

        print("Scatter plot: {}".format(title))

        if ax == plt:
            ax.clf()
        if len(categories) == 0:
            cvals = np.ones_like(xvals)
            ucats = list()
        else:
            ucats = np.array(sorted(list(set(categories))))
            cvals = [np.argmax(c == ucats) for c in categories]

        if bigDots:
            sz = plt.rcParams['lines.markersize'] ** 2 * 3
        else:
            sz = plt.rcParams['lines.markersize'] ** 2
        ax.scatter(xvals, yvals, c=cvals, zorder=2, s=sz)
        if ax == plt:
            ax.title(title)
            ax.xlabel(axesNames[0])
            ax.ylabel(axesNames[1])
        else:
            ax.set_title(title)
            ax.set_xlabel(axesNames[0])
            ax.set_ylabel(axesNames[1])
        if midline:
            mval = max(max(xvals), max(yvals))
            ax.plot([0, mval], [0, mval], color='black', zorder=1)

        if makeLegend:
            legend_elements = []
            if len(ucats) > 0:
                cs = np.linspace(0, 1, len(ucats))
                # print(cs)
            for i, cat in enumerate(ucats):
                c = plt.get_cmap()(cs[i])
                # print(cs, i, cs[i], c)
                legend_elements.append(Line2D([0], [0], marker='o', color=c, label=cat))
                # Line2D([0], [0], marker='o', color=cs[i], label=cat,
                # markerfacecolor='g', markersize=15)

            ax.legend(handles=legend_elements, loc='upper right')

        # for cat in ucats:
            # print(cat, "n = ", sum([i == cat for i in categories]))

        if ax != plt:
            return

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                if len(categories) == 0:
                    output_filename = os.path.join(self.output_dir, "_".join(
                        axesNames[::-1]))
                else:
                    output_filename = os.path.join(self.output_dir, "_".join(
                        axesNames[::-1]) + "_" + "_".join(sorted(list(ucats))))
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

    def makeASwarmPlot(self, xvals, yvals, axesNames, categories, output_filename="", title=""):
        if self.SKIP_SWARM_PLOTS:
            print("Warning, skipping swarm plots!")
            return

        print("Swarm plot: {}".format(title))

        s = pd.Series([xvals, yvals], index=axesNames)
        s['cat'] = categories
        plt.clf()
        sns.swarmplot(x=axesNames[0], y=axesNames[1],
                      hue='cat', data=s, palette="Set3")
        plt.title(title)
        plt.xlabel(axesNames[0])
        plt.ylabel(axesNames[1])

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, "_".join(
                    axesNames[::-1]) + title)
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

    def makeASwarmPlotByWell(self, session, valFunc, title, outputFName=None):
        xs = np.arange(len(self.all_well_names))
        ys = [valFunc(w) for w in self.all_well_names]
        titleNoSpace = title.replace(" ", "_")
        axesNames = ["well_idx", titleNoSpace]
        if outputFName is None:
            outputFName = "well_idx_" + title + "_" + session.name
        wellCategory = [("home" if w == session.home_well else (
            "away" if w in session.visited_away_wells else "other")) for w in self.all_well_names]

        self.makeASwarmPlot(xs, ys, axesNames, wellCategory,
                            output_filename=outputFName, title=title + " - " + session.name)

    # def makeAPersevMeasurePlotJustHome(self, measure_name, datafunc, output_filename="", title="", doStats=True, scaleValue=None, yAxisLabel=None):
    #     sessions_with_all_wells = list(
    #         filter(lambda sesh: len(sesh.visited_away_wells) > 0, all_sessions))

    #     home_vals = [datafunc(sesh, sesh.home_well)
    #                  for sesh in sessions_with_all_wells]

    #     n = len(sessions_with_all_wells)
    #     axesNames = ["Session_Type", measure_name]
    #     assert len(home_vals) == n
    #     session_type = [trial_label(sesh) for sesh in sessions_with_all_wells]

    #     if scaleValue is not None:
    #         for i in range(len(home_vals)):
    #             home_vals[i] *= scaleValue

    #     s = pd.Series([session_type, home_vals], index=axesNames)
    #     plt.clf()
    #     sns.boxplot(x=axesNames[0], y=axesNames[1], data=s,
    #                 hue=axesNames[0], palette="Set3")
    #     plt.title(title)
    #     plt.xlabel(axesNames[0])
    #     if yAxisLabel is None:
    #         plt.ylabel(axesNames[1])
    #     else:
    #         plt.ylabel(yAxisLabel)

    #     if doStats:
    #         anovaModel = ols(
    #             axesNames[1] + " ~ C(Session_Type)", data=s).fit()
    #         anova_table = anova_lm(anovaModel, typ=1)
    #         print("============================\n" + measure_name + " ANOVA:")
    #         print(anova_table)
    #         print("n ctrl:", session_type.count("CTRL"))
    #         print("n swr:", session_type.count("SWR"))

    #     if SHOW_OUTPUT_PLOTS:
    #         plt.show()
    #     if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
    #         if len(output_filename) == 0:
    #             output_filename = os.path.join(output_dir, measure_name + title)
    #         else:
    #             output_filename = os.path.join(output_dir, output_filename)
    #         plt.savefig(output_filename, dpi=800)

    def makeAPersevMeasurePlot(self, measure_name, datafunc, output_filename="", title="", doStats=True, scaleValue=None, yAxisLabel=None, alsoMakePerWellPerSessionPlot=True, includeNoProbeSessions=False, yAxisLims=None):
        print("Persev measure plot: {}".format(measure_name))

        seshs = self.all_sessions if includeNoProbeSessions else self.all_sessions_with_probe
        sessions_with_all_wells_all = list(
            filter(lambda sesh: len(sesh.visited_away_wells) > 0, seshs))

        sessions_with_all_wells = [
            sesh for sesh in sessions_with_all_wells_all if sesh.date_str != "20211004"]
        if len(sessions_with_all_wells) != len(sessions_with_all_wells_all):
            print("WARNING=========\nExcluding 10/04\n=================================")

        home_vals = [datafunc(sesh, sesh.home_well)
                     for sesh in sessions_with_all_wells]

        def m(a):
            if np.all(np.isnan(a)):
                return np.nan
            return np.nanmean(a)
        away_vals = [m(np.array([datafunc(sesh, aw) for aw in sesh.visited_away_wells]))
                     for sesh in sessions_with_all_wells]
        other_vals = [m(np.array([datafunc(sesh, ow) for ow in set(
            self.all_well_names) - set(sesh.visited_away_wells) - set([sesh.home_well])])) for sesh in sessions_with_all_wells]

        ll = [self.trial_label(sesh) for sesh in sessions_with_all_wells]
        # print([e for i, e in enumerate(home_vals) if ll[i] == "CTRL"])
        # print(away_vals)
        # print(other_vals)

        # home_vals = [sesh.__dict__[measure_name][sesh.home_well_idx_in_allwells]
        #  for sesh in sessions_with_all_wells]
        # away_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][np.argmax(
        #     all_well_names == awi)] for awi in sesh.visited_away_wells])) for sesh in sessions_with_all_wells]
        # other_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][np.argmax(
        #     all_well_names == awi)] for awi in set(all_well_names) - set(sesh.visited_away_wells) - set([sesh.home_well])])) for sesh in sessions_with_all_wells]

        n = len(sessions_with_all_wells)
        axesNames = ["Well_type", measure_name, "Session_Type"]
        assert len(home_vals) == len(away_vals) and len(
            away_vals) == len(other_vals) and len(away_vals) == n
        categories = ["home"] * n + ["away"] * n + ["other"] * n
        session_type = [self.trial_label(sesh) for sesh in sessions_with_all_wells] * 3

        yvals = home_vals + away_vals + other_vals
        if scaleValue is not None:
            for i in range(len(yvals)):
                yvals[i] *= scaleValue

        # print("Unsorted:")
        # print(categories)
        # print(yvals)
        # print(session_type)

        # Sorting them all here by session type so that the hues for each type are the same regardless of which type of session came first
        # Have to append the index otherwise the sorting order is ambiguous
        sortList = [x + str(xi) for xi, x in enumerate(session_type)]
        categories = [x for _, x in sorted(zip(sortList, categories))]
        yvals = [x for _, x in sorted(zip(sortList, yvals))]
        session_type = [x for _, x in sorted(zip(sortList, session_type))]

        s = pd.Series([categories, yvals, session_type], index=axesNames)
        # print("sort list")
        # print(sortList)
        # print("Sorted:")
        # print(categories)
        # print(yvals)
        # print(session_type)

        plt.clf()
        sns.boxplot(x=axesNames[0], y=axesNames[1], data=s,
                    hue="Session_Type", palette="Set3")
        sns.swarmplot(x=axesNames[0], y=axesNames[1], data=s,
                      color="0.25", hue="Session_Type", dodge=True)

        plt.title(title)
        plt.xlabel(axesNames[0])
        if yAxisLabel is None:
            plt.ylabel(axesNames[1])
        else:
            plt.ylabel(yAxisLabel)

        if yAxisLims is not None:
            plt.ylim(yAxisLims)

        if doStats:
            self.statsFile.write("============================\n" +
                                 measure_name + " ANOVA:")
            if s.size == 0 or np.count_nonzero(np.logical_not(np.isnan(np.array(yvals)))) == 0:
                self.statsFile.write("No data ... can't do stats")
            else:
                # print(s)
                anovaModel = ols(
                    axesNames[1] + " ~ C(Well_type) + C(Session_Type) + C(Well_type):C(Session_Type)", data=s).fit()
                anova_table = anova_lm(anovaModel, typ=2)
                self.statsFile.write(str(anova_table))
            self.statsFile.write("n ctrl: " + str(session_type.count("CTRL") / 3))
            self.statsFile.write("n swr: " + str(session_type.count("SWR") / 3))
            self.statsFile.write("\n\n")

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, measure_name + title)
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

        if alsoMakePerWellPerSessionPlot:
            for sesh in sessions_with_all_wells:
                self.makeASwarmPlotByWell(sesh, lambda w: datafunc(
                    sesh, w), measure_name + "_by_well")

    def quadrantOfWell(self, well_idx):
        if well_idx > 24:
            res = 2
        else:
            res = 0

        if (well_idx - 1) % 8 >= 4:
            res += 1

        return res

    def quadrantsExceptWell(self, well_idx):
        well_quad = self.quadrantOfWell(well_idx)
        return list(set([0, 1, 2, 3]) - set([well_quad]))

    def makeAQuadrantPersevMeasurePlot(self, measure_name, datafunc, output_filename="", title="", doStats=True, includeNoProbeSessions=False):
        print("Quadrant persev measure plot: {}".format(measure_name))

        seshs = self.all_sessions if includeNoProbeSessions else self.all_sessions_with_probe
        home_vals = [datafunc(sesh, "Q" + str(self.quadrantOfWell(sesh.home_well)))
                     for sesh in seshs]
        other_vals = [np.nanmean(np.array([datafunc(sesh, "Q" + str(qi)) for qi in self.quadrantsExceptWell(sesh.home_well)]))
                      for sesh in seshs]

        # home_vals = [sesh.__dict__[measure_name][quadrantOfWell(sesh.home_well)]
        #              for sesh in self.all_sessions]
        # other_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][oq]
        #                                    for oq in quadrantsExceptWell(sesh.home_well)])) for sesh in self.all_sessions]

        n = len(seshs)
        axesNames = ["Quad_type", measure_name, "Session_Type"]
        assert len(home_vals) == len(other_vals) and len(home_vals) == n
        categories = ["home"] * n + ["other"] * n
        session_type = [self.trial_label(sesh) for sesh in seshs] * 2
        yvals = home_vals + other_vals
        s = pd.Series([categories, yvals, session_type], index=axesNames)
        plt.clf()
        sns.boxplot(x=axesNames[0], y=axesNames[1], data=s,
                    hue="Session_Type", palette="Set3")
        plt.title(title)
        plt.xlabel(axesNames[0])
        plt.ylabel(axesNames[1])

        if doStats:
            self.statsFile.write("============================\n" + measure_name + " ANOVA:")
            if s.size == 0 or np.count_nonzero(np.logical_not(np.isnan(np.array(yvals)))) == 0:
                self.statsFile.write("No data ... can't do stats")
            else:
                # print(s)
                anovaModel = ols(
                    axesNames[1] + " ~ C(Quad_type) + C(Session_Type) + C(Quad_type):C(Session_Type)", data=s).fit()
                anova_table = anova_lm(anovaModel, typ=2)
                self.statsFile.write(str(anova_table))
            self.statsFile.write("n ctrl: " + str(session_type.count("CTRL") / 2))
            self.statsFile.write("n swr: " + str(session_type.count("SWR") / 2))
            self.statsFile.write("\n\n")

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, measure_name + title)
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

    def makeABinaryPersevBiasPlot(self, measure_name, output_filename="", title="", includeNoProbeSessions=False):
        if SKIP_BINARY_PERSEVBIAS_PLOTS:
            print("Warning, skipping binary persev bias plots!")
            return

        print("binary persev measure plot: {}".format(measure_name))

        seshs = self.all_sessions if includeNoProbeSessions else self.all_sessions_with_probe

        MAX_NUM_AWAY = 9
        cnt_home_pos = 0
        cnt_home_total = 0
        cnt_other_pos = 0
        cnt_other_total = 0
        cnt_away_pos = np.zeros((MAX_NUM_AWAY,))
        cnt_away_total = np.zeros((MAX_NUM_AWAY,))
        num_missing_away = 0
        for sesh in seshs:
            if len(sesh.visited_away_wells) == 0:
                # print("Warning, session {} does not have visited away wells recorded".format(sesh.name))
                num_missing_away += 1
                continue
            for i, wi in enumerate(self.all_well_names):
                if wi == sesh.home_well:
                    if sesh.__dict__[measure_name][i] > 0:
                        cnt_home_pos += 1
                    cnt_home_total += 1
                elif wi in sesh.visited_away_wells:
                    for ei, awi in enumerate(sesh.visited_away_wells):
                        if awi == wi:
                            if sesh.__dict__[measure_name][i] > 0:
                                cnt_away_pos[ei] += 1
                            cnt_away_total[ei] += 1
                            break
                else:
                    if sesh.__dict__[measure_name][i] > 0:
                        cnt_other_pos += 1
                    cnt_other_total += 1

        # if num_missing_away > 0:
        #     print("{} of {} sessions missing away wells".format(num_missing_away, len(all_sessions)))

        # print(cnt_home_pos, cnt_home_total)
        # print(cnt_away_pos, cnt_away_total)
        # print(cnt_other_pos, cnt_other_total)
        xvals = np.arange(MAX_NUM_AWAY+2)
        yvals = np.concatenate((np.array([float(cnt_home_pos) / float(cnt_home_total)]),
                                (cnt_away_pos / cnt_away_total), np.array([float(cnt_other_pos) / float(cnt_other_total)])))
        axesNames = ["Well type", measure_name]
        s = pd.Series([xvals, yvals], index=axesNames)
        s['cat'] = ["Home"] + ["Away"] * MAX_NUM_AWAY + ["Other"]
        plt.clf()
        sns.swarmplot(x=axesNames[0], y=axesNames[1],
                      hue='cat', data=s, palette="Set2")
        plt.title(title)
        plt.xlabel(axesNames[0])
        plt.ylabel(axesNames[1])

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, measure_name + title)
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

    def makeAHistogram(self, yvals, categories, output_filename="", title=""):
        if self.SKIP_HISTOGRAMS:
            print("Warning, skipping histograms!")
            return

        print("histogram: {}".format(title))

        plt.clf()
        yva = np.array(yvals)
        cata = np.array(categories)
        ucats = sorted(list(set(categories)))
        for cat in ucats:
            cat_yvals = yva[cata == cat]
            try:
                sns.distplot(cat_yvals, label=cat)
            except:
                kde_kws = {'bw': 1.5}
                sns.distplot(cat_yvals, label=cat, kde_kws=kde_kws)
        plt.title(title)
        plt.legend()
        # plt.xlabel(axesNames[0])
        # plt.ylabel(axesNames[1])
        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, "_".join(ucats) + title)
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

    def trial_label(self, sesh):
        if sesh.isRippleInterruption:
            return "SWR"
        else:
            return "CTRL"

    def makeAPlotOverDays(self, datafunc, labels, fname, plotPast=False, includeNoProbeSessions=False):
        seshs = self.all_sessions if includeNoProbeSessions else self.all_sessions_with_probe
        sessions_with_all_wells = list(
            filter(lambda sesh: len(sesh.visited_away_wells) > 0, seshs))
        # print("Considering {} out of {} sessions that have all away wells listed".format(
        # len(sessions_with_all_wells), len(all_sessions)))

        print("plot over days: {}".format(fname))

        plt.clf()
        for si, sesh in enumerate(sessions_with_all_wells):
            hw = sesh.home_well
            if labels[si] == "SWR":
                color = "orange"
            else:
                color = "blue"

            if plotPast:
                i1 = 0
            else:
                i1 = si

            x = np.arange(i1, len(sessions_with_all_wells))
            y = [datafunc(s, hw) for s in sessions_with_all_wells[i1:]]
            plt.plot(x, y, c=color)
            plt.scatter(si, datafunc(sesh, hw), c=color)

        away_vals = [np.array([datafunc(s, aw) for aw in s.visited_away_wells])
                     for s in sessions_with_all_wells]
        other_vals = [np.array([datafunc(s, ow) for ow in set(
            self.all_well_names) - set(s.visited_away_wells) - set([s.home_well])]) for s in sessions_with_all_wells]
        away_means = [np.nanmean(v) for v in away_vals]
        other_means = [np.nanmean(v) for v in other_vals]
        away_std = [np.nanstd(v) for v in away_vals]
        other_std = [np.nanstd(v) for v in other_vals]

        for si in range(len(sessions_with_all_wells)):
            a1 = away_means[si] - away_std[si]
            a2 = away_means[si] + away_std[si]
            o1 = other_means[si] - other_std[si]
            o2 = other_means[si] + other_std[si]
            plt.plot([si + 0.15, si + 0.15], [a1, a2], c="black")
            plt.plot([si + 0.3, si + 0.3], [o1, o2], c="grey")

        saveOrShow(fname)
