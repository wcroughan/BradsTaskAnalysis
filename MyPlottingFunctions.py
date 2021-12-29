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
        self.tlbls = [self.trial_label(sesh) for sesh in self.all_sessions]

    def saveOrShow(self, fname):
        if self.SAVE_OUTPUT_PLOTS:
            plt.savefig(os.path.join(self.output_dir, fname), dpi=800)
        if self.SHOW_OUTPUT_PLOTS:
            plt.show()

    def makeASimpleBoxPlot(self, valFunc, title, yAxisName=None):
        if yAxisName is None:
            yAxisName = title
        self.makeABoxPlot([valFunc(sesh) for sesh in self.all_sessions],
                          self.tlbls, ["Condition", yAxisName], title=title)

    def makeABoxPlot(self, yvals, categories, axesNames, output_filename="", title="", doStats=True, scaleValue=None):
        if self.SKIP_BOX_PLOTS:
            print("Warning, skipping box plots!")
            return

        axesNamesNoSpaces = [a.replace(" ", "_") for a in axesNames]

        s = pd.Series([categories, yvals], index=axesNamesNoSpaces)
        plt.clf()
        print(s)
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
            print(s)
            anovaModel = ols(
                axesNamesNoSpaces[1] + " ~ C(" + axesNamesNoSpaces[0] + ")", data=s).fit()
            anova_table = anova_lm(anovaModel, typ=1)
            print("============================\n" + output_filename + " ANOVA:")
            print(anova_table)
            for cat in ucats:
                print(cat, "n = ", sum([i == cat for i in categories]))

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            plt.savefig(output_filename, dpi=800)

    def makeAScatterPlot(self, xvals, yvals, axesNames, categories=list(), output_filename="", title="", midline=False, makeLegend=False, ax=plt, bigDots=True):
        if self.SKIP_SCATTER_PLOTS:
            print("Warning, skipping scatter plots!")
            return

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

        for cat in ucats:
            print(cat, "n = ", sum([i == cat for i in categories]))

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

    def makeAPersevMeasurePlot(self, measure_name, datafunc, output_filename="", title="", doStats=True, scaleValue=None, yAxisLabel=None):
        sessions_with_all_wells = list(
            filter(lambda sesh: len(sesh.visited_away_wells) > 0, self.all_sessions))

        print("WARNING=========\nExcluding 10/04\n=================================")
        sessions_with_all_wells = [
            sesh for sesh in sessions_with_all_wells if sesh.date_str != "20211004"]

        home_vals = [datafunc(sesh, sesh.home_well)
                     for sesh in sessions_with_all_wells]
        away_vals = [np.nanmean(np.array([datafunc(sesh, aw) for aw in sesh.visited_away_wells]))
                     for sesh in sessions_with_all_wells]
        other_vals = [np.nanmean(np.array([datafunc(sesh, ow) for ow in set(
            self.all_well_names) - set(sesh.visited_away_wells) - set([sesh.home_well])])) for sesh in sessions_with_all_wells]

        ll = [self.trial_label(sesh) for sesh in sessions_with_all_wells]
        print([e for i, e in enumerate(home_vals) if ll[i] == "CTRL"])
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

        s = pd.Series([categories, yvals, session_type], index=axesNames)
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

        if doStats:
            anovaModel = ols(
                axesNames[1] + " ~ C(Well_type) + C(Session_Type) + C(Well_type):C(Session_Type)", data=s).fit()
            anova_table = anova_lm(anovaModel, typ=2)
            print("============================\n" + measure_name + " ANOVA:")
            print(anova_table)
            print("n ctrl:", session_type.count("CTRL") / 3)
            print("n swr:", session_type.count("SWR") / 3)

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, measure_name + title)
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

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

    def makeAQuadrantPersevMeasurePlot(self, measure_name, datafunc, output_filename="", title="", doStats=True):
        home_vals = [datafunc(sesh, "Q" + str(self.quadrantOfWell(sesh.home_well)))
                     for sesh in self.all_sessions]
        other_vals = [np.nanmean(np.array([datafunc(sesh, "Q" + str(qi)) for qi in self.quadrantsExceptWell(sesh.home_well)]))
                      for sesh in self.all_sessions]

        # home_vals = [sesh.__dict__[measure_name][quadrantOfWell(sesh.home_well)]
        #              for sesh in self.all_sessions]
        # other_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][oq]
        #                                    for oq in quadrantsExceptWell(sesh.home_well)])) for sesh in self.all_sessions]

        n = len(self.all_sessions)
        axesNames = ["Quad_type", measure_name, "Session_Type"]
        assert len(home_vals) == len(other_vals) and len(home_vals) == n
        categories = ["home"] * n + ["other"] * n
        session_type = [self.trial_label(sesh) for sesh in self.all_sessions] * 2
        yvals = home_vals + other_vals
        s = pd.Series([categories, yvals, session_type], index=axesNames)
        plt.clf()
        sns.boxplot(x=axesNames[0], y=axesNames[1], data=s,
                    hue="Session_Type", palette="Set3")
        plt.title(title)
        plt.xlabel(axesNames[0])
        plt.ylabel(axesNames[1])

        if doStats:
            anovaModel = ols(
                axesNames[1] + " ~ C(Quad_type) + C(Session_Type) + C(Quad_type):C(Session_Type)", data=s).fit()
            anova_table = anova_lm(anovaModel, typ=2)
            print("============================\n" + measure_name + " ANOVA:")
            print(anova_table)
            print("n ctrl:", session_type.count("CTRL") / 2)
            print("n swr:", session_type.count("SWR") / 2)

        if self.SHOW_OUTPUT_PLOTS:
            plt.show()
        if self.SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
            if len(output_filename) == 0:
                output_filename = os.path.join(self.output_dir, measure_name + title)
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_filename, dpi=800)

    def makeABinaryPersevBiasPlot(self, measure_name, output_filename="", title=""):
        if SKIP_BINARY_PERSEVBIAS_PLOTS:
            print("Warning, skipping binary persev bias plots!")
            return
        MAX_NUM_AWAY = 9
        cnt_home_pos = 0
        cnt_home_total = 0
        cnt_other_pos = 0
        cnt_other_total = 0
        cnt_away_pos = np.zeros((MAX_NUM_AWAY,))
        cnt_away_total = np.zeros((MAX_NUM_AWAY,))
        num_missing_away = 0
        for sesh in self.all_sessions:
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

    def makeAPlotOverDays(self, datafunc, labels, fname, plotPast=False):
        sessions_with_all_wells = list(
            filter(lambda sesh: len(sesh.visited_away_wells) > 0, self.all_sessions))
        # print("Considering {} out of {} sessions that have all away wells listed".format(
        # len(sessions_with_all_wells), len(all_sessions)))

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
