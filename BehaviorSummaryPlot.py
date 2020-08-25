from BTData import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data_filename = "/media/WDC4/martindata/bradtask/martin_bradtask.dat"
alldata = BTData()
alldata.loadFromFile(data_filename)
all_sessions = alldata.getSessions()

output_dir = '/media/WDC4/martindata/processed_data'

SHOW_OUTPUT_PLOTS = False
SAVE_OUTPUT_PLOTS = True

SKIP_BOX_PLOTS = False
SKIP_SCATTER_PLOTS = False
SKIP_SWARM_PLOTS = False
SKIP_PERSEVBIAS_PLOTS = False
SKIP_PERSEV_MEASURE_PLOTS = False
SKIP_BINARY_PERSEVBIAS_PLOTS = False
SKIP_HISTOGRAMS = False
SKIP_AVG_SPEED_COMP_PLOTS = False
SKIP_AVG_SPEED_PLOTS = False
SKIP_PERSEV_QUAD_PLOTS = False

TRODES_SAMPLING_RATE = 30000


def saveOrShow(fname):
    if SAVE_OUTPUT_PLOTS:
        plt.savefig(os.path.join(output_dir, fname), dpi=800)
    if SHOW_OUTPUT_PLOTS:
        plt.show()


def makeABoxPlot(yvals, categories, axesNames, output_filename="", title=""):
    if SKIP_BOX_PLOTS:
        print("Warning, skipping box plots!")
        return

    s = pd.Series([categories, yvals], index=axesNames)
    plt.clf()
    sns.boxplot(x=axesNames[0], y=axesNames[1], data=s, palette="Set3")
    plt.title(title)
    plt.xlabel(axesNames[0])
    plt.ylabel(axesNames[1])
    ucats = set(categories)
    for cat in ucats:
        print(cat, "n = ", sum([i == cat for i in categories]))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, "_".join(
                axesNames[::-1]) + "_" + "_".join(sorted(list(ucats))))
        else:
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=800)


def makeAScatterPlot(xvals, yvals, axesNames, categories=list(), output_filename="", title="", midline=False):
    if SKIP_SCATTER_PLOTS:
        print("Warning, skipping scatter plots!")
        return

    plt.clf()
    if len(categories) == 0:
        cvals = np.ones_like(xvals)
        ucats = list()
    else:
        ucats = np.array(sorted(list(set(categories))))
        cvals = [np.argmax(c == ucats) for c in categories]
    plt.scatter(xvals, yvals, c=cvals, zorder=2)
    plt.title(title)
    plt.xlabel(axesNames[0])
    plt.ylabel(axesNames[1])
    if midline:
        mval = max(max(xvals), max(yvals))
        plt.plot([0, mval], [0, mval], color='black', zorder=1)

    for cat in ucats:
        print(cat, "n = ", sum([i == cat for i in categories]))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            if len(categories) == 0:
                output_filename = os.path.join(output_dir, "_".join(
                    axesNames[::-1]))
            else:
                output_filename = os.path.join(output_dir, "_".join(
                    axesNames[::-1]) + "_" + "_".join(sorted(list(ucats))))
        else:
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=800)


def makeASwarmPlot(xvals, yvals, axesNames, categories, output_filename="", title=""):
    if SKIP_SWARM_PLOTS:
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

    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, "_".join(
                axesNames[::-1]) + title)
        else:
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=800)


def makeAPersevMeasurePlot(measure_name, output_filename="", title="", doStats=True):
    sessions_with_all_wells = list(
        filter(lambda sesh: len(sesh.visited_away_wells) > 0, all_sessions))

    home_vals = [sesh.__dict__[measure_name][sesh.home_well_idx_in_allwells]
                 for sesh in sessions_with_all_wells]
    away_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][np.argmax(
        all_well_idxs == awi)] for awi in sesh.visited_away_wells])) for sesh in sessions_with_all_wells]
    other_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][np.argmax(
        all_well_idxs == awi)] for awi in set(all_well_idxs) - set(sesh.visited_away_wells) - set([sesh.home_well])])) for sesh in sessions_with_all_wells]

    n = len(sessions_with_all_wells)
    axesNames = ["Well_type", measure_name, "Session_Type"]
    assert len(home_vals) == len(away_vals) and len(
        away_vals) == len(other_vals) and len(away_vals) == n
    categories = ["home"] * n + ["away"] * n + ["other"] * n
    session_type = [trial_label(sesh) for sesh in sessions_with_all_wells] * 3
    yvals = home_vals + away_vals + other_vals
    s = pd.Series([categories, yvals, session_type], index=axesNames)
    plt.clf()
    sns.boxplot(x=axesNames[0], y=axesNames[1], data=s, hue="Session_Type", palette="Set3")
    plt.title(title)
    plt.xlabel(axesNames[0])
    plt.ylabel(axesNames[1])

    if doStats:
        anovaModel = ols(
            axesNames[1] + " ~ C(Well_type) + C(Session_Type) + C(Well_type):C(Session_Type)", data=s).fit()
        anova_table = anova_lm(anovaModel, typ=2)
        print("============================\n" + measure_name + " ANOVA:")
        print(anova_table)
        print("n ctrl:", session_type.count("CTRL") / 3)
        print("n swr:", session_type.count("SWR") / 3)

    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, measure_name + title)
        else:
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=800)


def quadrantOfWell(well_idx):
    if well_idx > 24:
        res = 2
    else:
        res = 0

    if (well_idx - 1) % 8 >= 4:
        res += 1

    return res


def quadrantsExceptWell(well_idx):
    well_quad = quadrantOfWell(well_idx)
    return list(set([0, 1, 2, 3]) - set([well_quad]))


def makeAQuadrantPersevMeasurePlot(measure_name, output_filename="", title="", doStats=True):
    home_vals = [sesh.__dict__[measure_name][quadrantOfWell(sesh.home_well)]
                 for sesh in all_sessions]
    other_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][oq]
                                       for oq in quadrantsExceptWell(sesh.home_well)])) for sesh in all_sessions]

    n = len(all_sessions)
    axesNames = ["Quad_type", measure_name, "Session_Type"]
    assert len(home_vals) == len(other_vals) and len(home_vals) == n
    categories = ["home"] * n + ["other"] * n
    session_type = [trial_label(sesh) for sesh in all_sessions] * 2
    yvals = home_vals + other_vals
    s = pd.Series([categories, yvals, session_type], index=axesNames)
    plt.clf()
    sns.boxplot(x=axesNames[0], y=axesNames[1], data=s, hue="Session_Type", palette="Set3")
    plt.title(title)
    plt.xlabel(axesNames[0])
    plt.ylabel(axesNames[1])

    if doStats:
        anovaModel = ols(
            axesNames[1] + " ~ C(Quad_type) + C(Session_Type) + C(Quad_type):C(Session_Type)", data=s).fit()
        anova_table = anova_lm(anovaModel, typ=2)
        print("============================\n" + measure_name + " ANOVA:")
        print(anova_table)
        print("n ctrl:", session_type.count("CTRL") / 3)
        print("n swr:", session_type.count("SWR") / 3)

    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, measure_name + title)
        else:
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=800)


def makeABinaryPersevBiasPlot(measure_name, output_filename="", title=""):
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
    for sesh in all_sessions:
        if len(sesh.visited_away_wells) == 0:
            # print("Warning, session {} does not have visited away wells recorded".format(sesh.name))
            num_missing_away += 1
            continue
        for i, wi in enumerate(all_well_idxs):
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

    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, measure_name + title)
        else:
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=800)


def makeAHistogram(yvals, categories, output_filename="", title=""):
    if SKIP_HISTOGRAMS:
        print("Warning, skipping histograms!")
        return

    plt.clf()
    yva = np.array(yvals)
    cata = np.array(categories)
    ucats = sorted(list(set(categories)))
    for cat in ucats:
        cat_yvals = yva[cata == cat]
        sns.distplot(cat_yvals, label=cat)
    plt.title(title)
    plt.legend()
    # plt.xlabel(axesNames[0])
    # plt.ylabel(axesNames[1])
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, "_".join(ucats) + title)
        else:
            output_filename = os.path.join(output_dir, output_filename)
        plt.savefig(output_filename, dpi=800)


def trial_label(sesh):
    if sesh.isRippleInterruption:
        return "SWR"
    else:
        return "CTRL"


plt.figure()
tlbls = [trial_label(sesh) for sesh in all_sessions]

makeABoxPlot([sesh.probe_home_well_entry_times.size for sesh in all_sessions],
             tlbls, ['Condition', 'NumHomeEntries'], title="Probe num home entries")
makeABoxPlot([sesh.probe_ctrl_home_well_entry_times.size for sesh in all_sessions],
             tlbls, ['Condition', 'NumCtrlHomeEntries'], title="Probe num ctrl home entries")

makeAScatterPlot([sesh.bt_home_well_entry_times.size for sesh in all_sessions],
                 [sesh.probe_home_well_entry_times.size for sesh in all_sessions],
                 ['BT num home well entries', 'Probe num home well entries'], tlbls)
makeAScatterPlot([sesh.bt_ctrl_home_well_entry_times.size for sesh in all_sessions],
                 [sesh.probe_ctrl_home_well_entry_times.size for sesh in all_sessions],
                 ['BT num ctrl home well entries', 'Probe num ctrl home well entries'], tlbls)

makeAScatterPlot([sesh.probe_ctrl_home_well_entry_times.size for sesh in all_sessions],
                 [sesh.probe_home_well_entry_times.size for sesh in all_sessions],
                 ['Probe num ctrl home well entries', 'Probe num home well entries'], tlbls, midline=True)
makeAScatterPlot([sesh.probe_well_total_dwell_times_1min[sesh.home_well_idx_in_allwells]
                  for sesh in all_sessions],
                 [sesh.bt_well_total_dwell_times[sesh.home_well_idx_in_allwells]
                  for sesh in all_sessions],
                 ['Probe total home well dwell time', 'Task total home well dwell time'], tlbls, midline=True)

makeABoxPlot([sesh.probe_mean_dist_to_home_well for sesh in all_sessions],
             tlbls, ['Condition', 'MeanDistToHomeWell'], title="Probe Mean Dist to Home")
makeABoxPlot([sesh.probe_mean_dist_to_ctrl_home_well for sesh in all_sessions],
             tlbls, ['Condition', 'MeanDistToCtrlHomeWell'], title="Probe Mean Dist to Ctrl Home")

makeABoxPlot([sesh.probe_mv_mean_dist_to_home_well for sesh in all_sessions],
             tlbls, ['Condition', 'MoveMeanDistToHomeWell'], title="Probe (move) Mean Dist to Home")
makeABoxPlot([sesh.probe_mv_mean_dist_to_ctrl_home_well for sesh in all_sessions],
             tlbls, ['Condition', 'MoveMeanDistToCtrlHomeWell'], title="Probe (move) Mean Dist to Ctrl Home")

makeABoxPlot([sesh.probe_still_mean_dist_to_home_well for sesh in all_sessions],
             tlbls, ['Condition', 'StillMeanDistToHomeWell'], title="Probe (still) Mean Dist to Home")
makeABoxPlot([sesh.probe_still_mean_dist_to_ctrl_home_well for sesh in all_sessions],
             tlbls, ['Condition', 'StillMeanDistToCtrlHomeWell'], title="Probe (still) Mean Dist to Ctrl Home")
makeAScatterPlot([sesh.bt_mean_dist_to_home_well for sesh in all_sessions],
                 [sesh.probe_mean_dist_to_home_well for sesh in all_sessions],
                 ['BT Mean Dist to Home', 'Probe Mean Dist to Home'], tlbls)
makeAScatterPlot([sesh.bt_mean_dist_to_ctrl_home_well for sesh in all_sessions],
                 [sesh.probe_mean_dist_to_ctrl_home_well for sesh in all_sessions],
                 ['BT Mean Dist to ctrl Home', 'Probe Mean Dist to ctrl Home'], tlbls)


# ===================================
# Analysis: active exploration vs chilling, dwell times, etc
# ===================================
all_well_idxs = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])
for sesh in all_sessions:
    well_idxs = []
    well_dwell_times = []
    well_category = []
    for i, wi in enumerate(all_well_idxs):
        for j in range(len(sesh.probe_well_entry_times[i])):
            well_idxs.append(wi)
            dwell_time = (sesh.probe_well_exit_times[i][j] -
                          sesh.probe_well_entry_times[i][j]) / TRODES_SAMPLING_RATE
            # if dwell_time <= 0:
            #     print(i, wi, j, sesh.probe_well_entry_times[i][j], sesh.probe_well_exit_times[i][j],
            #           sesh.probe_well_entry_idxs[i][j], sesh.probe_well_exit_idxs[i][j])
            well_dwell_times.append(dwell_time)
            if wi == sesh.home_well:
                well_category.append("home")
            elif wi in sesh.visited_away_wells:
                well_category.append("away")
            else:
                well_category.append("other")

    axesNames = ['Well number', 'Dwell Time']
    title = 'Dwell time by well, Probe - ' + sesh.name
    fname = "Probe_dwelltime_" + sesh.name
    makeASwarmPlot(well_idxs, well_dwell_times, axesNames,
                   well_category, output_filename=fname, title=title)

all_quadrants = [0, 1, 2, 3]
for sesh in all_sessions:
    quadrant_idxs = []
    quadrant_dwell_times = []
    quadrant_category = []
    for i, wi in enumerate(all_quadrants):
        for j in range(len(sesh.probe_quadrant_entry_times[i])):
            quadrant_idxs.append(wi)
            dwell_time = (sesh.probe_quadrant_exit_times[i][j] -
                          sesh.probe_quadrant_entry_times[i][j]) / TRODES_SAMPLING_RATE
            # if dwell_time <= 0:
            #     print(i, wi, j, sesh.probe_well_entry_times[i][j], sesh.probe_well_exit_times[i][j],
            #           sesh.probe_well_entry_idxs[i][j], sesh.probe_well_exit_idxs[i][j])
            quadrant_dwell_times.append(dwell_time)
            if wi == sesh.home_quadrant:
                quadrant_category.append("home")
            else:
                quadrant_category.append("other")

    axesNames = ['Quadrant number', 'Dwell Time']
    title = 'Dwell time by Quadrant, Probe - ' + sesh.name
    fname = "Probe_dwelltime_" + sesh.name
    makeASwarmPlot(quadrant_idxs, quadrant_dwell_times, axesNames,
                   quadrant_category, output_filename=fname, title=title)


for sesh in all_sessions:
    well_idxs = []
    well_dwell_times = []
    well_category = []
    for i, wi in enumerate(all_well_idxs):
        for j in range(len(sesh.bt_well_entry_times[i])):
            well_idxs.append(wi)
            dwell_time = (sesh.bt_well_exit_times[i][j] -
                          sesh.bt_well_entry_times[i][j]) / TRODES_SAMPLING_RATE
            # if dwell_time <= 0:
            #     print(i, wi, j, sesh.bt_well_entry_times[i][j], sesh.bt_well_exit_times[i][j],
            #           sesh.bt_well_entry_idxs[i][j], sesh.bt_well_exit_idxs[i][j])
            well_dwell_times.append(dwell_time)
            if wi == sesh.home_well:
                well_category.append("home")
            elif wi in sesh.visited_away_wells:
                well_category.append("away")
            else:
                well_category.append("other")

    axesNames = ['Well number', 'Dwell Time']
    title = 'Dwell time by well, BT - ' + sesh.name
    fname = "BT_dwelltime_" + sesh.name
    makeASwarmPlot(well_idxs, well_dwell_times, axesNames,
                   well_category, output_filename=fname, title=title)

# ===================================
# Perseveration bias measures
# ===================================
if SKIP_PERSEVBIAS_PLOTS:
    print("Warning skipping Persev bias plots!")
else:
    makeAPersevMeasurePlot("bt_persev_bias_mean_dist_to_well")
    makeAPersevMeasurePlot("bt_persev_bias_num_entries_to_well")
    makeAPersevMeasurePlot("bt_persev_bias_total_dwell_time")
    makeAPersevMeasurePlot("bt_persev_bias_avg_dwell_time")
    makeAPersevMeasurePlot("bt_persev_bias_total_dwell_time_excluding_reward")
    makeAPersevMeasurePlot("bt_persev_bias_avg_dwell_time_excluding_reward")
    makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well")
    makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well")
    makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time")
    makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time")
    makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well_1min")
    makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well_1min")
    makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time_1min")
    makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time_1min")
    makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well_30sec")
    makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well_30sec")
    makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time_30sec")
    makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time_30sec")

# ===================================
# Perseveration measures
# ===================================
if SKIP_PERSEV_MEASURE_PLOTS:
    print("Warning skipping Persev measure plots!")
else:
    makeAPersevMeasurePlot("bt_mean_dist_to_wells")
    makeAPersevMeasurePlot("bt_median_dist_to_wells")
    makeAPersevMeasurePlot("probe_mean_dist_to_wells")
    makeAPersevMeasurePlot("probe_median_dist_to_wells")
    makeAPersevMeasurePlot("probe_mean_dist_to_wells_1min")
    makeAPersevMeasurePlot("probe_median_dist_to_wells_1min")
    makeAPersevMeasurePlot("probe_mean_dist_to_wells_30sec")
    makeAPersevMeasurePlot("probe_median_dist_to_wells_30sec")
    makeAPersevMeasurePlot("bt_well_num_entries")
    makeAPersevMeasurePlot("bt_well_total_dwell_times")
    makeAPersevMeasurePlot("bt_well_avg_dwell_times")
    makeAPersevMeasurePlot("bt_well_total_dwell_times_excluding_reward")
    makeAPersevMeasurePlot("bt_well_avg_dwell_times_excluding_reward")
    makeAPersevMeasurePlot("probe_well_num_entries")
    makeAPersevMeasurePlot("probe_well_total_dwell_times")
    makeAPersevMeasurePlot("probe_well_avg_dwell_times")
    makeAPersevMeasurePlot("probe_well_num_entries_1min")
    makeAPersevMeasurePlot("probe_well_total_dwell_times_1min")
    makeAPersevMeasurePlot("probe_well_avg_dwell_times_1min")
    makeAPersevMeasurePlot("probe_well_num_entries_30sec")
    makeAPersevMeasurePlot("probe_well_total_dwell_times_30sec")
    makeAPersevMeasurePlot("probe_well_avg_dwell_times_30sec")
    makeAPersevMeasurePlot("bt_well_num_entries_ninc")
    makeAPersevMeasurePlot("bt_well_total_dwell_times_ninc")
    makeAPersevMeasurePlot("bt_well_avg_dwell_times_ninc")
    makeAPersevMeasurePlot("bt_well_total_dwell_times_excluding_reward_ninc")
    makeAPersevMeasurePlot("bt_well_avg_dwell_times_excluding_reward_ninc")
    makeAPersevMeasurePlot("probe_well_num_entries_ninc")
    makeAPersevMeasurePlot("probe_well_total_dwell_times_ninc")
    makeAPersevMeasurePlot("probe_well_avg_dwell_times_ninc")
    makeAPersevMeasurePlot("probe_well_num_entries_1min_ninc")
    makeAPersevMeasurePlot("probe_well_total_dwell_times_1min_ninc")
    makeAPersevMeasurePlot("probe_well_avg_dwell_times_1min_ninc")
    makeAPersevMeasurePlot("probe_well_num_entries_30sec_ninc")
    makeAPersevMeasurePlot("probe_well_total_dwell_times_30sec_ninc")
    makeAPersevMeasurePlot("probe_well_avg_dwell_times_30sec_ninc")


if SKIP_PERSEV_QUAD_PLOTS:
    print("Warning skipping quad plots!")
else:
    makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries_1min")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times_1min")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times_1min")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries_30sec")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times_30sec")
    makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times_30sec")

# ===================================
# What do dwell times look like?
# ===================================
dt_vals = [dt for sesh in all_sessions for dtlist in sesh.probe_dwell_times for dt in dtlist]
makeAHistogram(dt_vals, categories=["all"] * len(dt_vals), title="dwell times")

# ===================================
# Avg speed
# ===================================


def calcAvgSpeed(sesh, start, stop):
    if start <= 3 and stop <= 3:
        # This version uses 0-1 to symbolize times in task, 2-3 for times in probe
        # Just calls recursively using actual times
        if start <= 1:
            assert start >= 0 and stop <= 1 and stop >= 0
            xp = [0, 1]
            fp = [sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]]
        else:
            assert start >= 2 and start <= 3 and stop >= 2 and stop <= 3
            xp = [2, 3]
            fp = [sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]]
        start_t = np.interp(start, xp, fp)
        stop_t = np.interp(stop, xp, fp)
        return calcAvgSpeed(sesh, start_t, stop_t)
    if start < sesh.probe_pos_ts[0]:
        tref = sesh.bt_pos_ts[0:-1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        return np.mean(sesh.bt_vel_cm_s[start_i:stop_i])
    else:
        tref = sesh.probe_pos_ts[0:-1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        return np.mean(sesh.probe_vel_cm_s[start_i:stop_i])


def avg_speed_plot(interval, output_filename=""):
    if SKIP_AVG_SPEED_PLOTS:
        print("Warning: skipping avg speed comp plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0] for sesh in all_sessions])
    max_probe_dur = np.max([sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
    istep = interval * TRODES_SAMPLING_RATE
    bt_num_speeds = int(np.floor(max_bt_dur / istep))
    probe_num_speeds = int(np.floor(max_probe_dur / istep))

    all_bt_speeds = np.empty((len(all_sessions), bt_num_speeds))
    all_probe_speeds = np.empty((len(all_sessions), probe_num_speeds))
    all_bt_speeds[:] = np.nan
    all_probe_speeds[:] = np.nan

    for si, sesh in enumerate(all_sessions):
        bt_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                     for i1 in np.arange(sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]-istep, istep)]
        probe_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                        for i1 in np.arange(sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]-istep, istep)]

        all_bt_speeds[si, 0:len(bt_speeds)] = bt_speeds
        all_probe_speeds[si, 0:len(probe_speeds)] = probe_speeds

    bt_mvals = np.nanmean(all_bt_speeds, axis=0)
    probe_mvals = np.nanmean(all_probe_speeds, axis=0)
    bt_sems = np.nanstd(all_bt_speeds, axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(np.isnan(all_bt_speeds)), axis=0))
    probe_sems = np.nanstd(all_probe_speeds, axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(np.isnan(all_probe_speeds)), axis=0))

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * bt_num_speeds, bt_num_speeds), bt_mvals, bt_sems)
    plt.title("BT Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, "bt_avg_speed_{}".format(interval))
        else:
            output_filename = os.path.join(output_dir, output_filename + "_bt")
        plt.savefig(output_filename, dpi=800)

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * probe_num_speeds,
                             probe_num_speeds), probe_mvals, probe_sems)
    plt.title("Probe Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, "probe_avg_speed_{}".format(interval))
        else:
            output_filename = os.path.join(output_dir, output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


avg_speed_plot(30)
avg_speed_plot(60)
avg_speed_plot(300)


def avg_speed_plot_separate_conditions(interval, output_filename=""):
    if SKIP_AVG_SPEED_COMP_PLOTS:
        print("Warning: skipping avg speed comp plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0] for sesh in all_sessions])
    max_probe_dur = np.max([sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
    istep = interval * TRODES_SAMPLING_RATE
    bt_num_speeds = int(np.floor(max_bt_dur / istep))
    probe_num_speeds = int(np.floor(max_probe_dur / istep))

    all_bt_speeds = np.empty((len(all_sessions), bt_num_speeds))
    all_probe_speeds = np.empty((len(all_sessions), probe_num_speeds))
    all_bt_speeds[:] = np.nan
    all_probe_speeds[:] = np.nan
    session_type = np.empty((len(all_sessions), ))

    for si, sesh in enumerate(all_sessions):
        bt_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                     for i1 in np.arange(sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]-istep, istep)]
        probe_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                        for i1 in np.arange(sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]-istep, istep)]

        all_bt_speeds[si, 0:len(bt_speeds)] = bt_speeds
        all_probe_speeds[si, 0:len(probe_speeds)] = probe_speeds
        if sesh.isRippleInterruption:
            session_type[si] = 1
        else:
            session_type[si] = 0

    # print(all_bt_speeds)
    # print(all_bt_speeds[np.where(session_type == 1), :])
    # print(session_type)
    bt_mvals_swr = np.nanmean(all_bt_speeds[np.where(session_type == 1), :][0], axis=0)
    bt_mvals_ctrl = np.nanmean(all_bt_speeds[np.where(session_type == 0), :][0], axis=0)
    probe_mvals_swr = np.nanmean(all_probe_speeds[np.where(session_type == 1), :][0], axis=0)
    probe_mvals_ctrl = np.nanmean(all_probe_speeds[np.where(session_type == 0), :][0], axis=0)
    bt_sems_swr = np.nanstd(all_bt_speeds[np.where(session_type == 1), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_bt_speeds[np.where(session_type == 0), :][0])), axis=0))
    bt_sems_ctrl = np.nanstd(all_bt_speeds[np.where(session_type == 0), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_bt_speeds[np.where(session_type == 0), :][0])), axis=0))
    probe_sems_swr = np.nanstd(all_probe_speeds[np.where(session_type == 1), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_probe_speeds[np.where(session_type == 0), :][0])), axis=0))
    probe_sems_ctrl = np.nanstd(all_probe_speeds[np.where(session_type == 0), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_probe_speeds[np.where(session_type == 0), :][0])), axis=0))

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * bt_num_speeds,
                             bt_num_speeds), bt_mvals_swr, bt_sems_swr)
    plt.errorbar(np.linspace(interval, interval * bt_num_speeds,
                             bt_num_speeds), bt_mvals_ctrl, bt_sems_ctrl)
    plt.title("BT Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    plt.legend(["SWR", "CTRL"])
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "bt_avg_speed_{}_comparison".format(interval))
        else:
            output_filename = os.path.join(output_dir, output_filename + "_bt")
        plt.savefig(output_filename, dpi=800)

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * probe_num_speeds,
                             probe_num_speeds), probe_mvals_swr, probe_sems_swr)
    plt.errorbar(np.linspace(interval, interval * probe_num_speeds,
                             probe_num_speeds), probe_mvals_ctrl, probe_sems_ctrl)
    plt.title("Probe Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    plt.legend(["SWR", "CTRL"])
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "probe_avg_speed_{}_comparison".format(interval))
        else:
            output_filename = os.path.join(output_dir, output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


avg_speed_plot_separate_conditions(30)
avg_speed_plot_separate_conditions(60)
