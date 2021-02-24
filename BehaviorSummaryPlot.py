# TODO
# ANOVA spherical assumption is not met because spread of other < spread of away < spread of home
#   Need to do correct statistical test
# Reversal measure: how often visited wells ABC vs ABA when A is home or etc
#
# Comparisons to do:
# dwell time during tasks excluding reward
# num home checks excluding reward (maybe over time? Some way to account for probably increasing checks over course of run?)
#
# Model start of probe as [pause] -> run -> search. Isolate search parth, draw all paths cetered around home, color by start->finish of search path


from BTData import *
import sortednp
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data_filename = "/media/WDC1/martindata/bradtask/martin_bradtask.dat"
alldata = BTData()
alldata.loadFromFile(data_filename)
all_sessions = alldata.getSessions()

output_dir = '/media/WDC1/martindata/processed_data'

SHOW_OUTPUT_PLOTS = False
SAVE_OUTPUT_PLOTS = True

SKIP_BOX_PLOTS = False
SKIP_SCATTER_PLOTS = True
SKIP_SWARM_PLOTS = True
SKIP_PERSEVBIAS_PLOTS = True
SKIP_PERSEV_MEASURE_PLOTS = False
SKIP_PERSEV_BOX_PLOTS = True
SKIP_BINARY_PERSEVBIAS_PLOTS = True
SKIP_BOUT_PLOTS = True
SKIP_WAIT_PLOTS = True
SKIP_BALL_PLOTS = True
SKIP_CURVATURE_PLOTS = True
SKIP_HISTOGRAMS = True
SKIP_LINE_PLOTS = True
SKIP_AVG_SPEED_COMP_PLOTS = True
SKIP_AVG_SPEED_PLOTS = True
SKIP_PERSEV_QUAD_PLOTS = True
SKIP_BOUT_PROP_PLOTS = True
SKIP_HW_PLOT = True
SKIP_ORDER_PLOTS = True
SKIP_RIPPLE_PLACE_PLOTS = True
SKIP_EVERY_MINUTE_PLOTS = False

PRINT_TRIAL_INFO = False
SKIP_TO_MY_LOU_DARLIN = True

TRODES_SAMPLING_RATE = 30000

all_well_names = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])


def saveOrShow(fname):
    if SAVE_OUTPUT_PLOTS:
        plt.savefig(os.path.join(output_dir, fname), dpi=800)
    if SHOW_OUTPUT_PLOTS:
        plt.show()


def makeABoxPlot(yvals, categories, axesNames, output_filename="", title="", doStats=True, scaleValue=None):
    if SKIP_BOX_PLOTS:
        print("Warning, skipping box plots!")
        return

    s = pd.Series([categories, yvals], index=axesNames)
    plt.clf()
    print(s)
    sns.boxplot(x=axesNames[0], y=axesNames[1], data=s, palette="Set3")
    plt.title(title)
    plt.xlabel(axesNames[0])
    plt.ylabel(axesNames[1])
    ucats = set(categories)

    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(output_dir, "_".join(
                axesNames[::-1]) + "_" + "_".join(sorted(list(ucats))))
        else:
            output_filename = os.path.join(output_dir, output_filename)

    if doStats:
        if scaleValue is not None:
            for i in range(len(yvals)):
                yvals[i] *= scaleValue

        s = pd.Series([categories, yvals], index=axesNames)
        anovaModel = ols(
            axesNames[1] + " ~ C(" + axesNames[0] + ")", data=s).fit()
        anova_table = anova_lm(anovaModel, typ=1)
        print("============================\n" + output_filename + " ANOVA:")
        print(anova_table)
        for cat in ucats:
            print(cat, "n = ", sum([i == cat for i in categories]))

    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
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
    plt.scatter(xvals, yvals, c=cvals, zorder=2, s=plt.rcParams['lines.markersize'] ** 2 * 3)
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


# def makeAPersevMeasurePlotJustHome(measure_name, datafunc, output_filename="", title="", doStats=True, scaleValue=None, yAxisLabel=None):
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


def makeAPersevMeasurePlot(measure_name, datafunc, output_filename="", title="", doStats=True, scaleValue=None, yAxisLabel=None):
    sessions_with_all_wells = list(
        filter(lambda sesh: len(sesh.visited_away_wells) > 0, all_sessions))

    home_vals = [datafunc(sesh, sesh.home_well)
                 for sesh in sessions_with_all_wells]
    away_vals = [np.nanmean(np.array([datafunc(sesh, aw) for aw in sesh.visited_away_wells]))
                 for sesh in sessions_with_all_wells]
    other_vals = [np.nanmean(np.array([datafunc(sesh, ow) for ow in set(
        all_well_names) - set(sesh.visited_away_wells) - set([sesh.home_well])])) for sesh in sessions_with_all_wells]

    ll = [trial_label(sesh) for sesh in sessions_with_all_wells]
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
    session_type = [trial_label(sesh) for sesh in sessions_with_all_wells] * 3

    yvals = home_vals + away_vals + other_vals
    if scaleValue is not None:
        for i in range(len(yvals)):
            yvals[i] *= scaleValue

    s = pd.Series([categories, yvals, session_type], index=axesNames)
    plt.clf()
    sns.boxplot(x=axesNames[0], y=axesNames[1], data=s,
                hue="Session_Type", palette="Set3")
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


def makeAQuadrantPersevMeasurePlot(measure_name, datafunc, output_filename="", title="", doStats=True):
    home_vals = [datafunc(sesh, "Q" + str(quadrantOfWell(sesh.home_well)))
                 for sesh in all_sessions]
    other_vals = [np.nanmean(np.array([datafunc(sesh, "Q" + str(qi)) for qi in quadrantsExceptWell(sesh.home_well)]))
                  for sesh in all_sessions]

    # home_vals = [sesh.__dict__[measure_name][quadrantOfWell(sesh.home_well)]
    #              for sesh in all_sessions]
    # other_vals = [np.nanmean(np.array([sesh.__dict__[measure_name][oq]
    #                                    for oq in quadrantsExceptWell(sesh.home_well)])) for sesh in all_sessions]

    n = len(all_sessions)
    axesNames = ["Quad_type", measure_name, "Session_Type"]
    assert len(home_vals) == len(other_vals) and len(home_vals) == n
    categories = ["home"] * n + ["other"] * n
    session_type = [trial_label(sesh) for sesh in all_sessions] * 2
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
        for i, wi in enumerate(all_well_names):
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
        try:
            sns.distplot(cat_yvals, label=cat)
        except:
            kde_kws = {'bw': 1.5}
            sns.distplot(cat_yvals, label=cat, kde_kws=kde_kws)
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

if PRINT_TRIAL_INFO:
    for i, s in enumerate(all_sessions):
        print(i, s.name, tlbls[i])
    exit()


# Quick graphs made during lab meeting checking how task behavior and perseveration relate:
# Todo organize these in somewhere
if False:
    makeAScatterPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     [sesh.total_dwell_time(False, sesh.home_well)
                      for sesh in all_sessions],
                     ['Probe avg home dwell time 1min', 'Task total home well dwell time'], tlbls, midline=False)
    makeAScatterPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     [sesh.num_well_entries(False, sesh.home_well) for sesh in all_sessions],
                     ['Probe avg home dwell time 1min', 'Task num home well entries'], tlbls, midline=False)
    makeAScatterPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     [len(sesh.bt_pos_ts) for sesh in all_sessions],
                     ['Probe avg home dwell time 1min', 'Task length'], tlbls, midline=False)

    exit()


if not SKIP_TO_MY_LOU_DARLIN:
    makeAScatterPlot([sesh.total_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(False)
                      for sesh in all_sessions],
                     ['Probe total home dwell time 1min', 'Task Curvature at Home'], tlbls, midline=False)
    makeAScatterPlot([sesh.avg_curvature_at_home_well(True,  timeInterval=[0, 60])
                      for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(False)
                      for sesh in all_sessions],
                     ['Probe curvature at Home 1min', 'Task Curvature at Home'], tlbls, midline=False)

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

    makeAScatterPlot([sesh.total_dwell_time(True, sesh.home_well, [0, 60])
                      for sesh in all_sessions],
                     [sesh.total_dwell_time(False, sesh.home_well)
                         for sesh in all_sessions],
                     ['Probe total home well dwell time', 'Task total home well dwell time'], tlbls, midline=True)
    # makeAScatterPlot([sesh.total_dwell_time(True, sesh.home_well, [0, 60])
    # makeABoxPlot([sesh.probe_mean_dist_to_home_well for sesh in all_sessions],
    makeABoxPlot([sesh.avg_dist_to_home_well(True) for sesh in all_sessions],
                 tlbls, ['Condition', 'MeanDistToHomeWell'], title="Probe Mean Dist to Home")
    # makeABoxPlot([sesh.probe_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well) for sesh in all_sessions],
                 tlbls, ['Condition', 'MeanDistToCtrlHomeWell'], title="Probe Mean Dist to Ctrl Home")

    # makeABoxPlot([sesh.probe_mv_mean_dist_to_home_well for sesh in all_sessions],
    makeABoxPlot([sesh.avg_dist_to_home_well(True, moveFlag=BTSession.MOVE_FLAG_MOVING) for sesh in all_sessions],
                 tlbls, ['Condition', 'MoveMeanDistToHomeWell'], title="Probe (move) Mean Dist to Home")
    # makeABoxPlot([sesh.probe_mv_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well, moveFlag=BTSession.MOVE_FLAG_MOVING) for sesh in all_sessions],
                 tlbls, ['Condition', 'MoveMeanDistToCtrlHomeWell'], title="Probe (move) Mean Dist to Ctrl Home")

    # makeABoxPlot([sesh.probe_still_mean_dist_to_home_well for sesh in all_sessions],
    makeABoxPlot([sesh.avg_dist_to_home_well(True, moveFlag=BTSession.MOVE_FLAG_STILL) for sesh in all_sessions],
                 tlbls, ['Condition', 'StillMeanDistToHomeWell'], title="Probe (still) Mean Dist to Home")
    # makeABoxPlot([sesh.probe_still_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well, moveFlag=BTSession.MOVE_FLAG_STILL) for sesh in all_sessions],
                 tlbls, ['Condition', 'StillMeanDistToCtrlHomeWell'], title="Probe (still) Mean Dist to Ctrl Home")
    # makeAScatterPlot([sesh.bt_mean_dist_to_home_well for sesh in all_sessions],
    makeAScatterPlot([sesh.avg_dist_to_home_well(False) for sesh in all_sessions],
                     #  [sesh.probe_mean_dist_to_home_well for sesh in all_sessions],
                     [sesh.avg_dist_to_home_well(True) for sesh in all_sessions],
                     ['BT Mean Dist to Home', 'Probe Mean Dist to Home'], tlbls)
    # makeAScatterPlot([sesh.bt_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    makeAScatterPlot([sesh.avg_dist_to_well(False, sesh.ctrl_home_well) for sesh in all_sessions],
                     #  [sesh.probe_mean_dist_to_ctrl_home_well for sesh in all_sessions],
                     [sesh.avg_dist_to_well(True, sesh.ctrl_home_well) for sesh in all_sessions],
                     ['BT Mean Dist to ctrl Home', 'Probe Mean Dist to ctrl Home'], tlbls)

    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 10sec mean vel', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 30sec mean vel', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 60sec mean vel', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 120sec mean vel', 'probe avg home dwell time 90sec'], tlbls)

    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)

    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)

    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 10sec mean vel', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 30sec mean vel', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 60sec mean vel', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec mean vel', 'probe avg home curvature 90sec'], tlbls)

    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 10sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 30sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 60sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 120sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)

    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 10sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 10sec explore proportion', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 30sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 30sec explore proportion', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 60sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 60sec explore proportion', 'probe avg home curvature 90sec'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                      for sesh in all_sessions],
                     ['Probe 120sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                     [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                      for sesh in all_sessions],
                     ['Probe 120sec explore proportion', 'probe avg home curvature 90sec'], tlbls)


# ===================================
# Analysis: active exploration vs chilling, dwell times, etc
# ===================================
if SKIP_SWARM_PLOTS:
    print("Warning, skipping swarm plots!")
else:
    for sesh in all_sessions:
        well_idxs = []
        well_dwell_times = []
        well_category = []
        for i, wi in enumerate(all_well_names):
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
        fname = "Probe_quad_dwelltime_" + sesh.name
        makeASwarmPlot(quadrant_idxs, quadrant_dwell_times, axesNames,
                       quadrant_category, output_filename=fname, title=title)

    for sesh in all_sessions:
        well_idxs = []
        well_dwell_times = []
        well_category = []
        for i, wi in enumerate(all_well_names):
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
    if not SKIP_TO_MY_LOU_DARLIN:
        makeAPersevMeasurePlot("bt_persev_bias_mean_dist_to_well",
                               lambda s, w: s.avg_dist_to_well(False, w) - s.avg_dist_to_well(False, s.ctrl_well_for_well(w)))
        makeAPersevMeasurePlot("bt_persev_bias_num_entries_to_well",
                               lambda s, w: s.num_well_entries(False, w) - s.num_well_entries(False, s.ctrl_well_for_well(w)))
        makeAPersevMeasurePlot("bt_persev_bias_total_dwell_time",
                               lambda s, w: s.total_dwell_time(False, w) - s.total_dwell_time(False, s.ctrl_well_for_well(w)))
        makeAPersevMeasurePlot("bt_persev_bias_avg_dwell_time",
                               lambda s, w: s.avg_dwell_time(False, w) - s.avg_dwell_time(False, s.ctrl_well_for_well(w)))
        makeAPersevMeasurePlot("bt_persev_bias_total_dwell_time_excluding_reward",
                               lambda s, w: s.total_dwell_time(False, w, excludeReward=True) - s.total_dwell_time(False, s.ctrl_well_for_well(w), excludeReward=True))
        makeAPersevMeasurePlot("bt_persev_bias_avg_dwell_time_excluding_reward",
                               lambda s, w: s.avg_dwell_time(False, w, excludeReward=True) - s.avg_dwell_time(False, s.ctrl_well_for_well(w), excludeReward=True))
        makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well",
                               lambda s, w: s.avg_dist_to_well(True, w) - s.avg_dist_to_well(True, s.ctrl_well_for_well(w)))
        makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well",
                               lambda s, w: s.num_well_entries(True, w) - s.num_well_entries(True, s.ctrl_well_for_well(w)))
        makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time",
                               lambda s, w: s.total_dwell_time(True, w) - s.total_dwell_time(True, s.ctrl_well_for_well(w)))
        makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time",
                               lambda s, w: s.avg_dwell_time(True, w) - s.avg_dwell_time(True, s.ctrl_well_for_well(w)))

    makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well_1min",
                           lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 60]) - s.avg_dist_to_well(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well_1min",
                           lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]) - s.num_well_entries(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time_1min",
                           lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]) - s.total_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time_1min",
                           lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]) - s.avg_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well_30sec",
                           lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 30]) - s.avg_dist_to_well(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))
    makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well_30sec",
                           lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30]) - s.num_well_entries(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))
    makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time_30sec",
                           lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30]) - s.total_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))
    makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time_30sec",
                           lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30]) - s.avg_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))

# ===================================
# Perseveration measures
# ===================================
if SKIP_PERSEV_MEASURE_PLOTS:
    print("Warning skipping Persev measure plots!")
else:

    if not SKIP_TO_MY_LOU_DARLIN:
        makeAPersevMeasurePlot("probe_num_entries_to_well",
                               lambda s, w: s.num_well_entries(True, w))
        makeAPersevMeasurePlot("probe_num_entries_to_well_1min",
                               lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
        # makeAPersevMeasurePlot("probe_well_total_dwell_times_1min", scaleValue=1.0 /
        #    float(TRODES_SAMPLING_RATE), yAxisLabel="Probe 1st Min Total Dwell Time (sec)")
        # makeAPersevMeasurePlot("probe_well_avg_dwell_times_1min", scaleValue=1.0 /
        #    float(TRODES_SAMPLING_RATE), yAxisLabel="Probe 1st Min Average Dwell Time (sec)")
        makeAPersevMeasurePlot("bt_mean_dist_to_wells",
                               lambda s, w: s.avg_dist_to_well(False, w))
        makeAPersevMeasurePlot("bt_median_dist_to_wells",
                               lambda s, w: s.avg_dist_to_well(False, w, avgFunc=np.nanmedian))
        makeAPersevMeasurePlot("bt_num_entries_to_well",
                               lambda s, w: s.num_well_entries(False, w))
        makeAPersevMeasurePlot("bt_total_dwell_time",
                               lambda s, w: s.total_dwell_time(False, w))
        makeAPersevMeasurePlot("bt_avg_dwell_time",
                               lambda s, w: s.avg_dwell_time(False, w))
        makeAPersevMeasurePlot("bt_total_dwell_time_excluding_reward",
                               lambda s, w: s.total_dwell_time(False, w, excludeReward=True))
        makeAPersevMeasurePlot("bt_avg_dwell_time_excluding_reward",
                               lambda s, w: s.avg_dwell_time(False, w, excludeReward=True))
        makeAPersevMeasurePlot("probe_mean_dist_to_well",
                               lambda s, w: s.avg_dist_to_well(True, w))
        makeAPersevMeasurePlot("probe_median_dist_to_well",
                               lambda s, w: s.avg_dist_to_well(True, w, avgFunc=np.nanmedian))
        makeAPersevMeasurePlot("probe_num_entries_to_well",
                               lambda s, w: s.num_well_entries(True, w))
        makeAPersevMeasurePlot("probe_total_dwell_time",
                               lambda s, w: s.total_dwell_time(True, w))
        makeAPersevMeasurePlot("probe_avg_dwell_time",
                               lambda s, w: s.avg_dwell_time(True, w))
        makeAPersevMeasurePlot("probe_mean_dist_to_well_1min",
                               lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 60]))
        makeAPersevMeasurePlot("probe_median_dist_to_well_1min",
                               lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 60], avgFunc=np.nanmedian))
        makeAPersevMeasurePlot("probe_num_entries_to_well_1min",
                               lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
        makeAPersevMeasurePlot("probe_total_dwell_time_1min",
                               lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
        makeAPersevMeasurePlot("probe_avg_dwell_time_1min",
                               lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))
        makeAPersevMeasurePlot("probe_mean_dist_to_well_30sec",
                               lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 30]))
        makeAPersevMeasurePlot("probe_median_dist_to_well_30sec",
                               lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 30], avgFunc=np.nanmedian))
        makeAPersevMeasurePlot("probe_num_entries_to_well_30sec",
                               lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30]))
        makeAPersevMeasurePlot("probe_total_dwell_time_30sec",
                               lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30]))
        makeAPersevMeasurePlot("probe_avg_dwell_time_30sec",
                               lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30]))

        makeAPersevMeasurePlot("bt_num_entries_to_well_ninc",
                               lambda s, w: s.num_well_entries(False, w, includeNeighbors=True))
        makeAPersevMeasurePlot("bt_total_dwell_time_ninc",
                               lambda s, w: s.total_dwell_time(False, w, includeNeighbors=True))
        makeAPersevMeasurePlot("bt_avg_dwell_time_ninc",
                               lambda s, w: s.avg_dwell_time(False, w, includeNeighbors=True))
        makeAPersevMeasurePlot("probe_num_entries_to_well_ninc",
                               lambda s, w: s.num_well_entries(True, w, includeNeighbors=True))
        makeAPersevMeasurePlot("probe_total_dwell_time_ninc",
                               lambda s, w: s.total_dwell_time(True, w, includeNeighbors=True))
        makeAPersevMeasurePlot("probe_avg_dwell_time_ninc",
                               lambda s, w: s.avg_dwell_time(True, w, includeNeighbors=True))
        makeAPersevMeasurePlot("probe_num_entries_to_well_1min_ninc",
                               lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60], includeNeighbors=True))
        makeAPersevMeasurePlot("probe_num_entries_to_well_30sec_ninc",
                               lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30], includeNeighbors=True))
        makeAPersevMeasurePlot("probe_total_dwell_time_30sec_ninc",
                               lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30], includeNeighbors=True))
        makeAPersevMeasurePlot("probe_avg_dwell_time_30sec_ninc",
                               lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30], includeNeighbors=True))
        makeAPersevMeasurePlot("probe_avg_dwell_time_1min_ninc",
                               lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60], includeNeighbors=True))

        makeAPersevMeasurePlot("probe_avg_dwell_time_90sec",
                               lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 90]))
        makeAPersevMeasurePlot("probe_num_entries_to_well_90sec",
                               lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 90]))

        makeAPersevMeasurePlot("time_within_20_cm_of_well",
                               lambda s, w: s.total_time_near_well(True, w, radius=20, timeInterval=[0, 60]) / TRODES_SAMPLING_RATE)

        makeAPersevMeasurePlot("probe_total_dwell_time_1min",
                               lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
        makeAPersevMeasurePlot("probe_avg_dwell_time_60sec",
                               lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))

        makeAPersevMeasurePlot("probe_total_dwell_time_1min_ninc",
                               lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60], includeNeighbors=True))
        makeAPersevMeasurePlot("probe_avg_curve_1min_ninc",
                               lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 60], includeNeighbors=True))

if SKIP_PERSEV_BOX_PLOTS:
    print("Warning skipping Persev measure plots!")
else:
    makeABoxPlot([sesh.num_well_entries(True, sesh.home_well) for sesh in all_sessions],
                 tlbls, ['Condition', 'probe_num_entries_to_well'])

    if not SKIP_TO_MY_LOU_DARLIN:
        makeABoxPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60]) for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_avg_dwell_time_60sec'])
        makeABoxPlot([sesh.total_dwell_time(True, sesh.home_well, timeInterval=[0, 60]) for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_total_dwell_time_60sec'])
        makeABoxPlot([sesh.num_well_entries(True, sesh.home_well, timeInterval=[0, 60]) for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_num_entries_to_well_60sec'])

        makeABoxPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90]) for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_avg_dwell_time_90sec'])
        makeABoxPlot([sesh.total_dwell_time(True, sesh.home_well, timeInterval=[0, 90]) for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_total_dwell_time_90sec'])
        makeABoxPlot([sesh.num_well_entries(True, sesh.home_well, timeInterval=[0, 90]) for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_num_entries_to_well_90sec'])

        makeABoxPlot([(sesh.entry_exit_times(True, sesh.home_well)[0][0] - sesh.probe_pos_ts[0]) / TRODES_SAMPLING_RATE for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_latency_to_home'])
        makeABoxPlot([sesh.path_optimality(True, wellName=sesh.home_well) for sesh in all_sessions],
                     tlbls, ['Condition', 'probe_optimality_to_home'])

        makeABoxPlot([sesh.avg_dist_to_home_well(True) for sesh in all_sessions],
                     tlbls, ['Condition', 'mean_dist_to_home'])
        makeABoxPlot([sesh.avg_dist_to_home_well(True, timeInterval=[0, 60]) for sesh in all_sessions],
                     tlbls, ['Condition', 'mean_dist_to_home_1min'])

    makeABoxPlot([sesh.total_time_near_well(True, sesh.home_well, radius=20) / TRODES_SAMPLING_RATE for sesh in all_sessions],
                 tlbls, ['Condition', 'total_time_within_20_cm_home'])
    makeABoxPlot([sesh.total_time_near_well(True, sesh.home_well, radius=20, timeInterval=[0, 60]) / TRODES_SAMPLING_RATE for sesh in all_sessions],
                 tlbls, ['Condition', 'total_time_within_20_cm_home_1min'])

if SKIP_BOUT_PLOTS:
    print("Warning skipping exploration bout plots!")
else:
    makeAPersevMeasurePlot("probe_bout_count_by_well",
                           lambda s, w: s.num_bouts_where_well_was_visited(True, w))
    makeAPersevMeasurePlot("probe_bout_count_by_well_1min",
                           lambda s, w: s.num_bouts_where_well_was_visited(True, w, timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_bout_count_by_well_30sec",
                           lambda s, w: s.num_bouts_where_well_was_visited(True, w, timeInterval=[0, 30]))
    makeAPersevMeasurePlot("bt_bout_count_by_well",
                           lambda s, w: s.num_bouts_where_well_was_visited(False, w))

    makeAPersevMeasurePlot("probe_bout_pct_by_well",
                           lambda s, w: s.pct_bouts_where_well_was_visited(True, w))
    makeAPersevMeasurePlot("probe_bout_pct_by_well_1min",
                           lambda s, w: s.pct_bouts_where_well_was_visited(True, w, timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_bout_pct_by_well_30sec",
                           lambda s, w: s.pct_bouts_where_well_was_visited(True, w, timeInterval=[0, 30]))
    makeAPersevMeasurePlot("bt_bout_pct_by_well",
                           lambda s, w: s.pct_bouts_where_well_was_visited(False, w))

    num_bouts_1min = [sesh.num_bouts(True, timeInterval=[0, 60]) for sesh in all_sessions]
    num_bouts_10sec = [sesh.num_bouts(True, timeInterval=[0, 10]) for sesh in all_sessions]
    makeAHistogram(num_bouts_1min, categories=["all"] * len(num_bouts_1min), title="num bouts 1min")
    makeAHistogram(num_bouts_10sec, categories=["all"]
                   * len(num_bouts_10sec), title="num bouts 10sec")
    makeAPersevMeasurePlot("pct_first_1_bouts",
                           lambda s, w: s.pct_bouts_where_well_was_visited(True, w, boutsInterval=[0, 1]))
    makeAPersevMeasurePlot("pct_first_3_bouts",
                           lambda s, w: s.pct_bouts_where_well_was_visited(True, w, boutsInterval=[0, 3]))

if SKIP_WAIT_PLOTS:
    print("Warning skipping exploration wait plots!")
else:
    # makeAPersevMeasurePlot("probe_wait_count_by_well")
    # makeAPersevMeasurePlot("probe_wait_count_by_well_1min")
    # makeAPersevMeasurePlot("probe_wait_count_by_well_30sec")
    # makeAPersevMeasurePlot("bt_wait_count_by_well")

    # makeAPersevMeasurePlot("probe_wait_pct_by_well")
    # makeAPersevMeasurePlot("probe_wait_pct_by_well_1min")
    # makeAPersevMeasurePlot("probe_wait_pct_by_well_30sec")
    # makeAPersevMeasurePlot("bt_wait_pct_by_well")
    pass


if SKIP_BALL_PLOTS:
    print("Warning skipping ballisticity by well plots!")
else:
    # makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_time")
    # makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_visits")
    # makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_time_1min")
    # makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_visits_1min")
    # makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_time_30sec")
    # makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_visits_30sec")
    # makeAPersevMeasurePlot("bt_well_avg_ballisticity_over_time")
    # makeAPersevMeasurePlot("bt_well_avg_ballisticity_over_visits")
    pass

if SKIP_CURVATURE_PLOTS:
    print("Warning skipping curvature by well plots!")
else:
    if not SKIP_TO_MY_LOU_DARLIN:
        makeAPersevMeasurePlot("probe_well_avg_curvature_over_time",
                               lambda s, w: s.avg_curvature_at_well(True, w))
        makeAPersevMeasurePlot("probe_well_avg_curvature_over_visits",
                               lambda s, w: s.avg_curvature_at_well(True, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS))
        makeAPersevMeasurePlot("probe_well_avg_curvature_over_time_1min",
                               lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 60]))
        makeAPersevMeasurePlot("probe_well_avg_curvature_over_visits_1min",
                               lambda s, w: s.avg_curvature_at_well(True, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS, timeInterval=[0, 60]))
        makeAPersevMeasurePlot("probe_well_avg_curvature_over_time_30sec",
                               lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 30]))
        makeAPersevMeasurePlot("probe_well_avg_curvature_over_visits_30sec",
                               lambda s, w: s.avg_curvature_at_well(True, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS, timeInterval=[0, 30]))
        makeAPersevMeasurePlot("bt_well_avg_curvature_over_time",
                               lambda s, w: s.avg_curvature_at_well(False, w))
        makeAPersevMeasurePlot("bt_well_avg_curvature_over_visits",
                               lambda s, w: s.avg_curvature_at_well(False, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS))

    makeAPersevMeasurePlot("probe_well_avg_curvature_over_time_90sec",
                           lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 90]))


if SKIP_PERSEV_QUAD_PLOTS:
    print("Warning skipping quad plots!")
else:
    makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries",
                                   lambda s, w: s.num_well_entries(True, w))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times",
                                   lambda s, w: s.total_dwell_time(True, w))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times",
                                   lambda s, w: s.avg_dwell_time(True, w))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries_1min",
                                   lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times_1min",
                                   lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times_1min",
                                   lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries_30sec",
                                   lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30]))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times_30sec",
                                   lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30]))
    makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times_30sec",
                                   lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30]))

# ===================================
# What do dwell times look like?
# ===================================
if not SKIP_TO_MY_LOU_DARLIN:
    dt_vals = [
        dt for sesh in all_sessions for dtlist in [sesh.dwell_times(True, w) for w in all_well_names] for dt in dtlist]
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
        tref = sesh.bt_pos_ts[0: -1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        return np.mean(sesh.bt_vel_cm_s[start_i: stop_i])
    else:
        tref = sesh.probe_pos_ts[0: -1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        return np.mean(sesh.probe_vel_cm_s[start_i: stop_i])


def avg_speed_plot(interval, output_filename=""):
    if SKIP_AVG_SPEED_PLOTS:
        print("Warning: skipping avg speed comp plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]
                         for sesh in all_sessions])
    max_probe_dur = np.max(
        [sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
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

        all_bt_speeds[si, 0: len(bt_speeds)] = bt_speeds
        all_probe_speeds[si, 0: len(probe_speeds)] = probe_speeds

    bt_mvals = np.nanmean(all_bt_speeds, axis=0)
    probe_mvals = np.nanmean(all_probe_speeds, axis=0)
    bt_sems = np.nanstd(all_bt_speeds, axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_bt_speeds)), axis=0))
    probe_sems = np.nanstd(all_probe_speeds, axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_probe_speeds)), axis=0))

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * bt_num_speeds,
                             bt_num_speeds), bt_mvals, bt_sems)
    plt.title("BT Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "bt_avg_speed_{}".format(interval))
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
            output_filename = os.path.join(
                output_dir, "probe_avg_speed_{}".format(interval))
        else:
            output_filename = os.path.join(
                output_dir, output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


if not SKIP_TO_MY_LOU_DARLIN:
    avg_speed_plot(30)
    avg_speed_plot(60)
    avg_speed_plot(300)


def avg_speed_plot_separate_conditions(interval, output_filename=""):
    if SKIP_AVG_SPEED_COMP_PLOTS:
        print("Warning: skipping avg speed comp plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]
                         for sesh in all_sessions])
    max_probe_dur = np.max(
        [sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
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
    bt_mvals_swr = np.nanmean(
        all_bt_speeds[np.where(session_type == 1), :][0], axis=0)
    bt_mvals_ctrl = np.nanmean(
        all_bt_speeds[np.where(session_type == 0), :][0], axis=0)
    probe_mvals_swr = np.nanmean(
        all_probe_speeds[np.where(session_type == 1), :][0], axis=0)
    probe_mvals_ctrl = np.nanmean(
        all_probe_speeds[np.where(session_type == 0), :][0], axis=0)
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
            output_filename = os.path.join(
                output_dir, output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


if not SKIP_TO_MY_LOU_DARLIN:
    avg_speed_plot_separate_conditions(30)
    avg_speed_plot_separate_conditions(60)
    avg_speed_plot_separate_conditions(150)
    avg_speed_plot_separate_conditions(300)

# for sesh in all_sessions:
#     if SKIP_LINE_PLOTS:
#         print("Warning, skipping line plots")
#         break
#     plt.clf()

#     x = sesh.bt_ball_by_dist_to_home_xvals
#     y = sesh.bt_ball_by_dist_to_home
#     tse = sesh.bt_ball_by_dist_to_home_sem
#     # session.probe_ball_by_dist_to_ctrl_home
#     # session.probe_ball_by_dist_to_ctrl_home_sem

#     plt.plot(x, y)
#     # plt.fill_between(x, y-tse, y+tse)
#     plt.show()


def calcBoutProps(sesh, start, stop):
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
        return calcBoutProps(sesh, start_t, stop_t)
    if start < sesh.probe_pos_ts[0]:
        tref = sesh.bt_pos_ts[0:-1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        cats = sesh.bt_bout_category[start_i:stop_i]
        c0 = float(np.count_nonzero(cats == 0)) / float(cats.size)
        c1 = float(np.count_nonzero(cats == 1)) / float(cats.size)
        c2 = float(np.count_nonzero(cats == 2)) / float(cats.size)
        return c0, c1, c2
    else:
        tref = sesh.probe_pos_ts[0:-1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        cats = sesh.probe_bout_category[start_i:stop_i]
        c0 = float(np.count_nonzero(cats == 0)) / float(cats.size)
        c1 = float(np.count_nonzero(cats == 1)) / float(cats.size)
        c2 = float(np.count_nonzero(cats == 2)) / float(cats.size)
        return c0, c1, c2


def boutProportionPlot(interval, given_output_filename=""):
    if SKIP_BOUT_PROP_PLOTS:
        print("Warning: skipping bout prop plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]
                         for sesh in all_sessions])
    max_probe_dur = np.max(
        [sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
    istep = interval * TRODES_SAMPLING_RATE
    bt_num_cats = int(np.floor(max_bt_dur / istep))
    probe_num_cats = int(np.floor(max_probe_dur / istep))

    all_bt_cats = np.empty((len(all_sessions), bt_num_cats, 3))
    all_probe_cats = np.empty((len(all_sessions), probe_num_cats, 3))
    all_bt_cats[:] = np.nan
    all_probe_cats[:] = np.nan

    for si, sesh in enumerate(all_sessions):
        bt_cats = np.array([calcBoutProps(sesh, i1, i1+istep)
                            for i1 in np.arange(sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]-istep, istep)])
        probe_cats = np.array([calcBoutProps(sesh, i1, i1+istep)
                               for i1 in np.arange(sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]-istep, istep)])

        all_bt_cats[si, 0:bt_cats.shape[0], :] = bt_cats
        all_probe_cats[si, 0:probe_cats.shape[0], :] = probe_cats

    bt_mvals = np.nanmean(all_bt_cats, axis=0)
    probe_mvals = np.nanmean(all_probe_cats, axis=0)
    bt_sems = np.divide(np.nanstd(all_bt_cats, axis=0),
                        np.sqrt(np.count_nonzero(np.logical_not(np.isnan(all_bt_cats[:, :, 0])), axis=0))[:, None])
    probe_sems = np.divide(np.nanstd(all_probe_cats, axis=0),
                           np.sqrt(np.count_nonzero(np.logical_not(np.isnan(all_probe_cats[:, :, 0])), axis=0))[:, None])

    plt.clf()
    x = np.linspace(interval, interval * bt_num_cats, bt_num_cats)
    widths = (x[1]-x[0]) * 0.8
    p1 = plt.bar(x, bt_mvals[:, 0], widths, yerr=bt_sems[:, 0])
    p2 = plt.bar(x, bt_mvals[:, 1], widths,
                 yerr=bt_sems[:, 1], bottom=bt_mvals[:, 0])
    p3 = plt.bar(x, bt_mvals[:, 2], widths, yerr=bt_sems[:, 2],
                 bottom=bt_mvals[:, 0]+bt_mvals[:, 1])
    plt.title("BT bout props, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Proportion in each category")
    plt.legend((p1[0], p2[0], p3[0]), ('explore', 'rest', 'reward'))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(given_output_filename) > 0:
        if len(given_output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "bt_bout_proportions_{}".format(interval))
        else:
            output_filename = os.path.join(
                output_dir, given_output_filename + "_bt")
        plt.savefig(output_filename, dpi=800)

    plt.clf()
    x = np.linspace(interval, interval * probe_num_cats, probe_num_cats)
    widths = (x[1]-x[0]) * 0.8
    p1 = plt.bar(x, probe_mvals[:, 0], widths, yerr=probe_sems[:, 0])
    p2 = plt.bar(x, probe_mvals[:, 1], widths,
                 yerr=probe_sems[:, 1], bottom=probe_mvals[:, 0])
    p3 = plt.bar(x, probe_mvals[:, 2], widths, yerr=probe_sems[:, 2],
                 bottom=probe_mvals[:, 0]+probe_mvals[:, 1])
    plt.title("probe bout props, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Proportion in each category")
    plt.legend((p1[0], p2[0], p3[0]), ('explore', 'rest', 'reward'))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(given_output_filename) > 0:
        if len(given_output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "probe_bout_proportions_{}".format(interval))
        else:
            output_filename = os.path.join(
                output_dir, given_output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


boutProportionPlot(60)
boutProportionPlot(30)
boutProportionPlot(15)

if SKIP_HW_PLOT:
    print("Warning, skipping home well plot")
else:
    session_type = [0 if trial_label(
        sesh) == "SWR" else 1 for sesh in all_sessions]
    xvals = [sesh.home_x for sesh in all_sessions]
    yvals = [sesh.home_y for sesh in all_sessions]
    plt.clf()
    plt.xlim(0, 1200)
    plt.ylim(0, 1000)
    plt.scatter(xvals, yvals, c=session_type, zorder=2)
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "all_home_wells"), dpi=800)

if SKIP_ORDER_PLOTS:
    print("Warning, skipping trial order plots")
else:
    if not SKIP_TO_MY_LOU_DARLIN:
        makeAScatterPlot(list(range(len(all_sessions))),
                         [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                          for sesh in all_sessions],
                         ['Session idx', 'Home avg dwell time probe 1min'], tlbls)
        makeAScatterPlot(list(range(len(all_sessions))),
                         [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                          for sesh in all_sessions],
                         ['Session idx', 'home avg curvature over time 1min'], tlbls)

        makeAScatterPlot(list(range(len(all_sessions))),
                         [sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[
                             0, 60]) for sesh in all_sessions],
                         ['Session idx', 'Probe 60sec explore proportion'], tlbls)
        makeAScatterPlot(list(range(len(all_sessions))),
                         [sesh.mean_vel(True, timeInterval=[0, 30])
                          for sesh in all_sessions],
                         ['Session idx', 'Probe 30sec mean vel'], tlbls)
        makeAScatterPlot(list(range(len(all_sessions))),
                         [sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True)
                          for sesh in all_sessions],
                         ['Session idx', 'Probe 30sec moving mean vel'], tlbls)

        makeAScatterPlot(list(range(len(all_sessions))),
                         [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                          for sesh in all_sessions],
                         ['Session idx', 'Home avg dwell time probe 90sec'], tlbls)
        makeAScatterPlot(list(range(len(all_sessions))),
                         [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                          for sesh in all_sessions],
                         ['Session idx', 'home avg curvature over time 90sec'], tlbls)


if SKIP_RIPPLE_PLACE_PLOTS:
    print("Warning, skipping ripple place plots")
else:
    ripoutdir = os.path.join(output_dir, 'ripple_place')
    all_rip_speeds = np.array([])
    all_speeds = np.array([])
    for sesh in all_sessions:
        # print(sesh.home_well_find_pos_idxs)
        well_find_pos_idxs = np.concatenate(
            (sesh.home_well_find_pos_idxs, sesh.away_well_find_pos_idxs))
        well_find_pos_idxs.sort()
        # print(well_find_pos_idxs)

        posx = np.array(sesh.bt_pos_xs)
        posy = np.array(sesh.bt_pos_ys)
        post = np.array(sesh.bt_pos_ts)
        rip_ts = np.array(sesh.interruption_timestamps)

        if not SKIP_TO_MY_LOU_DARLIN:
            last_find_idx = 0
            for wi, next_find_idx in enumerate(well_find_pos_idxs):

                t1 = post[last_find_idx]
                t2 = post[next_find_idx]
                rts = rip_ts[np.logical_and(rip_ts >= t1, rip_ts < t2)]
                rip_idxs = np.searchsorted(post[last_find_idx:next_find_idx], rts)

                plt.clf()
                plt.plot(posx[last_find_idx:next_find_idx], posy[last_find_idx:next_find_idx])
                plt.scatter(posx[last_find_idx + rip_idxs],
                            posy[last_find_idx + rip_idxs], c=[[1, 0, 0, 1]], zorder=32)
                # print(wi, len(rts), len(rip_ts))
                # print(t1, t2, rip_ts[0], rip_ts[-1])
                plt.scatter(posx[last_find_idx], posy[last_find_idx], c=[[0, 1, 0, 1]],
                            s=plt.rcParams['lines.markersize'] ** 2 * 2, zorder=31)
                plt.gca().set_xlim(200, 1100)
                plt.gca().set_ylim(0, 1000)

                if SHOW_OUTPUT_PLOTS:
                    plt.show()
                if SAVE_OUTPUT_PLOTS:
                    output_filename = os.path.join(ripoutdir, sesh.name + "_" + str(wi))
                    plt.savefig(output_filename, dpi=800)

                last_find_idx = next_find_idx

        # speed
        rip_idxs = np.searchsorted(post, rip_ts[rip_ts < post[-1]])
        rip_speeds = sesh.bt_vel_cm_s[rip_idxs]
        all_rip_speeds = np.concatenate((all_rip_speeds, rip_speeds))
        all_speeds = np.concatenate((all_speeds, sesh.bt_vel_cm_s))

    plt.clf()
    # sns.distplot(speeds)
    # bin_max = max(np.max(all_speeds), np.max(all_rip_speeds))
    bin_max = 100
    x, bins, p = plt.hist((all_speeds, all_rip_speeds),
                          bins=np.arange(0, bin_max, 5), label=['all', 'stim'])
    plt.legend()
    # print(x, bins, p)
    # x, bins, p = plt.hist(all_speeds, bins=np.arange(0, np.max(all_speeds), 5))
    for item in p[0]:
        item.set_height(item.get_height()/np.sum(x[0]))
    for item in p[1]:
        item.set_height(item.get_height()/np.sum(x[1]))
    # x, bins, p = plt.hist(all_rip_speeds, bins=np.arange(0, np.max(all_rip_speeds), 5))
    # for item in p:
        # item.set_height(item.get_height()/sum(x))
    prop_txt = str(np.count_nonzero(all_rip_speeds < 5) / np.size(all_rip_speeds))
    plt.text(1, 1, prop_txt, horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.gca().set_ylim((0, 0.5))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS:
        output_filename = os.path.join(output_dir, "all_speeds_at_interruption")
        plt.savefig(output_filename, dpi=800)

    # distance to home well
    # in exploration bout?

if SKIP_EVERY_MINUTE_PLOTS:
    print("Warning, skipping plots that look at each minute of probe")
else:
    makeAPersevMeasurePlot("probe_curvature_full",
                           lambda s, w: s.avg_curvature_at_well(True, w))
    makeAPersevMeasurePlot("probe_curvature_0_60s",
                           lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_curvature_60_120s",
                           lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[60, 120]))
    makeAPersevMeasurePlot("probe_curvature_120_180s",
                           lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[120, 180]))
    makeAPersevMeasurePlot("probe_curvature_120_180s",
                           lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[120, 180]))
    makeAPersevMeasurePlot("probe_curvature_180_240",
                           lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[180, 240]))
    makeAPersevMeasurePlot("probe_curvature_240_300",
                           lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[240, 300]))

    makeAPersevMeasurePlot("probe_avg_dwell_full",
                           lambda s, w: s.avg_dwell_time(True, w))
    makeAPersevMeasurePlot("probe_avg_dwell_0_60s",
                           lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_avg_dwell_60_120s",
                           lambda s, w: s.avg_dwell_time(True, w, timeInterval=[60, 120]))
    makeAPersevMeasurePlot("probe_avg_dwell_120_180s",
                           lambda s, w: s.avg_dwell_time(True, w, timeInterval=[120, 180]))
    makeAPersevMeasurePlot("probe_avg_dwell_180_240s",
                           lambda s, w: s.avg_dwell_time(True, w, timeInterval=[180, 240]))
    makeAPersevMeasurePlot("probe_avg_dwell_240_300s",
                           lambda s, w: s.avg_dwell_time(True, w, timeInterval=[240, 300]))

    makeAPersevMeasurePlot("probe_num_entries_full",
                           lambda s, w: s.num_well_entries(True, w))
    makeAPersevMeasurePlot("probe_num_entries_0_60s",
                           lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
    makeAPersevMeasurePlot("probe_num_entries_60_120s",
                           lambda s, w: s.num_well_entries(True, w, timeInterval=[60, 120]))
    makeAPersevMeasurePlot("probe_num_entries_120_180s",
                           lambda s, w: s.num_well_entries(True, w, timeInterval=[120, 180]))
    makeAPersevMeasurePlot("probe_num_entries_180_240s",
                           lambda s, w: s.num_well_entries(True, w, timeInterval=[180, 240]))
    makeAPersevMeasurePlot("probe_num_entries_240_300s",
                           lambda s, w: s.num_well_entries(True, w, timeInterval=[240, 300]))
