from BTData import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_filename = "/media/WDC4/martindata/bradtask/martin_bradtask.dat"
alldata = BTData()
alldata.loadFromFile(data_filename)
all_sessions = alldata.getSessions()

output_dir = '/media/WDC4/martindata/processed_data'

SHOW_OUTPUT_PLOTS = False
SAVE_OUTPUT_PLOTS = True

SKIP_BOX_PLOTS = True
SKIP_SCATTER_PLOTS = True
SKIP_SWARM_PLOTS = False

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
