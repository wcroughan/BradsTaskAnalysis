import os
import numpy as np

from MyPlottingFunctions import MyPlottingFunctions
from BTData import BTData
from BTSession import BTSession

animals = ["B13", "B14"]

for animal_name in animals:
    if animal_name == "B13":
        data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
        output_dir = "/media/WDC7/B13/processed_data/behavior_figures/"
    elif animal_name == "B14":
        data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
        output_dir = "/media/WDC7/B14/processed_data/behavior_figures/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alldata = BTData()
    alldata.loadFromFile(data_filename)
    P = MyPlottingFunctions(alldata, output_dir)

    # ==========================================================================
    # Behavior traces
    # ==================
    # Raw behavior traces
    if False:
        P.makeALinePlot(lambda s: [(s.bt_pos_xs, s.bt_pos_ys)],
                        "Raw Behavior Task", axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs, s.probe_pos_ys)],
                        "Raw Behavior Probe", axisLims="environment")

    # ==================
    # behavior trace for each well during task
    if False:
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            [0] + (s.away_well_leave_pos_idxs if s.ended_on_home else s.away_well_leave_pos_idxs[0:-1]),
            s.home_well_find_pos_idxs
        )], "path_to_well_home", saveAllValuePairsSeparately=True, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            s.home_well_leave_pos_idxs[0:-1] if s.ended_on_home else s.home_well_leave_pos_idxs,
            s.away_well_find_pos_idxs
        )], "path_to_well_away", saveAllValuePairsSeparately=True, axisLims="environment")

    # ==================
    # behavior trace to first home well check in probe
    if False:
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[0:s.getLatencyToWell(True, s.home_well, returnIdxs=True)]),
                        s.bt_pos_ys[0:s.getLatencyToWell(True, s.home_well, returnIdxs=True)]], "first_home_check_in_probe", axisLims="environment")

    # ==================
    # behavior trace colored by nearest well
    if False:
        def f(sesh, w, inProbe):
            entryExitIdxs = list(zip(*sesh.entry_exit_times(inProbe, w, returnIdxs=True)))
            sxs = sesh.probe_pos_xs if inProbe else sesh.bt_pos_xs
            sys = sesh.probe_pos_ys if inProbe else sesh.bt_pos_ys
            xs = np.array([sxs[i1:i2] + [np.nan]
                           for (i1, i2) in entryExitIdxs], dtype=object)
            ys = np.array([sys[i1:i2] + [np.nan]
                           for (i1, i2) in entryExitIdxs], dtype=object)
            xflat = [i for a in xs for i in a]
            yflat = [i for a in ys for i in a]
            return xflat, yflat
        P.makeALinePlot(lambda s: [f(s, w, False)
                        for w in P.all_well_names], "trace_colored_by_well_task", axisLims="environment")
        P.makeALinePlot(lambda s: [f(s, w, True)
                        for w in P.all_well_names], "trace_colored_by_well_probe", axisLims="environment")

    # ==================
    # Truncated behavior trace where rat ran to me to exit environment
    if False:
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[s.bt_recall_pos_idx:],
                        s.bt_pos_ys[s.bt_recall_pos_idx:])], "Recall path task", axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[s.probe_recall_pos_idx:],
                        s.probe_pos_ys[s.probe_recall_pos_idx:])], "Recall path probe", axisLims="environment")

    # ==================
    # Behavior trace with line colored by curvature
    def lerp(v, c1, c2):
        return c1 * (1. - v) + v * c2
    P.makeALinePlot(lambda s: [(s.bt_pos_xs, s.bt_pos_ys)],
                    "task colored by curvature", colorFunc=lambda s: [[lerp(v, np.array([1, 0, 0, 1]), np.array([0, 1, 1, 1])) for v in s.bt_curvature]], axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs, s.probe_pos_ys)],
                    "probe colored by curvature", colorFunc=lambda s: [[lerp(v, np.array([1, 0, 0, 1]), np.array([0, 1, 1, 1])) for v in s.probe_curvature]], axisLims="environment")

    # ==================
    # Behavior trace of each exploration bout, each resting period
    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.bt_explore_bout_starts, s.bt_explore_bout_ends
    )], "exploration_bout_task", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.bt_explore_bout_starts, s.bt_explore_bout_ends
    )], "exploration_bout_task_combined", saveAllValuePairsSeparately=False, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.probe_explore_bout_starts, s.probe_explore_bout_ends
    )], "exploration_bout_probe", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.probe_explore_bout_starts, s.probe_explore_bout_ends
    )], "exploration_bout_probe_combined", saveAllValuePairsSeparately=False, axisLims="environment")

    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
        s.bt_explore_bout_ends[:-1]
        if s.bt_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_ends,
        s.bt_explore_bout_starts[1:]
        if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_starts +
        ([] if s.bt_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else [len(s.bt_bout_category)])
    )], "resting_states_task", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
        s.bt_explore_bout_ends[:-1]
        if s.bt_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_ends,
        s.bt_explore_bout_starts[1:]
        if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_starts +
        ([] if s.bt_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else [len(s.bt_bout_category)])
    )], "resting_states_task_combined", saveAllValuePairsSeparately=False, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
        s.probe_explore_bout_ends[:-1]
        if s.probe_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_ends,
        s.probe_explore_bout_starts[1:]
        if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_starts +
        ([] if s.probe_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else [len(s.probe_bout_category)])
    )], "resting_states_probe", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
        s.probe_explore_bout_ends[:-1]
        if s.probe_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_ends,
        s.probe_explore_bout_starts[1:]
        if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_starts +
        ([] if s.probe_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else [len(s.probe_bout_category)])
    )], "resting_states_probe_combined", saveAllValuePairsSeparately=False, axisLims="environment")

    # ==================
    # Behavior trace of each off-wall excursion during probe, and one plot per session with all excursions
    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.bt_excursion_starts, s.bt_excursion_ends
    )], "excursion_task", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.bt_excursion_starts, s.bt_excursion_ends
    )], "excursion_task_combined", saveAllValuePairsSeparately=False, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.probe_excursion_starts, s.probe_excursion_ends
    )], "excursion_probe", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        s.probe_excursion_starts, s.probe_excursion_ends
    )], "excursion_probe_combined", saveAllValuePairsSeparately=False, axisLims="environment")

    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else [0]) +
        s.bt_excursion_ends[:-1]
        if s.bt_excursion_category[-1] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.bt_excursion_ends,
        s.bt_excursion_starts[1:]
        if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.bt_excursion_starts +
        ([] if s.bt_excursion_category[-1] ==
         BTSession.EXCURSION_STATE_ON_EXCURSION else [len(s.bt_excursion_category)])
    )], "on_wall_paths_task", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else [0]) +
        s.bt_excursion_ends[:-1]
        if s.bt_excursion_category[-1] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.bt_excursion_ends,
        s.bt_excursion_starts[1:]
        if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.bt_excursion_starts +
        ([] if s.bt_excursion_category[-1] ==
         BTSession.EXCURSION_STATE_ON_EXCURSION else [len(s.bt_excursion_category)])
    )], "on_wall_paths_task_combined", saveAllValuePairsSeparately=False, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else [0]) +
        s.probe_excursion_ends[:-1]
        if s.probe_excursion_category[-1] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.probe_excursion_ends,
        s.probe_excursion_starts[1:]
        if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.probe_excursion_starts +
        ([] if s.probe_excursion_category[-1] ==
         BTSession.EXCURSION_STATE_ON_EXCURSION else [len(s.probe_excursion_category)])
    )], "on_wall_paths_probe", saveAllValuePairsSeparately=True, axisLims="environment")
    P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
        ([] if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else [0]) +
        s.probe_excursion_ends[:-1]
        if s.probe_excursion_category[-1] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.probe_excursion_ends,
        s.probe_excursion_starts[1:]
        if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_ON_EXCURSION else s.probe_excursion_starts +
        ([] if s.probe_excursion_category[-1] ==
         BTSession.EXCURSION_STATE_ON_EXCURSION else [len(s.probe_excursion_category)])
    )], "on_wall_paths_probe_combined", saveAllValuePairsSeparately=False, axisLims="environment")

    # ==========================================================================
    # Task behavior
    # ==================
    # Latency to each well during task (line plot)
    P.makeALinePlot(lambda s: [(np.arange(s.num_home_found) + 1,
                                (s.home_well_find_times -
                                ([s.bt_pos_ts[0]] + s.away_well_leave_times)) / BTSession.TRODES_SAMPLING_RATE
                                )], "home_find_times", individualSessions=False,
                    colorFunc=[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.home_well_find_times))])
    P.makeALinePlot(lambda s: [(np.arange(s.num_home_found) + 1,
                                (s.away_well_find_times - s.home_well_leave_times[0:len(
                                    s.away_well_find_times)]) / BTSession.TRODES_SAMPLING_RATE
                                )], "away_find_times", individualSessions=False,
                    colorFunc=[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.away_well_find_times))])

    # ==================
    # Task behavior comparisons between conditions
    P.makeASimpleBoxPlot(lambda s: np.sum(s.home_well_find_times -
                                          ([s.bt_pos_ts[0]] + s.away_well_leave_times)) / BTSession.TRODES_SAMPLING_RATE,
                         "total home search time")
    P.makeASimpleBoxPlot(lambda s: s.num_away_found, "num aways found")
    P.makeASimpleBoxPlot(lambda s: (
        s.bt_pos_ts[s.bt_recall_pos_idx] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE, "Total task time")

    P.makeASimpleBoxPlot(lambda s: np.sum(s.home_well_find_times -
                                          ([s.bt_pos_ts[0]] + s.away_well_leave_times)) / BTSession.TRODES_SAMPLING_RATE,
                         "total home search time NP inc", includeNoProbeSessions=True)
    P.makeASimpleBoxPlot(lambda s: s.num_away_found, "num aways found NP inc",
                         includeNoProbeSessions=True)
    P.makeASimpleBoxPlot(lambda s: (
        s.bt_pos_ts[s.bt_recall_pos_idx] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE, "Total task time NP inc", includeNoProbeSessions=True)

    # ==========================================================================
    # Animations
    # ==================
    # Animated exploration bout vs rest vs reward
    # ani = anim.FuncAnimation(fig, animate, range(len(sesh.bt_pos_ts) - 1),
    #  repeat=False, init_func=init_plot, interval=5)

    # ==================
    # Animated task with nearest well highlighted
    # ani = anim.FuncAnimation(fig, animate, range(len(sesh.bt_pos_ts) - 1),
    #  repeat=False, init_func=init_plot, interval=5)

    # ==========================================================================
    # Perseveration measures
    # ==================
    # dwell time
    interestingProbeTimes = [60, 90]
    for iv in interestingProbeTimes:
        P.makeAPersevMeasurePlot("probe_total_dwell_time_{}sec".format(iv),
                                 lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=True)
        P.makeAPersevMeasurePlot("probe_avg_dwell_time_{}sec".format(iv),
                                 lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=True)
    P.makeAPersevMeasurePlot("probe_total_dwell_time",
                             lambda s, w: s.total_dwell_time(True, w), alsoMakePerWellPerSessionPlot=True)
    P.makeAPersevMeasurePlot("probe_avg_dwell_time",
                             lambda s, w: s.avg_dwell_time(True, w), alsoMakePerWellPerSessionPlot=True)

    # ==================
    # num entries
    for iv in interestingProbeTimes:
        P.makeAPersevMeasurePlot("probe_num_entries_{}sec".format(iv),
                                 lambda s, w: s.num_well_entries(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=True)
    P.makeAPersevMeasurePlot("probe_num_entries",
                             lambda s, w: s.total_dwell_time(True, w), alsoMakePerWellPerSessionPlot=True)

    # ==================
    # latency to first entry
    P.makeASimpleBoxPlot(lambda s: s.getLatencyToWell(True, s.home_well),
                         "Probe latency to home", yAxisName="Latency to home")

    # ==================
    # mean dist to home
    P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_home_well(True),
                         "Probe Mean Dist to Home", yAxisName="Mean Dist to Home")
    P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_well(True, s.ctrl_home_well),
                         "Probe Mean Dist to Ctrl Home", yAxisName="Mean Dist to Ctrl Home")
    for iv in interestingProbeTimes:
        P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_home_well(True, timeInterval=[0, iv]),
                             "Probe Mean Dist to Home {}sec".format(iv), yAxisName="Mean Dist to Home")
        P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_well(True, s.ctrl_home_well, timeInterval=[0, iv]),
                             "Probe Mean Dist to Ctrl Home {}sec".format(iv), yAxisName="Mean Dist to Ctrl Home")

    # ==================
    # time within certain dist to home

    # ==================
    # curvature

    # ==================
    # pct exploration bouts home well visitied

    # ==================
    # pct excursions home well visitied

    # ==========================================================================
    # behavior during probe
    # ==================
    # avg vel over time
    # ==================
    # pct time exploring
