import os
import numpy as np
from itertools import groupby
from random import random
import MountainViewIO

from MyPlottingFunctions import MyPlottingFunctions
from BTData import BTData
from BTSession import BTSession
from BTRestSession import BTRestSession

PLOT_BEHAVIOR_TRACES = False
PLOT_BEHAVIOR_SUMMARIES = False
PLOT_PERSEV_MEASURES = False
PLOT_PROBE_BEHAVIOR_SUMMARIES = False
PLOT_ITI_LFP = False
PLOT_INDIVIDUAL_RIPPLES_ITI = False
PLOT_PROBE_LFP = False
PLOT_REST_LFP = False
PLOT_DWELL_TIME_OVER_TIME = True

animals = []
animals += ["B13"]
animals += ["B14"]
animals += ["Martin"]

for animal_name in animals:
    if animal_name == "B13":
        data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
        output_dir = "/media/WDC7/B13/processed_data/behavior_figures/"
    elif animal_name == "B14":
        data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
        output_dir = "/media/WDC7/B14/processed_data/behavior_figures/"
    elif animal_name == "Martin":
        data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
        output_dir = "/media/WDC7/Martin/processed_data/behavior_figures/"
    else:
        raise Exception("Unknown rat")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alldata = BTData()
    alldata.loadFromFile(data_filename)
    # P = MyPlottingFunctions(alldata, output_dir, showPlots=True)
    P = MyPlottingFunctions(alldata, output_dir)

    print("starting plots for", animal_name)

    if PLOT_BEHAVIOR_TRACES:
        # ==========================================================================
        # Where were the homes?
        P.makeAScatterPlotWithFunc(
            lambda s: [((s.home_well % 8) - 1 + random() * 0.25, (s.home_well // 8) + 1 + random() * 0.25)], "home_wells", axisLims=((0, 7), (0, 7)), individualSessions=False,
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"])

        # ==========================================================================
        # Behavior traces
        # ==================
        # Raw behavior traces
        P.makeALinePlot(lambda s: [(s.bt_pos_xs, s.bt_pos_ys)],
                        "Raw Behavior Task", axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs, s.probe_pos_ys)],
                        "Raw Behavior Probe", axisLims="environment")

        # ==================
        # behavior trace for each well during task
        P.makeALinePlot(lambda s: [] if not s.found_first_home else [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            np.hstack(
                ([0], s.away_well_leave_pos_idxs if s.ended_on_home else s.away_well_leave_pos_idxs[0:-1])),
            s.home_well_find_pos_idxs
        )], "path_to_well_home", saveAllValuePairsSeparately=True, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            s.home_well_leave_pos_idxs[0:-1] if s.ended_on_home else s.home_well_leave_pos_idxs,
            s.away_well_find_pos_idxs
        )], "path_to_well_away", saveAllValuePairsSeparately=True, axisLims="environment")

        # ==================
        # behavior trace to first home well check in probe
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[0:s.getLatencyToWell(True, s.home_well, returnIdxs=True, emptyVal=0)],
                        s.probe_pos_ys[0:s.getLatencyToWell(True, s.home_well, returnIdxs=True, emptyVal=0)])], "first_home_check_in_probe", axisLims="environment")

        # ==================
        # behavior trace colored by nearest well
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
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[s.bt_recall_pos_idx:],
                        s.bt_pos_ys[s.bt_recall_pos_idx:])], "Recall path task", axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[s.probe_recall_pos_idx:],
                        s.probe_pos_ys[s.probe_recall_pos_idx:])], "Recall path probe", axisLims="environment")

        # ==================
        # Behavior trace with line colored by curvature
        def lerp(v, c1, c2):
            return np.clip(c1 * (1. - v) + v * c2, 0, 1)

        def normalized(a):
            return (a + np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))
        P.makeALinePlot(lambda s: [(s.bt_pos_xs, s.bt_pos_ys)],
                        "task colored by curvature", colorFunc=lambda s: [[lerp(v, np.array([1, 0, 0, 1]), np.array([0, 1, 1, 1])) for v in normalized(s.bt_curvature)]], axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs, s.probe_pos_ys)],
                        "probe colored by curvature", colorFunc=lambda s: [[lerp(v, np.array([1, 0, 0, 1]), np.array([0, 1, 1, 1])) for v in normalized(s.probe_curvature)]], axisLims="environment")

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
            (np.array([], dtype=int) if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
            s.bt_explore_bout_ends[:-1]
            if s.bt_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_ends,
            s.bt_explore_bout_starts[1:]
            if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_starts +
            (np.array([], dtype=int) if s.bt_bout_category[-1] ==
                BTSession.BOUT_STATE_EXPLORE else [len(s.bt_bout_category)])
        )], "resting_states_task", saveAllValuePairsSeparately=True, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            (np.array([], dtype=int) if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
            s.bt_explore_bout_ends[:-1]
            if s.bt_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_ends,
            s.bt_explore_bout_starts[1:]
            if s.bt_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.bt_explore_bout_starts +
            (np.array([], dtype=int) if s.bt_bout_category[-1] ==
                BTSession.BOUT_STATE_EXPLORE else [len(s.bt_bout_category)])
        )], "resting_states_task_combined", saveAllValuePairsSeparately=False, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
            (np.array([], dtype=int) if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
            s.probe_explore_bout_ends[:-1]
            if s.probe_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_ends,
            s.probe_explore_bout_starts[1:]
            if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_starts +
            (np.array([], dtype=int) if s.probe_bout_category[-1] ==
                BTSession.BOUT_STATE_EXPLORE else [len(s.probe_bout_category)])
        )], "resting_states_probe", saveAllValuePairsSeparately=True, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
            (np.array([], dtype=int) if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else [0]) +
            s.probe_explore_bout_ends[:-1]
            if s.probe_bout_category[-1] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_ends,
            s.probe_explore_bout_starts[1:]
            if s.probe_bout_category[0] == BTSession.BOUT_STATE_EXPLORE else s.probe_explore_bout_starts +
            (np.array([], dtype=int) if s.probe_bout_category[-1] ==
                BTSession.BOUT_STATE_EXPLORE else [len(s.probe_bout_category)])
        )], "resting_states_probe_combined", saveAllValuePairsSeparately=False, axisLims="environment")

        # ==================
        # Behavior trace of each off-wall excursion during probe, and one plot per session with all excursions
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )], "excursion_probe_combined", saveAllValuePairsSeparately=False, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )], "excursion_probe", saveAllValuePairsSeparately=True, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )], "excursion_task_combined", saveAllValuePairsSeparately=False, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )], "excursion_task", saveAllValuePairsSeparately=True, axisLims="environment")

        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            np.hstack((np.array([], dtype=int) if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else [0],
                        s.bt_excursion_ends[:-1]
                        if s.bt_excursion_category[-1] == BTSession.EXCURSION_STATE_OFF_WALL else s.bt_excursion_ends)),
            np.hstack((s.bt_excursion_starts[1:]
                       if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else s.bt_excursion_starts,
                       np.array([], dtype=int) if s.bt_excursion_category[-1] ==
                       BTSession.EXCURSION_STATE_OFF_WALL else [len(s.bt_excursion_category)]))
        )], "on_wall_paths_task", saveAllValuePairsSeparately=True, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.bt_pos_xs[i1:i2], s.bt_pos_ys[i1:i2]) for (i1, i2) in zip(
            np.hstack((np.array([], dtype=int) if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else [0],
                        s.bt_excursion_ends[:-1]
                        if s.bt_excursion_category[-1] == BTSession.EXCURSION_STATE_OFF_WALL else s.bt_excursion_ends)),
            np.hstack((s.bt_excursion_starts[1:]
                       if s.bt_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else s.bt_excursion_starts,
                       np.array([], dtype=int) if s.bt_excursion_category[-1] ==
                       BTSession.EXCURSION_STATE_OFF_WALL else [len(s.bt_excursion_category)]))
        )], "on_wall_paths_task_combined", saveAllValuePairsSeparately=False, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
            np.hstack((np.array([], dtype=int) if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else [0],
                        s.probe_excursion_ends[:-1]
                        if s.probe_excursion_category[-1] == BTSession.EXCURSION_STATE_OFF_WALL else s.probe_excursion_ends)),
            np.hstack((s.probe_excursion_starts[1:]
                       if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else s.probe_excursion_starts,
                       np.array([], dtype=int) if s.probe_excursion_category[-1] ==
                       BTSession.EXCURSION_STATE_OFF_WALL else [len(s.probe_excursion_category)]))
        )], "on_wall_paths_probe", saveAllValuePairsSeparately=True, axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs[i1:i2], s.probe_pos_ys[i1:i2]) for (i1, i2) in zip(
            np.hstack((np.array([], dtype=int) if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else [0],
                        s.probe_excursion_ends[:-1]
                        if s.probe_excursion_category[-1] == BTSession.EXCURSION_STATE_OFF_WALL else s.probe_excursion_ends)),
            np.hstack((s.probe_excursion_starts[1:]
                       if s.probe_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL else s.probe_excursion_starts,
                       np.array([], dtype=int) if s.probe_excursion_category[-1] ==
                       BTSession.EXCURSION_STATE_OFF_WALL else [len(s.probe_excursion_category)]))
        )], "on_wall_paths_probe_combined", saveAllValuePairsSeparately=False, axisLims="environment")

    if PLOT_BEHAVIOR_SUMMARIES:
        # ==========================================================================
        # Task behavior
        # ==================
        # Latency to each well during task (line plot)
        P.makeALinePlot(lambda s: [(np.arange(s.num_home_found) + 1,
                                    (s.home_well_find_times -
                                    (np.hstack(([s.bt_pos_ts[0]], (s.away_well_leave_times if s.ended_on_home else s.away_well_leave_times[0:-1]))))) / BTSession.TRODES_SAMPLING_RATE
                                    )], "home_find_times", individualSessions=False,
                        colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.home_well_find_times))]], plotAverage=True)
        P.makeALinePlot(lambda s: [(np.arange(s.num_away_found) + 1,
                                    (s.away_well_find_times - s.home_well_leave_times[0:len(
                                        s.away_well_find_times)]) / BTSession.TRODES_SAMPLING_RATE
                                    )], "away_find_times", individualSessions=False,
                        colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.away_well_find_times))]], plotAverage=True)

        P.makeALinePlot(lambda s: [(np.arange(s.num_home_found) + 1,
                                    (s.home_well_find_times -
                                    (np.hstack(([s.bt_pos_ts[0]], (s.away_well_leave_times if s.ended_on_home else s.away_well_leave_times[0:-1]))))) / BTSession.TRODES_SAMPLING_RATE
                                    )] if s.isRippleInterruption else [], "home_find_times_just_swr", individualSessions=False,
                        colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.home_well_find_times))]], plotAverage=True)
        P.makeALinePlot(lambda s: [(np.arange(s.num_away_found) + 1,
                                    (s.away_well_find_times - s.home_well_leave_times[0:len(
                                        s.away_well_find_times)]) / BTSession.TRODES_SAMPLING_RATE
                                    )] if s.isRippleInterruption else [], "away_find_times_just_swr", individualSessions=False,
                        colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.away_well_find_times))]], plotAverage=True)

        P.makeALinePlot(lambda s: [(np.arange(s.num_home_found) + 1,
                                    (s.home_well_find_times -
                                    (np.hstack(([s.bt_pos_ts[0]], (s.away_well_leave_times if s.ended_on_home else s.away_well_leave_times[0:-1]))))) / BTSession.TRODES_SAMPLING_RATE
                                    )] if not s.isRippleInterruption else [], "home_find_times_just_delay", individualSessions=False,
                        colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.home_well_find_times))]], plotAverage=True)
        P.makeALinePlot(lambda s: [(np.arange(s.num_away_found) + 1,
                                    (s.away_well_find_times - s.home_well_leave_times[0:len(
                                        s.away_well_find_times)]) / BTSession.TRODES_SAMPLING_RATE
                                    )] if not s.isRippleInterruption else [], "away_find_times_just_delay", individualSessions=False,
                        colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(s.away_well_find_times))]], plotAverage=True)

        # ==================
        # Task behavior comparisons between conditions
        P.makeASimpleBoxPlot(lambda s: np.sum(s.home_well_find_times -
                                              np.hstack(([s.bt_pos_ts[0]], s.away_well_leave_times if s.ended_on_home else s.away_well_leave_times[0:-1]))) / BTSession.TRODES_SAMPLING_RATE,
                             "total home search time")
        P.makeASimpleBoxPlot(lambda s: s.num_away_found, "num aways found")
        P.makeASimpleBoxPlot(lambda s: (
            s.bt_pos_ts[s.bt_recall_pos_idx] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE, "Total task time")

        P.makeASimpleBoxPlot(lambda s: np.sum(s.home_well_find_times -
                                              np.hstack(([s.bt_pos_ts[0]], s.away_well_leave_times if s.ended_on_home else s.away_well_leave_times[0:-1]))) / BTSession.TRODES_SAMPLING_RATE,
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

    if PLOT_PERSEV_MEASURES:
        # ==========================================================================
        # Perseveration measures
        # ==================
        # dwell time
        # interestingProbeTimes = []

        interestingProbeTimes = [60, 90]
        for iv in interestingProbeTimes:
            P.makeAPersevMeasurePlot("probe_total_dwell_time_{}sec".format(iv),
                                     lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=True)
            P.makeAPersevMeasurePlot("probe_avg_dwell_time_{}sec".format(iv),
                                     lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, iv], emptyVal=np.nan), alsoMakePerWellPerSessionPlot=True)
            P.makeAPersevMeasurePlot("probe_avg_dwell_time_{}sec_outlier_fix".format(iv),
                                     lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, iv], emptyVal=np.nan), yAxisLims=(0, 10), alsoMakePerWellPerSessionPlot=False)
        P.makeAPersevMeasurePlot("probe_total_dwell_time",
                                 lambda s, w: s.total_dwell_time(True, w), alsoMakePerWellPerSessionPlot=True)
        P.makeAPersevMeasurePlot("probe_avg_dwell_time",
                                 lambda s, w: s.avg_dwell_time(True, w, emptyVal=np.nan), alsoMakePerWellPerSessionPlot=True)

        # ==================
        # num entries
        for iv in interestingProbeTimes:
            P.makeAPersevMeasurePlot("probe_num_entries_{}sec".format(iv),
                                     lambda s, w: s.num_well_entries(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=True)
        P.makeAPersevMeasurePlot("probe_num_entries",
                                 lambda s, w: s.total_dwell_time(True, w), alsoMakePerWellPerSessionPlot=True)

        # ==================
        # latency to first entry
        P.makeASimpleBoxPlot(lambda s: s.getLatencyToWell(True, s.home_well) / BTSession.TRODES_SAMPLING_RATE,
                             "Probe latency to home", yAxisName="Latency to home")

        # ==================
        # optimality to first entry
        P.makeASimpleBoxPlot(lambda s: s.path_optimality(True, wellName=s.home_well),
                             "Probe (lack of) optimality to home", yAxisName="Lack of optimality to home")

        # ==================
        # mean dist to home
        P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_home_well(True),
                             "Probe Mean Dist to Home", yAxisName="Mean Dist to Home")
        P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_well(True, s.ctrl_home_well),
                             "Probe Mean Dist to Ctrl Home", yAxisName="Mean Dist to Ctrl Home")
        for iv in interestingProbeTimes:
            P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_home_well(True, timeInterval=[0, iv]),
                                 "Probe Mean Dist to Home {}sec".format(iv), yAxisName="Mean Dist to Home {}sec".format(iv))
            P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_well(True, s.ctrl_home_well, timeInterval=[0, iv]),
                                 "Probe Mean Dist to Ctrl Home {}sec".format(iv), yAxisName="Mean Dist to Ctrl Home {}sec".format(iv))

        # ==================
        # perseveration vs task performance
        P.makeAScatterPlotWithFunc(lambda s: [((s.home_well_find_times[1] - s.away_well_leave_times[0]) / BTSession.TRODES_SAMPLING_RATE,
                                               s.avg_dwell_time(True, s.home_well, emptyVal=np.nan)
                                               )] if len(s.home_well_find_times) > 1 else [], "2nd_home_search_time_vs_avg_probe_home_dwell",
                                   colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)

        # ==================
        # time within certain dist to home
        radius = 18
        for iv in interestingProbeTimes:
            P.makeAPersevMeasurePlot("probe_time_near_home_{}sec_{}cm".format(iv, radius), lambda s, w: s.total_time_near_well(
                True, w, radius=radius, timeInterval=[0, iv]) / BTSession.TRODES_SAMPLING_RATE)
        P.makeAPersevMeasurePlot("probe_time_near_home_{}cm".format(radius), lambda s, w: s.total_time_near_well(
            True, w, radius=radius) / BTSession.TRODES_SAMPLING_RATE)

        # ==================
        # curvature
        for iv in interestingProbeTimes:
            P.makeAPersevMeasurePlot("probe_curvature_{}sec".format(
                iv), lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, iv]))
        P.makeAPersevMeasurePlot("probe_curvature".format(iv), lambda s,
                                 w: s.avg_curvature_at_well(True, w))

        # ==================
        # pct exploration bouts home well visitied
        def numVisits(well, nearestWells):
            return np.count_nonzero(np.array([k for k, g in groupby(nearestWells)]) == well)

        def didVisit(well, nearestWells):
            return any(np.array(nearestWells) == well)

        P.makeAPersevMeasurePlot("avg_visits_per_exploration_bout",
                                 lambda s, w: np.sum([numVisits(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_explore_bout_starts, s.probe_explore_bout_ends
                                 )]) / float(len(s.probe_explore_bout_starts)))
        P.makeAPersevMeasurePlot("pct_exploration_bouts_visited",
                                 lambda s, w: np.sum([didVisit(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_explore_bout_starts, s.probe_explore_bout_ends
                                 )]) / float(len(s.probe_explore_bout_starts)))

        # ==================
        # pct excursions home well visitied
        P.makeAPersevMeasurePlot("avg_visits_per_excursion",
                                 lambda s, w: np.sum([numVisits(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_excursion_starts, s.probe_excursion_ends
                                 )]) / float(len(s.probe_excursion_starts)))
        P.makeAPersevMeasurePlot("pct_excursions_visited",
                                 lambda s, w: np.sum([didVisit(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_excursion_starts, s.probe_excursion_ends
                                 )]) / float(len(s.probe_excursion_starts)))

    if PLOT_PROBE_BEHAVIOR_SUMMARIES:
        # ==========================================================================
        # behavior during probe
        # ==================
        # avg vel over time
        timeResolutions = [15, 30, 60]
        for tint in timeResolutions:
            intervalStarts = np.arange(0, 5*60, tint)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(True, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )], "probe_avg_vel_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True)

        # ==================
        # pct time exploring
        for tint in timeResolutions:
            intervalStarts = np.arange(0, 5*60, tint)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )], "probe_prop_time_explore_bout_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True)

    # ==========================================================================
    # session LFP
    def LFPPoints(sesh, idxs, marginFwd, marginBack, plotDiff=False):
        lfpFName = sesh.bt_lfp_fnames[-1]
        lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
        lfpV = lfpData[1]['voltage']
        lfpT = lfpData[0]['time']

        MARGIN_F_IDX = float(marginFwd) * BTRestSession.LFP_SAMPLING_RATE
        MARGIN_B_IDX = float(marginBack) * BTRestSession.LFP_SAMPLING_RATE

        ret = []
        for i in idxs:
            i1 = int(max(0, i - MARGIN_B_IDX))
            i2 = int(min(i+MARGIN_F_IDX, len(lfpV)))
            # print(i1, i2, i2-i1, MARGIN_B_IDX, MARGIN_F_IDX)
            x = lfpT[i1:i2].astype(float)
            x -= float(lfpT[i])
            y = lfpV[i1:i2].astype(float)
            if plotDiff:
                y = np.diff(y, prepend=y[0])
            ret.append((x, y))
            # print(x, y)

        return ret

    if PLOT_ITI_LFP:
        # ==================
        # Raw LFP
        P.makeALinePlot(lambda s: LFPPoints(
            s, [0], (s.probe_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE, 0), "Raw LFP")
        P.makeALinePlot(lambda s: LFPPoints(
            s, [0], (s.probe_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE, 0, plotDiff=True), "Raw LFP Diff")

        # ==================
        # Deflections and artifacts
        # P.makeALinePlot(lambda s: LFPPoints(s, s.artifactIdxs[s.artifactIdxs < s.ITIRippleIdxOffset], 0.3, 0.3),
        # "LFP artifacts", saveAllValuePairsSeparately=True)
        # P.makeALinePlot(lambda s: LFPPoints(s, s.interruptionIdxs[s.interruptionIdxs < s.ITIRippleIdxOffset], 0.3, 0.3),
        # "LFP interruptions", saveAllValuePairsSeparately=True)

        # P.makeALinePlot(lambda s: LFPPoints(s, s.interruptionIdxs[s.interruptionIdxs > s.ITIRippleIdxOffset], 0.3, 0.3),
        # "ITI LFP large deflections", saveAllValuePairsSeparately=True)

        # ==========================================================================
        # probe and ITI LFP
        # ==================
        # ITI
        # ==================
        # Just raw values
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipLens)), s.ITIRipLens)],
                        "riplens", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipStartIdxs)),
                        s.ITIRipStartIdxs)], "ripstartidxs", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipPeakIdxs)),
                        s.ITIRipPeakIdxs)], "rippeakidxs", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipPeakAmps)),
                        s.ITIRipPeakAmps)], "rippeakamps", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.array(s.ITIRipStartIdxs) / BTRestSession.LFP_SAMPLING_RATE,
                        s.ITIRipPeakAmps)], "rippeakamps over time", individualSessions=False)

        # ==================
        # cumulative ripple count
        P.makeALinePlot(lambda rs: [(np.hstack(([0], np.array(rs.ITIRipStartIdxs) / BTRestSession.LFP_SAMPLING_RATE)),
                        np.arange(len(rs.ITIRipStartIdxs)+1))], "iti cumulative num ripples",
                        colorFunc=lambda rs: [[
                            ("orange" if rs.isRippleInterruption else "cyan") for v in rs.ITIRipStartIdxs]],
                        individualSessions=False)

        # # cumulative ripple length
        # P.makeALinePlot(lambda rs: [(np.hstack(([0], np.array(rs.ITIRipStartIdxs) / BTRestSession.LFP_SAMPLING_RATE)),
        #                 np.hstack(([0], np.cumsum(rs.ITIRipLens))))], "iti cumulative ripple length",
        #                 colorFunc=lambda rs: [[
        #                     ("orange" if rs.isRippleInterruption else "cyan") for v in rs.ITIRipStartIdxs]],
        #                 individualSessions=False)

        # ==================
        # ripple length, amplitude, duration, rate
        P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.ITIRipLens) / BTRestSession.LFP_SAMPLING_RATE,
                             "iti Ripple length")
        P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.ITIRipPeakAmps),
                             "iti Ripple amp")
        P.makeASimpleBoxPlot(lambda rs: rs.ITIDuration, "iti duration")
        P.makeASimpleBoxPlot(lambda rs: float(len(rs.ITIRipLens)) /
                             rs.ITIDuration, "iti ripple rate")

        # ==================
        # same but detected using probe stats
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipLensProbeStats)), s.ITIRipLensProbeStats)],
                        "riplens", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipStartIdxsProbeStats)),
                        s.ITIRipStartIdxsProbeStats)], "ripstartidxs probestats", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipPeakIdxsProbeStats)),
                        s.ITIRipPeakIdxsProbeStats)], "rippeakidxs probestats", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.ITIRipPeakAmpsProbeStats)),
                        s.ITIRipPeakAmpsProbeStats)], "rippeakamps probestats", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.array(s.ITIRipStartIdxsProbeStats) / BTRestSession.LFP_SAMPLING_RATE,
                        s.ITIRipPeakAmpsProbeStats)], "rippeakamps over time probestats", individualSessions=False)

        # ==================
        # cumulative ripple count
        P.makeALinePlot(lambda rs: [(np.hstack(([0], np.array(rs.ITIRipStartIdxsProbeStats) / BTRestSession.LFP_SAMPLING_RATE)),
                        np.arange(len(rs.ITIRipStartIdxsProbeStats)+1))], "iti cumulative num ripples probestats",
                        colorFunc=lambda rs: [[
                            ("orange" if rs.isRippleInterruption else "cyan") for v in rs.ITIRipStartIdxsProbeStats]],
                        individualSessions=False)

        # ==================
        # ripple length, amplitude, duration, rate
        P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.ITIRipLensProbeStats) / BTRestSession.LFP_SAMPLING_RATE,
                             "iti Ripple length probestats")
        P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.ITIRipPeakAmpsProbeStats),
                             "iti Ripple amp probestats")
        P.makeASimpleBoxPlot(lambda rs: float(len(rs.ITIRipLensProbeStats)) /
                             rs.ITIDuration, "iti ripple rate probestats")

    if PLOT_INDIVIDUAL_RIPPLES_ITI:
        # ==================
        # Look at individual ripples
        def ITIRippleLFPPoints(sesh, useProbeStats=False):
            lfpFName = sesh.bt_lfp_fnames[-1]
            lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
            lfpV = lfpData[1]['voltage']
            lfpT = lfpData[0]['time']

            if useProbeStats:
                starts = sesh.ITIRipStartIdxsProbeStats + sesh.ITIRippleIdxOffset
                lens = sesh.ITIRipLensProbeStats
                peakIdxs = sesh.ITIRipPeakIdxsProbeStats + sesh.ITIRippleIdxOffset
            else:
                starts = sesh.ITIRipStartIdxs + sesh.ITIRippleIdxOffset
                lens = sesh.ITIRipLens
                peakIdxs = sesh.ITIRipPeakIdxs + sesh.ITIRippleIdxOffset

            MARGIN_MSEC = 20
            MARGIN_IDX = float(MARGIN_MSEC) / 1000.0 * BTRestSession.LFP_SAMPLING_RATE

            ret = []
            for (i, l, pi) in zip(starts, lens, peakIdxs):
                i1 = int(max(0, i - MARGIN_IDX))
                i2 = int(min(i+l+MARGIN_IDX, len(lfpV)))
                x = lfpT[i1:i2].astype(float)
                x -= float(lfpT[pi])
                # print(pi)
                # print(lfpT[pi])
                y = lfpV[i1:i2].astype(float)
                ret.append((x, y))

            return ret
        P.makeALinePlot(ITIRippleLFPPoints, "ITI ripples", saveAllValuePairsSeparately=True)
        P.makeALinePlot(lambda s: ITIRippleLFPPoints(s, True),
                        "ITI ripples probe stats", saveAllValuePairsSeparately=True)

    if PLOT_PROBE_LFP:
        # ==================
        # probe
        # ==================
        # Just raw values
        P.makeALinePlot(lambda s: [(np.arange(len(s.probeRipLens)), s.probeRipLens)],
                        "probe riplens", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.probeRipStartIdxs)),
                        s.probeRipStartIdxs)], "probe ripstartidxs", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.probeRipPeakIdxs)),
                        s.probeRipPeakIdxs)], "probe rippeakidxs", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.arange(len(s.probeRipPeakAmps)),
                        s.probeRipPeakAmps)], "probe rippeakamps", individualSessions=False)
        P.makeALinePlot(lambda s: [(np.array(s.probeRipStartIdxs) / BTRestSession.LFP_SAMPLING_RATE,
                        s.probeRipPeakAmps)], "probe rippeakamps over time", individualSessions=False)

        # ==================
        # cumulative ripple count
        P.makeALinePlot(lambda rs: [(np.hstack(([0], np.array(rs.probeRipStartIdxs) / BTRestSession.LFP_SAMPLING_RATE)),
                        np.arange(len(rs.probeRipStartIdxs)+1))], "probe cumulative num ripples",
                        colorFunc=lambda rs: [[
                            ("orange" if rs.isRippleInterruption else "cyan") for v in rs.probeRipStartIdxs]],
                        individualSessions=False)

        # # cumulative ripple length
        # P.makeALinePlot(lambda rs: [(np.hstack(([0], np.array(rs.probeRipStartIdxs) / BTRestSession.LFP_SAMPLING_RATE)),
        #                 np.hstack(([0], np.cumsum(rs.probeRipLens))))], "probe cumulative ripple length",
        #                 colorFunc=lambda rs: [[
        #                     ("orange" if rs.isRippleInterruption else "cyan") for v in rs.probeRipStartIdxs]],
        #                 individualSessions=False)

        # ==================
        # ripple length, amplitude, duration, rate
        P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.probeRipLens) / BTRestSession.LFP_SAMPLING_RATE,
                             "probe Ripple length")
        P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.probeRipPeakAmps),
                             "probe Ripple amp")
        P.makeASimpleBoxPlot(lambda rs: rs.probeDuration, "probe duration")
        P.makeASimpleBoxPlot(lambda rs: float(len(rs.probeRipLens)) /
                             rs.probeDuration, "probe ripple rate")

    if PLOT_REST_LFP:
        if len(alldata.allRestSessions) > 0:
            # ==========================================================================
            # Rest session LFP
            # ==================
            # raw vals
            P.makeALinePlot(lambda s: [(np.arange(len(s.ripLens)), s.ripLens)],
                            "rest riplens", individualSessions=False, restSessions=True)
            P.makeALinePlot(lambda s: [(np.arange(len(s.ripStartIdxs)),
                            s.ripStartIdxs)], "rest ripstartidxs", individualSessions=False, restSessions=True)
            P.makeALinePlot(lambda s: [(np.arange(len(s.ripPeakIdxs)),
                            s.ripPeakIdxs)], "rest rippeakidxs", individualSessions=False, restSessions=True)
            P.makeALinePlot(lambda s: [(np.arange(len(s.ripPeakAmps)),
                            s.ripPeakAmps)], "rest rippeakamps", individualSessions=False, restSessions=True)
            P.makeALinePlot(lambda s: [(np.array(s.ripStartIdxs) / BTRestSession.LFP_SAMPLING_RATE,
                            s.ripPeakAmps)], "rest rippeakamps over time", individualSessions=False, restSessions=True)

            # ==================
            # cumulative ripple count
            P.makeALinePlot(lambda rs: [(np.hstack(([0], np.array(rs.ripStartIdxs) / BTRestSession.LFP_SAMPLING_RATE)),
                            np.arange(len(rs.ripStartIdxs)+1))], "rest cumulative num ripples", restSessions=True,
                            colorFunc=lambda rs: [[
                                ("orange" if rs.btwpSession.isRippleInterruption else "cyan") for v in rs.ripStartIdxs]],
                            individualSessions=False)

            # # cumulative ripple length
            # P.makeALinePlot(lambda rs: [(np.hstack(([0], np.array(rs.ripStartIdxs) / BTRestSession.LFP_SAMPLING_RATE)),
            #                 np.hstack(([0], np.cumsum(rs.ripLens))))], "rest cumulative ripple length", restSessions=True,
            #                 colorFunc=lambda rs: [[
            #                     ("orange" if rs.btwpSession.isRippleInterruption else "cyan") for v in rs.ripStartIdxs]],
            #                 individualSessions=False)

            # ripple length, amplitude, duration, rate
            P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.ripLens) / BTRestSession.LFP_SAMPLING_RATE,
                                 "rest Ripple length", restSessions=True)
            P.makeASimpleBoxPlot(lambda rs: np.nanmean(rs.ripPeakAmps),
                                 "rest Ripple amp", restSessions=True)
            P.makeASimpleBoxPlot(lambda rs: rs.restDuration, "rest duration", restSessions=True)
            P.makeASimpleBoxPlot(lambda rs: float(len(rs.ripLens)) /
                                 rs.restDuration, "rest ripple rate", restSessions=True)

    if PLOT_DWELL_TIME_OVER_TIME:
        def valF(sesh):
            ret = []
            for wi, w in enumerate(P.all_well_names):
                xs = np.arange(sesh.num_well_entries(True, w))
                ys = (
                    sesh.probe_well_exit_times[wi] - sesh.probe_well_entry_times[wi]) / BTSession().TRODES_SAMPLING_RATE
                ret.append((xs, ys))

            return ret

        def colorF(sesh):
            ret = []
            for w in P.all_well_names:
                n = sesh.num_well_entries(True, w)
                if w == sesh.home_well:
                    ret.append(["green"] * n)
                elif w in sesh.visited_away_wells:
                    ret.append(["red"] * n)
                else:
                    ret.append(["gray"] * n)

            return ret

        # P.makeALinePlot(valF, "wellDwellTimesOverProbe", colorFunc=colorF, linewidth=0.5)
        P.makeALinePlot(lambda s: [(np.arange(s.num_well_entries(True, s.home_well)),
                                    (s.probe_well_exit_times[s.home_well_idx_in_allwells] - s.probe_well_entry_times[s.home_well_idx_in_allwells]) / BTSession.TRODES_SAMPLING_RATE)],
                        "home dwell time over probe", colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)

        def valF(sesh, emptyVal=-1.0):
            hwi = sesh.home_well_idx_in_allwells
            ents = sesh.probe_well_entry_times[hwi]
            exts = sesh.probe_well_exit_times[hwi]
            if len(ents) == 0:
                return [(emptyVal, emptyVal)]

            dwellTimes = (exts - ents) / BTSession.TRODES_SAMPLING_RATE
            if len(dwellTimes) == 1:
                return [(dwellTimes[0], emptyVal)]

            return [(dwellTimes[0], dwellTimes[1])]

        P.makeAScatterPlotWithFunc(valF, "First and second home dwell times",
                                   colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
