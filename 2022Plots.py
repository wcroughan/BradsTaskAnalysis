import os
import numpy as np
from itertools import groupby
from random import random
import MountainViewIO
import matplotlib.pyplot as plt

from MyPlottingFunctions import MyPlottingFunctions
from BTData import BTData
from BTSession import BTSession
from BTRestSession import BTRestSession

PLOT_BEHAVIOR_TRACES = 5
PLOT_BEHAVIOR_SUMMARIES = 3
PLOT_PERSEV_MEASURES = 2
PLOT_PROBE_BEHAVIOR_SUMMARIES = 2
PLOT_TASK_BEHAVIOR_SUMMARIES = 2
PLOT_PROBE_VS_TASK_BEHAVIOR_SUMMARIES = 2
PLOT_ITI_LFP = 7
PLOT_INDIVIDUAL_RIPPLES_ITI = 10
PLOT_PROBE_LFP = 9
PLOT_REST_LFP = 10
PLOT_DWELL_TIME_OVER_TIME = 7
PLOT_PER_SESSION_ADDITIONAL = 8
PLOT_INTERRUPTION_INFO = 3
PLOT_RAW_LFP = 3
PLOT_TASK_INDIVIDUAL_RIPPLES = 8
PLOT_RIPPLES_VS_PERSEV = 3
PLOT_THESIS_COMMITTEE_FIGURES = 1

ONLY_COMBO = False

animals = []
animals += ["combo"]
animals += ["B13"]
animals += ["B14"]
animals += ["Martin"]


possibleDataDirs = ["/media/WDC7/", "/media/fosterlab/WDC7/", "/home/wcroughan/data/"]
dataDir = None
for dd in possibleDataDirs:
    if os.path.exists(dd):
        dataDir = dd
        break

if dataDir == None:
    print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
    exit()


def numVisits(well, nearestWells):
    return np.count_nonzero(np.array([k for k, g in groupby(nearestWells)]) == well)


def didVisit(well, nearestWells):
    return any(np.array(nearestWells) == well)


def numWellsVisited(nearestWells, countReturns=False):
    g = groupby(nearestWells)
    if countReturns:
        return len([k for k, _ in g])
    else:
        return len(set([k for k, _ in g]))


for animal_name in animals:
    if animal_name != "combo":
        if ONLY_COMBO:
            continue

        if animal_name == "B13":
            data_filename = os.path.join(dataDir, "B13/processed_data/B13_bradtask.dat")
            output_dir = os.path.join(dataDir, "B13/processed_data/behavior_figures")
        elif animal_name == "B14":
            data_filename = os.path.join(dataDir, "B14/processed_data/B14_bradtask.dat")
            output_dir = os.path.join(dataDir, "B14/processed_data/behavior_figures")
        elif animal_name == "Martin":
            data_filename = os.path.join(dataDir, "Martin/processed_data/martin_bradtask.dat")
            output_dir = os.path.join(dataDir, "Martin/processed_data/behavior_figures")
        else:
            raise Exception("Unknown rat " + animal_name)

        print("Loading data for " + animal_name)
        ratData = BTData()
        ratData.loadFromFile(data_filename)
        P = MyPlottingFunctions(ratData, output_dir, showPlots=False)
        print("starting plots for", animal_name)

        FIG_LEVEL = 2

    else:
        output_dir = os.path.join(dataDir, "combined_figures")
        allData = []
        for an in animals:
            if an == "combo":
                continue
            elif an == "B13":
                data_filename = os.path.join(dataDir, "B13/processed_data/B13_bradtask.dat")
            elif an == "B14":
                data_filename = os.path.join(dataDir, "B14/processed_data/B14_bradtask.dat")
            elif an == "Martin":
                data_filename = os.path.join(dataDir, "Martin/processed_data/martin_bradtask.dat")
            else:
                raise Exception("Unknown rat " + an)

            print("Loading data for " + an)
            ratData = BTData()
            ratData.loadFromFile(data_filename)
            allData.append(ratData)

        P = MyPlottingFunctions(allData, output_dir, showPlots=False)
        print("starting plots for combo")
        FIG_LEVEL = 2

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    showAdditional = PLOT_PER_SESSION_ADDITIONAL < FIG_LEVEL

    if PLOT_BEHAVIOR_TRACES < FIG_LEVEL:
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

    if PLOT_BEHAVIOR_SUMMARIES < FIG_LEVEL:
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

    if PLOT_PERSEV_MEASURES < FIG_LEVEL:
        # ==========================================================================
        # Perseveration measures
        # ==================
        # dwell time
        # interestingProbeTimes = []

        interestingProbeTimes = [60, 90]
        for iv in interestingProbeTimes:
            P.makeAPersevMeasurePlot("probe_total_dwell_time_{}sec".format(iv),
                                     lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=showAdditional)
            P.makeAPersevMeasurePlot("probe_avg_dwell_time_{}sec".format(iv),
                                     lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, iv], emptyVal=np.nan), alsoMakePerWellPerSessionPlot=showAdditional)
            P.makeAPersevMeasurePlot("probe_avg_dwell_time_{}sec_outlier_fix".format(iv),
                                     lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, iv], emptyVal=np.nan), yAxisLims=(0, 10), alsoMakePerWellPerSessionPlot=False)
        P.makeAPersevMeasurePlot("probe_total_dwell_time",
                                 lambda s, w: s.total_dwell_time(True, w), alsoMakePerWellPerSessionPlot=showAdditional)
        P.makeAPersevMeasurePlot("probe_avg_dwell_time",
                                 lambda s, w: s.avg_dwell_time(True, w, emptyVal=np.nan), alsoMakePerWellPerSessionPlot=showAdditional)

        P.makeAPersevMeasurePlot("probe_max_dwell_time",
                                 lambda s, w: s.avg_dwell_time(True, w, emptyVal=0.0, avgFunc=np.nanmax), alsoMakePerWellPerSessionPlot=showAdditional)

        # ==================
        # num entries
        for iv in interestingProbeTimes:
            P.makeAPersevMeasurePlot("probe_num_entries_{}sec".format(iv),
                                     lambda s, w: s.num_well_entries(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=showAdditional)
        P.makeAPersevMeasurePlot("probe_num_entries",
                                 lambda s, w: s.total_dwell_time(True, w), alsoMakePerWellPerSessionPlot=showAdditional)

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
                True, w, radius=radius, timeInterval=[0, iv]) / BTSession.TRODES_SAMPLING_RATE, alsoMakePerWellPerSessionPlot=showAdditional)
        P.makeAPersevMeasurePlot("probe_time_near_home_{}cm".format(radius), lambda s, w: s.total_time_near_well(
            True, w, radius=radius) / BTSession.TRODES_SAMPLING_RATE, alsoMakePerWellPerSessionPlot=showAdditional)

        # ==================
        # curvature
        for iv in interestingProbeTimes:
            P.makeAPersevMeasurePlot("probe_curvature_{}sec".format(
                iv), lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, iv]), alsoMakePerWellPerSessionPlot=showAdditional)
        P.makeAPersevMeasurePlot("probe_curvature".format(iv), lambda s,
                                 w: s.avg_curvature_at_well(True, w), alsoMakePerWellPerSessionPlot=showAdditional)

        # ==================
        # pct exploration bouts home well visitied
        P.makeAPersevMeasurePlot("avg_visits_per_exploration_bout",
                                 lambda s, w: np.sum([numVisits(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_explore_bout_starts, s.probe_explore_bout_ends
                                 )]) / float(len(s.probe_explore_bout_starts)), alsoMakePerWellPerSessionPlot=showAdditional)
        P.makeAPersevMeasurePlot("pct_exploration_bouts_visited",
                                 lambda s, w: np.sum([didVisit(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_explore_bout_starts, s.probe_explore_bout_ends
                                 )]) / float(len(s.probe_explore_bout_starts)), alsoMakePerWellPerSessionPlot=showAdditional)

        # ==================
        # pct excursions home well visitied
        P.makeAPersevMeasurePlot("avg_visits_per_excursion",
                                 lambda s, w: np.sum([numVisits(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_excursion_starts, s.probe_excursion_ends
                                 )]) / float(len(s.probe_excursion_starts)), alsoMakePerWellPerSessionPlot=showAdditional)
        P.makeAPersevMeasurePlot("pct_excursions_visited",
                                 lambda s, w: np.sum([didVisit(w, s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
                                     s.probe_excursion_starts, s.probe_excursion_ends
                                 )]) / float(len(s.probe_excursion_starts)), alsoMakePerWellPerSessionPlot=showAdditional)

    timeResolutions = [15, 30, 60]
    if PLOT_TASK_BEHAVIOR_SUMMARIES < FIG_LEVEL:
        # ==========================================================================
        # behavior during task (same measures as probe stuff below)
        # ==================
        # avg vel over time
        for tint in timeResolutions:
            intervalStarts = np.arange(0, 5*60, tint)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(False, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )], "task_avg_vel_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(False, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "task_avg_vel_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem")
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1*tint,
                                        [s.mean_vel(False, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "task_avg_vel_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem")

            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True)
                                            for i1 in intervalStarts]
                                        )], "task_avg_vel_{}sec_just_moving".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True)
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "task_avg_vel_by_cond_just_moving_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem")
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1*tint,
                                        [s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True)
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "task_avg_vel_by_cond_just_moving_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem")

            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [min(s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )], "task_avg_vel_{}sec_just_moving_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [min(s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "task_avg_vel_by_cond_just_moving_{}sec_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem")
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1*tint,
                                        [min(s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "task_avg_vel_by_cond_just_moving_{}sec_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem")

        # ==================
        # pct time exploring
        for tint in timeResolutions:
            intervalStarts = np.arange(0, 5*60, tint)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.prop_time_in_bout_state(False, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )], "task_prop_time_explore_bout_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True, onlyPlotAverage=True)

            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.prop_time_in_bout_state(False, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "task_prop_time_explore_bout_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem", onlyPlotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1 * tint,
                                        [s.prop_time_in_bout_state(False, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "task_prop_time_explore_bout_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem", onlyPlotAverage=True)

        # ==================
        # number of bouts/excursions
        P.makeASimpleBoxPlot(lambda s: len(s.bt_explore_bout_lens_secs), "num explore bouts task")
        P.makeASimpleBoxPlot(lambda s: len(s.bt_excursion_starts), "num excursions task")

        # ==================
        # how long did they last?
        P.makeASimpleBoxPlot(lambda s: np.nanmean(
            s.bt_explore_bout_lens_secs), "len explore bouts task")
        P.makeASimpleBoxPlot(lambda s: np.nanmean(s.bt_excursion_ends -
                             s.bt_excursion_starts), "len excursions task")

        # ==================
        # How many wells visited?
        P.makeASimpleBoxPlot(lambda s: numWellsVisited(
            s.bt_nearest_wells), "num wells visited full task")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.bt_explore_bout_starts, s.bt_explore_bout_ends
        )]), "num wells visited per explore bout task")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )]), "num wells visited per excursion task")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.bt_explore_bout_starts, s.bt_explore_bout_ends
        )]), "num wells visited per explore bout with repeats task")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )]), "num wells visited per excursion with repeats task")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.bt_explore_bout_starts, s.bt_explore_bout_ends
        )]), "num repeats per explore bout task")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )]), "num repeats per excursion task")

    if PLOT_PROBE_BEHAVIOR_SUMMARIES < FIG_LEVEL:
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
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(True, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "probe_avg_vel_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem")
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1*tint,
                                        [s.mean_vel(True, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "probe_avg_vel_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem")

            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(True, timeInterval=[i1, i1+tint], onlyMoving=True)
                                            for i1 in intervalStarts]
                                        )], "probe_avg_vel_{}sec_just_moving".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.mean_vel(True, timeInterval=[i1, i1+tint], onlyMoving=True)
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "probe_avg_vel_by_cond_just_moving_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem")
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1*tint,
                                        [s.mean_vel(True, timeInterval=[i1, i1+tint], onlyMoving=True)
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "probe_avg_vel_by_cond_just_moving_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem")

            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [min(s.mean_vel(True, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )], "probe_avg_vel_{}sec_just_moving_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [min(s.mean_vel(True, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "probe_avg_vel_by_cond_just_moving_{}sec_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem")
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1*tint,
                                        [min(s.mean_vel(True, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "probe_avg_vel_by_cond_just_moving_{}sec_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem")

        # ==================
        # pct time exploring
        for tint in timeResolutions:
            intervalStarts = np.arange(0, 5*60, tint)
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )], "probe_prop_time_explore_bout_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: [[("orange" if s.isRippleInterruption else "cyan") for v in range(len(intervalStarts))]], plotAverage=True, onlyPlotAverage=True)

            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "probe_prop_time_explore_bout_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem", onlyPlotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1 * tint,
                                        [s.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "probe_prop_time_explore_bout_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem", onlyPlotAverage=True)

        # ==================
        # number of bouts/excursions
        P.makeASimpleBoxPlot(lambda s: len(s.probe_explore_bout_lens_secs),
                             "num explore bouts probe")
        P.makeASimpleBoxPlot(lambda s: len(s.probe_excursion_starts), "num excursions probe")

        # ==================
        # how long did they last?
        P.makeASimpleBoxPlot(lambda s: np.nanmean(
            s.probe_explore_bout_lens_secs), "len explore bouts probe")
        P.makeASimpleBoxPlot(lambda s: np.nanmean(s.probe_excursion_ends -
                             s.probe_excursion_starts), "len excursions probe")

        # ==================
        # How many wells visited?
        P.makeASimpleBoxPlot(lambda s: numWellsVisited(
            s.probe_nearest_wells), "num wells visited full probe")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.probe_explore_bout_starts, s.probe_explore_bout_ends
        )]), "num wells visited per explore bout")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )]), "num wells visited per excursion")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.probe_explore_bout_starts, s.probe_explore_bout_ends
        )]), "num wells visited per explore bout with repeats")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )]), "num wells visited per excursion with repeats")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.probe_explore_bout_starts, s.probe_explore_bout_ends
        )]), "num repeats per explore bout")
        P.makeASimpleBoxPlot(lambda s: np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )]), "num repeats per excursion")

        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.probe_explore_bout_starts, s.probe_explore_bout_ends
        )]), s.avg_dwell_time(True, s.home_well, timeInterval=[0, 90], emptyVal=-1.0))], "num repeats per explore bout vs avg dwell time 90sec",
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)

    if PLOT_PROBE_VS_TASK_BEHAVIOR_SUMMARIES < FIG_LEVEL:
        # ==================
        # How are task and probe exploration related?
        P.makeAScatterPlotWithFunc(lambda s: [(len(s.bt_explore_bout_lens_secs), len(s.probe_explore_bout_lens_secs))], "probe vs bt num bouts",
                                   colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(len(s.bt_excursion_lens_secs), len(s.probe_excursion_lens_secs))], "probe vs bt num excursions",
                                   colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean(s.bt_explore_bout_lens_secs), np.nanmean(s.probe_explore_bout_lens_secs))], "probe vs bt bouts lens",
                                   colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean(s.bt_excursion_lens_secs), np.nanmean(s.probe_excursion_lens_secs))], "probe vs bt excursions lens",
                                   colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.bt_explore_bout_starts, s.bt_explore_bout_ends
        )]), np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.probe_explore_bout_starts, s.probe_explore_bout_ends
        )]))], "probe vs bt num wells visited per explore bout",
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )]), np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2]) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )]))], "probe vs bt num wells visited per excursion",
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.bt_explore_bout_starts, s.bt_explore_bout_ends
        )]), np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.probe_explore_bout_starts, s.probe_explore_bout_ends
        )]))], "probe vs bt num wells visited per explore bout with repeats",
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )]), np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )]))], "probe vs bt num wells visited per excursion with repeats",
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.bt_explore_bout_starts, s.bt_explore_bout_ends
        )]), np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.probe_explore_bout_starts, s.probe_explore_bout_ends
        )]))], "probe vs bt num repeats per explore bout",
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)
        P.makeAScatterPlotWithFunc(lambda s: [(np.nanmean([numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.bt_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.bt_excursion_starts, s.bt_excursion_ends
        )]), np.nanmean([numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=True) - numWellsVisited(s.probe_nearest_wells[i1:i2], countReturns=False) for (i1, i2) in zip(
            s.probe_excursion_starts, s.probe_excursion_ends
        )]))], "probe vs bt num repeats per excursion",
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"], individualSessions=False)

    # ==========================================================================
    # session LFP

    def LFPPoints(sesh, idxs, marginFwd, marginBack, step=1, plotDiff=False):
        print("{} LFP idxs".format(len(idxs)))
        lfpFName = sesh.bt_lfp_fnames[-1]
        lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
        lfpV = lfpData[1]['voltage']
        lfpT = lfpData[0]['time']

        MARGIN_F_IDX = float(marginFwd) * BTRestSession.LFP_SAMPLING_RATE
        MARGIN_B_IDX = float(marginBack) * BTRestSession.LFP_SAMPLING_RATE

        ret = []
        for i in range(0, len(idxs), step):
            idx = idxs[i]
            i1 = int(max(0, idx - MARGIN_B_IDX))
            i2 = int(min(idx+MARGIN_F_IDX, len(lfpV)))
            # print(i1, i2, i2-i1, MARGIN_B_IDX, MARGIN_F_IDX)
            x = lfpT[i1:i2].astype(float)
            x -= float(lfpT[idx])
            y = lfpV[i1:i2].astype(float)
            if plotDiff:
                y = np.diff(y, prepend=y[0])
            ret.append((x, y))
            # print(x, y)

        return ret

    if PLOT_RAW_LFP < FIG_LEVEL:
        # ==================
        # Raw LFP
        P.makeALinePlot(lambda s: LFPPoints(
            s, [0], (s.probe_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE, 0), "Raw LFP")
        # P.makeALinePlot(lambda s: LFPPoints(
        # s, [0], (s.probe_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE, 0, plotDiff=True), "Raw LFP Diff")

    if PLOT_TASK_INDIVIDUAL_RIPPLES < FIG_LEVEL:
        P.makeALinePlot(lambda s: LFPPoints(
            s, s.btRipStartIdxsProbeStats, 0.4, 0.4, step=100
        ), "Task Ripples probe stats", saveAllValuePairsSeparately=True)

    if PLOT_RIPPLES_VS_PERSEV < FIG_LEVEL:
        P.makeAScatterPlotWithFunc(
            lambda s: [(len(s.btRipStartIdxsProbeStats) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE),
                        s.avg_dwell_time(True, s.home_well, timeInterval=[0, 90], emptyVal=-1.0))],
            "total ripple rate vs home dwell in 90sec", individualSessions=False,
            colorFunc=lambda s: "orange" if s.isRippleInterruption else "cyan")
        P.makeAScatterPlotWithFunc(
            lambda s: [(len(s.bt_interruption_pos_idxs) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE),
                        s.avg_dwell_time(True, s.home_well, timeInterval=[0, 90], emptyVal=-1.0))],
            "total stim rate vs home dwell in 90sec", individualSessions=False,
            colorFunc=lambda s: "orange" if s.isRippleInterruption else "cyan")
        P.makeAScatterPlotWithFunc(
            lambda s: [((len(s.btRipStartIdxsProbeStats) + len(s.bt_interruption_pos_idxs)) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE),
                        s.avg_dwell_time(True, s.home_well, timeInterval=[0, 90], emptyVal=-1.0))],
            "total ripple plus stim rate vs home dwell in 90sec", individualSessions=False,
            colorFunc=lambda s: "orange" if s.isRippleInterruption else "cyan")

        P.makeAScatterPlotWithFunc(
            lambda s: [(len(s.btRipStartIdxsProbeStats) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE),
                        s.avg_dwell_time(True, s.home_well, emptyVal=-1.0, avgFunc=np.nanmax))],
            "total ripple rate vs max home dwell", individualSessions=False,
            colorFunc=lambda s: "orange" if s.isRippleInterruption else "cyan")
        P.makeAScatterPlotWithFunc(
            lambda s: [(len(s.bt_interruption_pos_idxs) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE),
                        s.avg_dwell_time(True, s.home_well, emptyVal=-1.0, avgFunc=np.nanmax))],
            "total stim rate vs max home dwell", individualSessions=False,
            colorFunc=lambda s: "orange" if s.isRippleInterruption else "cyan")
        P.makeAScatterPlotWithFunc(
            lambda s: [((len(s.btRipStartIdxsProbeStats) + len(s.bt_interruption_pos_idxs)) / ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE),
                        s.avg_dwell_time(True, s.home_well, emptyVal=-1.0, avgFunc=np.nanmax))],
            "total ripple plus stim rate vs max home dwell", individualSessions=False,
            colorFunc=lambda s: "orange" if s.isRippleInterruption else "cyan")

    if PLOT_ITI_LFP < FIG_LEVEL:
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

    if PLOT_INDIVIDUAL_RIPPLES_ITI < FIG_LEVEL:
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

    if PLOT_PROBE_LFP < FIG_LEVEL:
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

    if PLOT_REST_LFP < FIG_LEVEL:
        if len(P.all_rest_sessions) > 0:
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

    if PLOT_DWELL_TIME_OVER_TIME < FIG_LEVEL:
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

    # where interrupted, how much?, cumulative num interruptions, compare conditions
    if PLOT_INTERRUPTION_INFO < FIG_LEVEL:
        if False:
            # ==================
            # stim locations
            print("stim locations")
            for sesh in P.all_sessions_with_probe:
                plt.clf()
                plt.plot(sesh.bt_pos_xs, sesh.bt_pos_ys)
                xs = np.array(sesh.bt_pos_xs)
                ys = np.array(sesh.bt_pos_ys)
                its = sesh.interruption_timestamps
                its = its[its < sesh.bt_pos_ts[-1]]
                ids = np.searchsorted(sesh.bt_pos_ts, its)
                plt.scatter(xs[ids], ys[ids], c=(
                    "orange" if sesh.isRippleInterruption else "cyan"), zorder=2)
                P.saveOrShow("{}_{}".format("stim locations", sesh.name))

            # ==================
            # stim count and rate by well type
            P.makeAPersevMeasurePlot("stim count by well type",
                                     lambda s, w: s.numStimsAtWell(w))
            P.makeAPersevMeasurePlot("stim rate by well type", lambda s,
                                     w: s.numStimsAtWell(w) / s.total_dwell_time(False, w))

            # ==================
            # cumulative stim count over task
            def valF(sesh):
                ts = np.array(sesh.interruption_timestamps)
                ts = ts[np.logical_and(ts > sesh.bt_pos_ts[0], ts <
                                       sesh.bt_pos_ts[-1])] - sesh.bt_pos_ts[0]
                ts /= BTSession.TRODES_SAMPLING_RATE
                ts = np.hstack(([0], ts))
                return [(ts, np.arange(len(ts)))]

            P.makeALinePlot(valF, "task cumulative num stims",
                            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"],
                            individualSessions=False)

            # ==================
            # stim count rate and rate by condition
            P.makeASimpleBoxPlot(lambda s: len(s.bt_interruption_pos_idxs) /
                                 ((s.bt_pos_ts[-1] - s.bt_pos_ts[0]) / BTSession.TRODES_SAMPLING_RATE), "total stim rate")

        # ==================
        # compare interruption count/rate to dwell time in probe
        P.makeAScatterPlotWithFunc(
            lambda s: [(s.numStimsAtWell(s.home_well), s.avg_dwell_time(
                True, s.home_well, timeInterval=[0, 90], emptyVal=-1.0))],
            "stim count vs home dwell in 90sec", individualSessions=False,
            colorFunc=lambda s: ["orange" if s.isRippleInterruption else "cyan"])

        P.makeAScatterPlotWithFunc(
            lambda s: [(s.numStimsAtWell(w), s.avg_dwell_time(True, w, timeInterval=[
                        0, 90], emptyVal=-1.0)) for w in P.all_well_names],
            "stim count vs dwell in 90sec all wells", individualSessions=True,
            colorFunc=lambda s: ["green" if w == s.home_well else ("red" if w in s.visited_away_wells else "gray") for w in P.all_well_names])

        P.makeAScatterPlotWithFunc(
            lambda s: [(s.numStimsAtWell(w), s.avg_dwell_time(True, w, timeInterval=[
                        0, 90], emptyVal=-1.0)) for w in (s.visited_away_wells + [s.home_well])],
            "stim count vs dwell in 90sec rewarded wells", individualSessions=False, bigDots=False,
            colorFunc=lambda s: ["green" if w == s.home_well else ("red" if w in s.visited_away_wells else "gray") for w in (s.visited_away_wells + [s.home_well])])

        P.makeAScatterPlotWithFunc(
            lambda s: [(s.numStimsAtWell(s.home_well) / s.total_dwell_time(False, s.home_well), s.avg_dwell_time(
                True, s.home_well, timeInterval=[0, 90], emptyVal=-1.0))],
            "stim rate vs home dwell in 90sec", individualSessions=False,
            colorFunc=lambda s: "orange" if s.isRippleInterruption else "cyan")

        P.makeAScatterPlotWithFunc(
            lambda s: [(s.numStimsAtWell(w) / s.total_dwell_time(False, w), s.avg_dwell_time(True, w, timeInterval=[
                        0, 90], emptyVal=-1.0)) for w in P.all_well_names],
            "stim rate vs dwell in 90sec all wells", individualSessions=True,
            colorFunc=lambda s: ["green" if w == s.home_well else ("red" if w in s.visited_away_wells else "gray") for w in P.all_well_names])

        P.makeAScatterPlotWithFunc(
            lambda s: [(s.numStimsAtWell(w) / s.total_dwell_time(False, w), s.avg_dwell_time(True, w, timeInterval=[
                        0, 90], emptyVal=-1.0)) for w in (s.visited_away_wells + [s.home_well])],
            "stim rate vs dwell in 90sec rewarded wells", individualSessions=False, bigDots=False,
            colorFunc=lambda s: ["green" if w == s.home_well else ("red" if w in s.visited_away_wells else "gray") for w in (s.visited_away_wells + [s.home_well])])

    if PLOT_THESIS_COMMITTEE_FIGURES < FIG_LEVEL:
        # behavior trace
        # probe traces, labeled/grouped by condition
        P.makeALinePlot(lambda s: [(s.bt_pos_xs, s.bt_pos_ys)],
                        "Raw Behavior Task", axisLims="environment")
        P.makeALinePlot(lambda s: [(s.probe_pos_xs, s.probe_pos_ys)],
                        "Raw Behavior Probe", axisLims="environment")

        # home vs away latencies
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

        for tint in timeResolutions:
            intervalStarts = np.arange(0, 5*60, tint)
            # pct time exploring
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [s.prop_time_in_bout_state(False, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "task_prop_time_explore_bout_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem", onlyPlotAverage=True)
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1 * tint,
                                        [s.prop_time_in_bout_state(False, BTSession.BOUT_STATE_EXPLORE, timeInterval=[i1, i1+tint])
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "task_prop_time_explore_bout_by_cond_{}sec".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem", onlyPlotAverage=True)

            # velocity
            P.makeALinePlot(lambda s: [(intervalStarts,
                                        [min(s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )] if s.isRippleInterruption else [], "task_avg_vel_by_cond_just_moving_{}sec_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["orange"], plotAverage=True, avgError="sem")
            P.makeALinePlot(lambda s: [(intervalStarts + 0.1*tint,
                                        [min(s.mean_vel(False, timeInterval=[i1, i1+tint], onlyMoving=True), 45)
                                            for i1 in intervalStarts]
                                        )] if not s.isRippleInterruption else [], "task_avg_vel_by_cond_just_moving_{}sec_outlier_fix".format(tint), individualSessions=False,
                            colorFunc=lambda s: ["cyan"], plotAverage=True, holdLastPlot=True, avgError="sem")

        # num wells visited
        P.makeASimpleBoxPlot(lambda s: numWellsVisited(
            s.probe_nearest_wells), "num wells visited full probe")
