from BTData import *
import pandas as pd
import numpy as np
import os
import csv
import glob
import json
import matplotlib.pyplot as plt
import random
# import MountainViewIO
# import InterruptionAnalysis

INSPECT_IN_DETAIL = []
# INSPECT_IN_DETAIL = ["20200526"]
INSPECT_NANVALS = False
INSPECT_PROBE_BEHAVIOR_PLOT = False
INSPECT_PLOT_WELL_OCCUPANCIES = False
ENFORCE_DIFFERENT_WELL_COLORS = False
RUN_JUST_SPECIFIED = False
SPECIFIED_DAYS = ["20200604"]

TEST_NEAREST_WELL = False

data_dir = '/media/WDC4/martindata/bradtask/'
all_data_dirs = sorted(os.listdir(data_dir), key=lambda s: (
    s.split('_')[0], s.split('_')[1]))
behavior_notes_dir = os.path.join(data_dir, 'behavior_notes')

excluded_dates = ["20200528", "20200630", "20200702", "20200703"]
excluded_dates += ["20200531",  "20200603", "20200602",
                   "20200606", "20200605", "20200601"]
excluded_dates += ["20200526"]

excluded_sessions = ["20200624_1", "20200624_2", "20200628_2"]

animal_name = 'Martin'
output_dir = '/media/WDC4/martindata/bradtask/'
out_filename = "martin_bradtask.dat"

if os.path.exists(os.path.join(output_dir, out_filename)):
    # confirm = input("Output file exists already. Overwrite? (y/n):")
    confirm = "y"
    if confirm != "y":
        exit()

all_well_idxs = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])
all_quadrant_idxs = [0, 1, 2, 3]

MAX_JUMP_DISTANCE = 50
N_CLEANING_REPS = 2
X_START = 200
X_FINISH = 1175
Y_START = 20
Y_FINISH = 1275
RADIUS = 50  # pixels
VEL_THRESH = 10  # cm/s
PIXELS_PER_CM = 5.0
TRODES_SAMPLING_RATE = 30000
# 0.8 means transition from well a -> b requires rat dist to b to be 0.8 * dist to a
SWITCH_WELL_FACTOR = 0.8

# Typical observed amplitude of LFP deflection on stimulation
DEFLECTION_THRESHOLD = 2000.0
SAMPLING_RATE = 30000.0  # Rate at which timestamp data is sampled
# LFP is subsampled. The timestamps give time according to SAMPLING_RATE above
LFP_SAMPLING_RATE = 1500.0
# Typical duration of the stimulation artifact - For peak detection
MIN_ARTIFACT_PERIOD = int(0.1 * LFP_SAMPLING_RATE)
# Typical duration of a Sharp-Wave Ripple
ACCEPTED_RIPPLE_LENGTH = int(0.2 * LFP_SAMPLING_RATE)


# def readPositionData(data_filename):
#     trajectory_data = None
#     try:
#         with open(data_filename, 'r') as data_file:
#             timestamp_data = list()
#             x_data = list()
#             y_data = list()
#             csv_reader = csv.reader(data_file)
#             n_elements = 0
#             for data_row in csv_reader:
#                 if data_row:
#                     n_elements += 1
#                     timestamp_data.append(int(data_row[0]))
#                     x_data.append(int(data_row[1]))
#                     y_data.append(int(data_row[2]))
#             trajectory_data = np.empty((n_elements, 3), dtype=np.uint32)
#             trajectory_data[:, 0] = timestamp_data[:]
#             trajectory_data[:, 2] = x_data[:]
#             trajectory_data[:, 1] = y_data[:]
#     except Exception as err:
#         print(err)
#     return trajectory_data

# well_coords_file = '/home/wcroughan/repos/BradsTaskAnalysis/well_locations.csv'
def readWellCoordsFile(well_coords_file):
    with open(well_coords_file, 'r') as wcf:
        well_coords_map = {}
        csv_reader = csv.reader(wcf)
        for data_row in csv_reader:
            try:
                well_coords_map[int(data_row[0])] = (
                    int(data_row[1]), int(data_row[2]))
            except Exception as err:
                if data_row[1] != '':
                    print(err)

        return well_coords_map


def readRawPositionData(data_filename):
    try:
        with open(data_filename, 'rb') as datafile:
            dt = np.dtype([('timestamp', np.uint32), ('x1', np.uint16),
                           ('y1', np.uint16), ('x2', np.uint16), ('y2', np.uint16)])
            l = ""
            max_iter = 8
            iter = 0
            while l != b'<end settings>\n':
                l = datafile.readline().lower()
                # print(l)
                iter += 1
                if iter > max_iter:
                    raise Exception
            return np.fromfile(datafile, dtype=dt)
    except Exception as err:
        print(err)
        return 0


def readClipData(data_filename):
    time_clips = None
    try:
        with open(data_filename, 'r') as data_file:
            start_times = list()
            finish_times = list()
            csv_reader = csv.reader(data_file)
            n_time_clips = 0
            for data_row in csv_reader:
                if data_row:
                    n_time_clips += 1
                    start_times.append(int(data_row[1]))
                    finish_times.append(int(data_row[2]))
            time_clips = np.empty((n_time_clips, 2), dtype=np.uint32)
            time_clips[:, 0] = start_times[:]
            time_clips[:, 1] = finish_times[:]
    except Exception as err:
        print(err)
    return time_clips


def processPosData(position_data):
    x_pos = np.array(position_data['x1'], dtype=float)
    y_pos = np.array(position_data['y1'], dtype=float)

    # Interpolate the position data into evenly sampled time points
    x = np.linspace(position_data['timestamp'][0],
                    position_data['timestamp'][-1], position_data.shape[0])
    xp = position_data['timestamp']
    x_pos = np.interp(x, xp, position_data['x1'])
    y_pos = np.interp(x, xp, position_data['y1'])
    position_sampling_frequency = TRODES_SAMPLING_RATE/np.diff(x)[0]
    # Interpolated Timestamps:
    position_data['timestamp'] = x

    # Remove large jumps in position (tracking errors)
    for _ in range(N_CLEANING_REPS):
        jump_distance = np.sqrt(np.square(np.diff(x_pos, prepend=x_pos[0])) +
                                np.square(np.diff(y_pos, prepend=y_pos[0])))
        # print(jump_distance)
        points_in_range = (x_pos > X_START) & (x_pos < X_FINISH) &\
            (y_pos > Y_START) & (y_pos < Y_FINISH)
        clean_points = jump_distance < MAX_JUMP_DISTANCE

    # substitute them with NaNs then interpolate
    x_pos[np.logical_not(clean_points & points_in_range)] = np.nan
    y_pos[np.logical_not(clean_points & points_in_range)] = np.nan

    # try:
    #     assert not np.isnan(x_pos[0])
    #     assert not np.isnan(y_pos[0])
    #     assert not np.isnan(x_pos[-1])
    #     assert not np.isnan(y_pos[-1])
    # except:
    #     nans = np.argwhere(np.isnan(x_pos))
    #     print("nans (", np.size(nans), "):", nans)
    #     exit()

    nanpos = np.isnan(x_pos)
    notnanpos = np.logical_not(nanpos)
    x_pos = np.interp(x, x[notnanpos], x_pos[notnanpos])
    y_pos = np.interp(x, x[notnanpos], y_pos[notnanpos])

    return list(x_pos), list(y_pos), list(x)


def get_well_coordinates(well_num, well_coords_map):
    return well_coords_map[well_num]


def getMeanDistToWell(xs, ys, wellx, welly, duration=-1, ts=np.array([])):
    # Note nan values are ignored. This is intentional, so caller
    # can just consider some time points by making all other values nan
    # If duration == -1, use all times points. Otherwise, take only duration in seconds
    if duration != -1:
        assert xs.shape == ts.shape
        dur_idx = np.searchsorted(ts, ts[0] + duration)
        xs = xs[0:dur_idx]
        ys = ys[0:dur_idx]

    dist_to_well = np.sqrt(np.power(wellx - np.array(xs), 2) +
                           np.power(welly - np.array(ys), 2))
    return np.nanmean(dist_to_well)


def getMedianDistToWell(xs, ys, wellx, welly, duration=-1, ts=np.array([])):
    # Note nan values are ignored. This is intentional, so caller
    # can just consider some time points by making all other values nan
    # If duration == -1, use all times points. Otherwise, take only duration in seconds
    if duration != -1:
        assert xs.shape == ts.shape
        dur_idx = np.searchsorted(ts, ts[0] + duration)
        xs = xs[0:dur_idx]
        ys = ys[0:dur_idx]

    dist_to_well = np.sqrt(np.power(wellx - np.array(xs), 2) +
                           np.power(welly - np.array(ys), 2))
    return np.nanmedian(dist_to_well)


def getMeanDistToWells(xs, ys, well_coords, duration=-1, ts=np.array([])):
    res = []
    for wi in all_well_idxs:
        wx, wy = get_well_coordinates(wi, well_coords)
        res.append(getMeanDistToWell(np.array(xs), np.array(
            ys), wx, wy, duration=duration, ts=np.array(ts)))

    return res


def getMedianDistToWells(xs, ys, well_coords, duration=-1, ts=np.array([])):
    res = []
    for wi in all_well_idxs:
        wx, wy = get_well_coordinates(wi, well_coords)
        res.append(getMedianDistToWell(np.array(xs), np.array(
            ys), wx, wy, duration=duration, ts=np.array(ts)))

    return res


def getNearestWell(xs, ys, well_coords, well_idxs=all_well_idxs):
    well_coords = np.array([get_well_coordinates(i, well_coords) for i in well_idxs])
    tiled_x = np.tile(xs, (len(well_idxs), 1)).T  # each row is one time point
    tiled_y = np.tile(ys, (len(well_idxs), 1)).T

    tiled_wells_x = np.tile(well_coords[:, 0], (len(xs), 1))
    tiled_wells_y = np.tile(well_coords[:, 1], (len(ys), 1))

    delta_x = tiled_wells_x - tiled_x
    delta_y = tiled_wells_y - tiled_y
    delta = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

    raw_nearest_wells = np.argmin(delta, axis=1)
    nearest_well = raw_nearest_wells
    curr_well = nearest_well[0]
    for i in range(np.shape(xs)[0]):
        if curr_well != nearest_well[i]:
            if delta[i, nearest_well[i]] < SWITCH_WELL_FACTOR * delta[i, curr_well]:
                curr_well = nearest_well[i]
            else:
                nearest_well[i] = curr_well

    # if TEST_NEAREST_WELL:
    #     print("delta_x", delta_x)
    #     print("delta_y", delta_y)
    #     print("delta", delta)
    #     print("raw_nearest_wells", raw_nearest_wells)
    #     print("nearest_well", nearest_well)

    return well_idxs[nearest_well]


if __name__ == "__main__" and TEST_NEAREST_WELL:
    raise Exception("Unimplemented")
    # w1 = 2
    # w2 = 3
    # numpts = 20
    # # numpts = 3
    # w1x, w1y = get_well_coordinates(w1)
    # w2x, w2y = get_well_coordinates(w2)
    # xs = np.linspace(w1x, w2x, numpts)
    # ys = np.linspace(w1y, w2y, numpts)
    # ws = getNearestWell(xs, ys)
    # print(xs)
    # print(ys)
    # print(ws)
    # plt.clf()
    # plt.scatter(xs, ys, c=ws)
    # plt.show()
    # exit()


def getWellEntryAndExitTimes(nearest_wells, ts, well_idxs=all_well_idxs, include_neighbors=False):
    entry_times = []
    exit_times = []
    entry_idxs = []
    exit_idxs = []

    ts = np.array(ts)
    for wi in well_idxs:
        # last data point should count as an exit, so appending a false
        # same for first point should count as entry, prepending
        if include_neighbors:
            neighbors = list({wi, wi-1, wi+1, wi-7, wi-8, wi-9,
                              wi+7, wi+8, wi+9}.intersection(all_well_idxs))
            near_well = np.concatenate(([False], np.isin(nearest_wells, neighbors), [False]))
        else:
            near_well = np.concatenate(([False], nearest_wells == wi, [False]))
        idx = np.argwhere(np.diff(np.array(near_well, dtype=float)) == 1)
        idx2 = np.argwhere(np.diff(np.array(near_well, dtype=float)) == -1) - 1
        entry_idxs.append(idx.T[0])
        exit_idxs.append(idx2.T[0])
        entry_times.append(ts[idx.T[0]])
        exit_times.append(ts[idx2.T[0]])

    return entry_idxs, exit_idxs, entry_times, exit_times


def getSingleWellEntryAndExitTimes(xs, ys, ts, wellx, welly):
    """
    returns tuple of entry and exit times
    """
    # Note nan values are filled in. Cannot use nan as a valid way to
    # mask part of the values, just pass in the relevant portions
    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(ts)
    nanmask = np.logical_or(np.isnan(xs), np.isnan(ys))
    notnanmask = np.logical_not(nanmask)
    xs[nanmask] = np.interp(ts[nanmask], ts[notnanmask], xs[notnanmask])
    ys[nanmask] = np.interp(ts[nanmask], ts[notnanmask], ys[notnanmask])
    dist_to_well = np.sqrt(np.power(wellx - np.array(xs), 2) +
                           np.power(welly - np.array(ys), 2))

    near_well = dist_to_well < RADIUS
    idx = np.argwhere(np.diff(np.array(near_well, dtype=float)) == 1)
    idx2 = np.argwhere(np.diff(np.array(near_well, dtype=float)) == -1)
    return ts[idx.T[0]], ts[idx2.T[0]]


# 2 3
# 0 1
def quadrantOfWell(well_idx):
    if well_idx > 24:
        res = 2
    else:
        res = 0

    if (well_idx - 1) % 8 >= 4:
        res += 1

    return res


if __name__ == "__main__":
    dataob = BTData()

    # ===================================
    # Filter to just relevant directories
    # ===================================
    filtered_data_dirs = []
    for session_idx, session_dir in enumerate(all_data_dirs):
        if session_dir == "behavior_notes" or "numpy_objs" in session_dir:
            continue

        if not os.path.isdir(os.path.join(data_dir, session_dir)):
            print("skipping this file ... dirs only")
            continue

        dir_split = session_dir.split('_')
        if dir_split[-1] == "Probe" or dir_split[-1] == "ITI":
            # will deal with this in another loop
            print("skipping, is probe or iti")
            continue

        date_str = dir_split[0][-8:]
        if date_str in excluded_dates:
            print("skipping, excluded date")
            continue

        if RUN_JUST_SPECIFIED and date_str not in SPECIFIED_DAYS:
            continue

        filtered_data_dirs.append(session_dir)

    print("\n".join(filtered_data_dirs))

    for session_idx, session_dir in enumerate(filtered_data_dirs):

        # ======================================================================
        # ===================================
        # Create new session and import raw data
        # ===================================
        # ======================================================================
        session = BTSession()

        dir_split = session_dir.split('_')
        date_str = dir_split[0][-8:]
        session_name_pfx = dir_split[0][0:-8]
        session.date_str = date_str
        session.name = session_dir

        # check for files that belong to this date
        session.bt_dir = session_dir
        gl = data_dir + session_name_pfx + date_str + "*"
        dir_list = glob.glob(gl)
        for d in dir_list:
            if d.split('_')[-1] == "ITI":
                session.iti_dir = d
                session.separate_iti_file = True
                session.recorded_iti = True
            elif d.split('_')[-1] == "Probe":
                session.probe_dir = d
                session.separate_probe_file = True

        if not session.separate_probe_file:
            session.recorded_iti = True
            session.probe_dir = os.path.join(data_dir, session_dir)
        if not session.separate_iti_file and session.recorded_iti:
            session.iti_dir = session_dir

        file_str = os.path.join(data_dir, session_dir, session_dir)
        # lfp_data = []
        session.bt_lfp_fnames = []
        for i in range(len(session.ripple_detection_tetrodes)):
            session.bt_lfp_fnames.append(os.path.join(file_str + ".LFP", session_dir +
                                                      ".LFP_nt" + session.ripple_detection_tetrodes[i] + "ch1.dat"))
            # lfp_data.append(MountainViewIO.loadLFP(data_file=session.bt_lfp_fnames[-1]))

        position_data = readRawPositionData(
            file_str + '.1.videoPositionTracking')
        if session.separate_probe_file:
            probe_file_str = os.path.join(
                session.probe_dir, os.path.basename(session.probe_dir))
            bt_time_clips = readClipData(file_str + '.1.clips')[0]
            probe_time_clips = readClipData(probe_file_str + '.1.clips')[0]
        else:
            time_clips = readClipData(file_str + '.1.clips')
            bt_time_clips = time_clips[0]
            probe_time_clips = time_clips[1]

        xs, ys, ts = processPosData(position_data)
        bt_start_idx = np.searchsorted(ts, bt_time_clips[0])
        bt_end_idx = np.searchsorted(ts, bt_time_clips[1])
        session.bt_pos_xs = xs[bt_start_idx:bt_end_idx]
        session.bt_pos_ys = ys[bt_start_idx:bt_end_idx]
        session.bt_pos_ts = ts[bt_start_idx:bt_end_idx]

        if session.separate_probe_file:
            position_data = readRawPositionData(
                probe_file_str + '.1.videoPositionTracking')
            xs, ys, ts = processPosData(position_data)

        probe_start_idx = np.searchsorted(ts, probe_time_clips[0])
        probe_end_idx = np.searchsorted(ts, probe_time_clips[1])
        session.probe_pos_xs = xs[probe_start_idx:probe_end_idx]
        session.probe_pos_ys = ys[probe_start_idx:probe_end_idx]
        session.probe_pos_ts = ts[probe_start_idx:probe_end_idx]

        # ===================================
        # Get flags and info from info file
        # ===================================
        info_file = os.path.join(behavior_notes_dir, date_str + ".txt")
        if not os.path.exists(info_file):
            # Switched numbering scheme once started doing multiple sessions a day
            seshs_on_this_day = sorted(
                list(filter(lambda seshdir: session.date_str + "_" in seshdir, filtered_data_dirs)))
            num_on_this_day = len(seshs_on_this_day)
            for i in range(num_on_this_day):
                if seshs_on_this_day[i] == session_dir:
                    sesh_idx_within_day = i
            info_file = os.path.join(behavior_notes_dir, date_str + "_" +
                                     str(sesh_idx_within_day+1) + ".txt")

        if "".join(os.path.basename(info_file).split(".")[0:-1]) in excluded_sessions:
            print(session_dir, " excluded session, skipping")
            continue

        try:
            with open(info_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    lineparts = line.split(":")
                    if len(lineparts) != 2:
                        session.notes.append(line)
                        continue

                    field_name = lineparts[0]
                    field_val = lineparts[1]

                    if field_name.lower() == "home":
                        session.home_well = int(field_val)
                    elif field_name.lower() == "aways":
                        session.away_wells = [int(w)
                                              for w in field_val.strip().split(' ')]
                        # print(line)
                        # print(field_name, field_val)
                        # print(session.away_wells)
                    elif field_name.lower() == "condition":
                        type_in = field_val
                        if 'Ripple' in type_in or 'Interruption' in type_in:
                            session.isRippleInterruption = True
                        elif 'None' in type_in:
                            session.isNoInterruption = True
                        elif 'Delayed' in type_in:
                            session.isDelayedInterruption = True
                        else:
                            print("Couldn't recognize Condition {} in file {}".format(
                                type_in, info_file))
                    elif field_name.lower() == "thresh":
                        if "Low" in field_val:
                            session.ripple_detection_threshold = 2.5
                        elif "High" in field_val:
                            session.ripple_detection_threshold = 4
                        else:
                            session.ripple_detection_threshold = float(field_val)
                    elif field_name.lower() == "last away":
                        session.last_away_well = float(field_val)
                        # print(field_val)
                    elif field_name.lower() == "last well":
                        ended_on = field_val
                        if 'H' in field_val:
                            session.ended_on_home = True
                        elif 'A' in field_val:
                            session.ended_on_home = False
                        else:
                            print("Couldn't recognize last well {} in file {}".format(
                                field_val, info_file))
                    elif field_name.lower() == "iti stim on":
                        if 'Y' in field_val:
                            session.ITI_stim_on = True
                        elif 'N' in field_val:
                            session.ITI_stim_on = False
                        else:
                            print("Couldn't recognize ITI Stim condition {} in file {}".format(
                                field_val, info_file))
                    elif field_name.lower() == "probe stim on":
                        if 'Y' in field_val:
                            session.probe_stim_on = True
                        elif 'N' in field_val:
                            session.probe_stim_on = False
                        else:
                            print("Couldn't recognize Probe Stim condition {} in file {}".format(
                                field_val, info_file))
                    else:
                        session.notes.append(line)

        except FileNotFoundError as err:
            print(err)
            print("Couldn't read from info file " + info_file)
            # cbool = input("Would you like to skip this session (Y/N)?")
            # if cbool.lower() == "y":
            # continue
            continue

            print("Getting some info by hand")
            session.home_well = 0
            while session.home_well < 1 or session.home_well > 48:
                # session.home_well = int(input("Home well:"))
                session.home_well = 10

            type_in = 'X'
            while not (type_in in ['R', 'N', 'D']):
                # type_in = input(
                # "Type of trial ([R]ipple interruption/[N]o stim/[D]elayed stim):").upper()
                type_in = 'R'

            if type_in == 'R':
                session.isRippleInterruption = True
            elif type_in == 'N':
                session.isNoInterruption = True
            elif type_in == 'D':
                session.isDelayedInterruption = True

        except Exception as err:
            print(info_file)
            raise err

        if session.home_well == 0:
            print("Home well not listed in notes file, skipping")
            continue

        well_coords_file_name = file_str + '.1.wellLocations.csv'
        session.well_coords_map = readWellCoordsFile(well_coords_file_name)
        session.home_x, session.home_y = get_well_coordinates(
            session.home_well, session.well_coords_map)

        # ======================================================================
        # ===================================
        # Analyze data
        # ===================================
        # ======================================================================

        # ===================================
        # which away wells were visited?
        # ===================================
        session.num_away_found = next((i for i in range(
            len(session.away_wells)) if session.away_wells[i] == session.last_away_well), -1) + 1
        session.visited_away_wells = session.away_wells[0:session.num_away_found]
        # print(session.last_away_well)
        session.num_home_found = session.num_away_found
        if session.ended_on_home:
            session.num_home_found += 1

        # ===================================
        # Well visit times
        # ===================================
        rewardClipsFile = file_str + '.1.rewardClips'
        if not os.path.exists(rewardClipsFile):
            print("Well find times not marked for session {}".format(session.name))
        else:
            well_visit_times = readClipData(rewardClipsFile)
            assert session.num_away_found + session.num_home_found == np.shape(well_visit_times)[0]
            session.home_well_find_times = well_visit_times[::2, 0]
            session.home_well_leave_times = well_visit_times[::2, 1]
            session.away_well_find_times = well_visit_times[1::2, 0]
            session.away_well_leave_times = well_visit_times[1::2, 1]

            if len(session.home_well_leave_times) == len(session.away_well_find_times):
                session.away_well_latencies = np.array(session.away_well_find_times) - \
                    np.array(session.home_well_leave_times)
                session.home_well_latencies = np.array(session.home_well_find_times) - \
                    np.append([session.bt_pos_ts[0]], session.away_well_leave_times[0:-1])
            else:
                session.away_well_latencies = np.array(session.away_well_find_times) - \
                    np.array(session.home_well_leave_times[0:-1])
                session.home_well_latencies = np.array(session.home_well_find_times) - \
                    np.append([session.bt_pos_ts[0]], session.away_well_leave_times)

        # ===================================
        # separating movement time from still time
        # ===================================
        bt_vel = np.sqrt(np.power(np.diff(session.bt_pos_xs), 2) +
                         np.power(np.diff(session.bt_pos_ys), 2))
        session.bt_vel_cm_s = np.divide(bt_vel, np.diff(session.bt_pos_ts) /
                                        TRODES_SAMPLING_RATE) / PIXELS_PER_CM
        bt_is_mv = session.bt_vel_cm_s > VEL_THRESH
        bt_is_mv = np.append(bt_is_mv, np.array(bt_is_mv[-1]))
        session.bt_is_mv = bt_is_mv
        session.bt_mv_xs = np.array(session.bt_pos_xs)
        session.bt_mv_xs[np.logical_not(bt_is_mv)] = np.nan
        session.bt_still_xs = np.array(session.bt_pos_xs)
        session.bt_still_xs[bt_is_mv] = np.nan
        session.bt_mv_ys = np.array(session.bt_pos_ys)
        session.bt_mv_ys[np.logical_not(bt_is_mv)] = np.nan
        session.bt_still_ys = np.array(session.bt_pos_ys)
        session.bt_still_ys[bt_is_mv] = np.nan

        probe_vel = np.sqrt(np.power(np.diff(session.probe_pos_xs), 2) +
                            np.power(np.diff(session.probe_pos_ys), 2))
        session.probe_vel_cm_s = np.divide(probe_vel, np.diff(session.probe_pos_ts) /
                                           TRODES_SAMPLING_RATE) / PIXELS_PER_CM
        probe_is_mv = session.probe_vel_cm_s > VEL_THRESH
        probe_is_mv = np.append(probe_is_mv, np.array(probe_is_mv[-1]))
        session.probe_is_mv = probe_is_mv
        session.probe_mv_xs = np.array(session.probe_pos_xs)
        session.probe_mv_xs[np.logical_not(probe_is_mv)] = np.nan
        session.probe_still_xs = np.array(session.probe_pos_xs)
        session.probe_still_xs[probe_is_mv] = np.nan
        session.probe_mv_ys = np.array(session.probe_pos_ys)
        session.probe_mv_ys[np.logical_not(probe_is_mv)] = np.nan
        session.probe_still_ys = np.array(session.probe_pos_ys)
        session.probe_still_ys[probe_is_mv] = np.nan

        # ===================================
        # Perseveration measures
        # ===================================
        session.ctrl_home_well = 49 - session.home_well
        session.ctrl_home_x, session.ctrl_home_y = get_well_coordinates(
            session.ctrl_home_well, session.well_coords_map)

        # ===================================
        # Well and quadrant entry and exit times
        # ===================================
        session.bt_nearest_wells = getNearestWell(
            session.bt_pos_xs, session.bt_pos_ys, session.well_coords_map)

        session.bt_quadrants = np.array(
            [quadrantOfWell(wi) for wi in session.bt_nearest_wells])
        session.home_quadrant = quadrantOfWell(session.home_well)

        session.bt_well_entry_idxs, session.bt_well_exit_idxs, \
            session.bt_well_entry_times, session.bt_well_exit_times = \
            getWellEntryAndExitTimes(
                session.bt_nearest_wells, session.bt_pos_ts)

        # ninc stands for neighbors included
        session.bt_well_entry_idxs_ninc, session.bt_well_exit_idxs_ninc, \
            session.bt_well_entry_times_ninc, session.bt_well_exit_times_ninc = \
            getWellEntryAndExitTimes(
                session.bt_nearest_wells, session.bt_pos_ts, include_neighbors=True)

        session.bt_quadrant_entry_idxs, session.bt_quadrant_exit_idxs, \
            session.bt_quadrant_entry_times, session.bt_quadrant_exit_times = \
            getWellEntryAndExitTimes(
                session.bt_quadrants, session.bt_pos_ts, well_idxs=[0, 1, 2, 3])

        for i in range(len(all_well_idxs)):
            # print("well {} had {} entries, {} exits".format(
            #     all_well_idxs[i], len(session.bt_well_entry_idxs[i]), len(session.bt_well_exit_idxs[i])))
            assert len(session.bt_well_entry_times[i]) == len(
                session.bt_well_exit_times[i])

        session.home_well_idx_in_allwells = np.argmax(all_well_idxs == session.home_well)
        session.ctrl_home_well_idx_in_allwells = np.argmax(all_well_idxs == session.ctrl_home_well)

        session.bt_home_well_entry_times = session.bt_well_entry_times[session.home_well_idx_in_allwells]
        session.bt_home_well_exit_times = session.bt_well_exit_times[session.home_well_idx_in_allwells]

        session.bt_ctrl_home_well_entry_times = session.bt_well_entry_times[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_ctrl_home_well_exit_times = session.bt_well_exit_times[session.ctrl_home_well_idx_in_allwells]

        # same for during probe
        session.probe_nearest_wells = getNearestWell(
            session.probe_pos_xs, session.probe_pos_ys, session.well_coords_map)

        session.probe_well_entry_idxs, session.probe_well_exit_idxs, \
            session.probe_well_entry_times, session.probe_well_exit_times = getWellEntryAndExitTimes(
                session.probe_nearest_wells, session.probe_pos_ts)

        session.probe_well_entry_idxs_ninc, session.probe_well_exit_idxs_ninc, \
            session.probe_well_entry_times_ninc, session.probe_well_exit_times_ninc = getWellEntryAndExitTimes(
                session.probe_nearest_wells, session.probe_pos_ts, include_neighbors=True)

        session.probe_quadrants = np.array(
            [quadrantOfWell(wi) for wi in session.probe_nearest_wells])

        session.probe_quadrant_entry_idxs, session.probe_quadrant_exit_idxs, \
            session.probe_quadrant_entry_times, session.probe_quadrant_exit_times = getWellEntryAndExitTimes(
                session.probe_quadrants, session.probe_pos_ts, well_idxs=[0, 1, 2, 3])

        for i in range(len(all_well_idxs)):
            # print(i, len(session.probe_well_entry_times[i]), len(session.probe_well_exit_times[i]))
            assert len(session.probe_well_entry_times[i]) == len(
                session.probe_well_exit_times[i])

        session.probe_home_well_entry_times = session.probe_well_entry_times[session.home_well_idx_in_allwells]
        session.probe_home_well_exit_times = session.probe_well_exit_times[session.home_well_idx_in_allwells]

        session.probe_ctrl_home_well_entry_times = session.probe_well_entry_times[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_ctrl_home_well_exit_times = session.probe_well_exit_times[
            session.ctrl_home_well_idx_in_allwells]

        # ===================================
        # avg dist to wells
        # ===================================
        session.bt_mean_dist_to_wells = getMeanDistToWells(
            session.bt_pos_xs, session.bt_pos_ys, session.well_coords_map)
        session.bt_mean_dist_to_wells_1min = getMeanDistToWells(
            session.bt_pos_xs, session.bt_pos_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_mean_dist_to_wells_30sec = getMeanDistToWells(
            session.bt_pos_xs, session.bt_pos_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_mean_dist_to_home_well = session.bt_mean_dist_to_wells[session.home_well_idx_in_allwells]
        session.bt_mean_dist_to_home_well_1min = session.bt_mean_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.bt_mean_dist_to_home_well_30sec = session.bt_mean_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.bt_mean_dist_to_ctrl_home_well = session.bt_mean_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_mean_dist_to_ctrl_home_well_1min = session.bt_mean_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_mean_dist_to_ctrl_home_well_30sec = session.bt_mean_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]

        # Just moving times
        session.bt_mv_mean_dist_to_wells = getMeanDistToWells(
            session.bt_mv_xs, session.bt_mv_ys, session.well_coords_map)
        session.bt_mv_mean_dist_to_wells_1min = getMeanDistToWells(
            session.bt_mv_xs, session.bt_mv_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_mv_mean_dist_to_wells_30sec = getMeanDistToWells(
            session.bt_mv_xs, session.bt_mv_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_mv_mean_dist_to_home_well = session.bt_mv_mean_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.bt_mv_mean_dist_to_home_well_1min = session.bt_mv_mean_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.bt_mv_mean_dist_to_home_well_30sec = session.bt_mv_mean_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.bt_mv_mean_dist_to_ctrl_home_well = session.bt_mv_mean_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_mv_mean_dist_to_ctrl_home_well_1min = session.bt_mv_mean_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_mv_mean_dist_to_ctrl_home_well_30sec = session.bt_mv_mean_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]

        # just still times
        session.bt_still_mean_dist_to_wells = getMeanDistToWells(
            session.bt_still_xs, session.bt_still_ys, session.well_coords_map)
        session.bt_still_mean_dist_to_wells_1min = getMeanDistToWells(
            session.bt_still_xs, session.bt_still_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_still_mean_dist_to_wells_30sec = getMeanDistToWells(
            session.bt_still_xs, session.bt_still_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_still_mean_dist_to_home_well = session.bt_still_mean_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.bt_still_mean_dist_to_home_well_1min = session.bt_still_mean_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.bt_still_mean_dist_to_home_well_30sec = session.bt_still_mean_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.bt_still_mean_dist_to_ctrl_home_well = session.bt_still_mean_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_still_mean_dist_to_ctrl_home_well_1min = session.bt_still_mean_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_still_mean_dist_to_ctrl_home_well_30sec = session.bt_still_mean_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]

        # same for probe
        session.probe_mean_dist_to_wells = getMeanDistToWells(
            session.probe_pos_xs, session.probe_pos_ys, session.well_coords_map)
        session.probe_mean_dist_to_wells_1min = getMeanDistToWells(
            session.probe_pos_xs, session.probe_pos_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_mean_dist_to_wells_30sec = getMeanDistToWells(
            session.probe_pos_xs, session.probe_pos_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_mean_dist_to_home_well = session.probe_mean_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.probe_mean_dist_to_home_well_1min = session.probe_mean_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.probe_mean_dist_to_home_well_30sec = session.probe_mean_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.probe_mean_dist_to_ctrl_home_well = session.probe_mean_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_mean_dist_to_ctrl_home_well_1min = session.probe_mean_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_mean_dist_to_ctrl_home_well_30sec = session.probe_mean_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]

        # Just moving times
        session.probe_mv_mean_dist_to_wells = getMeanDistToWells(
            session.probe_mv_xs, session.probe_mv_ys, session.well_coords_map)
        session.probe_mv_mean_dist_to_wells_1min = getMeanDistToWells(
            session.probe_mv_xs, session.probe_mv_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_mv_mean_dist_to_wells_30sec = getMeanDistToWells(
            session.probe_mv_xs, session.probe_mv_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_mv_mean_dist_to_home_well = session.probe_mv_mean_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.probe_mv_mean_dist_to_home_well_1min = session.probe_mv_mean_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.probe_mv_mean_dist_to_home_well_30sec = session.probe_mv_mean_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.probe_mv_mean_dist_to_ctrl_home_well = session.probe_mv_mean_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_mv_mean_dist_to_ctrl_home_well_1min = session.probe_mv_mean_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_mv_mean_dist_to_ctrl_home_well_30sec = session.probe_mv_mean_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]

        # just still times
        session.probe_still_mean_dist_to_wells = getMeanDistToWells(
            session.probe_still_xs, session.probe_still_ys, session.well_coords_map)
        session.probe_still_mean_dist_to_wells_1min = getMeanDistToWells(
            session.probe_still_xs, session.probe_still_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_still_mean_dist_to_wells_30sec = getMeanDistToWells(
            session.probe_still_xs, session.probe_still_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_still_mean_dist_to_home_well = session.probe_still_mean_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.probe_still_mean_dist_to_home_well_1min = session.probe_still_mean_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.probe_still_mean_dist_to_home_well_30sec = session.probe_still_mean_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.probe_still_mean_dist_to_ctrl_home_well = session.probe_still_mean_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_still_mean_dist_to_ctrl_home_well_1min = session.probe_still_mean_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_still_mean_dist_to_ctrl_home_well_30sec = session.probe_still_mean_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]

        # Now all the same but median instead of mean
        session.bt_median_dist_to_wells = getMedianDistToWells(
            session.bt_pos_xs, session.bt_pos_ys, session.well_coords_map)
        session.bt_median_dist_to_wells_1min = getMedianDistToWells(
            session.bt_pos_xs, session.bt_pos_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_median_dist_to_wells_30sec = getMedianDistToWells(
            session.bt_pos_xs, session.bt_pos_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_median_dist_to_home_well = session.bt_median_dist_to_wells[session.home_well_idx_in_allwells]
        session.bt_median_dist_to_home_well_1min = session.bt_median_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.bt_median_dist_to_home_well_30sec = session.bt_median_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.bt_median_dist_to_ctrl_home_well = session.bt_median_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_median_dist_to_ctrl_home_well_1min = session.bt_median_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_median_dist_to_ctrl_home_well_30sec = session.bt_median_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_mv_median_dist_to_wells = getMedianDistToWells(
            session.bt_mv_xs, session.bt_mv_ys, session.well_coords_map)
        session.bt_mv_median_dist_to_wells_1min = getMedianDistToWells(
            session.bt_mv_xs, session.bt_mv_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_mv_median_dist_to_wells_30sec = getMedianDistToWells(
            session.bt_mv_xs, session.bt_mv_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_mv_median_dist_to_home_well = session.bt_mv_median_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.bt_mv_median_dist_to_home_well_1min = session.bt_mv_median_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.bt_mv_median_dist_to_home_well_30sec = session.bt_mv_median_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.bt_mv_median_dist_to_ctrl_home_well = session.bt_mv_median_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_mv_median_dist_to_ctrl_home_well_1min = session.bt_mv_median_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_mv_median_dist_to_ctrl_home_well_30sec = session.bt_mv_median_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_still_median_dist_to_wells = getMedianDistToWells(
            session.bt_still_xs, session.bt_still_ys, session.well_coords_map)
        session.bt_still_median_dist_to_wells_1min = getMedianDistToWells(
            session.bt_still_xs, session.bt_still_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_still_median_dist_to_wells_30sec = getMedianDistToWells(
            session.bt_still_xs, session.bt_still_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.bt_pos_ts)
        session.bt_still_median_dist_to_home_well = session.bt_still_median_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.bt_still_median_dist_to_home_well_1min = session.bt_still_median_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.bt_still_median_dist_to_home_well_30sec = session.bt_still_median_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.bt_still_median_dist_to_ctrl_home_well = session.bt_still_median_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_still_median_dist_to_ctrl_home_well_1min = session.bt_still_median_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_still_median_dist_to_ctrl_home_well_30sec = session.bt_still_median_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_median_dist_to_wells = getMedianDistToWells(
            session.probe_pos_xs, session.probe_pos_ys, session.well_coords_map)
        session.probe_median_dist_to_wells_1min = getMedianDistToWells(
            session.probe_pos_xs, session.probe_pos_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_median_dist_to_wells_30sec = getMedianDistToWells(
            session.probe_pos_xs, session.probe_pos_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_median_dist_to_home_well = session.probe_median_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.probe_median_dist_to_home_well_1min = session.probe_median_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.probe_median_dist_to_home_well_30sec = session.probe_median_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.probe_median_dist_to_ctrl_home_well = session.probe_median_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_median_dist_to_ctrl_home_well_1min = session.probe_median_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_median_dist_to_ctrl_home_well_30sec = session.probe_median_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_mv_median_dist_to_wells = getMedianDistToWells(
            session.probe_mv_xs, session.probe_mv_ys, session.well_coords_map)
        session.probe_mv_median_dist_to_wells_1min = getMedianDistToWells(
            session.probe_mv_xs, session.probe_mv_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_mv_median_dist_to_wells_30sec = getMedianDistToWells(
            session.probe_mv_xs, session.probe_mv_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_mv_median_dist_to_home_well = session.probe_mv_median_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.probe_mv_median_dist_to_home_well_1min = session.probe_mv_median_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.probe_mv_median_dist_to_home_well_30sec = session.probe_mv_median_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.probe_mv_median_dist_to_ctrl_home_well = session.probe_mv_median_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_mv_median_dist_to_ctrl_home_well_1min = session.probe_mv_median_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_mv_median_dist_to_ctrl_home_well_30sec = session.probe_mv_median_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_still_median_dist_to_wells = getMedianDistToWells(
            session.probe_still_xs, session.probe_still_ys, session.well_coords_map)
        session.probe_still_median_dist_to_wells_1min = getMedianDistToWells(
            session.probe_still_xs, session.probe_still_ys, session.well_coords_map, duration=60*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_still_median_dist_to_wells_30sec = getMedianDistToWells(
            session.probe_still_xs, session.probe_still_ys, session.well_coords_map, duration=30*TRODES_SAMPLING_RATE, ts=session.probe_pos_ts)
        session.probe_still_median_dist_to_home_well = session.probe_still_median_dist_to_wells[
            session.home_well_idx_in_allwells]
        session.probe_still_median_dist_to_home_well_1min = session.probe_still_median_dist_to_wells_1min[
            session.home_well_idx_in_allwells]
        session.probe_still_median_dist_to_home_well_30sec = session.probe_still_median_dist_to_wells_30sec[
            session.home_well_idx_in_allwells]
        session.probe_still_median_dist_to_ctrl_home_well = session.probe_still_median_dist_to_wells[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_still_median_dist_to_ctrl_home_well_1min = session.probe_still_median_dist_to_wells_1min[
            session.ctrl_home_well_idx_in_allwells]
        session.probe_still_median_dist_to_ctrl_home_well_30sec = session.probe_still_median_dist_to_wells_30sec[
            session.ctrl_home_well_idx_in_allwells]

        # ===================================
        # Dwell times
        # ===================================
        session.bt_dwell_times = []
        for i, wi in enumerate(all_well_idxs):
            session.bt_dwell_times.append(np.array(
                session.bt_well_exit_times[i]) - np.array(session.bt_well_entry_times[i]))

        session.bt_well_num_entries = []
        session.bt_well_total_dwell_times = []
        session.bt_well_avg_dwell_times = []
        session.bt_well_total_dwell_times_excluding_reward = []
        session.bt_well_avg_dwell_times_excluding_reward = []
        for i, wi in enumerate(all_well_idxs):
            nume = len(session.bt_well_entry_idxs[i])
            session.bt_well_num_entries.append(nume)
            total_dwell_time = np.sum(session.bt_dwell_times[i])
            session.bt_well_total_dwell_times.append(total_dwell_time)
            session.bt_well_avg_dwell_times.append(
                total_dwell_time / float(nume))

            total_dwell_time_without_reward = total_dwell_time
            if wi == session.home_well:
                if len(session.home_well_find_times) > 0:
                    # we know which visits were the rewarded ones
                    for ft, lt in zip(session.home_well_find_times, session.home_well_leave_times):
                        wt = (ft + lt) / 2.0
                        found = False
                        for ent, ext in zip(session.bt_well_entry_times[i], session.bt_well_exit_times[i]):
                            if wt >= ent and wt <= ext:
                                total_dwell_time_without_reward -= ext - ent
                                found = True
                                break

                        if not found:
                            # print(session.bt_well_entry_times[i], session.bt_well_exit_times[i])

                            # raise Exception(
                            print(
                                "Couldn't find home find time {} ({}, {}) in session {}".format(wt, ft, lt, session.name))

                else:
                    # we don't know which visits were rewarded, for now assume the longest ones. Will bias us to see less dwell time
                    dwell_times_sorted = sorted(
                        session.bt_dwell_times[i], reverse=True)
                    for i in range(session.num_home_found):
                        total_dwell_time_without_reward -= dwell_times_sorted[i]
            elif wi in session.visited_away_wells:
                if len(session.away_well_find_times) > 0:
                    # we know which visits were the rewarded ones
                    for ei, awi in enumerate(session.visited_away_wells):
                        if awi == wi:
                            ft = (session.away_well_find_times[ei] +
                                  session.away_well_leave_times[ei]) / 2.0
                    found = False
                    for ent, ext in zip(session.bt_well_entry_times[i], session.bt_well_exit_times[i]):
                        if ft >= ent and ft <= ext:
                            total_dwell_time_without_reward -= ext - ent
                            found = True
                            break

                    if not found:
                        print(session.bt_well_entry_times[i], session.bt_well_exit_times[i])
                        raise Exception(
                            # print(
                            "Couldn't find away find time {} for well {} in session {}".format(
                                ft, wi, session.name))

                else:
                    # we don't know which visits were rewarded, for now assume the longest ones. Will bias us to see less dwell time
                    dwell_times_sorted = sorted(
                        session.bt_dwell_times[i], reverse=True)
                    total_dwell_time_without_reward -= dwell_times_sorted[0]

            session.bt_well_total_dwell_times_excluding_reward.append(
                total_dwell_time_without_reward)
            session.bt_well_avg_dwell_times_excluding_reward.append(
                total_dwell_time_without_reward / float(nume))

        session.probe_dwell_times = []
        session.probe_dwell_times_1min = []
        session.probe_dwell_times_30sec = []
        for i, wi in enumerate(all_well_idxs):
            session.probe_dwell_times.append(np.array(
                session.probe_well_exit_times[i]) - np.array(session.probe_well_entry_times[i]))
            dts_1min = []
            dts_30sec = []
            for ent, ext in zip(session.probe_well_entry_times[i], session.probe_well_exit_times[i]):
                if ext <= session.probe_pos_ts[0] + 60*TRODES_SAMPLING_RATE:
                    dts_1min.append(ext - ent)
                if ext <= session.probe_pos_ts[0] + 30*TRODES_SAMPLING_RATE:
                    dts_30sec.append(ext - ent)
            session.probe_dwell_times_1min.append(dts_1min)
            session.probe_dwell_times_30sec.append(dts_30sec)

        session.probe_well_num_entries = []
        session.probe_well_total_dwell_times = []
        session.probe_well_avg_dwell_times = []
        session.probe_well_num_entries_1min = []
        session.probe_well_total_dwell_times_1min = []
        session.probe_well_avg_dwell_times_1min = []
        session.probe_well_num_entries_30sec = []
        session.probe_well_total_dwell_times_30sec = []
        session.probe_well_avg_dwell_times_30sec = []
        for i, wi in enumerate(all_well_idxs):
            nume = len(session.probe_dwell_times[i])
            session.probe_well_num_entries.append(nume)
            total_dwell_time = np.sum(session.probe_dwell_times[i])
            session.probe_well_total_dwell_times.append(total_dwell_time)
            session.probe_well_avg_dwell_times.append(
                total_dwell_time / float(nume))

            nume = len(session.probe_dwell_times_1min[i])
            session.probe_well_num_entries_1min.append(nume)
            total_dwell_time = np.sum(session.probe_dwell_times_1min[i])
            session.probe_well_total_dwell_times_1min.append(total_dwell_time)
            session.probe_well_avg_dwell_times_1min.append(
                total_dwell_time / float(nume))

            nume = len(session.probe_dwell_times_30sec[i])
            session.probe_well_num_entries_30sec.append(nume)
            total_dwell_time = np.sum(session.probe_dwell_times_30sec[i])
            session.probe_well_total_dwell_times_30sec.append(total_dwell_time)
            session.probe_well_avg_dwell_times_30sec.append(
                total_dwell_time / float(nume))

        # Same but with neighbors included
        session.bt_dwell_times_ninc = []
        for i, wi in enumerate(all_well_idxs):
            session.bt_dwell_times_ninc.append(np.array(
                session.bt_well_exit_times_ninc[i]) - np.array(session.bt_well_entry_times_ninc[i]))

        session.bt_well_num_entries_ninc = []
        session.bt_well_total_dwell_times_ninc = []
        session.bt_well_avg_dwell_times_ninc = []
        session.bt_well_total_dwell_times_excluding_reward_ninc = []
        session.bt_well_avg_dwell_times_excluding_reward_ninc = []
        for i, wi in enumerate(all_well_idxs):
            nume = len(session.bt_well_entry_idxs_ninc[i])
            session.bt_well_num_entries_ninc.append(nume)
            total_dwell_time = np.sum(session.bt_dwell_times_ninc[i])
            session.bt_well_total_dwell_times_ninc.append(total_dwell_time)
            session.bt_well_avg_dwell_times_ninc.append(
                total_dwell_time / float(nume))

            total_dwell_time_without_reward = total_dwell_time
            if wi == session.home_well:
                if len(session.home_well_find_times) > 0:
                    # we know which visits were the rewarded ones
                    for ft, lt in zip(session.home_well_find_times, session.home_well_leave_times):
                        wt = (ft + lt) / 2.0
                        found = False
                        for ent, ext in zip(session.bt_well_entry_times_ninc[i], session.bt_well_exit_times_ninc[i]):
                            if wt >= ent and wt <= ext:
                                total_dwell_time_without_reward -= ext - ent
                                found = True
                                break

                        if not found:
                            print(session.bt_well_entry_times[i], session.bt_well_exit_times[i])
                            print(session.bt_well_entry_times_ninc[i],
                                  session.bt_well_exit_times_ninc[i])

                            raise Exception(
                                # print(
                                "Couldn't find home find time {} in session {}".format(wt, session.name))

                else:
                    # we don't know which visits were rewarded, for now assume the longest ones. Will bias us to see less dwell time
                    dwell_times_sorted = sorted(
                        session.bt_dwell_times_ninc[i], reverse=True)
                    for i in range(session.num_home_found):
                        total_dwell_time_without_reward -= dwell_times_sorted[i]
            elif wi in session.visited_away_wells:
                if len(session.away_well_find_times) > 0:
                    # we know which visits were the rewarded ones
                    for ei, awi in enumerate(session.visited_away_wells):
                        if awi == wi:
                            ft = (session.away_well_find_times[ei] +
                                  session.away_well_leave_times[ei]) / 2.0
                    found = False
                    for ent, ext in zip(session.bt_well_entry_times_ninc[i], session.bt_well_exit_times_ninc[i]):
                        if ft >= ent and ft <= ext:
                            total_dwell_time_without_reward -= ext - ent
                            found = True
                            break

                    if not found:
                        print(session.bt_well_entry_times[i], session.bt_well_exit_times[i])
                        print(session.bt_well_entry_times_ninc[i],
                              session.bt_well_exit_times_ninc[i])
                        raise Exception(
                            # print(
                            "Couldn't find away find time {} for well {} in session {}".format(
                                ft, wi, session.name))

                else:
                    # we don't know which visits were rewarded, for now assume the longest ones. Will bias us to see less dwell time
                    dwell_times_sorted = sorted(
                        session.bt_dwell_times_ninc[i], reverse=True)
                    total_dwell_time_without_reward -= dwell_times_sorted[0]

            session.bt_well_total_dwell_times_excluding_reward_ninc.append(
                total_dwell_time_without_reward)
            session.bt_well_avg_dwell_times_excluding_reward_ninc.append(
                total_dwell_time_without_reward / float(nume))

        session.probe_dwell_times_ninc = []
        session.probe_dwell_times_1min_ninc = []
        session.probe_dwell_times_30sec_ninc = []
        for i, wi in enumerate(all_well_idxs):
            session.probe_dwell_times_ninc.append(np.array(
                session.probe_well_exit_times_ninc[i]) - np.array(session.probe_well_entry_times_ninc[i]))
            dts_1min = []
            dts_30sec = []
            for ent, ext in zip(session.probe_well_entry_times_ninc[i], session.probe_well_exit_times_ninc[i]):
                if ext <= session.probe_pos_ts[0] + 60*TRODES_SAMPLING_RATE:
                    dts_1min.append(ext - ent)
                if ext <= session.probe_pos_ts[0] + 30*TRODES_SAMPLING_RATE:
                    dts_30sec.append(ext - ent)
            session.probe_dwell_times_1min_ninc.append(dts_1min)
            session.probe_dwell_times_30sec_ninc.append(dts_30sec)

        session.probe_well_num_entries_ninc = []
        session.probe_well_total_dwell_times_ninc = []
        session.probe_well_avg_dwell_times_ninc = []
        session.probe_well_num_entries_1min_ninc = []
        session.probe_well_total_dwell_times_1min_ninc = []
        session.probe_well_avg_dwell_times_1min_ninc = []
        session.probe_well_num_entries_30sec_ninc = []
        session.probe_well_total_dwell_times_30sec_ninc = []
        session.probe_well_avg_dwell_times_30sec_ninc = []
        for i, wi in enumerate(all_well_idxs):
            nume = len(session.probe_dwell_times_ninc[i])
            session.probe_well_num_entries_ninc.append(nume)
            total_dwell_time = np.sum(session.probe_dwell_times_ninc[i])
            session.probe_well_total_dwell_times_ninc.append(total_dwell_time)
            session.probe_well_avg_dwell_times_ninc.append(
                total_dwell_time / float(nume))

            nume = len(session.probe_dwell_times_1min_ninc[i])
            session.probe_well_num_entries_1min_ninc.append(nume)
            total_dwell_time = np.sum(session.probe_dwell_times_1min_ninc[i])
            session.probe_well_total_dwell_times_1min_ninc.append(total_dwell_time)
            session.probe_well_avg_dwell_times_1min_ninc.append(
                total_dwell_time / float(nume))

            nume = len(session.probe_dwell_times_30sec_ninc[i])
            session.probe_well_num_entries_30sec_ninc.append(nume)
            total_dwell_time = np.sum(session.probe_dwell_times_30sec_ninc[i])
            session.probe_well_total_dwell_times_30sec_ninc.append(total_dwell_time)
            session.probe_well_avg_dwell_times_30sec_ninc.append(
                total_dwell_time / float(nume))

        # Same but for quadrants
        session.bt_quadrant_dwell_times = []
        for i, wi in enumerate(all_quadrant_idxs):
            session.bt_quadrant_dwell_times.append(np.array(
                session.bt_quadrant_exit_times[i]) - np.array(session.bt_quadrant_entry_times[i]))

        session.bt_quadrant_num_entries = []
        session.bt_quadrant_total_dwell_times = []
        session.bt_quadrant_avg_dwell_times = []
        for i, wi in enumerate(all_quadrant_idxs):
            nume = len(session.bt_quadrant_entry_idxs[i])
            session.bt_quadrant_num_entries.append(nume)
            total_dwell_time = np.sum(session.bt_quadrant_dwell_times[i])
            session.bt_quadrant_total_dwell_times.append(total_dwell_time)
            session.bt_quadrant_avg_dwell_times.append(
                total_dwell_time / float(nume))

        session.probe_quadrant_dwell_times = []
        session.probe_quadrant_dwell_times_1min = []
        session.probe_quadrant_dwell_times_30sec = []
        for i, wi in enumerate([0, 1, 2, 3]):
            session.probe_quadrant_dwell_times.append(np.array(
                session.probe_quadrant_exit_times[i]) - np.array(session.probe_quadrant_entry_times[i]))
            dts_1min = []
            dts_30sec = []
            for ent, ext in zip(session.probe_quadrant_entry_times[i], session.probe_quadrant_exit_times[i]):
                if ext <= session.probe_pos_ts[0] + 60*TRODES_SAMPLING_RATE:
                    dts_1min.append(ext - ent)
                if ext <= session.probe_pos_ts[0] + 30*TRODES_SAMPLING_RATE:
                    dts_30sec.append(ext - ent)
            session.probe_quadrant_dwell_times_1min.append(dts_1min)
            session.probe_quadrant_dwell_times_30sec.append(dts_30sec)

        session.probe_quadrant_num_entries = []
        session.probe_quadrant_total_dwell_times = []
        session.probe_quadrant_avg_dwell_times = []
        session.probe_quadrant_num_entries_1min = []
        session.probe_quadrant_total_dwell_times_1min = []
        session.probe_quadrant_avg_dwell_times_1min = []
        session.probe_quadrant_num_entries_30sec = []
        session.probe_quadrant_total_dwell_times_30sec = []
        session.probe_quadrant_avg_dwell_times_30sec = []
        for i, wi in enumerate([0, 1, 2, 3]):
            nume = len(session.probe_quadrant_dwell_times[i])
            session.probe_quadrant_num_entries.append(nume)
            total_dwell_time = np.sum(session.probe_quadrant_dwell_times[i])
            session.probe_quadrant_total_dwell_times.append(total_dwell_time)
            session.probe_quadrant_avg_dwell_times.append(
                total_dwell_time / float(nume))

            nume = len(session.probe_quadrant_dwell_times_1min[i])
            session.probe_quadrant_num_entries_1min.append(nume)
            total_dwell_time = np.sum(session.probe_quadrant_dwell_times_1min[i])
            session.probe_quadrant_total_dwell_times_1min.append(total_dwell_time)
            session.probe_quadrant_avg_dwell_times_1min.append(
                total_dwell_time / float(nume))

            nume = len(session.probe_quadrant_dwell_times_30sec[i])
            session.probe_quadrant_num_entries_30sec.append(nume)
            total_dwell_time = np.sum(session.probe_quadrant_dwell_times_30sec[i])
            session.probe_quadrant_total_dwell_times_30sec.append(total_dwell_time)
            session.probe_quadrant_avg_dwell_times_30sec.append(
                total_dwell_time / float(nume))

        # ===================================
        # Latency to well in probe
        # ===================================
        session.probe_latency_to_well = []
        for i, wi in enumerate(all_well_idxs):
            if len(session.probe_well_entry_times[i]) == 0:
                session.probe_latency_to_well.append(np.nan)
            else:
                session.probe_latency_to_well.append(
                    session.probe_well_entry_times[0])

        # ===================================
        # Perseveration bias: How high are perseveration measures for this well compared to the corresponding control well.
        # ===================================

        session.bt_persev_bias_mean_dist_to_well = []
        session.bt_persev_bias_num_entries_to_well = []
        session.bt_persev_bias_total_dwell_time = []
        session.bt_persev_bias_avg_dwell_time = []
        session.bt_persev_bias_total_dwell_time_excluding_reward = []
        session.bt_persev_bias_avg_dwell_time_excluding_reward = []
        session.probe_persev_bias_mean_dist_to_well = []
        session.probe_persev_bias_num_entries_to_well = []
        session.probe_persev_bias_total_dwell_time = []
        session.probe_persev_bias_avg_dwell_time = []
        session.probe_persev_bias_mean_dist_to_well_1min = []
        session.probe_persev_bias_num_entries_to_well_1min = []
        session.probe_persev_bias_total_dwell_time_1min = []
        session.probe_persev_bias_avg_dwell_time_1min = []
        session.probe_persev_bias_mean_dist_to_well_30sec = []
        session.probe_persev_bias_num_entries_to_well_30sec = []
        session.probe_persev_bias_total_dwell_time_30sec = []
        session.probe_persev_bias_avg_dwell_time_30sec = []
        for i, wi in enumerate(all_well_idxs):
            cwi = 49 - wi
            cw_idx = np.argmax(all_well_idxs == cwi)
            session.bt_persev_bias_mean_dist_to_well.append(session.bt_mean_dist_to_wells[i] -
                                                            session.bt_mean_dist_to_wells[cw_idx])
            session.bt_persev_bias_num_entries_to_well.append(
                session.bt_well_num_entries[i] - session.bt_well_num_entries[cw_idx])
            session.bt_persev_bias_total_dwell_time.append(
                session.bt_well_total_dwell_times[i] - session.bt_well_total_dwell_times[cw_idx])
            session.bt_persev_bias_avg_dwell_time.append(
                session.bt_well_avg_dwell_times[i] - session.bt_well_avg_dwell_times[cw_idx])
            session.bt_persev_bias_total_dwell_time_excluding_reward.append(
                session.bt_well_total_dwell_times_excluding_reward[i] - session.bt_well_total_dwell_times_excluding_reward[cw_idx])
            session.bt_persev_bias_avg_dwell_time_excluding_reward.append(
                session.bt_well_avg_dwell_times_excluding_reward[i] - session.bt_well_avg_dwell_times_excluding_reward[cw_idx])

            session.probe_persev_bias_mean_dist_to_well.append(session.probe_mean_dist_to_wells[i] -
                                                               session.probe_mean_dist_to_wells[cw_idx])
            session.probe_persev_bias_num_entries_to_well.append(
                session.probe_well_num_entries[i] - session.probe_well_num_entries[cw_idx])
            session.probe_persev_bias_total_dwell_time.append(
                session.probe_well_total_dwell_times[i] - session.probe_well_total_dwell_times[cw_idx])
            session.probe_persev_bias_avg_dwell_time.append(
                session.probe_well_avg_dwell_times[i] - session.probe_well_avg_dwell_times[cw_idx])

            session.probe_persev_bias_mean_dist_to_well_1min.append(session.probe_mean_dist_to_wells_1min[i] -
                                                                    session.probe_mean_dist_to_wells_1min[cw_idx])
            session.probe_persev_bias_num_entries_to_well_1min.append(
                session.probe_well_num_entries_1min[i] - session.probe_well_num_entries_1min[cw_idx])
            session.probe_persev_bias_total_dwell_time_1min.append(
                session.probe_well_total_dwell_times_1min[i] - session.probe_well_total_dwell_times_1min[cw_idx])
            session.probe_persev_bias_avg_dwell_time_1min.append(
                session.probe_well_avg_dwell_times_1min[i] - session.probe_well_avg_dwell_times_1min[cw_idx])

            session.probe_persev_bias_mean_dist_to_well_30sec.append(session.probe_mean_dist_to_wells_30sec[i] -
                                                                     session.probe_mean_dist_to_wells_30sec[cw_idx])
            session.probe_persev_bias_num_entries_to_well_30sec.append(
                session.probe_well_num_entries_30sec[i] - session.probe_well_num_entries_30sec[cw_idx])
            session.probe_persev_bias_total_dwell_time_30sec.append(
                session.probe_well_total_dwell_times_30sec[i] - session.probe_well_total_dwell_times_30sec[cw_idx])
            session.probe_persev_bias_avg_dwell_time_30sec.append(
                session.probe_well_avg_dwell_times_30sec[i] - session.probe_well_avg_dwell_times_30sec[cw_idx])

        # ======================================================================
        # TODO
        # Some perseveration measure during away trials to see if there's an effect during the actual task
        #
        # During task effect on home/away latencies?
        #
        # Is there a difference in how much they sniff around?
        # Knot measure: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000638
        # VTE
        #
        # average latencies for H1, A1, H2, ...
        #
        # difference in effect magnitude by distance from starting location to home well?
        #
        #
        # ======================================================================

        # ===================================
        # Now save this session
        # ===================================
        dataob.allSessions.append(session)

        # ===================================
        # extra plotting stuff
        # ===================================
        if session.date_str in INSPECT_IN_DETAIL:
            if INSPECT_NANVALS:
                print("{} nan xs, {} nan ys, {} nan ts".format(sum(np.isnan(session.probe_pos_xs)),
                                                               sum(np.isnan(
                                                                   session.probe_pos_xs)),
                                                               sum(np.isnan(session.probe_pos_xs))))

                nanxs = np.argwhere(np.isnan(session.probe_pos_xs))
                nanxs = nanxs.T[0]
                num_nan_x = np.size(nanxs)
                if num_nan_x > 0:
                    print(nanxs, num_nan_x, nanxs[0])
                    for i in range(min(5, num_nan_x)):
                        ni = nanxs[i]
                        print("index {}, x=({},nan,{}), y=({},{},{}), t=({},{},{})".format(
                            ni, session.probe_pos_xs[ni -
                                                     1], session.probe_pos_xs[ni + 1],
                            session.probe_pos_ys[ni -
                                                 1], session.probe_pos_ys[ni],
                            session.probe_pos_ys[ni +
                                                 1], session.probe_pos_ts[ni - 1],
                            session.probe_pos_ts[ni], session.probe_pos_ts[ni+1]
                        ))

            if INSPECT_PROBE_BEHAVIOR_PLOT:
                plt.clf()
                plt.scatter(session.home_x, session.home_y,
                            color='green', zorder=2)
                plt.plot(session.probe_pos_xs, session.probe_pos_ys, zorder=0)
                plt.grid('on')
                plt.show()

        if session.date_str in INSPECT_IN_DETAIL and INSPECT_PLOT_WELL_OCCUPANCIES:
            cmap = plt.get_cmap('Set3')
            make_colors = True
            while make_colors:
                well_colors = np.zeros((48, 4))
                for i in all_well_idxs:
                    well_colors[i-1, :] = cmap(random.uniform(0, 1))

                make_colors = False

                if ENFORCE_DIFFERENT_WELL_COLORS:
                    for i in all_well_idxs:
                        neighs = [i-8, i-1, i+8, i+1]
                        for n in neighs:
                            if n in all_well_idxs and np.all(well_colors[i-1, :] == well_colors[n-1, :]):
                                make_colors = True
                                print("gotta remake the colors!")
                                break

                        if make_colors:
                            break

            # print(session.bt_well_entry_idxs)
            # print(session.bt_well_exit_idxs)

            plt.clf()
            for i, wi in enumerate(all_well_idxs):
                color = well_colors[wi-1, :]
                for j in range(len(session.bt_well_entry_times[i])):
                    i1 = session.bt_well_entry_idxs[i][j]
                    try:
                        i2 = session.bt_well_exit_idxs[i][j]
                        plt.plot(
                            session.bt_pos_xs[i1:i2], session.bt_pos_ys[i1:i2], color=color)
                    except:
                        print("well {} had {} entries, {} exits".format(
                            wi, len(session.bt_well_entry_idxs[i]), len(session.bt_well_exit_idxs[i])))

            plt.show()

    # save all sessions to disk
    dataob.saveToFile(os.path.join(output_dir, out_filename))
