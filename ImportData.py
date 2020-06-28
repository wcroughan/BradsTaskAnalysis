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

# INSPECT_IN_DETAIL = []
INSPECT_IN_DETAIL = ["20200526"]
INSPECT_NANVALS = False
INSPECT_PROBE_BEHAVIOR_PLOT = False
INSPECT_PLOT_WELL_OCCUPANCIES = False
ENFORCE_DIFFERENT_WELL_COLORS = False

TEST_NEAREST_WELL = False

data_dir = '/media/WDC4/martindata/bradtask/'
all_data_dirs = sorted(os.listdir(data_dir), key=lambda s: (s.split('_')[0], s.split('_')[1]))
behavior_notes_dir = os.path.join(data_dir, 'behavior_notes')

excluded_dates = []

animal_name = 'Martin'
output_dir = '/media/WDC4/martindata/bradtask/'
out_filename = "martin_bradtask.dat"

if os.path.exists(os.path.join(output_dir, out_filename)):
    # confirm = input("Output file exists already. Overwrite? (y/n):")
    confirm = "y"
    if confirm != "y":
        exit()

well_coords_file = '/home/wcroughan/repos/BradsTaskAnalysis/well_locations.csv'
well_coords_map = {}
with open(well_coords_file, 'r') as wcf:
    csv_reader = csv.reader(wcf)
    for data_row in csv_reader:
        try:
            well_coords_map[int(data_row[0])] = (int(data_row[1]), int(data_row[2]))
        except Exception as err:
            if data_row[1] != '':
                print(err)

all_well_idxs = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])

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
SWITCH_WELL_FACTOR = 0.8  # 0.8 means transition from well a -> b requires rat dist to b to be 0.8 * dist to a

DEFLECTION_THRESHOLD = 2000.0  # Typical observed amplitude of LFP deflection on stimulation
SAMPLING_RATE = 30000.0  # Rate at which timestamp data is sampled
LFP_SAMPLING_RATE = 1500.0  # LFP is subsampled. The timestamps give time according to SAMPLING_RATE above
# Typical duration of the stimulation artifact - For peak detection
MIN_ARTIFACT_PERIOD = int(0.1 * LFP_SAMPLING_RATE)
ACCEPTED_RIPPLE_LENGTH = int(0.2 * LFP_SAMPLING_RATE)  # Typical duration of a Sharp-Wave Ripple


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
    for n_reps in range(N_CLEANING_REPS):
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


def get_well_coordinates(well_num):
    return well_coords_map[well_num]


def getMeanDistToWell(xs, ys, wellx, welly):
    # Note nan values are ignored. This is intentional, so caller
    # can just consider some time points by making all other values nan
    dist_to_well = np.sqrt(np.power(wellx - np.array(xs), 2) +
                           np.power(welly - np.array(ys), 2))
    return np.nanmean(dist_to_well)


def getNearestWell(xs, ys, well_idxs=all_well_idxs):
    well_coords = np.array([get_well_coordinates(i) for i in well_idxs])
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
    w1 = 2
    w2 = 3
    numpts = 20
    # numpts = 3
    w1x, w1y = get_well_coordinates(w1)
    w2x, w2y = get_well_coordinates(w2)
    xs = np.linspace(w1x, w2x, numpts)
    ys = np.linspace(w1y, w2y, numpts)
    ws = getNearestWell(xs, ys)
    print(xs)
    print(ys)
    print(ws)
    plt.clf()
    plt.scatter(xs, ys, c=ws)
    plt.show()
    exit()


def getWellEntryAndExitTimes(nearest_wells, ts, well_idxs=all_well_idxs):
    entry_times = []
    exit_times = []
    entry_idxs = []
    exit_idxs = []

    ts = np.array(ts)
    for wi in well_idxs:
        # last data point should count as an exit, so appending a false
        # same for first point should count as entry, prepending
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


if __name__ == "__main__":
    dataob = BTData()

    for session_idx, session_dir in enumerate(all_data_dirs):
        print(session_dir)
        # ===================================
        # Filter to just relevant directories
        # ===================================
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

        session_name_pfx = dir_split[0][0:-8]

        # if date_str != "20200602":
        #     print("skipping, is not my birthday!")
        #     print(date_str)
        #     continue

        # ===================================
        # Create new session and import raw data
        # ===================================
        session = BTSession()
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

        position_data = readRawPositionData(file_str + '.1.videoPositionTracking')
        if session.separate_probe_file:
            probe_file_str = os.path.join(session.probe_dir, os.path.basename(session.probe_dir))
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
            position_data = readRawPositionData(probe_file_str + '.1.videoPositionTracking')
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
        try:
            with open(info_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Home: "):
                        session.home_well = int(line.split(' ')[1])
                    elif line.startswith("Aways: "):
                        session.away_wells = [int(w) for w in line.split(' ')[1:]]
                    elif line.startswith("Condition: "):
                        type_in = line.split(' ')[1]
                        if 'Ripple' in type_in or 'Interruption' in type_in:
                            session.isRippleInterruption = True
                        elif 'None' in type_in:
                            session.isNoInterruption = True
                        elif 'Delayed' in type_in:
                            session.isDelayedInterruption = True
                        else:
                            print("Couldn't recognize Condition {} in file {}".format(type_in, info_file))
                    elif line.startswith("Thresh: "):
                        session.ripple_detection_threshold = float(line.split(' ')[1])
                    elif line.startswith("Last Away: "):
                        session.last_away_well = float(line.split(' ')[2])
                    elif line.startswith("Last well: "):
                        ended_on = line.split(' ')[2]
                        if 'H' in ended_on:
                            session.ended_on_home = True
                        elif 'A' in ended_on:
                            session.ended_on_home = False
                        else:
                            print("Couldn't recognize last well {} in file {}".format(ended_on, info_file))
                    elif line.startswith("ITI Stim On: "):
                        iti_on = line.split(' ')[3]
                        if 'Y' in iti_on:
                            session.ITI_stim_on = True
                        elif 'N' in iti_on:
                            session.ITI_stim_on = False
                        else:
                            print("Couldn't recognize ITI Stim condition {} in file {}".format(
                                iti_on, info_file))
                    elif line.startswith("Probe Stim On: "):
                        probe_on = line.split(' ')[3]
                        if 'Y' in probe_on:
                            session.probe_stim_on = True
                        elif 'N' in probe_on:
                            session.probe_stim_on = False
                        else:
                            print("Couldn't recognize Probe Stim condition {} in file {}".format(
                                probe_on, info_file))
                    else:
                        session.notes.append(line)

        except Exception as err:
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

        if session.home_well == 0:
            print("Home well not listed in notes file, skipping")
            continue

        session.home_x, session.home_y = get_well_coordinates(session.home_well)

        # ===================================
        # Analyze data
        # ===================================
        # which away wells were visited?
        session.num_away_found = next((i for i in range(
            len(session.away_wells)) if session.away_wells[i] == session.last_away_well), -1) + 1
        session.visited_away_wells = session.away_wells[0:session.num_away_found]

        # separating movement time from still time
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

        if session.date_str in INSPECT_IN_DETAIL:
            if INSPECT_NANVALS:
                print("{} nan xs, {} nan ys, {} nan ts".format(sum(np.isnan(session.probe_pos_xs)),
                                                               sum(np.isnan(session.probe_pos_xs)),
                                                               sum(np.isnan(session.probe_pos_xs))))

                nanxs = np.argwhere(np.isnan(session.probe_pos_xs))
                nanxs = nanxs.T[0]
                num_nan_x = np.size(nanxs)
                if num_nan_x > 0:
                    print(nanxs, num_nan_x, nanxs[0])
                    for i in range(min(5, num_nan_x)):
                        ni = nanxs[i]
                        print("index {}, x=({},nan,{}), y=({},{},{}), t=({},{},{})".format(
                            ni, session.probe_pos_xs[ni - 1], session.probe_pos_xs[ni + 1],
                            session.probe_pos_ys[ni - 1], session.probe_pos_ys[ni],
                            session.probe_pos_ys[ni + 1], session.probe_pos_ts[ni - 1],
                            session.probe_pos_ts[ni], session.probe_pos_ts[ni+1]
                        ))

            if INSPECT_PROBE_BEHAVIOR_PLOT:
                plt.clf()
                plt.scatter(session.home_x, session.home_y, color='green', zorder=2)
                plt.plot(session.probe_pos_xs, session.probe_pos_ys, zorder=0)
                plt.grid('on')
                plt.show()

        # avg dist to home and times at which rat entered home region
        session.ctrl_home_well = 49 - session.home_well
        session.ctrl_home_x, session.ctrl_home_y = get_well_coordinates(session.ctrl_home_well)

        session.bt_nearest_wells = getNearestWell(session.bt_pos_xs, session.bt_pos_ys)
        # if session.date_str in INSPECT_IN_DETAIL and INSPECT_PLOT_WELL_OCCUPANCIES:
        #     plt.clf()
        #     plt.scatter(session.bt_pos_xs, session.bt_pos_ys, s=1, c=session.bt_nearest_wells)
        #     plt.show()
        #     exit()

        session.bt_well_entry_idxs, session.bt_well_exit_idxs, \
            session.bt_well_entry_times, session.bt_well_exit_times = \
            getWellEntryAndExitTimes(session.bt_nearest_wells, session.bt_pos_ts)

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
                        plt.plot(session.bt_pos_xs[i1:i2], session.bt_pos_ys[i1:i2], color=color)
                    except:
                        print("well {} had {} entries, {} exits".format(
                            wi, len(session.bt_well_entry_idxs[i]), len(session.bt_well_exit_idxs[i])))

            plt.show()

        for i in range(len(all_well_idxs)):
            # print("well {} had {} entries, {} exits".format(
            #     all_well_idxs[i], len(session.bt_well_entry_idxs[i]), len(session.bt_well_exit_idxs[i])))
            assert len(session.bt_well_entry_times[i]) == len(session.bt_well_exit_times[i])

        session.bt_home_well_entry_times = session.bt_well_entry_times[np.argmax(
            all_well_idxs == session.home_well)]
        session.bt_home_well_exit_times = session.bt_well_exit_times[np.argmax(
            all_well_idxs == session.home_well)]

        session.bt_mean_dist_to_home_well = getMeanDistToWell(session.bt_pos_xs, session.bt_pos_ys,
                                                              session.home_x, session.home_y)
        session.bt_mv_mean_dist_to_home_well = getMeanDistToWell(session.bt_mv_xs, session.bt_mv_ys,
                                                                 session.home_x, session.home_y)
        session.bt_still_mean_dist_to_home_well = getMeanDistToWell(session.bt_still_xs, session.bt_still_ys,
                                                                    session.home_x, session.home_y)

        session.bt_ctrl_home_well_entry_times = session.bt_well_entry_times[np.argmax(
            all_well_idxs == session.ctrl_home_well)]
        session.bt_ctrl_home_well_exit_times = session.bt_well_exit_times[np.argmax(
            all_well_idxs == session.ctrl_home_well)]

        session.bt_mean_dist_to_ctrl_home_well = getMeanDistToWell(session.bt_pos_xs, session.bt_pos_ys,
                                                                   session.ctrl_home_x, session.ctrl_home_y)
        session.bt_mv_mean_dist_to_ctrl_home_well = getMeanDistToWell(session.bt_mv_xs, session.bt_mv_ys,
                                                                      session.ctrl_home_x, session.ctrl_home_y)
        session.bt_still_mean_dist_to_ctrl_home_well = getMeanDistToWell(session.bt_still_xs, session.bt_still_ys,
                                                                         session.ctrl_home_x, session.ctrl_home_y)

        # same for during probe
        session.probe_nearest_wells = getNearestWell(session.probe_pos_xs, session.probe_pos_ys)

        session.probe_well_entry_idxs, session.probe_well_exit_idxs, \
            session.probe_well_entry_times, session.probe_well_exit_times = \
            getWellEntryAndExitTimes(session.probe_nearest_wells, session.probe_pos_ts)

        for i in range(len(all_well_idxs)):
            # print(i, len(session.probe_well_entry_times[i]), len(session.probe_well_exit_times[i]))
            assert len(session.probe_well_entry_times[i]) == len(session.probe_well_exit_times[i])

        session.probe_home_well_entry_times = session.probe_well_entry_times[np.argmax(
            all_well_idxs == session.home_well)]
        session.probe_home_well_exit_times = session.probe_well_exit_times[np.argmax(
            all_well_idxs == session.home_well)]

        session.probe_mean_dist_to_home_well = getMeanDistToWell(session.probe_pos_xs, session.probe_pos_ys,
                                                                 session.home_x, session.home_y)
        session.probe_mv_mean_dist_to_home_well = getMeanDistToWell(session.probe_mv_xs, session.probe_mv_ys,
                                                                    session.home_x, session.home_y)
        session.probe_still_mean_dist_to_home_well = getMeanDistToWell(session.probe_still_xs, session.probe_still_ys,
                                                                       session.home_x, session.home_y)

        session.probe_ctrl_home_well_entry_times = session.probe_well_entry_times[np.argmax(
            all_well_idxs == session.ctrl_home_well)]
        session.probe_ctrl_home_well_exit_times = session.probe_well_exit_times[np.argmax(
            all_well_idxs == session.ctrl_home_well)]

        session.probe_mean_dist_to_ctrl_home_well = getMeanDistToWell(session.probe_pos_xs, session.probe_pos_ys,
                                                                      session.ctrl_home_x, session.ctrl_home_y)
        session.probe_mv_mean_dist_to_ctrl_home_well = getMeanDistToWell(session.probe_mv_xs, session.probe_mv_ys,
                                                                         session.ctrl_home_x, session.ctrl_home_y)
        session.probe_still_mean_dist_to_ctrl_home_well = getMeanDistToWell(session.probe_still_xs, session.probe_still_ys,
                                                                            session.ctrl_home_x, session.ctrl_home_y)

        # ===================================
        # Now save this session
        # ===================================
        dataob.allSessions.append(session)

    # save all sessions to disk
    dataob.saveToFile(os.path.join(output_dir, out_filename))
