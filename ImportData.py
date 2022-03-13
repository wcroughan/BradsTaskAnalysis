from BTData import *
import pandas as pd
import numpy as np
import os
import csv
import glob
import json
import matplotlib.pyplot as plt
import random
import scipy
from scipy import stats, signal
from itertools import groupby
import MountainViewIO
from scipy.ndimage.filters import gaussian_filter
from datetime import datetime
import sys

from consts import all_well_names, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE
from UtilFunctions import readWellCoordsFile, readRawPositionData, readClipData, \
    processPosData, getWellCoordinates, getNearestWell, getWellEntryAndExitTimes, \
    quadrantOfWell, getListOfVisitedWells, onWall, getRipplePower, detectRipples, \
    getInfoForAnimal, AnimalInfo, generateFoundWells, getUSBVideoFile
from ClipsMaker import runPositionAnnotator
from TrodesCameraExtrator import getTrodesLightTimes, processRawTrodesVideo

INSPECT_ALL = False
INSPECT_IN_DETAIL = []
# INSPECT_IN_DETAIL = ["20200526"]
INSPECT_NANVALS = False
INSPECT_PROBE_BEHAVIOR_PLOT = False
INSPECT_PLOT_WELL_OCCUPANCIES = False
ENFORCE_DIFFERENT_WELL_COLORS = False
RUN_JUST_SPECIFIED = False
SPECIFIED_DAYS = ["20200604"]
INSPECT_BOUTS = True
SAVE_DONT_SHOW = True
SHOW_CURVATURE_VIDEO = False
SKIP_LFP = False
SKIP_PREV_SESSION = True
JUST_EXTRACT_TRODES_DATA = False

TEST_NEAREST_WELL = False

if len(sys.argv) == 2:
    animal_name = sys.argv[1]
else:
    animal_name = 'B13'

print("Importing data for animal ", animal_name)
animalInfo = getInfoForAnimal(animal_name)

if not os.path.exists(animalInfo.output_dir):
    os.mkdir(animalInfo.output_dir)

all_data_dirs = sorted(os.listdir(animalInfo.data_dir), key=lambda s: (
    s.split('_')[0], s.split('_')[1]))
behavior_notes_dir = os.path.join(animalInfo.data_dir, 'behavior_notes')

if os.path.exists(os.path.join(animalInfo.output_dir, animalInfo.out_filename)):
    # confirm = input("Output file exists already. Overwrite? (y/n):")
    confirm = "y"
    if confirm != "y":
        exit()

all_quadrant_idxs = [0, 1, 2, 3]
well_name_to_idx = np.empty((np.max(all_well_names)+1))
well_name_to_idx[:] = np.nan
for widx, wname in enumerate(all_well_names):
    well_name_to_idx[wname] = widx

VEL_THRESH = 10  # cm/s
PIXELS_PER_CM = 5.0

# Typical observed amplitude of LFP deflection on stimulation
DEFLECTION_THRESHOLD_HI = 10000.0
DEFLECTION_THRESHOLD_LO = 2000.0
MIN_ARTIFACT_DISTANCE = int(0.05 * LFP_SAMPLING_RATE)

# # Typical duration of the stimulation artifact - For peak detection
# MIN_ARTIFACT_PERIOD = int(0.1 * LFP_SAMPLING_RATE)
# # Typical duration of a Sharp-Wave Ripple
# ACCEPTED_RIPPLE_LENGTH = int(0.2 * LFP_SAMPLING_RATE)

# constants for exploration bout analysis
# raise Exception("try longer sigmas here")
BOUT_VEL_SM_SIGMA_SECS = 1.5
PAUSE_MAX_SPEED_CM_S = 8.0
MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS = 2.5
MIN_EXPLORE_TIME_SECS = 3.0
MIN_EXPLORE_NUM_WELLS = 4
# COUNT_ONE_WELL_VISIT_PER_BOUT = False

# constants for ballisticity
# BALL_TIME_INTERVALS = list(range(1, 12, 3))
BALL_TIME_INTERVALS = list(range(1, 24))
# KNOT_H_CM = 20.0
KNOT_H_CM = 8.0
KNOT_H_POS = KNOT_H_CM * PIXELS_PER_CM

# well_coords_file = '/home/wcroughan/repos/BradsTaskAnalysis/well_locations.csv'

dataob = BTData()

# ===================================
# Filter to just relevant directories
# ===================================
filtered_data_dirs = []
prevSessionDirs = []
prevSession = None
for session_idx, session_dir in enumerate(all_data_dirs):
    if session_dir == "behavior_notes" or "numpy_objs" in session_dir:
        continue

    if not os.path.isdir(os.path.join(animalInfo.data_dir, session_dir)):
        print("skipping this file ... dirs only")
        continue

    dir_split = session_dir.split('_')
    if dir_split[-1] == "Probe" or dir_split[-1] == "ITI":
        # will deal with this in another loop
        print("skipping, is probe or iti")
        continue

    date_str = dir_split[0][-8:]

    if RUN_JUST_SPECIFIED and date_str not in SPECIFIED_DAYS:
        continue

    if date_str in animalInfo.excluded_dates:
        print("skipping, excluded date")
        prevSession = session_dir
        continue

    if animalInfo.minimum_date is not None and date_str < animalInfo.minimum_date:
        print("skipping date {}, is before minimum date {}".format(date_str, animalInfo.minimum_date))
        prevSession = session_dir
        continue

    filtered_data_dirs.append(session_dir)
    prevSessionDirs.append(prevSession)

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
    time_str = dir_split[1]
    session_name_pfx = dir_split[0][0:-8]
    session.date_str = date_str
    session.name = session_dir
    session.time_str = time_str
    s = "{}_{}".format(date_str, time_str)
    # print(s)
    session.date = datetime.strptime(s, "%Y%m%d_%H%M%S")
    print(session.date)

    session.prevSessionDir = prevSessionDirs[session_idx]

    # check for files that belong to this date
    session.bt_dir = session_dir
    gl = animalInfo.data_dir + session_name_pfx + date_str + "*"
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
        session.probe_dir = os.path.join(animalInfo.data_dir, session_dir)
    if not session.separate_iti_file and session.recorded_iti:
        session.iti_dir = session_dir

    file_str = os.path.join(animalInfo.data_dir, session_dir, session_dir)
    all_lfp_data = []
    session.bt_lfp_fnames = []

    # check if this session is excluded
    info_file = os.path.join(behavior_notes_dir, date_str + ".txt")
    if not os.path.exists(info_file):
        # Switched numbering scheme once started doing multiple sessions a day
        seshs_on_this_day = sorted(
            list(filter(lambda seshdir: session.date_str + "_" in seshdir, filtered_data_dirs)))
        num_on_this_day = len(seshs_on_this_day)
        # print("Looking for info file for {}".format(session_dir))
        for i in range(num_on_this_day):
            print("\t{}".format(seshs_on_this_day[i]))
            if seshs_on_this_day[i] == session_dir:
                sesh_idx_within_day = i
        possible_info_files = sorted(
            glob.glob(os.path.join(behavior_notes_dir, date_str + "_*.txt")))
        # print(possible_info_files)
        # info_file = os.path.join(behavior_notes_dir, date_str + "_" +
        #  str(sesh_idx_within_day+1) + ".txt")
        info_file = possible_info_files[sesh_idx_within_day]
        print(info_file)

    if "".join(os.path.basename(info_file).split(".")[0:-1]) in animalInfo.excluded_sessions:
        print(session_dir, " excluded session, skipping")
        continue

    # ===================================
    # Get flags and info from info file
    # ===================================

    prevsession_dir = prevSessionDirs[session_idx]
    if prevsession_dir is None or SKIP_PREV_SESSION:
        prevSessionInfoFile = None
    else:
        prevdir_split = prevsession_dir.split('_')
        prevdate_str = prevdir_split[0][-8:]
        prevSessionInfoFile = os.path.join(behavior_notes_dir, prevdate_str + ".txt")
        if not os.path.exists(prevSessionInfoFile):
            # Switched numbering scheme once started doing multiple sessions a day
            seshs_on_this_day = sorted(
                list(filter(lambda seshdir: session.date_str + "_" in seshdir, filtered_data_dirs)))
            num_on_this_day = len(seshs_on_this_day)
            for i in range(num_on_this_day):
                if seshs_on_this_day[i] == prevsession_dir:
                    sesh_idx_within_day = i
            prevSessionInfoFile = os.path.join(behavior_notes_dir, prevdate_str + "_" +
                                               str(sesh_idx_within_day+1) + ".txt")

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

                if JUST_EXTRACT_TRODES_DATA and field_name.lower() not in ["reference", "ref", "baseline"]:
                    continue

                if field_name.lower() == "home":
                    session.home_well = int(field_val)
                elif field_name.lower() == "aways":
                    session.away_wells = [int(w) for w in field_val.strip().split(' ')]
                    # print(line)
                    # print(field_name, field_val)
                    # print(session.away_wells)
                elif field_name.lower() == "condition":
                    type_in = field_val.lower()
                    if 'ripple' in type_in or 'interruption' in type_in:
                        session.isRippleInterruption = True
                    elif 'none' in type_in:
                        session.isNoInterruption = True
                    elif 'delay' in type_in:
                        session.isDelayedInterruption = True
                    else:
                        print("Couldn't recognize Condition {} in file {}".format(
                            type_in, info_file))
                elif field_name.lower() == "thresh":
                    if "low" in field_val.lower():
                        session.ripple_detection_threshold = 2.5
                    elif "high" in field_val.lower():
                        session.ripple_detection_threshold = 4
                    elif "med" in field_val.lower():
                        session.ripple_detection_threshold = 3
                    else:
                        session.ripple_detection_threshold = float(
                            field_val)
                elif field_name.lower() == "last away":
                    if field_val.strip() == "None":
                        session.last_away_well = None
                    else:
                        session.last_away_well = float(field_val)
                    # print(field_val)
                elif field_name.lower() == "reference" or field_name.lower() == "ref":
                    session.ripple_detection_tetrodes = [int(field_val)]
                    # print(field_val)
                elif field_name.lower() == "baseline":
                    session.ripple_baseline_tetrode = int(field_val)
                elif field_name.lower() == "last well":
                    if field_val.strip() == "None":
                        session.found_first_home = False
                        session.ended_on_home = False
                    else:
                        session.found_first_home = True
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
                elif field_name.lower() == "probe performed":
                    if 'Y' in field_val:
                        session.probe_performed = True
                    elif 'N' in field_val:
                        session.probe_performed = False
                        if animal_name == "Martin":
                            raise Exception("I thought all martin runs had a probe")
                    else:
                        print("Couldn't recognize Probe performed val {} in file {}".format(
                            field_val, info_file))
                elif field_name.lower() == "task ended at":
                    session.bt_ended_at_well = int(field_val)
                elif field_name.lower() == "probe ended at":
                    session.probe_ended_at_well = int(field_val)
                elif field_name.lower() == "weight":
                    session.weight = float(field_val)
                else:
                    session.notes.append(line)

        if not JUST_EXTRACT_TRODES_DATA:
            if session.probe_performed and session.probe_ended_at_well is None:
                raise Exception(
                    "Didn't mark the probe end well for session {}".format(session.name))

            if not (session.last_away_well == session.away_wells[-1] and session.ended_on_home) and session.bt_ended_at_well is None:
                raise Exception(
                    "Didn't find all the wells but task end well not marked for session {}".format(session.name))

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

    if prevSessionInfoFile is not None:
        try:
            with open(prevSessionInfoFile, 'r') as f:
                session.prevSessionInfoParsed = True
                lines = f.readlines()
                for line in lines:
                    lineparts = line.split(":")
                    if len(lineparts) != 2:
                        continue

                    field_name = lineparts[0]
                    field_val = lineparts[1]

                    if field_name.lower() == "home":
                        session.prevSessionHome = int(field_val)
                    elif field_name.lower() == "aways":
                        session.prevSessionAways = [int(w)
                                                    for w in field_val.strip().split(' ')]
                    elif field_name.lower() == "condition":
                        type_in = field_val
                        if 'Ripple' in type_in or 'Interruption' in type_in:
                            session.prevSessionIsRippleInterruption = True
                        elif 'None' in type_in:
                            session.prevSessionIsNoInterruption = True
                        elif 'Delay' in type_in:
                            session.prevSessionIsDelayedInterruption = True
                        else:
                            print("Couldn't recognize Condition {} in file {}".format(
                                type_in, prevSessionInfoFile))
                    elif field_name.lower() == "thresh":
                        if "Low" in field_val:
                            session.prevSession_ripple_detection_threshold = 2.5
                        elif "High" in field_val:
                            session.prevSession_ripple_detection_threshold = 4
                        else:
                            session.prevSession_ripple_detection_threshold = float(
                                field_val)
                    elif field_name.lower() == "last away":
                        session.prevSession_last_away_well = float(field_val)
                        # print(field_val)
                    elif field_name.lower() == "last well":
                        ended_on = field_val
                        if 'H' in field_val:
                            session.prevSession_ended_on_home = True
                        elif 'A' in field_val:
                            session.prevSession_ended_on_home = False
                        else:
                            print("Couldn't recognize last well {} in file {}".format(
                                field_val, prevSessionInfoFile))
                    elif field_name.lower() == "iti stim on":
                        if 'Y' in field_val:
                            session.prevSession_ITI_stim_on = True
                        elif 'N' in field_val:
                            session.prevSession_ITI_stim_on = False
                        else:
                            print("Couldn't recognize ITI Stim condition {} in file {}".format(
                                field_val, prevSessionInfoFile))
                    elif field_name.lower() == "probe stim on":
                        if 'Y' in field_val:
                            session.prevSession_probe_stim_on = True
                        elif 'N' in field_val:
                            session.prevSession_probe_stim_on = False
                        else:
                            print("Couldn't recognize Probe Stim condition {} in file {}".format(
                                field_val, prevSessionInfoFile))
                    else:
                        pass

        except FileNotFoundError as err:
            session.prevSessionInfoParsed = False
            print("Couldn't read from prev session info file " + prevSessionInfoFile)

    else:
        session.prevSessionInfoParsed = False

    if session.home_well == 0 and not JUST_EXTRACT_TRODES_DATA:
        print("Home well not listed in notes file, skipping")
        continue

    if session.foundWells is None:
        session.foundWells = generateFoundWells(
            session.home_well, session.away_wells, session.last_away_well, session.ended_on_home, session.found_first_home)

    if not JUST_EXTRACT_TRODES_DATA:
        # ===================================
        # get well coordinates
        # ===================================
        well_coords_file_name = file_str + '.1.wellLocations.csv'
        if not os.path.exists(well_coords_file_name):
            well_coords_file_name = os.path.join(animalInfo.data_dir, 'well_locations.csv')
            print("Specific well locations not found, falling back to file {}".format(well_coords_file_name))
        session.well_coords_map = readWellCoordsFile(well_coords_file_name)
        session.home_x, session.home_y = getWellCoordinates(
            session.home_well, session.well_coords_map)

        # ===================================
        # Read position data
        # ===================================
        trackingFile = file_str + '.1.videoPositionTracking'
        if os.path.exists(trackingFile):
            position_data_metadata, position_data = readRawPositionData(
                file_str + '.1.videoPositionTracking')

            if "source" not in position_data_metadata or "TrodesCameraExtractor" not in position_data_metadata["source"]:
                position_data = None
                os.rename(file_str + '.1.videoPositionTracking', file_str +
                          '.1.videoPositionTracking.manualOutput')
        else:
            position_data = None

        if position_data is None:
            print("running trodes camera extractor!")
            processRawTrodesVideo(file_str + '.1.h264')
            position_data_metadata, position_data = readRawPositionData(
                file_str + '.1.videoPositionTracking')

        if position_data is None:
            print("Warning: skipping position data")
            session.hasPositionData = False
        else:
            session.hasPositionData = True
            xs, ys, ts = processPosData(position_data, xLim=(
                animalInfo.X_START, animalInfo.X_FINISH), yLim=(animalInfo.Y_START, animalInfo.Y_FINISH))

            if "lightonframe" in position_data_metadata:
                trodesLightOnFrame = position_data_metadata['lightonframe']
                if trodesLightOnFrame[-3:] == "\\n'":
                    trodesLightOnFrame = trodesLightOnFrame[0:-3]
                trodesLightOnFrame = int(trodesLightOnFrame)
                trodesLightOffFrame = position_data_metadata['lightoffframe']
                if trodesLightOffFrame[-3:] == "\\n'":
                    trodesLightOffFrame = trodesLightOffFrame[0:-3]
                trodesLightOffFrame = int(trodesLightOffFrame)
                session.trodesLightOnFrame = trodesLightOnFrame
                session.trodesLightOffFrame = trodesLightOffFrame
                print(session.trodesLightOffFrame, session.trodesLightOnFrame)
                print(len(ts))
                session.trodesLightOnTime = ts[session.trodesLightOnFrame]
                session.trodesLightOffTime = ts[session.trodesLightOffFrame]
            else:
                # was a separate metadatafile made?
                positionMetadataFile = file_str + '.1.justLights'
                if os.path.exists(positionMetadataFile):
                    lightInfo = np.fromfile(positionMetadataFile, sep=",").astype(int)
                    session.trodesLightOffFrame = lightInfo[0]
                    session.trodesLightOnFrame = lightInfo[1]
                    print(lightInfo)
                    print(len(ts))
                    session.trodesLightOnTime = ts[session.trodesLightOnFrame]
                    session.trodesLightOffTime = ts[session.trodesLightOffFrame]
                else:
                    print("doing the lights")
                    session.trodesLightOffFrame, session.trodesLightOnFrame = getTrodesLightTimes(
                        file_str + '.1.h264', showVideo=True)
                    print(session.trodesLightOffFrame, session.trodesLightOnFrame)
                    print(len(ts))
                    session.trodesLightOnTime = ts[session.trodesLightOnFrame]
                    session.trodesLightOffTime = ts[session.trodesLightOffFrame]

            possibleDirectories = [
                "/media/WDC6/{}/".format(animal_name), "/media/WDC7/{}/".format(animal_name)]
            session.usbVidFile = getUSBVideoFile(session.name, possibleDirectories)

            clipsFileName = file_str + '.1.clips'
            if not os.path.exists(clipsFileName) and len(session.foundWells) > 0:
                print("clips file not found, gonna launch clips generator")
                all_pos_nearest_wells = getNearestWell(
                    xs, ys, session.well_coords_map)
                all_pos_well_entry_idxs, all_pos_well_exit_idxs, \
                    all_pos_well_entry_times, all_pos_well_exit_times = \
                    getWellEntryAndExitTimes(
                        all_pos_nearest_wells, ts)

                runPositionAnnotator(xs, ys, ts, session.trodesLightOffTime, session.trodesLightOnTime,
                                     session.usbVidFile, all_pos_well_entry_times, all_pos_well_exit_times,
                                     session.foundWells, not session.probe_performed, file_str + '.1')

            if session.separate_probe_file:
                probe_file_str = os.path.join(
                    session.probe_dir, os.path.basename(session.probe_dir))
                bt_time_clips = readClipData(file_str + '.1.clips')[0]
                probe_time_clips = readClipData(probe_file_str + '.1.clips')[0]
            else:
                time_clips = readClipData(file_str + '.1.clips')
                bt_time_clips = time_clips[0]
                if session.probe_performed:
                    probe_time_clips = time_clips[1]

            bt_start_idx = np.searchsorted(ts, bt_time_clips[0])
            bt_end_idx = np.searchsorted(ts, bt_time_clips[1])
            session.bt_pos_xs = xs[bt_start_idx:bt_end_idx]
            session.bt_pos_ys = ys[bt_start_idx:bt_end_idx]
            session.bt_pos_ts = ts[bt_start_idx:bt_end_idx]

            if session.probe_performed:
                if session.separate_probe_file:
                    _, position_data = readRawPositionData(
                        probe_file_str + '.1.videoPositionTracking')
                    xs, ys, ts = processPosData(position_data, xLim=(
                        animalInfo.X_START, animalInfo.X_FINISH), yLim=(animalInfo.Y_START, animalInfo.Y_FINISH))

                probe_start_idx = np.searchsorted(ts, probe_time_clips[0])
                probe_end_idx = np.searchsorted(ts, probe_time_clips[1])
                session.probe_pos_xs = xs[probe_start_idx:probe_end_idx]
                session.probe_pos_ys = ys[probe_start_idx:probe_end_idx]
                session.probe_pos_ts = ts[probe_start_idx:probe_end_idx]

                if np.nanmax(session.probe_pos_ys) < 550 and np.nanmax(session.probe_pos_ys) > 400:
                    print("Position data just covers top of the environment")
                    session.positionOnlyTop = True
                else:
                    pass
                    # print("min max")
                    # print(np.nanmin(session.probe_pos_ys))
                    # print(np.nanmax(session.probe_pos_ys))
                    # plt.plot(session.bt_pos_xs, session.bt_pos_ys)
                    # plt.show()
                    # plt.plot(session.probe_pos_xs, session.probe_pos_ys)
                    # plt.show()

        # for i in range(len(session.ripple_detection_tetrodes)):
        #     spkdir = file_str + ".spikes"
        #     if not os.path.exists(spkdir):
        #         print(spkdir, "doesn't exists, gonna try and extract the spikes")
        #         syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportspikes -rec " + file_str + ".rec"
        #         print(syscmd)
        #         os.system(syscmd)

    if len(session.ripple_detection_tetrodes) == 0:
        session.ripple_detection_tetrodes = [animalInfo.DEFAULT_RIP_DET_TET]

    for i in range(len(session.ripple_detection_tetrodes)):
        lfpdir = file_str + ".LFP"
        if not os.path.exists(lfpdir):
            print(lfpdir, "doesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + file_str + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + file_str + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + file_str + ".rec"
            print(syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + session_dir + ".LFP_nt" + \
            str(session.ripple_detection_tetrodes[i]) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        session.bt_lfp_fnames.append(lfpfilename)
        all_lfp_data.append(MountainViewIO.loadLFP(data_file=session.bt_lfp_fnames[-1]))

    if session.ripple_baseline_tetrode is None:
        session.ripple_baseline_tetrode = animalInfo.DEFAULT_RIP_BAS_TET

    # I think Martin didn't have this baseline tetrode? Need to check
    if session.ripple_baseline_tetrode is None:
        lfpdir = file_str + ".LFP"
        if not os.path.exists(lfpdir):
            print(lfpdir, "doesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + file_str + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + file_str + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + file_str + ".rec"
            print(syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + session_dir + ".LFP_nt" + \
            str(session.ripple_baseline_tetrode) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        session.bt_lfp_baseline_fname = lfpfilename
        baseline_lfp_data = MountainViewIO.loadLFP(data_file=session.bt_lfp_baseline_fname)

    if JUST_EXTRACT_TRODES_DATA:
        continue

    # ======================================================================
    # ===================================
    # Analyze data
    # ===================================
    # ======================================================================

    # ===================================
    # LFP
    # ===================================
    if not SKIP_LFP:
        lfp_data = all_lfp_data[0][1]['voltage']
        lfp_timestamps = all_lfp_data[0][0]['time']

        # Deflections represent interruptions from stimulation, artifacts include these and also weird noise
        lfp_deflections = signal.find_peaks(-lfp_data, height=DEFLECTION_THRESHOLD_HI,
                                            distance=MIN_ARTIFACT_DISTANCE)
        interruption_idxs = lfp_deflections[0]
        session.interruption_timestamps = lfp_timestamps[interruption_idxs]
        session.interruptionIdxs = interruption_idxs

        session.bt_interruption_pos_idxs = np.searchsorted(
            session.bt_pos_ts, session.interruption_timestamps)
        session.bt_interruption_pos_idxs = session.bt_interruption_pos_idxs[
            session.bt_interruption_pos_idxs < len(session.bt_pos_ts)]

        # ========
        # dealing with weird Martin sessions:
        showPlot = False
        print("{} interruptions detected".format(len(session.bt_interruption_pos_idxs)))
        if len(session.bt_interruption_pos_idxs) < 50:
            if (session.isRippleInterruption):
                print(
                    "WARNING: IGNORING BEHAVIOR NOTES FILE BECAUSE SEEING FEWER THAN 50 INTERRUPTIONS, CALLING THIS A CONTROL SESSION")
            else:
                print(
                    "WARNING: very few interruptions. This was a delay control but is basically a no-stim control")
            # showPlot = True
            session.isRippleInterruption = False
        elif len(session.bt_interruption_pos_idxs) < 100:
            print("50-100 interruptions: not overriding label")
            # showPlot = True

        print("Condition - {}".format("SWR" if session.isRippleInterruption else "Ctrl"))
        lfp_deflections = signal.find_peaks(np.abs(
            np.diff(lfp_data, prepend=lfp_data[0])), height=DEFLECTION_THRESHOLD_LO, distance=MIN_ARTIFACT_DISTANCE)
        lfp_artifact_idxs = lfp_deflections[0]
        session.artifact_timestamps = lfp_timestamps[lfp_artifact_idxs]
        session.artifactIdxs = lfp_artifact_idxs

        bt_lfp_start_idx = np.searchsorted(lfp_timestamps, session.bt_pos_ts[0])
        bt_lfp_end_idx = np.searchsorted(lfp_timestamps, session.bt_pos_ts[-1])
        btLFPData = lfp_data[bt_lfp_start_idx:bt_lfp_end_idx]
        # bt_lfp_artifact_idxs = lfp_artifact_idxs - bt_lfp_start_idx
        bt_lfp_artifact_idxs = interruption_idxs - bt_lfp_start_idx
        bt_lfp_artifact_idxs = bt_lfp_artifact_idxs[bt_lfp_artifact_idxs > 0]

        pre_bt_interruption_idxs = interruption_idxs[interruption_idxs < bt_lfp_start_idx]
        pre_bt_interruption_idxs_first_half = interruption_idxs[interruption_idxs < int(
            bt_lfp_start_idx/2)]

        _, _, session.prebtMeanRipplePower, session.prebtStdRipplePower = getRipplePower(
            lfp_data[0:bt_lfp_start_idx], omit_artifacts=False)
        _, ripple_power, _, _ = getRipplePower(
            btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=session.prebtMeanRipplePower, stdPower=session.prebtStdRipplePower, showPlot=showPlot)
        session.btRipStartIdxsPreStats, session.btRipLensPreStats, session.btRipPeakIdxsPreStats, session.btRipPeakAmpsPreStats, session.btRipCrossThreshIdxsPreStats = \
            detectRipples(ripple_power)
        session.btRipStartTimestampsPreStats = lfp_timestamps[session.btRipStartIdxsPreStats + bt_lfp_start_idx]

        _, _, session.prebtMeanRipplePowerArtifactsRemoved, session.prebtStdRipplePowerArtifactsRemoved = getRipplePower(
            lfp_data[0:bt_lfp_start_idx], lfp_deflections=pre_bt_interruption_idxs)
        _, _, session.prebtMeanRipplePowerArtifactsRemovedFirstHalf, session.prebtStdRipplePowerArtifactsRemovedFirstHalf = getRipplePower(
            lfp_data[0:int(bt_lfp_start_idx/2)], lfp_deflections=pre_bt_interruption_idxs)

        if session.probe_performed and not session.separate_iti_file and not session.separate_probe_file:
            ITI_MARGIN = 5  # units: seconds
            itiLfpStart_ts = session.bt_pos_ts[-1] + TRODES_SAMPLING_RATE * ITI_MARGIN
            itiLfpEnd_ts = session.probe_pos_ts[0] - TRODES_SAMPLING_RATE * ITI_MARGIN
            itiLfpStart_idx = np.searchsorted(lfp_timestamps, itiLfpStart_ts)
            itiLfpEnd_idx = np.searchsorted(lfp_timestamps, itiLfpEnd_ts)
            itiLFPData = lfp_data[itiLfpStart_idx:itiLfpEnd_idx]
            session.ITIRippleIdxOffset = itiLfpStart_idx
            session.itiLfpStart_ts = itiLfpStart_ts
            session.itiLfpEnd_ts = itiLfpEnd_ts

            # in general none, but there's a few right at the start of where this is defined
            itiStimIdxs = interruption_idxs - itiLfpStart_idx
            zeroIdx = np.searchsorted(itiStimIdxs, 0)
            itiStimIdxs = itiStimIdxs[zeroIdx:]

            _, ripple_power, session.ITIMeanRipplePower, session.ITIStdRipplePower = getRipplePower(
                itiLFPData, lfp_deflections=itiStimIdxs)
            session.ITIRipStartIdxs, session.ITIRipLens, session.ITIRipPeakIdxs, session.ITIRipPeakAmps, session.ITIRipCrossThreshIdxs = \
                detectRipples(ripple_power)
            session.ITIRipStartTimestamps = lfp_timestamps[session.ITIRipStartIdxs + itiLfpStart_idx]

            session.ITIDuration = (itiLfpEnd_ts - itiLfpStart_ts) / \
                BTSession.TRODES_SAMPLING_RATE

            probeLfpStart_ts = session.probe_pos_ts[0]
            probeLfpEnd_ts = session.probe_pos_ts[-1]
            probeLfpStart_idx = np.searchsorted(lfp_timestamps, probeLfpStart_ts)
            probeLfpEnd_idx = np.searchsorted(lfp_timestamps, probeLfpEnd_ts)
            probeLFPData = lfp_data[probeLfpStart_idx:probeLfpEnd_idx]
            session.probeRippleIdxOffset = probeLfpStart_idx
            session.probeLfpStart_ts = probeLfpStart_ts
            session.probeLfpEnd_ts = probeLfpEnd_ts

            _, ripple_power, session.probeMeanRipplePower, session.probeStdRipplePower = getRipplePower(
                probeLFPData, omit_artifacts=False)
            session.probeRipStartIdxs, session.probeRipLens, session.probeRipPeakIdxs, session.probeRipPeakAmps, session.probeRipCrossThreshIdxs = \
                detectRipples(ripple_power)
            session.probeRipStartTimestamps = lfp_timestamps[session.probeRipStartIdxs + probeLfpStart_idx]

            session.probeDuration = (probeLfpEnd_ts - probeLfpStart_ts) / \
                BTSession.TRODES_SAMPLING_RATE

            _, itiRipplePowerProbeStats, _, _ = getRipplePower(
                itiLFPData, lfp_deflections=itiStimIdxs, meanPower=session.probeMeanRipplePower, stdPower=session.probeStdRipplePower)
            session.ITIRipStartIdxsProbeStats, session.ITIRipLensProbeStats, session.ITIRipPeakIdxsProbeStats, session.ITIRipPeakAmpsProbeStats, session.ITIRipCrossThreshIdxsProbeStats = \
                detectRipples(itiRipplePowerProbeStats)
            session.ITIRipStartTimestampsProbeStats = lfp_timestamps[
                session.ITIRipStartIdxsProbeStats + itiLfpStart_idx]

            _, btRipplePowerProbeStats, _, _ = getRipplePower(
                btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=session.probeMeanRipplePower, stdPower=session.probeStdRipplePower, showPlot=showPlot)
            session.btRipStartIdxsProbeStats, session.btRipLensProbeStats, session.btRipPeakIdxsProbeStats, session.btRipPeakAmpsProbeStats, session.btRipCrossThreshIdxsProbeStats = \
                detectRipples(btRipplePowerProbeStats)
            session.btRipStartTimestampsProbeStats = lfp_timestamps[
                session.btRipStartIdxsProbeStats + bt_lfp_start_idx]
    # ripple_power = getRipplePower(lfp_data, omit_artifacts=True,
    #                                 causal_smoothing=False, lfp_deflections=lfp_artifact_idxs)

    # ripple_power = ripple_power[0]
    # # ripple_power /= np.nanmax(ripple_power)
    # # ripple_power *= np.nanmax(lfp_data)

    # lfp_diff = np.diff(lfp_data, prepend=lfp_data[0])

    # plt.clf()
    # plt.plot(lfp_timestamps, lfp_data)
    # # # # plt.plot(lfp_timestamps, np.abs(lfp_diff))
    # # # plt.plot(lfp_timestamps, ripple_power)
    # plt.scatter(lfp_timestamps[lfp_artifact_idxs],
    #             # #             # np.abs(lfp_diff[lfp_artifact_idxs]), c=[[1, 0, 0, 1]], zorder=30)
    #             lfp_data[lfp_artifact_idxs], c=[[1, 0, 0, 1]], zorder=30)

    # if session.isDelayedInterruption:
    #     type_txt = 'Delayed'
    # elif session.isNoInterruption:
    #     type_txt = 'None'
    # elif session.isRippleInterruption:
    #     type_txt = 'Ripple'
    # else:
    #     type_txt = 'unknown'
    #     print("unknown session type")
    # plt.text(1, 1, type_txt, horizontalalignment='right',
    #          verticalalignment='top', transform=plt.gca().transAxes)
    # plt.show()

    # ARP_SZ = int(0.3 * float(LFP_SAMPLING_RATE))
    # aligned_rip_power = np.empty((len(interruption_idxs), ARP_SZ))
    # for ai, a in enumerate(interruption_idxs):
    #     aligned_rip_power[ai, :] = ripple_power[a - ARP_SZ:a]

    # avg_arp = np.nanmean(aligned_rip_power, axis=0)

    # xvals = np.linspace(-300, 0, ARP_SZ)
    # # plt.clf()
    # # plt.plot(xvals, avg_arp)
    # # plt.text(1, 1, type_txt, horizontalalignment='right',
    # #          verticalalignment='top', transform=plt.gca().transAxes)
    # # plt.text(1, 0.9, str(len(all_lfp_data)), horizontalalignment='right',
    # #          verticalalignment='top', transform=plt.gca().transAxes)
    # # plt.show()

    # continue

            print("{} ripples found during task".format(
                len(session.btRipStartTimestampsProbeStats)))

        elif session.probe_performed:
            print("Probe performed but LFP in a separate file for session", session.name)

    # ===================================
    # which away wells were visited?
    # ===================================
    if session.last_away_well is None:
        session.num_away_found = 0
    else:
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
        if session.num_home_found > 0:
            print("Well find times not marked for session {}".format(session.name))
    else:
        well_visit_times = readClipData(rewardClipsFile)
        assert session.num_away_found + \
            session.num_home_found == np.shape(well_visit_times)[0]
        session.home_well_find_times = well_visit_times[::2, 0]
        session.home_well_leave_times = well_visit_times[::2, 1]
        session.away_well_find_times = well_visit_times[1::2, 0]
        session.away_well_leave_times = well_visit_times[1::2, 1]

        session.home_well_find_pos_idxs = np.searchsorted(
            session.bt_pos_ts, session.home_well_find_times)
        session.home_well_leave_pos_idxs = np.searchsorted(
            session.bt_pos_ts, session.home_well_leave_times)
        session.away_well_find_pos_idxs = np.searchsorted(
            session.bt_pos_ts, session.away_well_find_times)
        session.away_well_leave_pos_idxs = np.searchsorted(
            session.bt_pos_ts, session.away_well_leave_times)

        if len(session.home_well_leave_times) == len(session.away_well_find_times):
            session.away_well_latencies = np.array(session.away_well_find_times) - \
                np.array(session.home_well_leave_times)
            session.home_well_latencies = np.array(session.home_well_find_times) - \
                np.append([session.bt_pos_ts[0]],
                          session.away_well_leave_times[0:-1])
        else:
            session.away_well_latencies = np.array(session.away_well_find_times) - \
                np.array(session.home_well_leave_times[0:-1])
            session.home_well_latencies = np.array(session.home_well_find_times) - \
                np.append([session.bt_pos_ts[0]],
                          session.away_well_leave_times)

    if session.hasPositionData:
        # ===================================
        # separating movement time from still time
        # ===================================
        bt_vel = np.sqrt(np.power(np.diff(session.bt_pos_xs), 2) +
                         np.power(np.diff(session.bt_pos_ys), 2))
        session.bt_vel_cm_s = np.divide(bt_vel, np.diff(session.bt_pos_ts) /
                                        TRODES_SAMPLING_RATE) / PIXELS_PER_CM
        bt_is_mv = session.bt_vel_cm_s > VEL_THRESH
        if len(bt_is_mv) > 0:
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

        if session.probe_performed:
            probe_vel = np.sqrt(np.power(np.diff(session.probe_pos_xs), 2) +
                                np.power(np.diff(session.probe_pos_ys), 2))
            session.probe_vel_cm_s = np.divide(probe_vel, np.diff(session.probe_pos_ts) /
                                               TRODES_SAMPLING_RATE) / PIXELS_PER_CM
            probe_is_mv = session.probe_vel_cm_s > VEL_THRESH
            if len(probe_is_mv) > 0:
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
        session.ctrl_home_x, session.ctrl_home_y = getWellCoordinates(
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

        for i in range(len(all_well_names)):
            # print("well {} had {} entries, {} exits".format(
            #     all_well_names[i], len(session.bt_well_entry_idxs[i]), len(session.bt_well_exit_idxs[i])))
            assert len(session.bt_well_entry_times[i]) == len(
                session.bt_well_exit_times[i])

        session.home_well_idx_in_allwells = np.argmax(
            all_well_names == session.home_well)
        session.ctrl_home_well_idx_in_allwells = np.argmax(
            all_well_names == session.ctrl_home_well)

        session.bt_home_well_entry_times = session.bt_well_entry_times[
            session.home_well_idx_in_allwells]
        session.bt_home_well_exit_times = session.bt_well_exit_times[
            session.home_well_idx_in_allwells]

        session.bt_ctrl_home_well_entry_times = session.bt_well_entry_times[
            session.ctrl_home_well_idx_in_allwells]
        session.bt_ctrl_home_well_exit_times = session.bt_well_exit_times[
            session.ctrl_home_well_idx_in_allwells]

        # same for during probe
        if session.probe_performed:
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

            for i in range(len(all_well_names)):
                # print(i, len(session.probe_well_entry_times[i]), len(session.probe_well_exit_times[i]))
                assert len(session.probe_well_entry_times[i]) == len(
                    session.probe_well_exit_times[i])

            session.probe_home_well_entry_times = session.probe_well_entry_times[
                session.home_well_idx_in_allwells]
            session.probe_home_well_exit_times = session.probe_well_exit_times[
                session.home_well_idx_in_allwells]

            session.probe_ctrl_home_well_entry_times = session.probe_well_entry_times[
                session.ctrl_home_well_idx_in_allwells]
            session.probe_ctrl_home_well_exit_times = session.probe_well_exit_times[
                session.ctrl_home_well_idx_in_allwells]

        # ===================================
        # truncate path when rat was recalled (as he's running to be picked up) based on visit time of marked well
        # Note need to check sessions that ended near 7
        # ===================================
        # bt_recall_pos_idx
        if session.bt_ended_at_well is None:
            assert session.last_away_well == session.away_wells[-1] and session.ended_on_home
            session.bt_ended_at_well = session.home_well

        bt_ended_at_well_idx = np.argmax(all_well_names == session.bt_ended_at_well)
        session.bt_recall_pos_idx = session.bt_well_exit_idxs[bt_ended_at_well_idx][-1]

        if session.probe_performed:
            probe_ended_at_well_idx = np.argmax(all_well_names == session.probe_ended_at_well)
            session.probe_recall_pos_idx = session.probe_well_exit_idxs[probe_ended_at_well_idx][-1]

    # ===================================
    # Sniff times marked by hand from USB camera (Not available for Martin)
    # ===================================

    gl = animalInfo.data_dir + session_dir + '/*.rgs'
    dir_list = glob.glob(gl)
    if len(dir_list) == 0:
        print("Couldn't find sniff times files")
        session.hasSniffTimes = False
    else:
        session.hasSniffTimes = True

        session.sniffTimesFile = dir_list[0]
        if len(dir_list) > 1:
            print("Warning, multiple rgs files found: {}".format(dir_list))

        if not os.path.exists(session.sniffTimesFile):
            raise Exception("Couldn't find rgs file")
        else:
            print("getting rgs from {}".format(session.sniffTimesFile))
            with open(session.sniffTimesFile, 'r') as stf:
                streader = csv.reader(stf)
                sniffData = [v for v in streader]

                session.well_sniff_times_entry = [[] for _ in all_well_names]
                session.well_sniff_times_exit = [[] for _ in all_well_names]
                session.bt_well_sniff_times_entry = [[] for _ in all_well_names]
                session.bt_well_sniff_times_exit = [[] for _ in all_well_names]
                session.probe_well_sniff_times_entry = [[] for _ in all_well_names]
                session.probe_well_sniff_times_exit = [[] for _ in all_well_names]

                session.sniff_pre_trial_light_off = int(sniffData[0][0])
                session.sniff_trial_start = int(sniffData[0][1])
                session.sniff_trial_stop = int(sniffData[0][2])
                session.sniff_probe_start = int(sniffData[1][0])
                session.sniff_probe_stop = int(sniffData[1][1])
                session.sniff_post_probe_light_on = int(sniffData[1][2])

                for i in sniffData[2:]:
                    w = int(well_name_to_idx[int(i[2])])
                    entry_time = int(i[0])
                    exit_time = int(i[1])
                    session.well_sniff_times_entry[w].append(entry_time)
                    session.well_sniff_times_exit[w].append(exit_time)

                    if exit_time < entry_time:
                        print("mismatched interval: {} - {}".format(entry_time, exit_time))
                        assert False
                    if exit_time < session.sniff_trial_stop:
                        session.bt_well_sniff_times_entry[w].append(entry_time)
                        session.bt_well_sniff_times_exit[w].append(exit_time)
                    else:
                        session.probe_well_sniff_times_entry[w].append(entry_time)
                        session.probe_well_sniff_times_exit[w].append(exit_time)

    if session.hasPositionData:
        # ===================================
        # Ballisticity of movements
        # ===================================
        furthest_interval = max(BALL_TIME_INTERVALS)
        assert np.all(np.diff(np.array(BALL_TIME_INTERVALS)) > 0)
        assert BALL_TIME_INTERVALS[0] > 0

        # idx is (time, interval len, dimension)
        d1 = len(session.bt_pos_ts) - furthest_interval
        delta = np.empty((d1, furthest_interval, 2))
        delta[:] = np.nan
        dx = np.diff(session.bt_pos_xs)
        dy = np.diff(session.bt_pos_ys)
        delta[:, 0, 0] = dx[0:d1]
        delta[:, 0, 1] = dy[0:d1]

        for i in range(1, furthest_interval):
            delta[:, i, 0] = delta[:, i-1, 0] + dx[i:i+d1]
            delta[:, i, 1] = delta[:, i-1, 1] + dy[i:i+d1]

        displacement = np.sqrt(np.sum(np.square(delta), axis=2))
        displacement[displacement == 0] = np.nan
        last_displacement = displacement[:, -1]
        session.bt_ball_displacement = last_displacement

        x = np.log(np.tile(np.arange(furthest_interval) + 1, (d1, 1)))
        y = np.log(displacement)
        assert np.all(np.logical_or(
            np.logical_not(np.isnan(y)), np.isnan(displacement)))
        y[y == -np.inf] = np.nan
        np.nan_to_num(y, copy=False, nan=np.nanmin(y))

        x = x - np.tile(np.nanmean(x, axis=1), (furthest_interval, 1)).T
        y = y - np.tile(np.nanmean(y, axis=1), (furthest_interval, 1)).T
        def m(arg): return np.mean(arg, axis=1)
        # beta = ((m(y * x) - m(x) * m(y))/m(x*x) - m(x)**2)
        beta = m(y*x)/m(np.power(x, 2))
        session.bt_ballisticity = beta
        assert np.sum(np.isnan(session.bt_ballisticity)) == 0

        if session.probe_performed:
            # idx is (time, interval len, dimension)
            d1 = len(session.probe_pos_ts) - furthest_interval
            delta = np.empty((d1, furthest_interval, 2))
            delta[:] = np.nan
            dx = np.diff(session.probe_pos_xs)
            dy = np.diff(session.probe_pos_ys)
            delta[:, 0, 0] = dx[0:d1]
            delta[:, 0, 1] = dy[0:d1]

            for i in range(1, furthest_interval):
                delta[:, i, 0] = delta[:, i-1, 0] + dx[i:i+d1]
                delta[:, i, 1] = delta[:, i-1, 1] + dy[i:i+d1]

            displacement = np.sqrt(np.sum(np.square(delta), axis=2))
            displacement[displacement == 0] = np.nan
            last_displacement = displacement[:, -1]
            session.probe_ball_displacement = last_displacement

            x = np.log(np.tile(np.arange(furthest_interval) + 1, (d1, 1)))
            y = np.log(displacement)
            assert np.all(np.logical_or(
                np.logical_not(np.isnan(y)), np.isnan(displacement)))
            y[y == -np.inf] = np.nan
            np.nan_to_num(y, copy=False, nan=np.nanmin(y))

            x = x - np.tile(np.nanmean(x, axis=1), (furthest_interval, 1)).T
            y = y - np.tile(np.nanmean(y, axis=1), (furthest_interval, 1)).T
            # beta = ((m(y * x) - m(x) * m(y))/m(x*x) - m(x)**2)
            beta = m(y*x)/m(np.power(x, 2))
            session.probe_ballisticity = beta
            assert np.sum(np.isnan(session.probe_ballisticity)) == 0

        # ===================================
        # Knot-path-curvature as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000638
        # ===================================

        dx = np.diff(session.bt_pos_xs)
        dy = np.diff(session.bt_pos_ys)

        if SHOW_CURVATURE_VIDEO:
            cmap = plt.cm.get_cmap('coolwarm')
            fig = plt.figure()
            plt.ion()

        session.bt_curvature = np.empty((dx.size+1))
        session.bt_curvature_i1 = np.empty((dx.size+1))
        session.bt_curvature_i2 = np.empty((dx.size+1))
        session.bt_curvature_dxf = np.empty((dx.size+1))
        session.bt_curvature_dyf = np.empty((dx.size+1))
        session.bt_curvature_dxb = np.empty((dx.size+1))
        session.bt_curvature_dyb = np.empty((dx.size+1))
        for pi in range(dx.size+1):
            x0 = session.bt_pos_xs[pi]
            y0 = session.bt_pos_ys[pi]
            ii = pi
            dxf = 0.0
            dyf = 0.0
            while ii < dx.size:
                dxf += dx[ii]
                dyf += dy[ii]
                magf = dxf * dxf + dyf * dyf
                if magf >= KNOT_H_POS * KNOT_H_POS:
                    break
                ii += 1
            if ii == dx.size:
                session.bt_curvature[pi] = np.nan
                session.bt_curvature_i1[pi] = np.nan
                session.bt_curvature_i2[pi] = np.nan
                session.bt_curvature_dxf[pi] = np.nan
                session.bt_curvature_dyf[pi] = np.nan
                session.bt_curvature_dxb[pi] = np.nan
                session.bt_curvature_dyb[pi] = np.nan
                continue
            i2 = ii

            ii = pi - 1
            dxb = 0.0
            dyb = 0.0
            while ii >= 0:
                dxb += dx[ii]
                dyb += dy[ii]
                magb = dxb * dxb + dyb * dyb
                if magb >= KNOT_H_POS * KNOT_H_POS:
                    break
                ii -= 1
            if ii == -1:
                session.bt_curvature[pi] = np.nan
                session.bt_curvature_i1[pi] = np.nan
                session.bt_curvature_i2[pi] = np.nan
                session.bt_curvature_dxf[pi] = np.nan
                session.bt_curvature_dyf[pi] = np.nan
                session.bt_curvature_dxb[pi] = np.nan
                session.bt_curvature_dyb[pi] = np.nan
                continue
            i1 = ii

            uxf = dxf / np.sqrt(magf)
            uyf = dyf / np.sqrt(magf)
            uxb = dxb / np.sqrt(magb)
            uyb = dyb / np.sqrt(magb)
            dotprod = uxf * uxb + uyf * uyb
            session.bt_curvature[pi] = np.arccos(dotprod)

            session.bt_curvature_i1[pi] = i1
            session.bt_curvature_i2[pi] = i2
            session.bt_curvature_dxf[pi] = dxf
            session.bt_curvature_dyf[pi] = dyf
            session.bt_curvature_dxb[pi] = dxb
            session.bt_curvature_dyb[pi] = dyb

            if SHOW_CURVATURE_VIDEO:
                plt.clf()
                plt.xlim(0, 1200)
                plt.ylim(0, 1000)
                plt.plot(session.bt_pos_xs[i1:i2], session.bt_pos_ys[i1:i2])
                c = np.array(
                    cmap(session.bt_curvature[pi] / 3.15)).reshape(1, -1)
                plt.scatter(session.bt_pos_xs[pi], session.bt_pos_ys[pi], c=c)
                plt.show()
                plt.pause(0.01)

        if session.probe_performed:
            dx = np.diff(session.probe_pos_xs)
            dy = np.diff(session.probe_pos_ys)

            session.probe_curvature = np.empty((dx.size+1))
            session.probe_curvature_i1 = np.empty((dx.size+1))
            session.probe_curvature_i2 = np.empty((dx.size+1))
            session.probe_curvature_dxf = np.empty((dx.size+1))
            session.probe_curvature_dyf = np.empty((dx.size+1))
            session.probe_curvature_dxb = np.empty((dx.size+1))
            session.probe_curvature_dyb = np.empty((dx.size+1))
            for pi in range(dx.size+1):
                x0 = session.probe_pos_xs[pi]
                y0 = session.probe_pos_ys[pi]
                ii = pi
                dxf = 0.0
                dyf = 0.0
                while ii < dx.size:
                    dxf += dx[ii]
                    dyf += dy[ii]
                    magf = np.sqrt(dxf * dxf + dyf * dyf)
                    if magf >= KNOT_H_POS:
                        break
                    ii += 1
                if ii == dx.size:
                    session.probe_curvature[pi] = np.nan
                    session.probe_curvature_i1[pi] = np.nan
                    session.probe_curvature_i2[pi] = np.nan
                    session.probe_curvature_dxf[pi] = np.nan
                    session.probe_curvature_dyf[pi] = np.nan
                    session.probe_curvature_dxb[pi] = np.nan
                    session.probe_curvature_dyb[pi] = np.nan
                    continue
                i2 = ii

                ii = pi - 1
                dxb = 0.0
                dyb = 0.0
                while ii >= 0:
                    dxb += dx[ii]
                    dyb += dy[ii]
                    magb = np.sqrt(dxb * dxb + dyb * dyb)
                    if magb >= KNOT_H_POS:
                        break
                    ii -= 1
                if ii == -1:
                    session.probe_curvature[pi] = np.nan
                    session.probe_curvature_i1[pi] = np.nan
                    session.probe_curvature_i2[pi] = np.nan
                    session.probe_curvature_dxf[pi] = np.nan
                    session.probe_curvature_dyf[pi] = np.nan
                    session.probe_curvature_dxb[pi] = np.nan
                    session.probe_curvature_dyb[pi] = np.nan
                    continue
                i1 = ii

                uxf = dxf / magf
                uyf = dyf / magf
                uxb = dxb / magb
                uyb = dyb / magb
                dotprod = uxf * uxb + uyf * uyb
                session.probe_curvature[pi] = np.arccos(dotprod)

                session.probe_curvature_i1[pi] = i1
                session.probe_curvature_i2[pi] = i2
                session.probe_curvature_dxf[pi] = dxf
                session.probe_curvature_dyf[pi] = dyf
                session.probe_curvature_dxb[pi] = dxb
                session.probe_curvature_dyb[pi] = dyb

        session.bt_well_curvatures = []
        session.bt_well_avg_curvature_over_time = []
        session.bt_well_avg_curvature_over_visits = []
        for i, wi in enumerate(all_well_names):
            session.bt_well_curvatures.append([])
            for ei, (weni, wexi) in enumerate(zip(session.bt_well_entry_idxs[i], session.bt_well_exit_idxs[i])):
                if wexi > session.bt_curvature.size:
                    continue
                if weni == wexi:
                    continue
                session.bt_well_curvatures[i].append(
                    session.bt_curvature[weni:wexi])

            if len(session.bt_well_curvatures[i]) > 0:
                session.bt_well_avg_curvature_over_time.append(
                    np.mean(np.concatenate(session.bt_well_curvatures[i])))
                session.bt_well_avg_curvature_over_visits.append(
                    np.mean([np.mean(x) for x in session.bt_well_curvatures[i]]))
            else:
                session.bt_well_avg_curvature_over_time.append(np.nan)
                session.bt_well_avg_curvature_over_visits.append(np.nan)

        if session.probe_performed:
            session.probe_well_curvatures = []
            session.probe_well_avg_curvature_over_time = []
            session.probe_well_avg_curvature_over_visits = []
            session.probe_well_curvatures_1min = []
            session.probe_well_avg_curvature_over_time_1min = []
            session.probe_well_avg_curvature_over_visits_1min = []
            session.probe_well_curvatures_30sec = []
            session.probe_well_avg_curvature_over_time_30sec = []
            session.probe_well_avg_curvature_over_visits_30sec = []
            for i, wi in enumerate(all_well_names):
                session.probe_well_curvatures.append([])
                session.probe_well_curvatures_1min.append([])
                session.probe_well_curvatures_30sec.append([])
                for ei, (weni, wexi) in enumerate(zip(session.probe_well_entry_idxs[i], session.probe_well_exit_idxs[i])):
                    if wexi > session.probe_curvature.size:
                        continue
                    if weni == wexi:
                        continue
                    session.probe_well_curvatures[i].append(
                        session.probe_curvature[weni:wexi])
                    if session.probe_pos_ts[weni] <= session.probe_pos_ts[0] + 60*TRODES_SAMPLING_RATE:
                        session.probe_well_curvatures_1min[i].append(
                            session.probe_curvature[weni:wexi])
                    if session.probe_pos_ts[weni] <= session.probe_pos_ts[0] + 30*TRODES_SAMPLING_RATE:
                        session.probe_well_curvatures_30sec[i].append(
                            session.probe_curvature[weni:wexi])

                if len(session.probe_well_curvatures[i]) > 0:
                    session.probe_well_avg_curvature_over_time.append(
                        np.mean(np.concatenate(session.probe_well_curvatures[i])))
                    session.probe_well_avg_curvature_over_visits.append(
                        np.mean([np.mean(x) for x in session.probe_well_curvatures[i]]))
                else:
                    session.probe_well_avg_curvature_over_time.append(np.nan)
                    session.probe_well_avg_curvature_over_visits.append(np.nan)

                if len(session.probe_well_curvatures_1min[i]) > 0:
                    session.probe_well_avg_curvature_over_time_1min.append(
                        np.mean(np.concatenate(session.probe_well_curvatures_1min[i])))
                    session.probe_well_avg_curvature_over_visits_1min.append(
                        np.mean([np.mean(x) for x in session.probe_well_curvatures_1min[i]]))
                else:
                    session.probe_well_avg_curvature_over_time_1min.append(np.nan)
                    session.probe_well_avg_curvature_over_visits_1min.append(
                        np.nan)

                if len(session.probe_well_curvatures_30sec[i]) > 0:
                    session.probe_well_avg_curvature_over_time_30sec.append(
                        np.mean(np.concatenate(session.probe_well_curvatures_30sec[i])))
                    session.probe_well_avg_curvature_over_visits_30sec.append(
                        np.mean([np.mean(x) for x in session.probe_well_curvatures_30sec[i]]))
                else:
                    session.probe_well_avg_curvature_over_time_30sec.append(np.nan)
                    session.probe_well_avg_curvature_over_visits_30sec.append(
                        np.nan)

        # ===================================
        # Latency to well in probe
        # ===================================

        if session.probe_performed:
            session.probe_latency_to_well = []
            for i, wi in enumerate(all_well_names):
                if len(session.probe_well_entry_times[i]) == 0:
                    session.probe_latency_to_well.append(np.nan)
                else:
                    session.probe_latency_to_well.append(
                        session.probe_well_entry_times[0])

        # ===================================
        # exploration bouts
        # ===================================

        POS_FRAME_RATE = stats.mode(
            np.diff(session.bt_pos_ts))[0] / float(TRODES_SAMPLING_RATE)
        BOUT_VEL_SM_SIGMA = BOUT_VEL_SM_SIGMA_SECS / POS_FRAME_RATE
        MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS = 1.0
        MIN_PAUSE_TIME_FRAMES = int(
            MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS / POS_FRAME_RATE)
        MIN_EXPLORE_TIME_FRAMES = int(MIN_EXPLORE_TIME_SECS / POS_FRAME_RATE)

        bt_sm_vel = scipy.ndimage.gaussian_filter1d(
            session.bt_vel_cm_s, BOUT_VEL_SM_SIGMA)
        session.bt_sm_vel = bt_sm_vel

        bt_is_explore_local = bt_sm_vel > PAUSE_MAX_SPEED_CM_S
        dil_filt = np.ones((MIN_PAUSE_TIME_FRAMES), dtype=int)
        in_pause_bout = np.logical_not(signal.convolve(
            bt_is_explore_local.astype(int), dil_filt, mode='same').astype(bool))
        # now just undo dilation to get flags
        session.bt_is_in_pause = signal.convolve(
            in_pause_bout.astype(int), dil_filt, mode='same').astype(bool)
        session.bt_is_in_explore = np.logical_not(session.bt_is_in_pause)

        # explicitly adjust flags at reward consumption times
        for wft, wlt in zip(session.home_well_find_times, session.home_well_leave_times):
            pidx1 = np.searchsorted(session.bt_pos_ts[0:-1], wft)
            pidx2 = np.searchsorted(session.bt_pos_ts[0:-1], wlt)
            session.bt_is_in_pause[pidx1:pidx2] = True
            session.bt_is_in_explore[pidx1:pidx2] = False

        for wft, wlt in zip(session.away_well_find_times, session.away_well_leave_times):
            pidx1 = np.searchsorted(session.bt_pos_ts[0:-1], wft)
            pidx2 = np.searchsorted(session.bt_pos_ts[0:-1], wlt)
            session.bt_is_in_pause[pidx1:pidx2] = True
            session.bt_is_in_explore[pidx1:pidx2] = False

        assert np.sum(np.isnan(session.bt_is_in_pause)) == 0
        assert np.sum(np.isnan(session.bt_is_in_explore)) == 0
        assert np.all(np.logical_or(
            session.bt_is_in_pause, session.bt_is_in_explore))
        assert not np.any(np.logical_and(
            session.bt_is_in_pause, session.bt_is_in_explore))

        start_explores = np.where(
            np.diff(session.bt_is_in_explore.astype(int)) == 1)[0] + 1
        if session.bt_is_in_explore[0]:
            start_explores = np.insert(start_explores, 0, 0)

        stop_explores = np.where(
            np.diff(session.bt_is_in_explore.astype(int)) == -1)[0] + 1
        if session.bt_is_in_explore[-1]:
            stop_explores = np.append(
                stop_explores, len(session.bt_is_in_explore))

        bout_len_frames = stop_explores - start_explores

        long_enough = bout_len_frames >= MIN_EXPLORE_TIME_FRAMES
        bout_num_wells_visited = np.zeros((len(start_explores)))
        for i, (bst, ben) in enumerate(zip(start_explores, stop_explores)):
            bout_num_wells_visited[i] = len(
                getListOfVisitedWells(session.bt_nearest_wells[bst:ben], True))
            # bout_num_wells_visited[i] = len(set(session.bt_nearest_wells[bst:ben]))
        enough_wells = bout_num_wells_visited >= MIN_EXPLORE_NUM_WELLS

        keep_bout = np.logical_and(long_enough, enough_wells)
        session.bt_explore_bout_starts = start_explores[keep_bout]
        session.bt_explore_bout_ends = stop_explores[keep_bout]
        session.bt_explore_bout_lens = session.bt_explore_bout_ends - \
            session.bt_explore_bout_starts
        ts = np.array(session.bt_pos_ts)
        session.bt_explore_bout_lens_secs = (ts[session.bt_explore_bout_ends] -
                                             ts[session.bt_explore_bout_starts]) / TRODES_SAMPLING_RATE

        # add a category at each behavior time point for easy reference later:
        session.bt_bout_category = np.zeros_like(session.bt_pos_xs)
        last_stop = 0
        for bst, ben in zip(session.bt_explore_bout_starts, session.bt_explore_bout_ends):
            session.bt_bout_category[last_stop:bst] = 1
            last_stop = ben
        session.bt_bout_category[last_stop:] = 1
        for wft, wlt in zip(session.home_well_find_times, session.home_well_leave_times):
            pidx1 = np.searchsorted(session.bt_pos_ts, wft)
            pidx2 = np.searchsorted(session.bt_pos_ts, wlt)
            session.bt_bout_category[pidx1:pidx2] = 2
        for wft, wlt in zip(session.away_well_find_times, session.away_well_leave_times):
            pidx1 = np.searchsorted(session.bt_pos_ts, wft)
            pidx2 = np.searchsorted(session.bt_pos_ts, wlt)
            session.bt_bout_category[pidx1:pidx2] = 2

        if session.probe_performed:
            probe_sm_vel = scipy.ndimage.gaussian_filter1d(
                session.probe_vel_cm_s, BOUT_VEL_SM_SIGMA)
            session.probe_sm_vel = probe_sm_vel

            probe_is_explore_local = probe_sm_vel > PAUSE_MAX_SPEED_CM_S
            dil_filt = np.ones((MIN_PAUSE_TIME_FRAMES), dtype=int)
            in_pause_bout = np.logical_not(signal.convolve(
                probe_is_explore_local.astype(int), dil_filt, mode='same').astype(bool))
            # now just undo dilation to get flags
            session.probe_is_in_pause = signal.convolve(
                in_pause_bout.astype(int), dil_filt, mode='same').astype(bool)
            session.probe_is_in_explore = np.logical_not(session.probe_is_in_pause)

            assert np.sum(np.isnan(session.probe_is_in_pause)) == 0
            assert np.sum(np.isnan(session.probe_is_in_explore)) == 0
            assert np.all(np.logical_or(session.probe_is_in_pause,
                                        session.probe_is_in_explore))
            assert not np.any(np.logical_and(
                session.probe_is_in_pause, session.probe_is_in_explore))

            start_explores = np.where(
                np.diff(session.probe_is_in_explore.astype(int)) == 1)[0] + 1
            if session.probe_is_in_explore[0]:
                start_explores = np.insert(start_explores, 0, 0)

            stop_explores = np.where(
                np.diff(session.probe_is_in_explore.astype(int)) == -1)[0] + 1
            if session.probe_is_in_explore[-1]:
                stop_explores = np.append(
                    stop_explores, len(session.probe_is_in_explore))

            bout_len_frames = stop_explores - start_explores

            long_enough = bout_len_frames >= MIN_EXPLORE_TIME_FRAMES
            bout_num_wells_visited = np.zeros((len(start_explores)))
            for i, (bst, ben) in enumerate(zip(start_explores, stop_explores)):
                bout_num_wells_visited[i] = len(
                    getListOfVisitedWells(session.probe_nearest_wells[bst:ben], True))
            enough_wells = bout_num_wells_visited >= MIN_EXPLORE_NUM_WELLS

            keep_bout = np.logical_and(long_enough, enough_wells)
            session.probe_explore_bout_starts = start_explores[keep_bout]
            session.probe_explore_bout_ends = stop_explores[keep_bout]
            session.probe_explore_bout_lens = session.probe_explore_bout_ends - session.probe_explore_bout_starts
            ts = np.array(session.probe_pos_ts)
            session.probe_explore_bout_lens_secs = (ts[session.probe_explore_bout_ends] -
                                                    ts[session.probe_explore_bout_starts]) / TRODES_SAMPLING_RATE

            # add a category at each behavior time point for easy reference later:
            session.probe_bout_category = np.zeros_like(session.probe_pos_xs)
            last_stop = 0
            for bst, ben in zip(session.probe_explore_bout_starts, session.probe_explore_bout_ends):
                session.probe_bout_category[last_stop:bst] = 1
                last_stop = ben
            session.probe_bout_category[last_stop:] = 1
            for wft, wlt in zip(session.home_well_find_times, session.home_well_leave_times):
                pidx1 = np.searchsorted(session.probe_pos_ts, wft)
                pidx2 = np.searchsorted(session.probe_pos_ts, wlt)
                session.probe_bout_category[pidx1:pidx2] = 2
            for wft, wlt in zip(session.away_well_find_times, session.away_well_leave_times):
                pidx1 = np.searchsorted(session.probe_pos_ts, wft)
                pidx2 = np.searchsorted(session.probe_pos_ts, wlt)
                session.probe_bout_category[pidx1:pidx2] = 2

        # And a similar thing, but value == 0 for rest/reward, or i when rat is in ith bout (starting at 1)
        session.bt_bout_label = np.zeros_like(session.bt_pos_xs)
        for bi, (bst, ben) in enumerate(zip(session.bt_explore_bout_starts, session.bt_explore_bout_ends)):
            session.bt_bout_label[bst:ben] = bi + 1

        if session.probe_performed:
            session.probe_bout_label = np.zeros_like(session.probe_pos_xs)
            for bi, (bst, ben) in enumerate(zip(session.probe_explore_bout_starts, session.probe_explore_bout_ends)):
                session.probe_bout_label[bst:ben] = bi + 1

        # ===================================
        # excursions (when the rat left the wall-wells and searched the middle)
        # ===================================
        session.bt_excursion_category = np.array([BTSession.EXCURSION_STATE_ON_WALL if onWall(
            w) else BTSession.EXCURSION_STATE_OFF_WALL for w in session.bt_nearest_wells])
        session.bt_excursion_starts = np.where(np.diff(
            (session.bt_excursion_category == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == 1)[0] + 1
        if session.bt_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL:
            session.bt_excursion_starts = np.insert(session.bt_excursion_starts, 0, 0)
        session.bt_excursion_ends = np.where(np.diff(
            (session.bt_excursion_category == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == -1)[0] + 1
        if session.bt_excursion_category[-1] == BTSession.EXCURSION_STATE_OFF_WALL:
            session.bt_excursion_ends = np.append(
                session.bt_excursion_ends, len(session.bt_excursion_category))
        ts = np.array(session.bt_pos_ts + [session.bt_pos_ts[-1]])
        session.bt_excursion_lens_secs = (
            ts[session.bt_excursion_ends] - ts[session.bt_excursion_starts]) / TRODES_SAMPLING_RATE

        if session.probe_performed:
            session.probe_excursion_category = np.array([BTSession.EXCURSION_STATE_ON_WALL if onWall(
                w) else BTSession.EXCURSION_STATE_OFF_WALL for w in session.probe_nearest_wells])
            session.probe_excursion_starts = np.where(np.diff(
                (session.probe_excursion_category == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == 1)[0] + 1
            if session.probe_excursion_category[0] == BTSession.EXCURSION_STATE_OFF_WALL:
                session.probe_excursion_starts = np.insert(session.probe_excursion_starts, 0, 0)
            session.probe_excursion_ends = np.where(np.diff(
                (session.probe_excursion_category == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == -1)[0] + 1
            if session.probe_excursion_category[-1] == BTSession.EXCURSION_STATE_OFF_WALL:
                session.probe_excursion_ends = np.append(
                    session.probe_excursion_ends, len(session.probe_excursion_category))
            ts = np.array(session.probe_pos_ts + [session.probe_pos_ts[-1]])
            session.probe_excursion_lens_secs = (
                ts[session.probe_excursion_ends] - ts[session.probe_excursion_starts]) / TRODES_SAMPLING_RATE

    # ======================================================================
    # TODO
    # Some perseveration measure during away trials to see if there's an effect during the actual task
    #
    # During task effect on home/away latencies?
    #
    # average latencies for H1, A1, H2, ...
    #
    # difference in effect magnitude by distance from starting location to home well?
    #
    # Where do ripples happen? Would inform whether to split by away vs home, etc in future experiments
    #   Only during rest? Can set velocity threshold in future interruptions?
    #
    # Based on speed or exploration, is there a more principled way to choose period of probe that is measured? B/c could vary by rat
    #
    # avg speed by condition ... could explain higher visits per bout everywhere on SWR trials
    #
    # latency to home well in probe, directness of path, etc maybe?
    #   probably will be nothing, but will probably also be asked for
    #
    # Any differences b/w early vs later experiments?
    #
    # ======================================================================

    # ===================================
    # Now save this session
    # ===================================
    dataob.allSessions.append(session)
    print("done with", session.name)

    # ===================================
    # extra plotting stuff
    # ===================================
    if session.date_str in INSPECT_IN_DETAIL or INSPECT_ALL:
        if INSPECT_BOUTS and len(session.away_well_find_times) > 0:
            # print("{} probe bouts, {} bt bouts".format(
            # session.probe_num_bouts, session.bt_num_bouts))
            x = session.bt_pos_ts[0:-1]
            y = bt_sm_vel
            x1 = np.array(x, copy=True)
            y1 = np.copy(y)
            x2 = np.copy(x1)
            y2 = np.copy(y1)
            x1[session.bt_is_in_explore] = np.nan
            y1[session.bt_is_in_explore] = np.nan
            x2[session.bt_is_in_pause] = np.nan
            y2[session.bt_is_in_pause] = np.nan
            plt.clf()
            plt.plot(x1, y1)
            plt.plot(x2, y2)

            for wft, wlt in zip(session.home_well_find_times, session.home_well_leave_times):
                pidx1 = np.searchsorted(x, wft)
                pidx2 = np.searchsorted(x, wlt)
                plt.scatter(x[pidx1], y[pidx1], c='green')
                plt.scatter(x[pidx2], y[pidx2], c='green')

            print(len(session.away_well_find_times))
            for wft, wlt in zip(session.away_well_find_times, session.away_well_leave_times):
                pidx1 = np.searchsorted(x, wft)
                pidx2 = np.searchsorted(x, wlt)
                plt.scatter(x[pidx1], y[pidx1], c='green')
                plt.scatter(x[pidx2], y[pidx2], c='green')

            mny = np.min(bt_sm_vel)
            mxy = np.max(bt_sm_vel)
            for bst, ben in zip(session.bt_explore_bout_starts, session.bt_explore_bout_ends):
                plt.plot([x[bst], x[bst]], [mny, mxy], 'b')
                plt.plot([x[ben-1], x[ben-1]], [mny, mxy], 'r')
            if SAVE_DONT_SHOW:
                plt.savefig(os.path.join(animalInfo.fig_output_dir, "bouts",
                                         session.name + "_bouts_over_time"), dpi=800)
            else:
                plt.show()

            for i, (bst, ben) in enumerate(zip(session.bt_explore_bout_starts, session.bt_explore_bout_ends)):
                plt.clf()
                plt.plot(session.bt_pos_xs, session.bt_pos_ys)
                plt.plot(
                    session.bt_pos_xs[bst:ben], session.bt_pos_ys[bst:ben])

                wells_visited = getListOfVisitedWells(
                    session.bt_nearest_wells[bst:ben], True)
                for w in wells_visited:
                    wx, wy = getWellCoordinates(
                        w, session.well_coords_map)
                    plt.scatter(wx, wy, c='red')

                if SAVE_DONT_SHOW:
                    plt.savefig(os.path.join(animalInfo.fig_output_dir, "bouts",
                                             session.name + "_bouts_" + str(i)), dpi=800)
                else:
                    plt.show()

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

        if INSPECT_PLOT_WELL_OCCUPANCIES:
            cmap = plt.get_cmap('Set3')
            make_colors = True
            while make_colors:
                well_colors = np.zeros((48, 4))
                for i in all_well_names:
                    well_colors[i-1, :] = cmap(random.uniform(0, 1))

                make_colors = False

                if ENFORCE_DIFFERENT_WELL_COLORS:
                    for i in all_well_names:
                        neighs = [i-8, i-1, i+8, i+1]
                        for n in neighs:
                            if n in all_well_names and np.all(well_colors[i-1, :] == well_colors[n-1, :]):
                                make_colors = True
                                print("gotta remake the colors!")
                                break

                        if make_colors:
                            break

            # print(session.bt_well_entry_idxs)
            # print(session.bt_well_exit_idxs)

            plt.clf()
            for i, wi in enumerate(all_well_names):
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
print("Saving to file: {}".format(animalInfo.out_filename))
dataob.saveToFile(os.path.join(animalInfo.output_dir, animalInfo.out_filename))
print("Saved sessions:")
for sesh in dataob.allSessions:
    print(sesh.name)
