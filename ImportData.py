from BTData import BTData
from BTSession import BTSession
import numpy as np
import os
import csv
import glob
import scipy
from scipy import stats, signal
import MountainViewIO
from datetime import datetime
import sys
from PyQt5.QtWidgets import QApplication

from consts import all_well_names, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE
from UtilFunctions import readWellCoordsFile, readRawPositionData, readClipData, \
    processPosData, getWellCoordinates, getNearestWell, getWellEntryAndExitTimes, \
    quadrantOfWell, getListOfVisitedWells, onWall, getRipplePower, detectRipples, \
    getInfoForAnimal, generateFoundWells, getUSBVideoFile, processPosData_coords, \
    parseCmdLineAnimalNames
from ClipsMaker import AnnotatorWindow
from TrodesCameraExtrator import getTrodesLightTimes, processRawTrodesVideo, processUSBVideoData

importParentApp = QApplication(sys.argv)


# Returns a list of directories that correspond to runs for analysis. Unless runJustSpecified is True,
# only ever filters by day. Thus index within day of returned list can be used to find corresponding behavior notes file
def getSessionDirs(animalInfo, importOptions):
    filtered_data_dirs = []
    prevSessionDirs = []
    prevSession = None
    all_data_dirs = sorted(os.listdir(animalInfo.data_dir), key=lambda s: (
        s.split('_')[0], s.split('_')[1]))

    for session_idx, session_dir in enumerate(all_data_dirs):
        if session_dir == "behavior_notes" or "numpy_objs" in session_dir:
            continue

        if not os.path.isdir(os.path.join(animalInfo.data_dir, session_dir)):
            continue

        dir_split = session_dir.split('_')
        if dir_split[-1] == "Probe" or dir_split[-1] == "ITI":
            # will deal with this in another loop
            print(f"skipping {session_dir}, is probe or iti")
            continue

        date_str = dir_split[0][-8:]

        if importOptions["runJustSpecified"] and date_str not in importOptions["specifiedDays"] and \
                session_dir not in importOptions["specifiedRuns"]:
            prevSession = session_dir
            continue

        if date_str in animalInfo.excluded_dates:
            print(f"skipping, excluded date {date_str}")
            prevSession = session_dir
            continue

        if animalInfo.minimum_date is not None and date_str < animalInfo.minimum_date:
            print("skipping date {}, is before minimum date {}".format(
                date_str, animalInfo.minimum_date))
            prevSession = session_dir
            continue

        filtered_data_dirs.append(session_dir)
        prevSessionDirs.append(prevSession)
        prevSession = session_dir

    return filtered_data_dirs, prevSessionDirs


def makeSessionObj(seshDir, prevSeshDir, sessionNumber, prevSessionNumber, animalInfo):
    sesh = BTSession()
    sesh.prevSessionDir = prevSeshDir

    dir_split = seshDir.split('_')
    date_str = dir_split[0][-8:]
    time_str = dir_split[1]
    session_name_pfx = dir_split[0][0:-8]
    sesh.date_str = date_str
    sesh.name = seshDir
    sesh.time_str = time_str
    s = "{}_{}".format(date_str, time_str)
    sesh.date = datetime.strptime(s, "%Y%m%d_%H%M%S")

    # check for files that belong to this date
    sesh.bt_dir = seshDir
    gl = animalInfo.data_dir + session_name_pfx + date_str + "*"
    dir_list = glob.glob(gl)
    for d in dir_list:
        if d.split('_')[-1] == "ITI":
            sesh.iti_dir = d
            sesh.separate_iti_file = True
            sesh.recorded_iti = True
        elif d.split('_')[-1] == "Probe":
            sesh.probe_dir = d
            sesh.separate_probe_file = True
    if not sesh.separate_probe_file:
        sesh.recorded_iti = True
        sesh.probe_dir = os.path.join(animalInfo.data_dir, seshDir)
    if not sesh.separate_iti_file and sesh.recorded_iti:
        sesh.iti_dir = seshDir

    # Get behavior_notes info file name
    behaviorNotesDir = os.path.join(animalInfo.data_dir, 'behavior_notes')
    sesh.infoFileName = os.path.join(behaviorNotesDir, date_str + ".txt")
    if not os.path.exists(sesh.infoFileName):
        # Switched numbering scheme once started doing multiple sessions a day
        sesh.infoFileName = os.path.join(
            behaviorNotesDir, f"{date_str}_{sessionNumber}.txt")
    sesh.seshIdx = sessionNumber

    if prevSeshDir is None:
        sesh.prevInfoFileName = None
        sesh.prevSeshIdx = None
    else:
        dir_split = prevSeshDir.split('_')
        prevSeshDateStr = dir_split[0][-8:]
        sesh.prevInfoFileName = os.path.join(
            behaviorNotesDir, prevSeshDateStr + ".txt")
        if not os.path.exists(sesh.prevInfoFileName):
            # Switched numbering scheme once started doing multiple sessions a day
            sesh.prevInfoFileName = os.path.join(
                behaviorNotesDir, f"{prevSeshDateStr}_{prevSessionNumber}.txt")
        sesh.prevSeshIdx = prevSessionNumber

    sesh.fileStartString = os.path.join(animalInfo.data_dir, seshDir, seshDir)
    sesh.animalInfo = animalInfo

    return sesh


def parseInfoFiles(sesh):
    with open(sesh.infoFileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineparts = line.split(":")
            if len(lineparts) != 2:
                sesh.notes.append(line)
                continue

            fieldName = lineparts[0]
            fieldVal = lineparts[1]

            # if sesh.importOptions["justExtractData"]  and fieldName.lower() not in ["reference", "ref", "baseline", "home", \
            # "aways", "last away", "last well", "probe performed"]:
            # continue

            if fieldName.lower() == "home":
                sesh.home_well = int(fieldVal)
            elif fieldName.lower() == "aways":
                sesh.away_wells = [int(w) for w in fieldVal.strip().split(' ')]
                # print(line)
                # print(fieldName, fieldVal)
                # print(sesh.away_wells)
            elif fieldName.lower() == "foundwells":
                sesh.foundWells = [int(w) for w in fieldVal.strip().split(' ')]
            elif fieldName.lower() == "condition":
                type_in = fieldVal.lower()
                if 'ripple' in type_in or 'interruption' in type_in:
                    sesh.isRippleInterruption = True
                elif 'none' in type_in:
                    sesh.isNoInterruption = True
                elif 'delay' in type_in:
                    sesh.isDelayedInterruption = True
                else:
                    print("Couldn't recognize Condition {} in file {}".format(
                        type_in, sesh.infoFileName))
            elif fieldName.lower() == "thresh":
                if "low" in fieldVal.lower():
                    sesh.ripple_detection_threshold = 2.5
                elif "high" in fieldVal.lower():
                    sesh.ripple_detection_threshold = 4
                elif "med" in fieldVal.lower():
                    sesh.ripple_detection_threshold = 3
                else:
                    sesh.ripple_detection_threshold = float(
                        fieldVal)
            elif fieldName.lower() == "last away":
                if fieldVal.strip() == "None":
                    sesh.last_away_well = None
                else:
                    sesh.last_away_well = float(fieldVal)
                # print(fieldVal)
            elif fieldName.lower() == "reference" or fieldName.lower() == "ref":
                sesh.ripple_detection_tetrodes = [int(fieldVal)]
                # print(fieldVal)
            elif fieldName.lower() == "baseline":
                sesh.ripple_baseline_tetrode = int(fieldVal)
            elif fieldName.lower() == "last well":
                if fieldVal.strip() == "None":
                    sesh.found_first_home = False
                    sesh.ended_on_home = False
                else:
                    sesh.found_first_home = True
                    if 'h' in fieldVal.lower():
                        sesh.ended_on_home = True
                    elif 'a' in fieldVal.lower():
                        sesh.ended_on_home = False
                    else:
                        print("Couldn't recognize last well {} in file {}".format(
                            fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "iti stim on":
                if 'Y' in fieldVal:
                    sesh.ITI_stim_on = True
                elif 'N' in fieldVal:
                    sesh.ITI_stim_on = False
                else:
                    print("Couldn't recognize ITI Stim condition {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "probe stim on":
                if 'Y' in fieldVal:
                    sesh.probe_stim_on = True
                elif 'N' in fieldVal:
                    sesh.probe_stim_on = False
                else:
                    print("Couldn't recognize Probe Stim condition {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "probe performed":
                if 'Y' in fieldVal:
                    sesh.probe_performed = True
                elif 'N' in fieldVal:
                    sesh.probe_performed = False
                    if sesh.animalName == "Martin":
                        raise Exception(
                            "I thought all martin runs had a probe")
                else:
                    print("Couldn't recognize Probe performed val {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "task ended at":
                sesh.bt_ended_at_well = int(fieldVal)
            elif fieldName.lower() == "probe ended at":
                sesh.probe_ended_at_well = int(fieldVal)
            elif fieldName.lower() == "weight":
                sesh.weight = float(fieldVal)
            elif fieldName.lower() == "conditiongroup":
                sesh.conditionGroup = int(fieldVal)
            elif fieldName.lower() == "probe home fill time":
                sesh.probe_fill_time = int(fieldVal)
            else:
                sesh.notes.append(line)

    if sesh.probe_performed is None and sesh.animalName == "B12":
        sesh.probe_performed = True

    if sesh.probe_performed and sesh.probe_ended_at_well is None:
        sesh.missingProbeEndedAtWell = True
        # raise Exception(
        #     "Didn't mark the probe end well for sesh {}".format(sesh.name))
    else:
        sesh.missingProbeEndedAtWell = False

    if not (sesh.last_away_well == sesh.away_wells[-1] and sesh.ended_on_home) and sesh.bt_ended_at_well is None:
        # raise Exception(
        #     "Didn't find all the wells but task end well not marked for sesh {}".format(sesh.name))
        sesh.missingEndedAtWell = True
    else:
        sesh.missingEndedAtWell = False

    if sesh.prevInfoFileName is not None:
        with open(sesh.prevInfoFileName, 'r') as f:
            sesh.prevSessionInfoParsed = True
            lines = f.readlines()
            for line in lines:
                lineparts = line.split(":")
                if len(lineparts) != 2:
                    continue

                field_name = lineparts[0]
                field_val = lineparts[1]

                if field_name.lower() == "home":
                    sesh.prevSessionHome = int(field_val)
                elif field_name.lower() == "aways":
                    sesh.prevSessionAways = [int(w)
                                             for w in field_val.strip().split(' ')]
                elif field_name.lower() == "condition":
                    type_in = field_val.lower()
                    if 'ripple' in type_in or 'interruption' in type_in:
                        sesh.prevSessionIsRippleInterruption = True
                    elif 'none' in type_in:
                        sesh.prevSessionIsNoInterruption = True
                    elif 'delay' in type_in:
                        sesh.prevSessionIsDelayedInterruption = True
                    else:
                        print("Couldn't recognize Condition {} in file {}".format(
                            type_in, sesh.prevInfoFileName))
                elif field_name.lower() == "thresh":
                    if "low" in field_val.lower():
                        sesh.prevSession_ripple_detection_threshold = 2.5
                    elif "high" in field_val.lower():
                        sesh.prevSession_ripple_detection_threshold = 4
                    elif "med" in field_val.lower():
                        sesh.prevSession_ripple_detection_threshold = 3
                    else:
                        sesh.prevSession_ripple_detection_threshold = float(
                            field_val)
                elif field_name.lower() == "last away":
                    sesh.prevSession_last_away_well = float(field_val)
                    # print(field_val)
                elif field_name.lower() == "last well":
                    if 'h' in field_val.lower():
                        sesh.prevSession_ended_on_home = True
                    elif 'a' in field_val.lower():
                        sesh.prevSession_ended_on_home = False
                    else:
                        print("Couldn't recognize last well {} in file {}".format(
                            field_val, sesh.prevInfoFileName))
                elif field_name.lower() == "iti stim on":
                    if 'Y' in field_val:
                        sesh.prevSession_ITI_stim_on = True
                    elif 'N' in field_val:
                        sesh.prevSession_ITI_stim_on = False
                    else:
                        print("Couldn't recognize ITI Stim condition {} in file {}".format(
                            field_val, sesh.prevInfoFileName))
                elif field_name.lower() == "probe stim on":
                    if 'Y' in field_val:
                        sesh.prevSession_probe_stim_on = True
                    elif 'N' in field_val:
                        sesh.prevSession_probe_stim_on = False
                    else:
                        print("Couldn't recognize Probe Stim condition {} in file {}".format(
                            field_val, sesh.prevInfoFileName))
                else:
                    pass

    else:
        sesh.prevSessionInfoParsed = False

    if not sesh.importOptions["justExtractData"]:
        assert sesh.home_well != 0

    if sesh.foundWells is None:
        sesh.foundWells = generateFoundWells(
            sesh.home_well, sesh.away_wells, sesh.last_away_well, sesh.ended_on_home, sesh.found_first_home)


def parseDLCData(sesh):
    btPosFileName = sesh.animalInfo.DLC_dir + sesh.name + "_task.npz"
    probePosFileName = sesh.animalInfo.DLC_dir + sesh.name + "_probe.npz"
    if not os.path.exists(btPosFileName) or not os.path.exists(probePosFileName):
        print(f"{sesh.name} has no DLC file!")
        return None

    bt_pos = np.load(btPosFileName)
    sesh.bt_pos_xs, sesh.bt_pos_ys, sesh.bt_pos_ts = \
        processPosData_coords(bt_pos["x1"], bt_pos["y1"],
                              bt_pos["timestamp"], xLim=None, yLim=None, smooth=3)

    probe_pos = np.load(probePosFileName)
    sesh.probe_pos_xs, sesh.probe_pos_ys, sesh.probe_pos_ts = \
        processPosData_coords(probe_pos["x1"], probe_pos["y1"],
                              probe_pos["timestamp"], xLim=None, yLim=None, smooth=3)
    # if True:
    #     plt.plot(sesh.probe_pos_xs, sesh.probe_pos_ys)
    #     plt.show()

    sesh.hasPositionData = True

    sesh.importOptions["PIXELS_PER_CM"] = 1.28

    xs = list(np.hstack((sesh.bt_pos_xs, sesh.probe_pos_xs)))
    ys = list(np.hstack((sesh.bt_pos_ys, sesh.probe_pos_ys)))
    ts = list(np.hstack((sesh.bt_pos_ts, sesh.probe_pos_ts)))

    return xs, ys, ts


def loadPositionData(sesh):
    sesh.hasPositionData = False
    position_data = None

    # First try DeepLabCut
    if sesh.animalInfo.DLC_dir is not None:
        position_data = parseDLCData(sesh)
        if position_data is None:
            assert sesh.positionFromDeepLabCut is None or not sesh.positionFromDeepLabCut
            sesh.positionFromDeepLabCut = False
        else:
            assert sesh.positionFromDeepLabCut is None or sesh.positionFromDeepLabCut
            sesh.positionFromDeepLabCut = True
            xs, ys, ts = position_data
            position_data_metadata = {}
            sesh.hasPositionData = True
    else:
        sesh.positionFromDeepLabCut = False

    if position_data is None:
        assert not sesh.positionFromDeepLabCut
        sesh.importOptions["PIXELS_PER_CM"] = 5.0
        trackingFile = sesh.fileStartString + '.1.videoPositionTracking'
        if os.path.exists(trackingFile):
            position_data_metadata, position_data = readRawPositionData(
                trackingFile)

            if sesh.importOptions["forceAllTrackingAuto"] and ("source" not in position_data_metadata or
                                                               "trodescameraextractor" not in position_data_metadata["source"]):
                # print(position_data_metadata)
                position_data = None
                os.rename(sesh.fileStartString + '.1.videoPositionTracking', sesh.fileStartString +
                          '.1.videoPositionTracking.manualOutput')
            else:
                print("Got position tracking")
                sesh.frameTimes = position_data['timestamp']
                xs, ys, ts = processPosData(position_data, xLim=(
                    sesh.animalInfo.X_START, sesh.animalInfo.X_FINISH), yLim=(sesh.animalInfo.Y_START, sesh.animalInfo.Y_FINISH))
                sesh.hasPositionData = True
        else:
            position_data = None

        if position_data is None:
            print("running trodes camera extractor!")
            processRawTrodesVideo(sesh.fileStartString + '.1.h264')
            position_data_metadata, position_data = readRawPositionData(
                trackingFile)
            sesh.frameTimes = position_data['timestamp']
            xs, ys, ts = processPosData(position_data, xLim=(
                sesh.animalInfo.X_START, sesh.animalInfo.X_FINISH), yLim=(sesh.animalInfo.Y_START, sesh.animalInfo.Y_FINISH))
            sesh.hasPositionData = True

    if sesh.missingEndedAtWell:
        raise Exception(
            f"{sesh.name} has position but no last well marked in behavior notes and didn't end on last home")
    if sesh.missingProbeEndedAtWell:
        raise Exception(
            "Didn't mark the probe end well for session {}".format(sesh.name))

    if sesh.positionFromDeepLabCut:
        well_coords_file_name = sesh.fileStartString + '.1.wellLocations_dlc.csv'
        if not os.path.exists(well_coords_file_name):
            well_coords_file_name = os.path.join(
                sesh.animalInfo.data_dir, 'well_locations_dlc.csv')
            print("Specific well locations not found, falling back to file {}".format(
                well_coords_file_name))
    else:
        well_coords_file_name = sesh.fileStartString + '.1.wellLocations.csv'
        if not os.path.exists(well_coords_file_name):
            well_coords_file_name = os.path.join(
                sesh.animalInfo.data_dir, 'well_locations.csv')
            print("Specific well locations not found, falling back to file {}".format(
                well_coords_file_name))
    sesh.well_coords_map = readWellCoordsFile(well_coords_file_name)
    sesh.home_x, sesh.home_y = getWellCoordinates(
        sesh.home_well, sesh.well_coords_map)

    if not sesh.hasPositionData:
        print(f"WARNING: {sesh.name} has no position data")
        return

    if sesh.importOptions["skipUSB"]:
        print("WARNING: Skipping USB ")
    else:
        if "lightonframe" in position_data_metadata:
            sesh.trodesLightOnFrame = int(
                position_data_metadata['lightonframe'])
            sesh.trodesLightOffFrame = int(
                position_data_metadata['lightoffframe'])
            print("metadata says trodes light frames {}, {} (/{})".format(sesh.trodesLightOffFrame,
                                                                          sesh.trodesLightOnFrame, len(ts)))
            sesh.trodesLightOnTime = sesh.frameTimes[sesh.trodesLightOnFrame]
            sesh.trodesLightOffTime = sesh.frameTimes[sesh.trodesLightOffFrame]
            print("metadata file says trodes light timestamps {}, {} (/{})".format(
                sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))
            # playFrames(sesh.fileStartString + '.1.h264', sesh.trodesLightOffFrame -
            #            20, sesh.trodesLightOffFrame + 20)
            # playFrames(sesh.fileStartString + '.1.h264', sesh.trodesLightOnFrame -
            #            20, sesh.trodesLightOnFrame + 20)
        else:
            # was a separate metadatafile made?
            positionMetadataFile = sesh.fileStartString + '.1.justLights'
            if os.path.exists(positionMetadataFile):
                lightInfo = np.fromfile(
                    positionMetadataFile, sep=",").astype(int)
                sesh.trodesLightOffTime = lightInfo[0]
                sesh.trodesLightOnTime = lightInfo[1]
                print("justlights file says trodes light timestamps {}, {} (/{})".format(
                    sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))
            else:
                print("doing the lights")
                sesh.trodesLightOffTime, sesh.trodesLightOnTime = getTrodesLightTimes(
                    sesh.fileStartString + '.1.h264', showVideo=False)
                print("trodesLightFunc says trodes light Time {}, {} (/{})".format(
                    sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))

        possibleDirectories = [
            "/media/WDC6/videos/B16-20/trimmed/{}/".format(sesh.animalName),
            "/media/WDC7/videos/B16-20/trimmed/{}/".format(sesh.animalName),
            "/media/WDC8/videos/B16-20/trimmed/{}/".format(sesh.animalName),
            "/media/WDC6/{}/".format(sesh.animalName),
            "/media/WDC7/{}/".format(sesh.animalName),
            "/media/WDC8/{}/".format(sesh.animalName),
            "/media/WDC4/lab_videos"
        ]
        sesh.usbVidFile = getUSBVideoFile(
            sesh.name, possibleDirectories, seshIdx=sesh.seshIdx, useSeshIdxDirectly=(sesh.animalName == "B18"))
        if sesh.usbVidFile is None:
            print("??????? usb vid file not found for session ", sesh.name)
        else:
            print("Running USB light time analysis, file", sesh.usbVidFile)
            sesh.usbLightOffFrame, sesh.usbLightOnFrame = processUSBVideoData(
                sesh.usbVidFile, overwriteMode="loadOld", showVideo=False)
            if sesh.usbLightOffFrame is None or sesh.usbLightOnFrame is None:
                raise Exception("exclude me pls")

        if not sesh.importOptions["justExtractData"] or sesh.importOptions["runInteractiveExtraction"]:
            clipsFileName = sesh.fileStartString + '.1.clips'
            if not os.path.exists(clipsFileName) and len(sesh.foundWells) > 0:
                print("clips file not found, gonna launch clips generator")

                all_pos_nearest_wells = getNearestWell(
                    xs, ys, sesh.well_coords_map)
                all_pos_well_entry_idxs, all_pos_well_exit_idxs, \
                    all_pos_well_entry_times, all_pos_well_exit_times = \
                    getWellEntryAndExitTimes(
                        all_pos_nearest_wells, ts)

                ann = AnnotatorWindow(xs, ys, ts, sesh.trodesLightOffTime, sesh.trodesLightOnTime,
                                      sesh.usbVidFile, all_pos_well_entry_times, all_pos_well_exit_times,
                                      sesh.foundWells, not sesh.probe_performed, sesh.fileStartString + '.1',
                                      sesh.well_coords_map)
                ann.resize(1600, 800)
                ann.show()
                importParentApp.exec()

        if not sesh.importOptions["justExtractData"]:
            if sesh.separate_probe_file:
                probe_file_str = os.path.join(
                    sesh.probe_dir, os.path.basename(sesh.probe_dir))
                bt_time_clips = readClipData(
                    sesh.fileStartString + '.1.clips')[0]
                probe_time_clips = readClipData(probe_file_str + '.1.clips')[0]
            else:
                time_clips = readClipData(sesh.fileStartString + '.1.clips')
                bt_time_clips = time_clips[0]
                if sesh.probe_performed:
                    probe_time_clips = time_clips[1]

            print("clips: {} (/{})".format(bt_time_clips, len(ts)))
            bt_start_idx = np.searchsorted(ts, bt_time_clips[0])
            bt_end_idx = np.searchsorted(ts, bt_time_clips[1])
            sesh.bt_pos_xs = xs[bt_start_idx:bt_end_idx]
            sesh.bt_pos_ys = ys[bt_start_idx:bt_end_idx]
            sesh.bt_pos_ts = ts[bt_start_idx:bt_end_idx]

            if sesh.probe_performed:
                if sesh.separate_probe_file:
                    _, position_data = readRawPositionData(
                        probe_file_str + '.1.videoPositionTracking')
                    xs, ys, ts = processPosData(position_data, xLim=(
                        sesh.animalInfo.X_START, sesh.animalInfo.X_FINISH), yLim=(sesh.animalInfo.Y_START, sesh.animalInfo.Y_FINISH))

                probe_start_idx = np.searchsorted(ts, probe_time_clips[0])
                probe_end_idx = np.searchsorted(ts, probe_time_clips[1])
                sesh.probe_pos_xs = xs[probe_start_idx:probe_end_idx]
                sesh.probe_pos_ys = ys[probe_start_idx:probe_end_idx]
                sesh.probe_pos_ts = ts[probe_start_idx:probe_end_idx]

                if np.nanmax(sesh.probe_pos_ys) < 550 and np.nanmax(sesh.probe_pos_ys) > 400:
                    print("Position data just covers top of the environment")
                    sesh.positionOnlyTop = True
                else:
                    pass
                    # print("min max")
                    # print(np.nanmin(sesh.probe_pos_ys))
                    # print(np.nanmax(sesh.probe_pos_ys))
                    # plt.plot(sesh.bt_pos_xs, sesh.bt_pos_ys)
                    # plt.show()
                    # plt.plot(sesh.probe_pos_xs, sesh.probe_pos_ys)
                    # plt.show()

    rewardClipsFile = sesh.fileStartString + '.1.rewardClips'
    if not os.path.exists(rewardClipsFile):
        rewardClipsFile = sesh.fileStartString + '.1.rewardclips'
    if not os.path.exists(rewardClipsFile):
        if sesh.num_home_found > 0:
            print("Well find times not marked for session {}".format(sesh.name))
            sesh.hasWellFindTimes = False
        else:
            sesh.hasWellFindTimes = True
    else:
        sesh.hasWellFindTimes = True
        well_visit_times = readClipData(rewardClipsFile)
        assert sesh.num_away_found + \
            sesh.num_home_found == np.shape(well_visit_times)[0]
        sesh.home_well_find_times = well_visit_times[::2, 0]
        sesh.home_well_leave_times = well_visit_times[::2, 1]
        sesh.away_well_find_times = well_visit_times[1::2, 0]
        sesh.away_well_leave_times = well_visit_times[1::2, 1]

        sesh.home_well_find_pos_idxs = np.searchsorted(
            sesh.bt_pos_ts, sesh.home_well_find_times)
        sesh.home_well_leave_pos_idxs = np.searchsorted(
            sesh.bt_pos_ts, sesh.home_well_leave_times)
        sesh.away_well_find_pos_idxs = np.searchsorted(
            sesh.bt_pos_ts, sesh.away_well_find_times)
        sesh.away_well_leave_pos_idxs = np.searchsorted(
            sesh.bt_pos_ts, sesh.away_well_leave_times)

    if sesh.importOptions["skipSniff"]:
        sesh.hasSniffTimes = False
    else:
        gl = sesh.animalInfo.data_dir + sesh.name + '/*.rgs'
        dir_list = glob.glob(gl)
        if len(dir_list) == 0:
            print("Couldn't find sniff times files")
            sesh.hasSniffTimes = False
        else:
            sesh.hasSniffTimes = True

            sesh.sniffTimesFile = dir_list[0]
            if len(dir_list) > 1:
                print("Warning, multiple rgs files found: {}".format(dir_list))

            if not os.path.exists(sesh.sniffTimesFile):
                raise Exception("Couldn't find rgs file")
            else:
                print("getting rgs from {}".format(sesh.sniffTimesFile))
                with open(sesh.sniffTimesFile, 'r') as stf:
                    streader = csv.reader(stf)
                    sniffData = [v for v in streader]

                    sesh.well_sniff_times_entry = [[] for _ in all_well_names]
                    sesh.well_sniff_times_exit = [[] for _ in all_well_names]
                    sesh.bt_well_sniff_times_entry = [
                        [] for _ in all_well_names]
                    sesh.bt_well_sniff_times_exit = [
                        [] for _ in all_well_names]
                    sesh.probe_well_sniff_times_entry = [
                        [] for _ in all_well_names]
                    sesh.probe_well_sniff_times_exit = [
                        [] for _ in all_well_names]

                    sesh.sniff_pre_trial_light_off = int(sniffData[0][0])
                    sesh.sniff_trial_start = int(sniffData[0][1])
                    sesh.sniff_trial_stop = int(sniffData[0][2])
                    sesh.sniff_probe_start = int(sniffData[1][0])
                    sesh.sniff_probe_stop = int(sniffData[1][1])
                    sesh.sniff_post_probe_light_on = int(sniffData[1][2])

                    well_name_to_idx = np.empty((np.max(all_well_names) + 1))
                    well_name_to_idx[:] = np.nan
                    for widx, wname in enumerate(all_well_names):
                        well_name_to_idx[wname] = widx

                    for i in sniffData[2:]:
                        w = int(well_name_to_idx[int(i[2])])
                        entry_time = int(i[0])
                        exit_time = int(i[1])
                        sesh.well_sniff_times_entry[w].append(entry_time)
                        sesh.well_sniff_times_exit[w].append(exit_time)

                        if exit_time < entry_time:
                            print(
                                "mismatched interval: {} - {}".format(entry_time, exit_time))
                            assert False
                        if exit_time < sesh.sniff_trial_stop:
                            sesh.bt_well_sniff_times_entry[w].append(
                                entry_time)
                            sesh.bt_well_sniff_times_exit[w].append(exit_time)
                        else:
                            sesh.probe_well_sniff_times_entry[w].append(
                                entry_time)
                            sesh.probe_well_sniff_times_exit[w].append(
                                exit_time)


def loadLFPData(sesh):
    lfpData = []

    if len(sesh.ripple_detection_tetrodes) == 0:
        sesh.ripple_detection_tetrodes = [sesh.animalInfo.DEFAULT_RIP_DET_TET]

    for i in range(len(sesh.ripple_detection_tetrodes)):
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print(lfpdir, "doesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            elif os.path.exists("/home/wcroughan/Software/Trodes21/linux/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            print(syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + sesh.name + ".LFP_nt" + \
            str(sesh.ripple_detection_tetrodes[i]) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.bt_lfp_fnames.append(lfpfilename)
        lfpData.append(MountainViewIO.loadLFP(
            data_file=sesh.bt_lfp_fnames[-1]))

    if sesh.ripple_baseline_tetrode is None:
        sesh.ripple_baseline_tetrode = sesh.animalInfo.DEFAULT_RIP_BAS_TET

    # I think Martin didn't have this baseline tetrode? Need to check
    if sesh.ripple_baseline_tetrode is not None:
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print(lfpdir, "doesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            print(syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + sesh.name + ".LFP_nt" + \
            str(sesh.ripple_baseline_tetrode) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.bt_lfp_baseline_fname = lfpfilename
        baselineLfpData = MountainViewIO.loadLFP(
            data_file=sesh.bt_lfp_baseline_fname)
    else:
        baselineLfpData = None

    return lfpData, baselineLfpData


def runLFPAnalyses(sesh, lfpData, baselineLfpData, showPlot=False):
    lfpV = lfpData[0][1]['voltage']
    lfpTimestamps = lfpData[0][0]['time']
    C = sesh.importOptions["consts"]

    # Deflections represent interruptions from stimulation, artifacts include these and also weird noise
    # Although maybe not really ... 2022-5-11 replacing this
    # lfp_deflections = signal.find_peaks(-lfpV, height=DEFLECTION_THRESHOLD_HI,
    # distance=MIN_ARTIFACT_DISTANCE)
    lfp_deflections = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=C["DEFLECTION_THRESHOLD_HI"], distance=C["MIN_ARTIFACT_DISTANCE"])
    interruption_idxs = lfp_deflections[0]
    sesh.interruption_timestamps = lfpTimestamps[interruption_idxs]
    sesh.interruptionIdxs = interruption_idxs

    sesh.bt_interruption_pos_idxs = np.searchsorted(
        sesh.bt_pos_ts, sesh.interruption_timestamps)
    sesh.bt_interruption_pos_idxs = sesh.bt_interruption_pos_idxs[sesh.bt_interruption_pos_idxs < len(
        sesh.bt_pos_ts)]

    print("Condition - {}".format("SWR" if sesh.isRippleInterruption else "Ctrl"))
    lfp_deflections = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=C["DEFLECTION_THRESHOLD_LO"], distance=C["MIN_ARTIFACT_DISTANCE"])
    lfp_artifact_idxs = lfp_deflections[0]
    sesh.artifact_timestamps = lfpTimestamps[lfp_artifact_idxs]
    sesh.artifactIdxs = lfp_artifact_idxs

    bt_lfp_start_idx = np.searchsorted(lfpTimestamps, sesh.bt_pos_ts[0])
    bt_lfp_end_idx = np.searchsorted(lfpTimestamps, sesh.bt_pos_ts[-1])
    btLFPData = lfpV[bt_lfp_start_idx:bt_lfp_end_idx]
    # bt_lfp_artifact_idxs = lfp_artifact_idxs - bt_lfp_start_idx
    bt_lfp_artifact_idxs = interruption_idxs - bt_lfp_start_idx
    bt_lfp_artifact_idxs = bt_lfp_artifact_idxs[bt_lfp_artifact_idxs > 0]
    sesh.bt_lfp_artifact_idxs = bt_lfp_artifact_idxs
    sesh.bt_lfp_start_idx = bt_lfp_start_idx
    sesh.bt_lfp_end_idx = bt_lfp_end_idx

    pre_bt_interruption_idxs = interruption_idxs[interruption_idxs <
                                                 bt_lfp_start_idx]
    # pre_bt_interruption_idxs_first_half = interruption_idxs[interruption_idxs < int(
    #     bt_lfp_start_idx / 2)]

    _, _, sesh.prebtMeanRipplePower, sesh.prebtStdRipplePower = getRipplePower(
        lfpV[0:bt_lfp_start_idx], omit_artifacts=False)
    _, ripple_power, _, _ = getRipplePower(
        btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=sesh.prebtMeanRipplePower,
        stdPower=sesh.prebtStdRipplePower, showPlot=showPlot)
    sesh.btRipStartIdxsPreStats, sesh.btRipLensPreStats, sesh.btRipPeakIdxsPreStats, sesh.btRipPeakAmpsPreStats, sesh.btRipCrossThreshIdxsPreStats = \
        detectRipples(ripple_power)
    # print(bt_lfp_start_idx, sesh.btRipStartIdxsPreStats)
    if len(sesh.btRipStartIdxsPreStats) == 0:
        sesh.btRipStartTimestampsPreStats = np.array([])
    else:
        sesh.btRipStartTimestampsPreStats = lfpTimestamps[
            sesh.btRipStartIdxsPreStats + bt_lfp_start_idx]

    _, _, sesh.prebtMeanRipplePowerArtifactsRemoved, sesh.prebtStdRipplePowerArtifactsRemoved = getRipplePower(
        lfpV[0:bt_lfp_start_idx], lfp_deflections=pre_bt_interruption_idxs)
    _, _, sesh.prebtMeanRipplePowerArtifactsRemovedFirstHalf, sesh.prebtStdRipplePowerArtifactsRemovedFirstHalf = getRipplePower(
        lfpV[0:int(bt_lfp_start_idx / 2)], lfp_deflections=pre_bt_interruption_idxs)

    if sesh.probe_performed and not sesh.separate_iti_file and not sesh.separate_probe_file:
        ITI_MARGIN = 5  # units: seconds
        itiLfpStart_ts = sesh.bt_pos_ts[-1] + TRODES_SAMPLING_RATE * ITI_MARGIN
        itiLfpEnd_ts = sesh.probe_pos_ts[0] - TRODES_SAMPLING_RATE * ITI_MARGIN
        itiLfpStart_idx = np.searchsorted(lfpTimestamps, itiLfpStart_ts)
        itiLfpEnd_idx = np.searchsorted(lfpTimestamps, itiLfpEnd_ts)
        itiLFPData = lfpV[itiLfpStart_idx:itiLfpEnd_idx]
        sesh.ITIRippleIdxOffset = itiLfpStart_idx
        sesh.itiLfpStart_ts = itiLfpStart_ts
        sesh.itiLfpEnd_ts = itiLfpEnd_ts

        # in general none, but there's a few right at the start of where this is defined
        itiStimIdxs = interruption_idxs - itiLfpStart_idx
        zeroIdx = np.searchsorted(itiStimIdxs, 0)
        itiStimIdxs = itiStimIdxs[zeroIdx:]

        _, ripple_power, sesh.ITIMeanRipplePower, sesh.ITIStdRipplePower = getRipplePower(
            itiLFPData, lfp_deflections=itiStimIdxs)
        sesh.ITIRipStartIdxs, sesh.ITIRipLens, sesh.ITIRipPeakIdxs, sesh.ITIRipPeakAmps, sesh.ITIRipCrossThreshIdxs = \
            detectRipples(ripple_power)
        if len(sesh.ITIRipStartIdxs) > 0:
            sesh.ITIRipStartTimestamps = lfpTimestamps[sesh.ITIRipStartIdxs + itiLfpStart_idx]
        else:
            sesh.ITIRipStartTimestamps = np.array([])

        sesh.ITIDuration = (itiLfpEnd_ts - itiLfpStart_ts) / \
            TRODES_SAMPLING_RATE

        probeLfpStart_ts = sesh.probe_pos_ts[0]
        probeLfpEnd_ts = sesh.probe_pos_ts[-1]
        probeLfpStart_idx = np.searchsorted(lfpTimestamps, probeLfpStart_ts)
        probeLfpEnd_idx = np.searchsorted(lfpTimestamps, probeLfpEnd_ts)
        probeLFPData = lfpV[probeLfpStart_idx:probeLfpEnd_idx]
        sesh.probeRippleIdxOffset = probeLfpStart_idx
        sesh.probeLfpStart_ts = probeLfpStart_ts
        sesh.probeLfpEnd_ts = probeLfpEnd_ts
        sesh.probeLfpStart_idx = probeLfpStart_idx
        sesh.probeLfpEnd_idx = probeLfpEnd_idx

        _, ripple_power, sesh.probeMeanRipplePower, sesh.probeStdRipplePower = getRipplePower(
            probeLFPData, omit_artifacts=False)
        sesh.probeRipStartIdxs, sesh.probeRipLens, sesh.probeRipPeakIdxs, sesh.probeRipPeakAmps, sesh.probeRipCrossThreshIdxs = \
            detectRipples(ripple_power)
        if len(sesh.probeRipStartIdxs) > 0:
            sesh.probeRipStartTimestamps = lfpTimestamps[sesh.probeRipStartIdxs + probeLfpStart_idx]
        else:
            sesh.probeRipStartTimestamps = np.array([])

        sesh.probeDuration = (probeLfpEnd_ts - probeLfpStart_ts) / \
            TRODES_SAMPLING_RATE

        _, itiRipplePowerProbeStats, _, _ = getRipplePower(
            itiLFPData, lfp_deflections=itiStimIdxs, meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower)
        sesh.ITIRipStartIdxsProbeStats, sesh.ITIRipLensProbeStats, sesh.ITIRipPeakIdxsProbeStats, sesh.ITIRipPeakAmpsProbeStats, sesh.ITIRipCrossThreshIdxsProbeStats = \
            detectRipples(itiRipplePowerProbeStats)
        if len(sesh.ITIRipStartIdxsProbeStats) > 0:
            sesh.ITIRipStartTimestampsProbeStats = lfpTimestamps[
                sesh.ITIRipStartIdxsProbeStats + itiLfpStart_idx]
        else:
            sesh.ITIRipStartTimestampsProbeStats = np.array([])

        _, btRipplePowerProbeStats, _, _ = getRipplePower(
            btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower, showPlot=showPlot)
        sesh.btRipStartIdxsProbeStats, sesh.btRipLensProbeStats, sesh.btRipPeakIdxsProbeStats, sesh.btRipPeakAmpsProbeStats, sesh.btRipCrossThreshIdxsProbeStats = \
            detectRipples(btRipplePowerProbeStats)
        if len(sesh.btRipStartIdxsProbeStats) > 0:
            sesh.btRipStartTimestampsProbeStats = lfpTimestamps[
                sesh.btRipStartIdxsProbeStats + bt_lfp_start_idx]
        else:
            sesh.btRipStartTimestampsProbeStats = np.array([])

        if sesh.bt_lfp_baseline_fname is not None:
            # With baseline tetrode, calculated the way activelink does it
            lfpData = MountainViewIO.loadLFP(
                data_file=sesh.bt_lfp_baseline_fname)
            baselfpV = lfpData[1]['voltage']
            # baselfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

            btRipPower, _, _, _ = getRipplePower(
                btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower, showPlot=showPlot)
            probeRipPower, _, _, _ = getRipplePower(
                probeLFPData, omit_artifacts=False)

            baselineProbeLFPData = baselfpV[probeLfpStart_idx:probeLfpEnd_idx]
            probeBaselinePower, _, baselineProbeMeanRipplePower, baselineProbeStdRipplePower = getRipplePower(
                baselineProbeLFPData, omit_artifacts=False)
            btBaselineLFPData = baselfpV[bt_lfp_start_idx:bt_lfp_end_idx]
            btBaselineRipplePower, _, _, _ = getRipplePower(
                btBaselineLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=baselineProbeMeanRipplePower, stdPower=baselineProbeStdRipplePower,
                showPlot=showPlot)

            probeRawPowerDiff = probeRipPower - probeBaselinePower
            zmean = np.nanmean(probeRawPowerDiff)
            zstd = np.nanstd(probeRawPowerDiff)

            rawPowerDiff = btRipPower - btBaselineRipplePower
            zPowerDiff = (rawPowerDiff - zmean) / zstd

            sesh.btWithBaseRipStartIdx, sesh.btWithBaseRipLens, sesh.btWithBaseRipPeakIdx, sesh.btWithBaseRipPeakAmps, sesh.btWithBaseRipCrossThreshIdxs = \
                detectRipples(zPowerDiff)
            if len(sesh.btWithBaseRipStartIdx) > 0:
                sesh.btWithBaseRipStartTimestamps = lfpTimestamps[
                    sesh.btWithBaseRipStartIdx + bt_lfp_start_idx]
            else:
                sesh.btWithBaseRipStartTimestamps = np.array([])

        print("{} ripples found during task".format(
            len(sesh.btRipStartTimestampsProbeStats)))

    elif sesh.probe_performed:
        print("Probe performed but LFP in a separate file for session", sesh.name)


def runSanityChecks(sesh, lfpData, baselineLfpData, showPlots=False, overrideNotes=True):
    print(f"Running sanity checks for {sesh.name}")
    if lfpData is None:
        print("No LFP data to look at")
    else:
        lfpV = lfpData[0][1]['voltage']
        lfpTimestamps = lfpData[0][0]['time']
        C = sesh.importOptions["consts"]

        lfp_deflections = signal.find_peaks(np.abs(np.diff(
            lfpV, prepend=lfpV[0])), height=C["DEFLECTION_THRESHOLD_HI"], distance=C["MIN_ARTIFACT_DISTANCE"])
        interruptionIdxs = lfp_deflections[0]
        interruptionTimestamps = lfpTimestamps[interruptionIdxs]
        btInterruptionPosIdxs = np.searchsorted(sesh.bt_pos_ts, interruptionTimestamps)
        btInterruptionPosIdxs = sesh.bt_interruption_pos_idxs[btInterruptionPosIdxs < len(
            sesh.bt_pos_ts)]

        numInterruptions = len(sesh.bt_interruption_pos_idxs)
        print("{} interruptions detected".format(numInterruptions))
        if numInterruptions < 50:
            if sesh.isRippleInterruption and numInterruptions > 0:
                print("WARNING: FEWER THAN 50 STIMS DETECTED ON AN INTERRUPTION SESSION")
            elif numInterruptions == 0:
                print(
                    "WARNING: IGNORING BEHAVIOR NOTES FILE BECAUSE SEEING 0 INTERRUPTIONS, CALLING THIS A CONTROL SESSION")
                if overrideNotes:
                    sesh.isRippleInterruption = False
            else:
                print(
                    "WARNING: very few interruptions. This was a delay control but is basically a no-stim control")
        elif numInterruptions < 100:
            print("50-100 interruptions: not overriding label")

        dt = np.diff(lfpTimestamps)
        gapThresh = 2.0 * float(TRODES_SAMPLING_RATE / LFP_SAMPLING_RATE)
        isBigGap = dt > gapThresh

        if not any(isBigGap):
            print("No gaps in LFP!")
        else:
            totalTime = (lfpTimestamps[-1] - lfpTimestamps[0]) / TRODES_SAMPLING_RATE
            totalGapTime = np.sum(dt[isBigGap])
            print(f"{totalGapTime}/{totalTime} ({int(100*totalGapTime/totalTime)}%) of lfp signal missing")

            maxGapIdx = np.argmax(dt)
            maxGapLen = dt[maxGapIdx] / TRODES_SAMPLING_RATE
            maxGapT1 = (lfpTimestamps[maxGapIdx] - lfpTimestamps[0]) / TRODES_SAMPLING_RATE
            maxGapT2 = (lfpTimestamps[maxGapIdx + 1] - lfpTimestamps[0]) / TRODES_SAMPLING_RATE
            print(f"Biggest gap: {maxGapLen}s long ({maxGapT1} - {maxGapT2})")


def posCalcVelocity(sesh):
    pixPerCm = sesh.importOptions["consts"]["PIXELS_PER_CM"]
    bt_vel = np.sqrt(np.power(np.diff(sesh.bt_pos_xs), 2) +
                     np.power(np.diff(sesh.bt_pos_ys), 2))
    sesh.bt_vel_cm_s = np.divide(bt_vel, np.diff(sesh.bt_pos_ts) /
                                 TRODES_SAMPLING_RATE) / pixPerCm
    bt_is_mv = sesh.bt_vel_cm_s > sesh.importOptions["consts"]["VEL_THRESH"]
    if len(bt_is_mv) > 0:
        bt_is_mv = np.append(bt_is_mv, np.array(bt_is_mv[-1]))
    sesh.bt_is_mv = bt_is_mv
    sesh.bt_mv_xs = np.array(sesh.bt_pos_xs)
    sesh.bt_mv_xs[np.logical_not(bt_is_mv)] = np.nan
    sesh.bt_still_xs = np.array(sesh.bt_pos_xs)
    sesh.bt_still_xs[bt_is_mv] = np.nan
    sesh.bt_mv_ys = np.array(sesh.bt_pos_ys)
    sesh.bt_mv_ys[np.logical_not(bt_is_mv)] = np.nan
    sesh.bt_still_ys = np.array(sesh.bt_pos_ys)
    sesh.bt_still_ys[bt_is_mv] = np.nan

    if sesh.probe_performed:
        probe_vel = np.sqrt(np.power(np.diff(sesh.probe_pos_xs), 2) +
                            np.power(np.diff(sesh.probe_pos_ys), 2))
        sesh.probe_vel_cm_s = np.divide(probe_vel, np.diff(sesh.probe_pos_ts) /
                                        TRODES_SAMPLING_RATE) / pixPerCm
        probe_is_mv = sesh.probe_vel_cm_s > sesh.importOptions["consts"]["VEL_THRESH"]
        if len(probe_is_mv) > 0:
            probe_is_mv = np.append(probe_is_mv, np.array(probe_is_mv[-1]))
        sesh.probe_is_mv = probe_is_mv
        sesh.probe_mv_xs = np.array(sesh.probe_pos_xs)
        sesh.probe_mv_xs[np.logical_not(probe_is_mv)] = np.nan
        sesh.probe_still_xs = np.array(sesh.probe_pos_xs)
        sesh.probe_still_xs[probe_is_mv] = np.nan
        sesh.probe_mv_ys = np.array(sesh.probe_pos_ys)
        sesh.probe_mv_ys[np.logical_not(probe_is_mv)] = np.nan
        sesh.probe_still_ys = np.array(sesh.probe_pos_ys)
        sesh.probe_still_ys[probe_is_mv] = np.nan


def posCalcEntryExitTimes(sesh):
    # ===================================
    # Well and quadrant entry and exit times
    # ===================================
    sesh.bt_nearest_wells = getNearestWell(
        sesh.bt_pos_xs, sesh.bt_pos_ys, sesh.well_coords_map)

    sesh.bt_quadrants = np.array(
        [quadrantOfWell(wi) for wi in sesh.bt_nearest_wells])
    sesh.home_quadrant = quadrantOfWell(sesh.home_well)

    sesh.bt_well_entry_idxs, sesh.bt_well_exit_idxs, \
        sesh.bt_well_entry_times, sesh.bt_well_exit_times = \
        getWellEntryAndExitTimes(
            sesh.bt_nearest_wells, sesh.bt_pos_ts)

    # ninc stands for neighbors included
    sesh.bt_well_entry_idxs_ninc, sesh.bt_well_exit_idxs_ninc, \
        sesh.bt_well_entry_times_ninc, sesh.bt_well_exit_times_ninc = \
        getWellEntryAndExitTimes(
            sesh.bt_nearest_wells, sesh.bt_pos_ts, include_neighbors=True)

    sesh.bt_quadrant_entry_idxs, sesh.bt_quadrant_exit_idxs, \
        sesh.bt_quadrant_entry_times, sesh.bt_quadrant_exit_times = \
        getWellEntryAndExitTimes(
            sesh.bt_quadrants, sesh.bt_pos_ts, well_idxs=[0, 1, 2, 3])

    for i in range(len(all_well_names)):
        # print("well {} had {} entries, {} exits".format(
        #     all_well_names[i], len(sesh.bt_well_entry_idxs[i]), len(sesh.bt_well_exit_idxs[i])))
        assert len(sesh.bt_well_entry_times[i]) == len(
            sesh.bt_well_exit_times[i])

    sesh.home_well_idx_in_allwells = np.argmax(
        all_well_names == sesh.home_well)
    sesh.ctrl_home_well_idx_in_allwells = np.argmax(
        all_well_names == sesh.ctrl_home_well)

    sesh.bt_home_well_entry_times = sesh.bt_well_entry_times[
        sesh.home_well_idx_in_allwells]
    sesh.bt_home_well_exit_times = sesh.bt_well_exit_times[
        sesh.home_well_idx_in_allwells]

    sesh.bt_ctrl_home_well_entry_times = sesh.bt_well_entry_times[
        sesh.ctrl_home_well_idx_in_allwells]
    sesh.bt_ctrl_home_well_exit_times = sesh.bt_well_exit_times[
        sesh.ctrl_home_well_idx_in_allwells]

    # same for during probe
    if sesh.probe_performed:
        sesh.probe_nearest_wells = getNearestWell(
            sesh.probe_pos_xs, sesh.probe_pos_ys, sesh.well_coords_map)

        sesh.probe_well_entry_idxs, sesh.probe_well_exit_idxs, \
            sesh.probe_well_entry_times, sesh.probe_well_exit_times = getWellEntryAndExitTimes(
                sesh.probe_nearest_wells, sesh.probe_pos_ts)
        sesh.probe_well_entry_idxs_ninc, sesh.probe_well_exit_idxs_ninc, \
            sesh.probe_well_entry_times_ninc, sesh.probe_well_exit_times_ninc = getWellEntryAndExitTimes(
                sesh.probe_nearest_wells, sesh.probe_pos_ts, include_neighbors=True)

        sesh.probe_quadrants = np.array(
            [quadrantOfWell(wi) for wi in sesh.probe_nearest_wells])

        sesh.probe_quadrant_entry_idxs, sesh.probe_quadrant_exit_idxs, \
            sesh.probe_quadrant_entry_times, sesh.probe_quadrant_exit_times = getWellEntryAndExitTimes(
                sesh.probe_quadrants, sesh.probe_pos_ts, well_idxs=[0, 1, 2, 3])

        for i in range(len(all_well_names)):
            # print(i, len(sesh.probe_well_entry_times[i]), len(sesh.probe_well_exit_times[i]))
            assert len(sesh.probe_well_entry_times[i]) == len(
                sesh.probe_well_exit_times[i])

        sesh.probe_home_well_entry_times = sesh.probe_well_entry_times[
            sesh.home_well_idx_in_allwells]
        sesh.probe_home_well_exit_times = sesh.probe_well_exit_times[
            sesh.home_well_idx_in_allwells]

        sesh.probe_ctrl_home_well_entry_times = sesh.probe_well_entry_times[
            sesh.ctrl_home_well_idx_in_allwells]
        sesh.probe_ctrl_home_well_exit_times = sesh.probe_well_exit_times[
            sesh.ctrl_home_well_idx_in_allwells]


def posCalcBallisticity(sesh):
    # ===================================
    # Ballisticity of movements
    # ===================================
    timeIntervals = sesh.importOptions["consts"]["BALL_TIME_INTERVALS"]
    furthest_interval = max(timeIntervals)
    assert np.all(np.diff(np.array(timeIntervals)) > 0)
    assert timeIntervals[0] > 0

    # idx is (time, interval len, dimension)
    d1 = len(sesh.bt_pos_ts) - furthest_interval
    delta = np.empty((d1, furthest_interval, 2))
    delta[:] = np.nan
    dx = np.diff(sesh.bt_pos_xs)
    dy = np.diff(sesh.bt_pos_ys)
    delta[:, 0, 0] = dx[0:d1]
    delta[:, 0, 1] = dy[0:d1]

    for i in range(1, furthest_interval):
        delta[:, i, 0] = delta[:, i - 1, 0] + dx[i:i + d1]
        delta[:, i, 1] = delta[:, i - 1, 1] + dy[i:i + d1]

    displacement = np.sqrt(np.sum(np.square(delta), axis=2))
    displacement[displacement == 0] = np.nan
    last_displacement = displacement[:, -1]
    sesh.bt_ball_displacement = last_displacement

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
    beta = m(y * x) / m(np.power(x, 2))
    sesh.bt_ballisticity = beta
    assert np.sum(np.isnan(sesh.bt_ballisticity)) == 0

    if sesh.probe_performed:
        # idx is (time, interval len, dimension)
        d1 = len(sesh.probe_pos_ts) - furthest_interval
        delta = np.empty((d1, furthest_interval, 2))
        delta[:] = np.nan
        dx = np.diff(sesh.probe_pos_xs)
        dy = np.diff(sesh.probe_pos_ys)
        delta[:, 0, 0] = dx[0:d1]
        delta[:, 0, 1] = dy[0:d1]

        for i in range(1, furthest_interval):
            delta[:, i, 0] = delta[:, i - 1, 0] + dx[i:i + d1]
            delta[:, i, 1] = delta[:, i - 1, 1] + dy[i:i + d1]

        displacement = np.sqrt(np.sum(np.square(delta), axis=2))
        displacement[displacement == 0] = np.nan
        last_displacement = displacement[:, -1]
        sesh.probe_ball_displacement = last_displacement

        x = np.log(np.tile(np.arange(furthest_interval) + 1, (d1, 1)))
        y = np.log(displacement)
        assert np.all(np.logical_or(
            np.logical_not(np.isnan(y)), np.isnan(displacement)))
        y[y == -np.inf] = np.nan
        np.nan_to_num(y, copy=False, nan=np.nanmin(y))

        x = x - np.tile(np.nanmean(x, axis=1), (furthest_interval, 1)).T
        y = y - np.tile(np.nanmean(y, axis=1), (furthest_interval, 1)).T
        # beta = ((m(y * x) - m(x) * m(y))/m(x*x) - m(x)**2)
        beta = m(y * x) / m(np.power(x, 2))
        sesh.probe_ballisticity = beta
        assert np.sum(np.isnan(sesh.probe_ballisticity)) == 0


# ===================================
# Knot-path-curvature as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000638
# ===================================
def getCurvature(x, y, H):
    H2 = H * H
    i1 = np.empty_like(x)
    i2 = np.empty_like(x)
    dxf = np.empty_like(x)
    dyf = np.empty_like(x)
    dxb = np.empty_like(x)
    dyb = np.empty_like(x)
    magf = np.empty_like(x)
    magb = np.empty_like(x)

    dx = np.diff(x)
    dy = np.diff(y)

    for pi in range(len(x)):
        # forward
        ii = pi
        dxfi = 0.0
        dyfi = 0.0
        while ii < len(dx):
            dxfi += dx[ii]
            dyfi += dy[ii]
            magfi = dxfi * dxfi + dyfi * dyfi
            if magfi >= H2:
                break
            ii += 1

        if ii == len(dx):
            i1[pi] = np.nan
            i2[pi] = np.nan
            dxf[pi] = np.nan
            dyf[pi] = np.nan
            dxb[pi] = np.nan
            dyb[pi] = np.nan
            magf[pi] = np.nan
            magb[pi] = np.nan
            continue
        i2[pi] = ii

        ii = pi - 1
        dxbi = 0.0
        dybi = 0.0
        while ii >= 0:
            dxbi += dx[ii]
            dybi += dy[ii]
            magb = dxbi * dxbi + dybi * dybi
            if magb >= H2:
                break
        if ii == -1:
            i1[pi] = np.nan
            i2[pi] = np.nan
            dxf[pi] = np.nan
            dyf[pi] = np.nan
            dxb[pi] = np.nan
            dyb[pi] = np.nan
            magf[pi] = np.nan
            magb[pi] = np.nan
            continue
        i1[pi] = ii

    uxf = dxf / np.sqrt(magf)
    uyf = dyf / np.sqrt(magf)
    uxb = dxb / np.sqrt(magb)
    uyb = dyb / np.sqrt(magb)
    dotprod = uxf * uxb + uyf * uyb
    curvature = np.arccos(dotprod)

    return curvature, i1, i2, dxf, dyf, dxb, dyb


def posCalcCurvature(sesh):
    KNOT_H_POS = sesh.importOptions["consts"]["KNOT_H_CM"] * \
        sesh.importOptions["consts"]["PIXELS_PER_CM"]

    sesh.bt_curvature, sesh.bt_curvature_i1, sesh.bt_curvature_i2, sesh.bt_curvature_dxf, sesh.bt_curvature_dyf, sesh.bt_curvature_dxb, \
        sesh.bt_curvature_dyb = getCurvature(sesh.bt_pos_xs, sesh.bt_pos_ys, KNOT_H_POS)

    if sesh.probe_performed:
        sesh.probe_curvature, sesh.probe_curvature_i1, sesh.probe_curvature_i2, sesh.probe_curvature_dxf, sesh.probe_curvature_dyf, sesh.probe_curvature_dxb, \
            sesh.probe_curvature_dyb = getCurvature(
                sesh.probe_pos_xs, sesh.probe_pos_ys, KNOT_H_POS)


def getExplorationCategories(ts, vel, nearestWells, consts, forcePauseIntervals=None):
    POS_FRAME_RATE = stats.mode(np.diff(ts))[0] / float(TRODES_SAMPLING_RATE)
    BOUT_VEL_SM_SIGMA = consts["BOUT_VEL_SM_SIGMA_SECS"] / POS_FRAME_RATE
    MIN_PAUSE_TIME_FRAMES = int(consts["MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS"] / POS_FRAME_RATE)
    MIN_EXPLORE_TIME_FRAMES = int(consts["MIN_EXPLORE_TIME_SECS"] / POS_FRAME_RATE)

    smoothVel = scipy.ndimage.gaussian_filter1d(vel, BOUT_VEL_SM_SIGMA)

    isExploreLocal = smoothVel > consts["PAUSE_MAX_SPEED_CM_S"]
    dilationFilter = np.ones((MIN_PAUSE_TIME_FRAMES), dtype=int)
    inPauseBout = ~ (signal.convolve(isExploreLocal.astype(int),
                     dilationFilter, mode='same').astype(bool))
    isInPause = signal.convolve(inPauseBout.astype(int), dilationFilter, mode='same').astype(bool)
    isInExplore = np.logical_not(isInPause)

    if forcePauseIntervals is not None:
        for i1, i2 in forcePauseIntervals:
            pidx1 = np.searchsorted(ts[0:-1], i1)
            pidx2 = np.searchsorted(ts[0:-1], i2)
            isInPause[pidx1:pidx2] = True
            isInExplore[pidx1:pidx2] = False

    assert np.sum(np.isnan(isInPause)) == 0
    assert np.sum(np.isnan(isInExplore)) == 0
    assert np.all(np.logical_or(isInPause, isInExplore))
    assert not np.any(np.logical_and(isInPause, isInExplore))

    startExplores = np.where(np.diff(isInExplore.astype(int)) == 1)[0] + 1
    if isInExplore[0]:
        startExplores = np.insert(startExplores, 0, 0)

    stopExplores = np.where(np.diff(isInExplore.astype(int)) == -1)[0] + 1
    if isInExplore[-1]:
        stopExplores = np.append(stopExplores, len(isInExplore))

    boutLenFrames = stopExplores - startExplores
    longEnough = boutLenFrames >= MIN_EXPLORE_TIME_FRAMES

    boutNumWellsVisited = np.zeros((len(startExplores)))
    for i, (bst, ben) in enumerate(zip(startExplores, stopExplores)):
        boutNumWellsVisited[i] = len(
            getListOfVisitedWells(nearestWells[bst:ben], True))
    enoughWells = boutNumWellsVisited >= consts["MIN_EXPLORE_NUM_WELLS"]

    keepBout = np.logical_and(longEnough, enoughWells)
    exploreBoutStarts = startExplores[keepBout]
    exploreBoutEnds = stopExplores[keepBout]
    exploreBoutLens = exploreBoutEnds - exploreBoutStarts
    ts = np.array(ts)
    exploreBoutLensSecs = (ts[exploreBoutEnds] - ts[exploreBoutStarts]) / TRODES_SAMPLING_RATE

    # add a category at each behavior time point for easy reference later:
    boutCategory = np.zeros_like(ts)
    last_stop = 0
    for bst, ben in zip(exploreBoutStarts, exploreBoutEnds):
        boutCategory[last_stop:bst] = 1
        last_stop = ben
    boutCategory[last_stop:] = 1
    for i1, i2 in forcePauseIntervals:
        pidx1 = np.searchsorted(ts[0:-1], i1)
        pidx2 = np.searchsorted(ts[0:-1], i2)
        boutCategory[pidx1:pidx2] = 2

    boutLabel = np.zeros_like(ts)
    for bi, (bst, ben) in enumerate(zip(exploreBoutStarts, exploreBoutEnds)):
        boutLabel[bst:ben] = bi + 1

    return smoothVel, isInPause, isInExplore, exploreBoutStarts, exploreBoutEnds, exploreBoutLens, \
        exploreBoutLensSecs, boutCategory, boutLabel


def posCalcExplorationBouts(sesh):
    sesh.bt_sm_vel, sesh.bt_is_in_pause, sesh.bt_is_in_explore, sesh.bt_explore_bout_starts, sesh.bt_explore_bout_ends, \
        sesh.bt_explore_bout_lens, sesh.bt_explore_bout_lens_secs, sesh.bt_bout_category, sesh.bt_bout_label = \
        getExplorationCategories(sesh.bt_pos_ts, sesh.bt_vel_cm_s, sesh.bt_nearest_wells, sesh.importOptions["consts"],
                                 forcePauseIntervals=(list(zip(sesh.home_well_find_times, sesh.home_well_leave_times)) +
                                                      list(zip(sesh.away_well_find_times, sesh.away_well_leave_times))))

    if sesh.probe_performed:
        sesh.probe_sm_vel, sesh.probe_is_in_pause, sesh.probe_is_in_explore, sesh.probe_explore_bout_starts, sesh.probe_explore_bout_ends, \
            sesh.probe_explore_bout_lens, sesh.probe_explore_bout_lens_secs, sesh.probe_bout_category, sesh.probe_bout_label = \
            getExplorationCategories(sesh.probe_pos_ts, sesh.probe_vel_cm_s,
                                     sesh.probe_nearest_wells, sesh.importOptions["consts"])


def getExcursions(nearestWells, ts):
    excursionCategory = np.array([BTSession.EXCURSION_STATE_ON_WALL if onWall(
        w) else BTSession.EXCURSION_STATE_OFF_WALL for w in nearestWells])
    excursionStarts = np.where(
        np.diff((excursionCategory == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == 1)[0] + 1
    if excursionCategory[0] == BTSession.EXCURSION_STATE_OFF_WALL:
        excursionStarts = np.insert(excursionStarts, 0, 0)
    excursionEnds = np.where(
        np.diff((excursionCategory == BTSession.EXCURSION_STATE_OFF_WALL).astype(int)) == -1)[0] + 1
    if excursionCategory[-1] == BTSession.EXCURSION_STATE_OFF_WALL:
        excursionEnds = np.append(excursionEnds, len(excursionCategory))
    t = np.array(ts + [ts[-1]])
    excursionLensSecs = (t[excursionEnds] - t[excursionStarts]) / TRODES_SAMPLING_RATE

    return excursionCategory, excursionStarts, excursionEnds, excursionLensSecs


def posCalcExcursions(sesh):
    sesh.bt_excursion_category, sesh.bt_excursion_starts, sesh.bt_excursion_ends, sesh.bt_excursion_lens_secs = \
        getExcursions(sesh.bt_nearest_wells, sesh.bt_pos_ts)

    if sesh.probe_performed:
        sesh.probe_excursion_category, sesh.probe_excursion_starts, sesh.probe_excursion_ends, sesh.probe_excursion_lens_secs = \
            getExcursions(sesh.probe_nearest_wells, sesh.probe_pos_ts)


def runPositionAnalyses(sesh):
    if sesh.last_away_well is None:
        sesh.num_away_found = 0
    else:
        sesh.num_away_found = next((i for i in range(
            len(sesh.away_wells)) if sesh.away_wells[i] == sesh.last_away_well), -1) + 1
    sesh.visited_away_wells = sesh.away_wells[0:sesh.num_away_found]
    # print(sesh.last_away_well)
    sesh.num_home_found = sesh.num_away_found
    if sesh.ended_on_home:
        sesh.num_home_found += 1

    if not sesh.hasPositionData:
        print("Can't run analyses without any data!")
        return

    if sesh.hasWellFindTimes:
        if len(sesh.home_well_leave_times) == len(sesh.away_well_find_times):
            sesh.away_well_latencies = np.array(sesh.away_well_find_times) - \
                np.array(sesh.home_well_leave_times)
            sesh.home_well_latencies = np.array(sesh.home_well_find_times) - \
                np.append([sesh.bt_pos_ts[0]],
                          sesh.away_well_leave_times[0:-1])
        else:
            sesh.away_well_latencies = np.array(sesh.away_well_find_times) - \
                np.array(sesh.home_well_leave_times[0:-1])
            sesh.home_well_latencies = np.array(sesh.home_well_find_times) - \
                np.append([sesh.bt_pos_ts[0]], sesh.away_well_leave_times)

    sesh.ctrl_home_well = 49 - sesh.home_well
    sesh.ctrl_home_x, sesh.ctrl_home_y = getWellCoordinates(
        sesh.ctrl_home_well, sesh.well_coords_map)

    posCalcVelocity(sesh)
    posCalcEntryExitTimes(sesh)

    # ===================================
    # truncate path when rat was recalled (as he's running to be picked up) based on visit time of marked well
    # Note need to check sessions that ended near 7
    # ===================================
    # bt_recall_pos_idx
    if sesh.bt_ended_at_well is None:
        assert sesh.last_away_well == sesh.away_wells[-1] and sesh.ended_on_home
        sesh.bt_ended_at_well = sesh.home_well

    bt_ended_at_well_idx = np.argmax(all_well_names == sesh.bt_ended_at_well)
    sesh.bt_recall_pos_idx = sesh.bt_well_exit_idxs[bt_ended_at_well_idx][-1]

    # if True:
    #     print(all_well_names[bt_ended_at_well_idx])
    #     print(sesh.bt_well_entry_idxs)
    #     colors = ["k", "r", "g", "c"]
    #     for wi in range(len(sesh.bt_well_entry_idxs)):
    #         ents = sesh.bt_well_entry_idxs[wi]
    #         exts = sesh.bt_well_exit_idxs[wi]
    #         for ent, ext in zip(ents, exts):
    #             plt.plot(sesh.bt_pos_xs[ent:ext],
    #                      sesh.bt_pos_ys[ent:ext], colors[wi % 4])
    #     plt.show()

    if sesh.probe_performed:
        probe_ended_at_well_idx = np.argmax(
            all_well_names == sesh.probe_ended_at_well)
        sesh.probe_recall_pos_idx = sesh.probe_well_exit_idxs[probe_ended_at_well_idx][-1]

    posCalcCurvature(sesh)
    # posCalcBallisticity(sesh)

    posCalcExplorationBouts(sesh)
    posCalcExcursions(sesh)
    # posCalcRewardCategories(sesh)


def extractAndSave(animalName, importOptions):
    numExcluded = 0
    numExtracted = 0

    print("=========================================================")
    print(f"Extracting data for animal {animalName}")
    print("=========================================================")
    animalInfo = getInfoForAnimal(animalName)
    print(
        f"\tdata_dir = {animalInfo.data_dir}\n\toutput_dir = {animalInfo.output_dir}")
    if not os.path.exists(animalInfo.output_dir):
        os.mkdir(animalInfo.output_dir)

    sessionDirs, prevSessionDirs = getSessionDirs(animalInfo, importOptions)
    dirListStr = "".join(
        [f"\t{s}\t{ss}\n" for s, ss in zip(sessionDirs, prevSessionDirs)])
    print("Session dirs:")
    print(dirListStr)

    dataObj = BTData()
    dataObj.importOptions = importOptions

    lastDateStr = ""
    lastPrevDateStr = ""
    sessionNumber = None
    prevSessionNumber = None
    for seshDir, prevSeshDir in zip(sessionDirs, prevSessionDirs):
        dateStr = seshDir.split('_')[0][-8:]
        if dateStr == lastDateStr:
            sessionNumber += 1
        else:
            lastDateStr = dateStr
            sessionNumber = 1

        if prevSeshDir is not None:
            prevDateStr = prevSeshDir.split('_')[0][-8:]
            if prevDateStr == lastPrevDateStr:
                prevSessionNumber += 1
            else:
                lastPrevDateStr = prevDateStr
                prevSessionNumber = 1

        sesh = makeSessionObj(seshDir, prevSeshDir,
                              sessionNumber, prevSessionNumber, animalInfo)
        sesh.animalName = animalName
        sesh.importOptions = importOptions
        print("\n=======================================")
        print(f"Starting session {sesh.name}:")
        print(f"\tfolder: {seshDir}:")
        print(f"\tinfo file name: {sesh.infoFileName}:")
        print(f"\tprev sesh info file name: {sesh.prevInfoFileName }:")
        if seshDir in animalInfo.excluded_sessions:
            print(seshDir, " excluded session, skipping")
            numExcluded += 1
            continue
        if "".join(os.path.basename(sesh.infoFileName).split(".")[0:-1]) in animalInfo.excluded_sessions:
            print(seshDir, " excluded session, skipping")
            numExcluded += 1
            continue

        parseInfoFiles(sesh)
        loadPositionData(sesh)
        if importOptions["skipLFP"]:
            lfpData = None
            baselineLfpData = None
        else:
            lfpData, baselineLfpData = loadLFPData(sesh)

        if importOptions["justExtractData"]:
            numExtracted += 1
            continue

        if not importOptions["skipLFP"]:
            runLFPAnalyses(sesh, lfpData, baselineLfpData)
        runPositionAnalyses(sesh)

        runSanityChecks(sesh, lfpData, baselineLfpData)

        dataObj.allSessions.append(sesh)

    if importOptions["justExtractData"]:
        print(
            f"Extracted data from {numExtracted} sessions, excluded {numExcluded}. Not analyzing or saving")
        return

    if not any([s.conditionGroup is not None for s in dataObj.allSessions]):
        for si, sesh in enumerate(dataObj.allSessions):
            sesh.conditionGroup = si

    # save all sessions to disk
    print("Saving to file: {}".format(animalInfo.out_filename))
    dataObj.saveToFile(os.path.join(
        animalInfo.output_dir, animalInfo.out_filename))
    print("Saved sessions:")
    for sesh in dataObj.allSessions:
        print(sesh.name)


if __name__ == "__main__":
    animalNames = parseCmdLineAnimalNames(default=["B18"])
    importOptions = {
        "skipLFP": False,
        "skipUSB": False,
        "skipPrevSession": True,
        "skipSniff": True,
        "forceAllTrackingAuto": False,
        "skipCurvature": False,
        "runJustSpecified": False,
        "specifiedDays": [],
        "specifiedRuns": [],
        "justExtractData": False,
        "runInteractiveExtraction": True,
        "consts": {
            "VEL_THRESH": 10,  # cm/s
            "PIXELS_PER_CM": None,

            # Typical observed amplitude of LFP deflection on stimulation
            "DEFLECTION_THRESHOLD_HI": 6000.0,
            "DEFLECTION_THRESHOLD_LO": 2000.0,
            "MIN_ARTIFACT_DISTANCE": int(0.05 * LFP_SAMPLING_RATE),

            # constants for exploration bout analysis
            "BOUT_VEL_SM_SIGMA_SECS": 1.5,
            "PAUSE_MAX_SPEED_CM_S": 8.0,
            "MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS": 1.0,
            "MIN_EXPLORE_TIME_SECS": 3.0,
            "MIN_EXPLORE_NUM_WELLS": 4,

            # constants for ballisticity
            "BALL_TIME_INTERVALS": list(range(1, 24)),
            "KNOT_H_CM": 8.0
        }
    }

    for animalName in animalNames:
        importOptions["skipUSB"] = animalName == "Martin"
        extractAndSave(animalName, importOptions)
