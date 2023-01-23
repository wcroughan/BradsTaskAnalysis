from BTData import BTData
from BTSession import BTSession
import numpy as np
import os
import csv
import glob
import scipy
from scipy import stats, signal
import MountainViewIO
from datetime import datetime, date
import sys
from PyQt5.QtWidgets import QApplication
# import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

from consts import allWellNames, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE, CM_PER_FT
from UtilFunctions import readWellCoordsFile, readRawPositionData, getListOfVisitedWells, onWall, getRipplePower, \
    detectRipples, getInfoForAnimal, getUSBVideoFile, quickPosPlot, parseCmdLineAnimalNames, AnimalInfo, \
    timeStrForTrodesTimestamp, quadrantOfWell, TimeThisFunction
from ClipsMaker import AnnotatorWindow
from TrodesCameraExtrator import getTrodesLightTimes, processRawTrodesVideo, processUSBVideoData

importParentApp = QApplication(sys.argv)

# TODO
# Set a maximum gap for previous sessions, so large gaps aren't counted and prev dir goes back to None
# Some "well visits" are just one time point. Need to smooth more and/or just generally change logic away from this
# visit stuff


# Returns a list of directories that correspond to runs for analysis. Unless runJustSpecified is True,
# only ever filters by day. Thus index within day of returned list can be used to find corresponding behavior notes file
def getSessionDirs(animalInfo, importOptions) -> Tuple[List[str], List[str]]:
    filtered_data_dirs = []
    prevSessionDirs = []
    prevSession = None
    all_data_dirs = sorted([d for d in os.listdir(animalInfo.data_dir) if d.count("_") > 0],
                           key=lambda s: (s.split('_')[0], s.split('_')[1]))

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

        dateStr = dir_split[0][-8:]

        if importOptions["runJustSpecified"] and dateStr not in importOptions["specifiedDays"] and \
                session_dir not in importOptions["specifiedRuns"]:
            prevSession = session_dir
            continue

        if dateStr in animalInfo.excluded_dates:
            print(f"skipping, excluded date {dateStr}")
            prevSession = session_dir
            continue

        if animalInfo.minimum_date is not None and dateStr < animalInfo.minimum_date:
            print("skipping date {}, is before minimum date {}".format(
                dateStr, animalInfo.minimum_date))
            prevSession = session_dir
            continue

        filtered_data_dirs.append(session_dir)
        if prevSession is not None:
            d1 = date(int(dateStr[0:4]), int(dateStr[4:6]), int(dateStr[6:8]))
            pds = prevSession.split('_')[0][-8:]
            d2 = date(int(pds[0:4]), int(pds[4:6]), int(pds[6:8]))
            daysBetweenSessions = (d1 - d2).days
            # print(d1, d2, daysBetweenSessions)
            if daysBetweenSessions > 2:
                prevSession = None
        prevSessionDirs.append(prevSession)
        prevSession = session_dir

    return filtered_data_dirs, prevSessionDirs


def makeSessionObj(seshDir: str, prevSeshDir: str,
                   sessionNumber: int, prevSessionNumber: int, animalInfo: AnimalInfo) -> BTSession:
    sesh = BTSession()
    sesh.prevSessionDir = prevSeshDir

    dir_split = seshDir.split('_')
    dateStr = dir_split[0][-8:]
    timeStr = dir_split[1]
    session_name_pfx = dir_split[0][0:-8]
    sesh.dateStr = dateStr
    sesh.name = seshDir
    sesh.timeStr = timeStr
    s = "{}_{}".format(dateStr, timeStr)
    sesh.date = datetime.strptime(s, "%Y%m%d_%H%M%S")

    # check for files that belong to this date
    sesh.btDir = seshDir
    gl = animalInfo.data_dir + session_name_pfx + dateStr + "*"
    dir_list = glob.glob(gl)
    for d in dir_list:
        if d.split('_')[-1] == "ITI":
            sesh.itiDir = d
            sesh.separateItiFile = True
            sesh.recordedIti = True
        elif d.split('_')[-1] == "Probe":
            sesh.probeDir = d
            sesh.separateProbeFile = True
    if not sesh.separateProbeFile:
        sesh.recordedIti = True
        sesh.probeDir = os.path.join(animalInfo.data_dir, seshDir)
    if not sesh.separateItiFile and sesh.recordedIti:
        sesh.itiDir = seshDir

    # Get behavior_notes info file name
    behaviorNotesDir = os.path.join(animalInfo.data_dir, 'behavior_notes')
    sesh.infoFileName = os.path.join(behaviorNotesDir, dateStr + ".txt")
    if not os.path.exists(sesh.infoFileName):
        # Switched numbering scheme once started doing multiple sessions a day
        sesh.infoFileName = os.path.join(
            behaviorNotesDir, f"{dateStr}_{sessionNumber}.txt")
    sesh.seshIdx = sessionNumber

    if prevSeshDir is None:
        sesh.prevInfoFileName = None
        sesh.prevSessionIdx = None
    else:
        dir_split = prevSeshDir.split('_')
        prevSeshDateStr = dir_split[0][-8:]
        sesh.prevInfoFileName = os.path.join(
            behaviorNotesDir, prevSeshDateStr + ".txt")
        if not os.path.exists(sesh.prevInfoFileName):
            # Switched numbering scheme once started doing multiple sessions a day
            sesh.prevInfoFileName = os.path.join(
                behaviorNotesDir, f"{prevSeshDateStr}_{prevSessionNumber}.txt")
        sesh.prevSessionIdx = prevSessionNumber

    sesh.fileStartString = os.path.join(animalInfo.data_dir, seshDir, seshDir)
    sesh.animalInfo = animalInfo

    return sesh


def generateFoundWells(home_well, away_wells, lastAwayWell, endedOnHome, foundFirstHome) -> List[int]:
    if not foundFirstHome:
        return []
    elif lastAwayWell is None:
        return [home_well]

    foundWells = []
    for aw in away_wells:
        foundWells += [home_well, aw]
        if aw == lastAwayWell:
            break
    if endedOnHome:
        foundWells.append(home_well)
    return foundWells


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

            # if sesh.importOptions["justExtractData"]  and fieldName.lower() not in ["reference", "ref", "baseline", \
            # "home", \
            # "aways", "last away", "last well", "probe performed"]:
            # continue

            if fieldName.lower() == "home":
                sesh.homeWell = int(fieldVal)
            elif fieldName.lower() == "aways":
                sesh.awayWells = [int(w) for w in fieldVal.strip().split(' ')]
                # print(line)
                # print(fieldName, fieldVal)
                # print(sesh.awayWells)
            elif fieldName.lower() == "foundwells":
                sesh.foundWells = [int(w) for w in fieldVal.strip().split(' ')]
            elif fieldName.lower() == "condition":
                typeIn = fieldVal.lower()
                if 'ripple' in typeIn or 'interruption' in typeIn:
                    # sesh.isRippleInterruption = True
                    sesh.condition = BTSession.CONDITION_INTERRUPTION
                elif 'none' in typeIn:
                    # sesh.isNoInterruption = True
                    sesh.condition = BTSession.CONDITION_NO_STIM
                elif 'delay' in typeIn:
                    # sesh.isDelayedInterruption = True
                    sesh.condition = BTSession.CONDITION_DELAY
                else:
                    print("\tCouldn't recognize Condition {} in file {}".format(
                        typeIn, sesh.infoFileName))
            elif fieldName.lower() == "thresh":
                if "low" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = 2.5
                elif "high" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = 4.0
                elif "med" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = 3.0
                elif "none" in fieldVal.lower():
                    sesh.rippleDetectionThreshold = None
                else:
                    sesh.rippleDetectionThreshold = float(
                        fieldVal)
            elif fieldName.lower() == "last away":
                if fieldVal.strip() == "None":
                    sesh.lastAwayWell = None
                else:
                    sesh.lastAwayWell = int(fieldVal)
            elif fieldName.lower() == "reference" or fieldName.lower() == "ref":
                if fieldVal.strip() == "None":
                    sesh.rippleDetectionTetrodes = None
                else:
                    sesh.rippleDetectionTetrodes = [int(fieldVal)]
            elif fieldName.lower() == "baseline":
                if fieldVal.strip() == "None":
                    sesh.rippleBaselineTetrode = None
                else:
                    sesh.rippleBaselineTetrode = int(fieldVal)
            elif fieldName.lower() == "last well":
                if fieldVal.strip() == "None":
                    sesh.foundFirstHome = False
                    sesh.endedOnHome = False
                else:
                    sesh.foundFirstHome = True
                    if 'h' in fieldVal.lower():
                        sesh.endedOnHome = True
                    elif 'a' in fieldVal.lower():
                        sesh.endedOnHome = False
                    else:
                        print("\tCouldn't recognize last well {} in file {}".format(
                            fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "iti stim on":
                if 'Y' in fieldVal:
                    sesh.ItiStimOn = True
                elif 'N' in fieldVal:
                    sesh.ItiStimOn = False
                else:
                    print("\tCouldn't recognize ITI Stim condition {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "probe stim on":
                if 'Y' in fieldVal:
                    sesh.probeStimOn = True
                elif 'N' in fieldVal:
                    sesh.probeStimOn = False
                else:
                    print("\tCouldn't recognize Probe Stim condition {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "probe performed":
                if 'Y' in fieldVal:
                    sesh.probePerformed = True
                elif 'N' in fieldVal:
                    sesh.probePerformed = False
                    if sesh.animalName == "Martin":
                        raise Exception(
                            "I thought all martin runs had a probe")
                else:
                    print("\tCouldn't recognize Probe performed val {} in file {}".format(
                        fieldVal, sesh.infoFileName))
            elif fieldName.lower() == "task ended at":
                sesh.btEndedAtWell = int(fieldVal)
            elif fieldName.lower() == "probe ended at":
                sesh.probeEndedAtWell = int(fieldVal)
            elif fieldName.lower() == "weight":
                sesh.weight = float(fieldVal)
            elif fieldName.lower() == "conditiongroup":
                cgs = fieldVal.strip().split("-")
                if not (len(cgs) == 2 and ("B" + cgs[0] == sesh.animalName or cgs[0] == sesh.animalName)):
                    print(fieldVal, cgs, sesh.animalName)
                    assert False
                sesh.conditionGroup = fieldVal
            elif fieldName.lower() == "probe home fill time":
                sesh.probeFillTime = int(fieldVal)
            elif fieldName.lower() == "trodes lights skip start":
                sesh.trodesLightsIgnoreSeconds = float(fieldVal)
            else:
                sesh.notes.append(line)

    if sesh.probePerformed is None and sesh.animalName == "B12":
        sesh.probePerformed = True

    if sesh.probePerformed and sesh.probeEndedAtWell is None:
        raise Exception(
            "Didn't mark the probe end well for sesh {}".format(sesh.name))

    if not (sesh.lastAwayWell == sesh.awayWells[-1] and sesh.endedOnHome) and sesh.btEndedAtWell is None:
        raise Exception(
            "Didn't find all the wells but task end well not marked for sesh {}".format(sesh.name))

    if sesh.prevInfoFileName is not None:
        with open(sesh.prevInfoFileName, 'r') as f:
            sesh.prevSessionInfoParsed = True
            lines = f.readlines()
            for line in lines:
                lineparts = line.split(":")
                if len(lineparts) != 2:
                    continue

                fieldName = lineparts[0]
                fieldVal = lineparts[1]

                if fieldName.lower() == "home":
                    sesh.prevSessionHome = int(fieldVal)
                elif fieldName.lower() == "aways":
                    sesh.prevSessionAways = [int(w)
                                             for w in fieldVal.strip().split(' ')]
                elif fieldName.lower() == "condition":
                    typeIn = fieldVal.lower()
                    if 'ripple' in typeIn or 'interruption' in typeIn:
                        # sesh.prevSessionIsRippleInterruption = True
                        sesh.prevSessionCondition = BTSession.CONDITION_INTERRUPTION
                    elif 'none' in typeIn:
                        # sesh.prevSessionIsNoInterruption = True
                        sesh.prevSessionCondition = BTSession.CONDITION_NO_STIM
                    elif 'delay' in typeIn:
                        # sesh.prevSessionIsDelayedInterruption = True
                        sesh.prevSessionCondition = BTSession.CONDITION_DELAY
                    else:
                        print("\tCouldn't recognize Condition {} in file {}".format(
                            typeIn, sesh.prevInfoFileName))
                elif fieldName.lower() == "thresh":
                    if "low" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = 2.5
                    elif "high" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = 4
                    elif "med" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = 3
                    elif "none" in fieldVal.lower():
                        sesh.prevSessionRippleDetectionThreshold = None
                    else:
                        sesh.prevSessionRippleDetectionThreshold = float(
                            fieldVal)
                elif fieldName.lower() == "last away":
                    if 'none' in fieldVal.lower():
                        sesh.prevSessionLastAwayWell = None
                    else:
                        sesh.prevSessionLastAwayWell = int(fieldVal)
                elif fieldName.lower() == "last well":
                    if 'h' in fieldVal.lower():
                        sesh.prevSessionEndedOnHome = True
                    elif 'a' in fieldVal.lower():
                        sesh.prevSessionEndedOnHome = False
                    elif 'none' in fieldVal.lower():
                        sesh.prevSessionEndedOnHome = None
                    else:
                        print("\tCouldn't recognize last well {} in file {}".format(
                            fieldVal, sesh.prevInfoFileName))
                elif fieldName.lower() == "iti stim on":
                    if 'Y' in fieldVal:
                        sesh.prevSessionItiStimOn = True
                    elif 'N' in fieldVal:
                        sesh.prevSessionItiStimOn = False
                    else:
                        print("\tCouldn't recognize ITI Stim condition {} in file {}".format(
                            fieldVal, sesh.prevInfoFileName))
                elif fieldName.lower() == "probe stim on":
                    if 'Y' in fieldVal:
                        sesh.prevSessionProbeStimOn = True
                    elif 'N' in fieldVal:
                        sesh.prevSessionProbeStimOn = False
                    else:
                        print("\tCouldn't recognize Probe Stim condition {} in file {}".format(
                            fieldVal, sesh.prevInfoFileName))
                else:
                    pass

    else:
        sesh.prevSessionInfoParsed = False

    if not sesh.importOptions["justExtractData"]:
        assert sesh.homeWell != 0

    if sesh.foundWells is None:
        sesh.foundWells = generateFoundWells(
            sesh.homeWell, sesh.awayWells, sesh.lastAwayWell, sesh.endedOnHome, sesh.foundFirstHome)


def parseDLCData(sesh):
    btPosFileName = sesh.animalInfo.DLC_dir + sesh.name + "_task.npz"
    probePosFileName = sesh.animalInfo.DLC_dir + sesh.name + "_probe.npz"
    if not os.path.exists(btPosFileName) or not os.path.exists(probePosFileName):
        print(f"\t{sesh.name} has no DLC file!")
        return None

    wellCoordsFileName = sesh.fileStartString + '.1.wellLocations_dlc.csv'
    if not os.path.exists(wellCoordsFileName):
        wellCoordsFileName = os.path.join(
            sesh.animalInfo.data_dir, 'well_locations_dlc.csv')
        print("\tSpecific well locations not found, falling back to file {}".format(
            wellCoordsFileName))
    wellCoordsMap = readWellCoordsFile(wellCoordsFileName)

    btPos = np.load(btPosFileName)
    sesh.btPosXs, sesh.btPosYs, sesh.btPos_ts = \
        processPosData(btPos["x1"], btPos["y1"],
                       btPos["timestamp"], wellCoordsMap, xLim=None, yLim=None, smooth=3)

    probePos = np.load(probePosFileName)
    sesh.probePosXs, sesh.probePosYs, sesh.probePos_ts = \
        processPosData(probePos["x1"], probePos["y1"],
                       probePos["timestamp"], wellCoordsMap, xLim=None, yLim=None, smooth=3)

    sesh.hasPositionData = True

    xs = list(np.hstack((sesh.btPosXs, sesh.probePosXs)))
    ys = list(np.hstack((sesh.btPosYs, sesh.probePosYs)))
    ts = list(np.hstack((sesh.btPos_ts, sesh.probePos_ts)))

    return xs, ys, ts, wellCoordsMap


def cleanupPos(tpts, xPos, yPos, xLim=None, yLim=None, excludeBoxes=None,
               maxJumpDistance=None, makePlots=False, minCleanTimeFrames=None):
    # only in bounds points pls
    pointsInRange = np.ones_like(xPos).astype(bool)
    if xLim is not None:
        pointsInRange &= (xPos > xLim[0]) & (xPos < xLim[1])
    if yLim is not None:
        pointsInRange &= (yPos > yLim[0]) & (yPos < yLim[1])
    xPos[~ pointsInRange] = np.nan
    yPos[~ pointsInRange] = np.nan
    if makePlots:
        quickPosPlot(tpts, xPos, yPos, "in bounds only")

    # exclude boxes
    if excludeBoxes is not None:
        for x1, y1, x2, y2 in excludeBoxes:
            inBox = (xPos > x1) & (xPos < x2) & (yPos > y1) & (yPos < y2)
            xPos[inBox] = np.nan
            yPos[inBox] = np.nan
        if makePlots:
            quickPosPlot(tpts, xPos, yPos, "excluded boxes")

    # Remove large jumps in position (tracking errors)
    if maxJumpDistance is not None:
        jumpDistance = np.sqrt(np.square(np.diff(xPos, prepend=xPos[0])) +
                               np.square(np.diff(yPos, prepend=yPos[0])))
        cleanPoints = jumpDistance < maxJumpDistance
        xPos[~ cleanPoints] = np.nan
        yPos[~ cleanPoints] = np.nan
        if makePlots:
            quickPosPlot(tpts, xPos, yPos, "no jumps (single)")

    # Dilate the excluded parts but only inward toward the noise
    if minCleanTimeFrames is not None:
        nanidx = np.argwhere(np.isnan(xPos)).reshape(-1)
        nidiff = np.diff(nanidx)
        takeout = np.argwhere((nidiff > 1) & (nidiff < minCleanTimeFrames))
        for t in takeout:
            xPos[nanidx[t][0]:nanidx[t+1][0]] = np.nan
            yPos[nanidx[t][0]:nanidx[t+1][0]] = np.nan
        if makePlots:
            quickPosPlot(tpts, xPos, yPos, "no jumps (dilated)")


def interpNanPositions(tpts, xPos, yPos):
    nanpos = np.isnan(xPos)
    ynanpos = np.isnan(yPos)
    assert all(ynanpos == nanpos)
    notnanpos = np.logical_not(nanpos)
    xPos = np.interp(tpts, tpts[notnanpos], xPos[notnanpos])
    yPos = np.interp(tpts, tpts[notnanpos], yPos[notnanpos])
    return xPos, yPos


def getWellCameraCoordinates(well_num: int, well_coords_map: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    return well_coords_map[str(well_num)]


def correctFishEye(wellCoordsMap: Dict[str, Tuple[int, int]], xs, ys):
    # returns position given by xs, ys in the context of sesh but corrected so
    # well locations would lay on a grid at x and y values 0.5, 1.5, ... 5.5

    # Here's the lattice we're mapping to, including imaginary wells that encircle the real wells
    xv, yv = np.meshgrid(np.linspace(-0.5, 6.5, 8), np.linspace(-0.5, 6.5, 8))
    xpts = xv.reshape((-1))
    ypts = yv.reshape((-1))

    def getWellCoords(xp, yp):
        if xp > 0 and xp < 6 and yp > 0 and yp < 6:
            wellName = int(2 + (xp - 0.5) + 8 * (yp - 0.5))
            return np.array(getWellCameraCoordinates(wellName, wellCoordsMap))
        if xp > 0 and xp < 6:
            # vertical extrapolation. p1 next well, p2 is further
            if yp > 6:
                p1 = getWellCoords(xp, yp-1)
                p2 = getWellCoords(xp, yp-2)
            else:
                p1 = getWellCoords(xp, yp+1)
                p2 = getWellCoords(xp, yp+2)
        else:
            if xp > 6:
                p1 = getWellCoords(xp-1, yp)
                p2 = getWellCoords(xp-2, yp)
            else:
                p1 = getWellCoords(xp+1, yp)
                p2 = getWellCoords(xp+2, yp)

        return p1 + (p1 - p2)

    cameraWellLocations = np.array([getWellCoords(x, y) for x, y in zip(xpts, ypts)])
    pathCoords = np.vstack((np.array(xs), np.array(ys))).T
    xres = griddata(cameraWellLocations, xpts, pathCoords, method="cubic")
    yres = griddata(cameraWellLocations, ypts, pathCoords, method="cubic")

    return xres, yres


def integrateCorrectionFiles(ts, xs, ys, correctionDirectory, wellCoordsMap: Dict[str, Tuple[int, int]],
                             minCorrectionSecs=1.0):
    validTpts = ts[~np.isnan(xs)]
    notnandiff = np.diff(validTpts)
    MIN_CORRECTION_TS = minCorrectionSecs * TRODES_SAMPLING_RATE
    needsCorrection = np.argwhere((notnandiff >= MIN_CORRECTION_TS)).reshape(-1)
    correctionRegions = []
    correctedFlag = []
    # print("\tLarge gaps that are being interped:")
    for nc in needsCorrection:
        t1 = int(validTpts[nc])
        t2 = int(validTpts[nc+1])

        timeStr1 = timeStrForTrodesTimestamp(t1)
        timeStr2 = timeStrForTrodesTimestamp(t2)

        entry = (
            t1,
            t2,
            timeStr1,
            timeStr2,
            (t2-t1) / TRODES_SAMPLING_RATE
        )
        correctionRegions.append(entry)
        correctedFlag.append(False)

        # print("\t" + "\t".join([str(s) for s in entry]))

    if correctionDirectory is None:
        response = input("No corrections provided, interp all these gaps anyway (y/N)?")
        if response != "y":
            exit()
        return

    if not os.path.exists(correctionDirectory):
        os.makedirs(correctionDirectory)

    gl = correctionDirectory + '/*.videoPositionTracking'
    cfiles = glob.glob(gl)
    # print(gl, cfiles)
    numCorrectionsIntegrated = 0
    for cf in cfiles:
        _, posdata = readRawPositionData(cf)
        ct = np.array(posdata["timestamp"]).astype(float)
        cx = np.array(posdata["x1"]).astype(float)
        cy = np.array(posdata["y1"]).astype(float)
        cx, cy = correctFishEye(wellCoordsMap, cx, cy)

        # Seems like it includes all the video timestamps, but just extends wherever tracking happened for
        # some reason
        cd = np.abs(np.diff(cx, prepend=cx[0])) + np.abs(np.diff(cy, prepend=cy[0]))
        nzcd = np.nonzero(cd)[0]
        ci1 = nzcd[0]
        ci2 = nzcd[-1]
        if False:
            quickPosPlot(ct, cx, cy, "incoming correction points")
            quickPosPlot(
                ct, cx, cy, "incoming correction points (just with tracking data)", irange=(ci1, ci2))
        ct = ct[ci1:ci2]
        cx = cx[ci1:ci2]
        cy = cy[ci1:ci2]
        # quickPosPlot(ct, cx, cy, "incoming correction points")

        posStart_ts = ct[0]
        posEnd_ts = ct[-1]
        # print(f"\tchecking {cf}\n\t\t{posStart_ts} - {posEnd_ts}")
        for entryi, entry in enumerate(correctionRegions):
            t1 = entry[0]
            t2 = entry[1]
            # print("\t\tagainst entry\t", t1, t2)
            if t1 > posStart_ts and t2 < posEnd_ts:
                correctedFlag[entryi] = True
                numCorrectionsIntegrated += 1
                # print("\tfound correction for", "\t".join([str(s) for s in entry]))
                # integrate this bit

                tpi1 = np.searchsorted(ts, t1)
                tpi2 = np.searchsorted(ts, t2)
                cpi1 = np.searchsorted(ct, t1)
                cpi2 = np.searchsorted(ct, t2)

                # MARGIN = 20
                # quickPosPlot(ct, cx, cy, "incoming correction points, full")
                # quickPosPlot(ts, xs, ys, "correction region original",
                #              irange=(max(0, tpi1 - MARGIN), min(len(ts)-1, tpi2 + MARGIN)))
                # quickPosPlot(ct, cx, cy, "incoming correction points", irange=(cpi1, cpi2))
                for t in ct:
                    assert t in ts
                for ci, pi in zip(range(cpi1, cpi2), range(tpi1, tpi2)):
                    if np.isnan(xs[pi]):
                        xs[pi] = cx[ci]
                        ys[pi] = cy[ci]
                # quickPosPlot(ts, xs, ys, "correction region integrated",
                # irange=(max(0, tpi1 - MARGIN), min(len(ts)-1, tpi2 + MARGIN)))

    print(f"\tCorrected {numCorrectionsIntegrated} regions with corrections files")
    # print("\tRemaining regions that are uncorrected:")
    lastEnd = None
    lastStart = None
    optimizedTimes = []
    SPLITGAP = 90
    COMBINEGAP = 30
    MAXREC = 60*5
    for entryi, entry in enumerate(correctionRegions):
        if not correctedFlag[entryi]:
            # print("\t" + "\t".join([str(s) for s in entry]))
            # oe format: (start time str, end time str, num included regions, length of biggest gap, gap start, gap end)

            if lastEnd is not None and (
                (entry[0] - lastEnd) / TRODES_SAMPLING_RATE < COMBINEGAP or
                ((entry[0] - lastEnd) / TRODES_SAMPLING_RATE < SPLITGAP and
                    (entry[1] - lastStart) / TRODES_SAMPLING_RATE < MAXREC)):
                # combine this with last entry
                currentOptimizedEntry = optimizedTimes[-1]
                gapLen = entry[0] - lastEnd
                if gapLen > currentOptimizedEntry[3]:
                    gapStart = currentOptimizedEntry[1]
                    gapEnd = entry[2]
                else:
                    gapLen = currentOptimizedEntry[3]
                    gapStart = currentOptimizedEntry[4]
                    gapEnd = currentOptimizedEntry[5]
                optimizedTimes[-1] = (currentOptimizedEntry[0], entry[3],
                                      currentOptimizedEntry[2]+1, gapLen, gapStart, gapEnd)
                lastEnd = entry[1]
            else:
                # new entry
                lastStart = entry[0]
                lastEnd = entry[1]
                optimizedTimes.append((entry[2], entry[3], 1, -1, None, None))

    print("\toptimized corrections = ")
    for oe in optimizedTimes:
        print(f"\t\t({oe[2]}) {oe[0]}\t{oe[1]}\t\t{oe[4]}\t{oe[5]}")

    correctionsFileName = os.path.join(correctionDirectory, "optimized.txt")
    # print(f"\tsaving optimized list to file {correctionsFileName}")
    print("\tsaving optimized list to file")
    with open(correctionsFileName, 'w') as f:
        f.writelines([f"{oe[0]} - {oe[1]}\n" for oe in optimizedTimes])

    # TODO it'd be helpful to return all the uncorrected entries (not optimized), filter based on clips
    # so can exclude ones during ITI or before or after behavior, then make optimized times and save them


def processPosData(x, y, t, wellCoordsMap: Dict[str, Tuple[int, int]], maxJumpDistance=0.25,
                   xLim=(100, 1050), yLim=(20, 900), smooth=None,
                   excludeBoxes=None, correctionDirectory=None, minCleanTimeFrames=15):
    xPos = np.array(x, dtype=float)
    yPos = np.array(y, dtype=float)

    # Interpolate the position data into evenly sampled time points

    tpts = np.array(t).astype(float)
    xPos = np.array(x).astype(float)
    yPos = np.array(y).astype(float)

    # Remove out of bounds and excluded position points
    cleanupPos(tpts, xPos, yPos, xLim=xLim, yLim=yLim, excludeBoxes=excludeBoxes)

    # Run fisheye correction
    xPos, yPos = correctFishEye(wellCoordsMap, xPos, yPos)

    # Exclude large jumps
    cleanupPos(tpts, xPos, yPos, maxJumpDistance=maxJumpDistance,
               minCleanTimeFrames=minCleanTimeFrames)

    # Now check for large gaps that were removed and fill in with correction files
    integrateCorrectionFiles(tpts, xPos, yPos, correctionDirectory, wellCoordsMap)

    # Now interp all remaining unknown positions
    xPos, yPos = interpNanPositions(tpts, xPos, yPos)

    # quickPosPlot(tpts, xPos, yPos, "interp")

    if smooth is not None:
        xPos = gaussian_filter1d(xPos, smooth)
        yPos = gaussian_filter1d(yPos, smooth)

        # quickPosPlot(tpts, xPos, yPos, "smooth")

    return xPos, yPos, tpts


def getWellEntryAndExitTimes(nearestWells, ts, quads=False, includeNeighbors=False):
    entryTimes = []
    exitTimes = []
    entryIdxs = []
    exitIdxs = []

    ts = np.append(np.array(ts), ts[-1] + (ts[-1] - ts[-2]))
    if quads:
        wellIdxs = [0, 1, 2, 3]
    else:
        wellIdxs = range(48)

    for wi in wellIdxs:
        # last data point should count as an exit, so appending a false
        # same for first point should count as entry, prepending
        if includeNeighbors:
            neighbors = list({wi, wi - 1, wi + 1, wi - 7, wi - 8, wi - 9,
                              wi + 7, wi + 8, wi + 9}.intersection(allWellNames))
            nearWell = np.concatenate(
                ([False], np.isin(nearestWells, neighbors), [False]))
        else:
            nearWell = np.concatenate(([False], nearestWells == wi, [False]))
        idx = np.argwhere(np.diff(np.array(nearWell, dtype=float)) == 1)
        # removing the -1 here so it's now exclusive exit time (exit and entrance will be the same, not 1 off anymore)
        idx2 = np.argwhere(np.diff(np.array(nearWell, dtype=float)) == -1)
        entryIdxs.append(idx.T[0])
        exitIdxs.append(idx2.T[0])
        entryTimes.append(ts[idx.T[0]])
        exitTimes.append(ts[idx2.T[0]])

    return entryIdxs, exitIdxs, entryTimes, exitTimes


def getNearestWell(xs, ys, well_idxs=allWellNames, switchWellFactor=0.8):
    # switchWellFactor of 0.8 means transition from well a -> b requires rat dist to b to be 0.8 * dist to a
    xv, yv = np.meshgrid(np.linspace(0.5, 5.5, 6), np.linspace(0.5, 5.5, 6))
    xpts = xv.reshape((-1, 1))
    ypts = yv.reshape((-1, 1))
    keepWell = np.array([wn in well_idxs for wn in allWellNames])
    xpts = xpts[keepWell]
    ypts = ypts[keepWell]
    well_coords = np.hstack((xpts, ypts))

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
            if delta[i, nearest_well[i]] < switchWellFactor * delta[i, curr_well]:
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


def loadPositionData(sesh: BTSession):
    sesh.hasPositionData = False
    positionData = None

    # First try DeepLabCut
    if sesh.animalInfo.DLC_dir is not None:
        print(f"\tLoading from DLC, DLC_dir = {sesh.animalInfo.DLC_dir}")
        positionData = parseDLCData(sesh)
        if positionData is None:
            assert sesh.positionFromDeepLabCut is None or not sesh.positionFromDeepLabCut
            sesh.positionFromDeepLabCut = False
        else:
            assert sesh.positionFromDeepLabCut is None or sesh.positionFromDeepLabCut
            sesh.positionFromDeepLabCut = True
            xs, ys, ts, wellCoordsMap = positionData
            positionDataMetadata = {}
            sesh.hasPositionData = True
            raise Exception("Unimplemented new position standards for DLC (scale, fisheye, etc)")
    else:
        sesh.positionFromDeepLabCut = False

    if positionData is None:
        assert not sesh.positionFromDeepLabCut
        print("\tUsing standard tracking, not DLC")
        trackingFile = sesh.fileStartString + '.1.videoPositionTracking'
        possibleOtherTrackingFile = sesh.fileStartString + '.1'
        if not os.path.exists(trackingFile) and os.path.exists(possibleOtherTrackingFile):
            print(
                f"\tFound a possible tracking file {possibleOtherTrackingFile}\n\tand no file {trackingFile}")
            if sesh.importOptions["skipConfirmTrackingFileExtension"]:
                print("\tAdding file extension automatically")
                renameFile = True
            else:
                response = input(
                    "Would you like to rename the .1 file to a .1.videoPositionTracking file (y/N)?")
                renameFile = response.strip() == "y"

            if renameFile:
                os.rename(possibleOtherTrackingFile, trackingFile)

        if os.path.exists(trackingFile):
            positionDataMetadata, positionData = readRawPositionData(
                trackingFile)

            if sesh.importOptions["forceAllTrackingAuto"] and \
                ("source" not in positionDataMetadata or
                 "trodescameraextractor" not in positionDataMetadata["source"]):
                print("\tForcing all tracking to be auto")
                positionData = None
                os.rename(sesh.fileStartString + '.1.videoPositionTracking', sesh.fileStartString +
                          '.1.videoPositionTracking.manualOutput')
            else:
                if "source" in positionDataMetadata and \
                        "trodescameraextractor" in positionDataMetadata["source"]:
                    print(
                        f"WARNING!!! {sesh.name} tracking was done with my script, not camera module")
                print("\tFound existing tracking data")
                sesh.frameTimes = positionData['timestamp']
                correctionDirectory = os.path.join(os.path.dirname(
                    sesh.fileStartString), "trackingCorrections")
                # print(f"\tcorrectionDirectory is {correctionDirectory} ")

                wellCoordsFileName = sesh.fileStartString + '.1.wellLocations.csv'
                if not os.path.exists(wellCoordsFileName):
                    wellCoordsFileName = os.path.join(
                        sesh.animalInfo.data_dir, 'well_locations.csv')
                    print("\tSpecific well locations not found, falling back to file {}".format(
                        wellCoordsFileName))
                wellCoordsMap = readWellCoordsFile(wellCoordsFileName)

                xs, ys, ts = processPosData(positionData["x1"], positionData["y1"], positionData["timestamp"],
                                            wellCoordsMap,
                                            xLim=(sesh.animalInfo.X_START,
                                                  sesh.animalInfo.X_FINISH),
                                            yLim=(sesh.animalInfo.Y_START,
                                                  sesh.animalInfo.Y_FINISH),
                                            excludeBoxes=sesh.animalInfo.excludeBoxes,
                                            correctionDirectory=correctionDirectory)
                sesh.hasPositionData = True
        else:
            positionData = None

        if positionData is None:
            raise Exception("Do everything by hand now!")
            print("\trunning trodes camera extractor!")
            processRawTrodesVideo(sesh.fileStartString + '.1.h264')
            positionDataMetadata, positionData = readRawPositionData(
                trackingFile)
            sesh.frameTimes = positionData['timestamp']
            xs, ys, ts = processPosData(positionData["x1"], positionData["y1"], positionData["timestamp"],
                                        wellCoordsMap,
                                        xLim=(sesh.animalInfo.X_START,
                                              sesh.animalInfo.X_FINISH),
                                        yLim=(sesh.animalInfo.Y_START,
                                              sesh.animalInfo.Y_FINISH),
                                        excludeBoxes=sesh.animalInfo.excludeBoxes)
            sesh.hasPositionData = True

    if not sesh.hasPositionData:
        print(f"!!!!!!!\tWARNING: {sesh.name} has no position data\t!!!!!!")
        return

    if sesh.importOptions["skipUSB"]:
        print("!!!!!!\tWARNING: Skipping USB\t!!!!!!")
    else:
        if "lightonframe" in positionDataMetadata:
            sesh.trodesLightOnFrame = int(
                positionDataMetadata['lightonframe'])
            sesh.trodesLightOffFrame = int(
                positionDataMetadata['lightoffframe'])
            print("\tmetadata says trodes light frames {}, {} (/{})".format(sesh.trodesLightOffFrame,
                                                                            sesh.trodesLightOnFrame, len(ts)))
            sesh.trodesLightOnTime = sesh.frameTimes[sesh.trodesLightOnFrame]
            sesh.trodesLightOffTime = sesh.frameTimes[sesh.trodesLightOffFrame]
            print("\tmetadata file says trodes light timestamps {}, {} (/{})".format(
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
                print("\tjustlights file says trodes light timestamps {}, {} (/{})".format(
                    sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))
            else:
                print(f"\tdoing the lights with file {sesh.fileStartString + '.1.h264'}")
                sesh.trodesLightOffTime, sesh.trodesLightOnTime = getTrodesLightTimes(
                    sesh.fileStartString + '.1.h264', showVideo=False,
                    ignoreFirstSeconds=sesh.trodesLightsIgnoreSeconds)
                print("\ttrodesLightFunc says trodes light Time {}, {} (/{})".format(
                    sesh.trodesLightOffTime, sesh.trodesLightOnTime, len(ts)))

        possibleDirectories = [
            f"/media/WDC8/videos/labvideos/trimmed/{sesh.animalName}/",
            "/media/WDC6/videos/B16-20/trimmed/{}/".format(sesh.animalName),
            "/media/WDC7/videos/B16-20/trimmed/{}/".format(sesh.animalName),
            "/media/WDC8/videos/B16-20/trimmed/{}/".format(sesh.animalName),
            "/media/WDC6/{}/".format(sesh.animalName),
            "/media/WDC7/{}/".format(sesh.animalName),
            "/media/WDC8/{}/".format(sesh.animalName),
            "/media/WDC8/videos/",
            "/media/WDC7/videos/B16-20/",
            "/media/WDC4/lab_videos",
            f"/home/wcroughan/data/videos/{sesh.animalName}/"
        ]
        useSeshIdxDirectly = sesh.animalName == "B18" or sesh.dateStr > "20221100"
        sesh.usbVidFile = getUSBVideoFile(
            sesh.name, possibleDirectories, seshIdx=sesh.seshIdx, useSeshIdxDirectly=useSeshIdxDirectly)
        if sesh.usbVidFile is None:
            print("??????? usb vid file not found for session ", sesh.name)
            print(possibleDirectories, sesh.seshIdx, useSeshIdxDirectly)
            assert False
        else:
            print("\tRunning USB light time analysis, file", sesh.usbVidFile)
            sesh.usbLightOffFrame, sesh.usbLightOnFrame = processUSBVideoData(
                sesh.usbVidFile, overwriteMode="loadOld", showVideo=False)
            # if sesh.usbLightOffFrame is None or sesh.usbLightOnFrame is None:
            #     raise Exception("exclude me pls")

        if not sesh.importOptions["justExtractData"] or sesh.importOptions["runInteractiveExtraction"]:
            clipsFileName = sesh.fileStartString + '.1.clips'
            if not os.path.exists(clipsFileName) and len(sesh.foundWells) > 0:
                print("\tclips file not found, gonna launch clips generator")

                allPosNearestWells = getNearestWell(xs, ys)
                _, _, allPosWellEntryTimes, allPosWellExitTimes = \
                    getWellEntryAndExitTimes(
                        allPosNearestWells, ts)

                ann = AnnotatorWindow(xs, ys, ts, sesh.trodesLightOffTime, sesh.trodesLightOnTime,
                                      sesh.usbVidFile, allPosWellEntryTimes, allPosWellExitTimes,
                                      sesh.foundWells, not sesh.probePerformed, sesh.fileStartString + '.1',
                                      wellCoordsMap)
                ann.resize(1600, 800)
                ann.show()
                importParentApp.exec()

    if sesh.separateProbeFile:
        probeFileStr = os.path.join(
            sesh.probeDir, os.path.basename(sesh.probeDir))
        btTimeClips = readClipData(
            sesh.fileStartString + '.1.clips')[0]
        probeTimeClips = readClipData(probeFileStr + '.1.clips')[0]
    else:
        timeClips = readClipData(sesh.fileStartString + '.1.clips')
        if timeClips is None:
            btTimeClips = None
            if sesh.probePerformed:
                probeTimeClips = None
        else:
            btTimeClips = timeClips[0]
            if sesh.probePerformed:
                probeTimeClips = timeClips[1]

    if btTimeClips is None:
        print("\t!!WARNING: Clips couldn't be loaded")
        if sesh.importOptions["justExtractData"]:
            return
        else:
            raise FileNotFoundError()

    print("\tclips: {} (/{})".format(btTimeClips, len(ts)))
    btStart_posIdx = np.searchsorted(ts, btTimeClips[0])
    btEnd_posIdx = np.searchsorted(ts, btTimeClips[1])
    sesh.btPosXs = xs[btStart_posIdx:btEnd_posIdx]
    sesh.btPosYs = ys[btStart_posIdx:btEnd_posIdx]
    sesh.btPos_ts = ts[btStart_posIdx:btEnd_posIdx]

    if sesh.probePerformed:
        if sesh.separateProbeFile:
            assert not sesh.positionFromDeepLabCut
            _, positionData = readRawPositionData(
                probeFileStr + '.1.videoPositionTracking')
            xs, ys, ts = processPosData(positionData["x1"], positionData["y1"], positionData["timestamp"],
                                        wellCoordsMap,
                                        xLim=(sesh.animalInfo.X_START,
                                              sesh.animalInfo.X_FINISH),
                                        yLim=(sesh.animalInfo.Y_START,
                                              sesh.animalInfo.Y_FINISH),
                                        excludeBoxes=sesh.animalInfo.excludeBoxes)

        probeStart_posIdx = np.searchsorted(ts, probeTimeClips[0])
        probeEnd_posIdx = np.searchsorted(ts, probeTimeClips[1])
        sesh.probePosXs = xs[probeStart_posIdx:probeEnd_posIdx]
        sesh.probePosYs = ys[probeStart_posIdx:probeEnd_posIdx]
        sesh.probePos_ts = ts[probeStart_posIdx:probeEnd_posIdx]

    rewardClipsFile = sesh.fileStartString + '.1.rewardClips'
    if not os.path.exists(rewardClipsFile):
        rewardClipsFile = sesh.fileStartString + '.1.rewardclips'
    if not os.path.exists(rewardClipsFile):
        if sesh.numHomeFound > 0:
            print("!!!!!!\tWell find times not marked for session {}\t!!!!!!!".format(sesh.name))
            sesh.hasRewardFindTimes = False
        else:
            sesh.hasRewardFindTimes = True
    else:
        sesh.hasRewardFindTimes = True
        wellVisitTimes = readClipData(rewardClipsFile)
        sesh.homeRewardEnter_ts = wellVisitTimes[::2, 0]
        sesh.homeRewardExit_ts = wellVisitTimes[::2, 1]
        sesh.awayRewardEnter_ts = wellVisitTimes[1::2, 0]
        sesh.awayRewardExit_ts = wellVisitTimes[1::2, 1]

        sesh.homeRewardEnter_posIdx = np.searchsorted(
            sesh.btPos_ts, sesh.homeRewardEnter_ts)
        sesh.homeRewardExit_posIdx = np.searchsorted(
            sesh.btPos_ts, sesh.homeRewardExit_ts)
        sesh.awayRewardEnter_posIdx = np.searchsorted(
            sesh.btPos_ts, sesh.awayRewardEnter_ts)
        sesh.awayRewardExit_posIdx = np.searchsorted(
            sesh.btPos_ts, sesh.awayRewardExit_ts)


def loadLFPData(sesh):
    lfpData = []

    if len(sesh.rippleDetectionTetrodes) == 0:
        sesh.rippleDetectionTetrodes = [sesh.animalInfo.DEFAULT_RIP_DET_TET]

    for i in range(len(sesh.rippleDetectionTetrodes)):
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print("\t" + lfpdir, "\tdoesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            elif os.path.exists("/home/wcroughan/Software/Trodes21/linux/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            # elif os.path.exists("/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP"):
            #     syscmd = "/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP -rec " + \
            #         sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            print("\t" + syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + sesh.name + ".LFP_nt" + \
            str(sesh.rippleDetectionTetrodes[i]) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.btLfpFnames.append(lfpfilename)
        lfpData.append(MountainViewIO.loadLFP(
            data_file=sesh.btLfpFnames[-1]))

    if sesh.rippleBaselineTetrode is None:
        sesh.rippleBaselineTetrode = sesh.animalInfo.DEFAULT_RIP_BAS_TET

    # I think Martin didn't have this baseline tetrode? Need to check
    if sesh.rippleBaselineTetrode is not None:
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print("\t" + lfpdir, "\tdoesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            # elif os.path.exists("/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP"):
            #     syscmd = "/home/wcroughan/Software/Trodes_2-3-2_Ubuntu2004/exportLFP -rec " + \
            #         sesh.fileStartString + ".rec"
            elif os.path.exists("/home/wcroughan/Software/Trodes21/linux/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804/exportLFP -rec " + \
                    sesh.fileStartString + ".rec"
            print("\t" + syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + sesh.name + ".LFP_nt" + \
            str(sesh.rippleBaselineTetrode) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.btLfpBaselineFname = lfpfilename
        baselineLfpData = MountainViewIO.loadLFP(
            data_file=sesh.btLfpBaselineFname)
    else:
        baselineLfpData = None

    return lfpData, baselineLfpData


def runLFPAnalyses(sesh, lfpData, baselineLfpData, showPlot=False):
    lfpV = lfpData[0][1]['voltage']
    lfp_ts = lfpData[0][0]['time']
    C = sesh.importOptions["consts"]

    # Deflections represent interruptions from stimulation, artifacts include these and also weird noise
    # Although maybe not really ... 2022-5-11 replacing this
    # lfp_deflections = signal.find_peaks(-lfpV, height=DEFLECTION_THRESHOLD_HI,
    # distance=MIN_ARTIFACT_DISTANCE)
    lfpDeflections = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=C["DEFLECTION_THRESHOLD_HI"], distance=C["MIN_ARTIFACT_DISTANCE"])
    interruptions_lfpIdx = lfpDeflections[0]
    sesh.interruption_ts = lfp_ts[interruptions_lfpIdx]
    sesh.interruptions_lfpIdx = interruptions_lfpIdx

    sesh.btInterruption_posIdx = np.searchsorted(sesh.btPos_ts, sesh.interruption_ts)
    sesh.btInterruption_posIdx = sesh.btInterruption_posIdx[sesh.btInterruption_posIdx < len(
        sesh.btPos_ts)]

    print("\tCondition - {}".format("SWR" if sesh.isRippleInterruption else "Ctrl"))
    lfpDeflections = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=C["DEFLECTION_THRESHOLD_LO"], distance=C["MIN_ARTIFACT_DISTANCE"])
    lfpArtifacts_lfpIdx = lfpDeflections[0]
    sesh.artifact_ts = lfp_ts[lfpArtifacts_lfpIdx]
    sesh.artifacts_lfpIdx = lfpArtifacts_lfpIdx

    btLfpStart_lfpIdx = np.searchsorted(lfp_ts, sesh.btPos_ts[0])
    btLfpEnd_lfpIdx = np.searchsorted(lfp_ts, sesh.btPos_ts[-1])
    btLFPData = lfpV[btLfpStart_lfpIdx:btLfpEnd_lfpIdx]
    btLfpArtifacts_lfpIdx = interruptions_lfpIdx - btLfpStart_lfpIdx
    btLfpArtifacts_lfpIdx = btLfpArtifacts_lfpIdx[btLfpArtifacts_lfpIdx > 0]
    sesh.btLfpArtifacts_lfpIdx = btLfpArtifacts_lfpIdx
    sesh.btLfpStart_lfpIdx = btLfpStart_lfpIdx
    sesh.btLfpEnd_lfpIdx = btLfpEnd_lfpIdx

    _, _, sesh.prebtMeanRipplePower, sesh.prebtStdRipplePower = getRipplePower(
        lfpV[0:btLfpStart_lfpIdx], omitArtifacts=False)
    _, ripplePower, _, _ = getRipplePower(
        btLFPData, lfpDeflections=btLfpArtifacts_lfpIdx, meanPower=sesh.prebtMeanRipplePower,
        stdPower=sesh.prebtStdRipplePower, showPlot=showPlot)
    sesh.btRipStartsPreStats_lfpIdx, sesh.btRipLensPreStats_lfpIdx, sesh.btRipPeaksPreStats_lfpIdx, \
        sesh.btRipPeakAmpsPreStats, sesh.btRipCrossThreshPreStats_lfpIdx = \
        detectRipples(ripplePower)
    if len(sesh.btRipStartsPreStats_lfpIdx) == 0:
        sesh.btRipStartsPreStats_ts = np.array([])
    else:
        sesh.btRipStartsPreStats_ts = lfp_ts[sesh.btRipStartsPreStats_lfpIdx + btLfpStart_lfpIdx]

    # preBtInterruptions_lfpIdx = interruptions_lfpIdx[interruptions_lfpIdx <
    #                                                  btLfpStart_lfpIdx]
    # _, _, sesh.prebtMeanRipplePowerArtifactsRemoved, sesh.prebtStdRipplePowerArtifactsRemoved = getRipplePower(
    #     lfpV[0:btLfpStart_lfpIdx], lfpDeflections=preBtInterruptions_lfpIdx)
    # _, _, sesh.prebtMeanRipplePowerArtifactsRemovedFirstHalf, sesh.prebtStdRipplePowerArtifactsRemovedFirstHalf = \
    #     getRipplePower(lfpV[0:int(btLfpStart_lfpIdx / 2)],
    #                    lfpDeflections=preBtInterruptions_lfpIdx)

    if sesh.probePerformed and not sesh.separateItiFile and not sesh.separateProbeFile:
        ITI_MARGIN = 10  # units: seconds
        ITI_INCLUDE_SECS = 60
        itiLfpEnd_ts = sesh.probePos_ts[0] - TRODES_SAMPLING_RATE * ITI_MARGIN
        itiLfpStart_ts = itiLfpEnd_ts - TRODES_SAMPLING_RATE * ITI_INCLUDE_SECS
        itiLfpStart_lfpIdx = np.searchsorted(lfp_ts, itiLfpStart_ts)
        itiLfpEnd_lfpIdx = np.searchsorted(lfp_ts, itiLfpEnd_ts)
        itiLFPData = lfpV[itiLfpStart_lfpIdx:itiLfpEnd_lfpIdx]
        sesh.itiLfpStart_lfpIdx = itiLfpStart_lfpIdx
        sesh.itiLfpStart_ts = itiLfpStart_ts
        sesh.itiLfpEnd_ts = itiLfpEnd_ts

        # in general none, but there's a few right at the start of where this is defined
        itiStims_lfpIdx = interruptions_lfpIdx - itiLfpStart_lfpIdx
        zeroIdx = np.searchsorted(itiStims_lfpIdx, 0)
        itiStims_lfpIdx = itiStims_lfpIdx[zeroIdx:]

        _, ripplePower, sesh.itiMeanRipplePower, sesh.itiStdRipplePower = getRipplePower(
            itiLFPData, lfpDeflections=itiStims_lfpIdx)
        sesh.itiRipStarts_lfpIdx, sesh.itiRipLens_lfpIdx, sesh.itiRipPeaks_lfpIdx, sesh.itiRipPeakAmps, \
            sesh.itiRipCrossThresh_lfpIdxs = detectRipples(ripplePower)
        if len(sesh.itiRipStarts_lfpIdx) > 0:
            sesh.itiRipStarts_ts = lfp_ts[sesh.itiRipStarts_lfpIdx + itiLfpStart_lfpIdx]
        else:
            sesh.itiRipStarts_ts = np.array([])

        sesh.itiDuration = (itiLfpEnd_ts - itiLfpStart_ts) / \
            TRODES_SAMPLING_RATE

        probeLfpStart_ts = sesh.probePos_ts[0]
        probeLfpEnd_ts = sesh.probePos_ts[-1]
        probeLfpStart_lfpIdx = np.searchsorted(lfp_ts, probeLfpStart_ts)
        probeLfpEnd_lfpIdx = np.searchsorted(lfp_ts, probeLfpEnd_ts)
        probeLFPData = lfpV[probeLfpStart_lfpIdx:probeLfpEnd_lfpIdx]
        sesh.probeRippleIdxOffset = probeLfpStart_lfpIdx
        sesh.probeLfpStart_ts = probeLfpStart_ts
        sesh.probeLfpEnd_ts = probeLfpEnd_ts
        sesh.probeLfpStart_lfpIdx = probeLfpStart_lfpIdx
        sesh.probeLfpEnd_lfpIdx = probeLfpEnd_lfpIdx

        _, ripplePower, sesh.probeMeanRipplePower, sesh.probeStdRipplePower = getRipplePower(
            probeLFPData, omitArtifacts=False)
        sesh.probeRipStarts_lfpIdx, sesh.probeRipLens_lfpIdx, sesh.probeRipPeakIdxs_lfpIdx, sesh.probeRipPeakAmps, \
            sesh.probeRipCrossThreshIdxs_lfpIdx = detectRipples(ripplePower)
        if len(sesh.probeRipStarts_lfpIdx) > 0:
            sesh.probeRipStarts_ts = lfp_ts[sesh.probeRipStarts_lfpIdx + probeLfpStart_lfpIdx]
        else:
            sesh.probeRipStarts_ts = np.array([])

        sesh.probeDuration = (probeLfpEnd_ts - probeLfpStart_ts) / \
            TRODES_SAMPLING_RATE

        _, itiRipplePowerProbeStats, _, _ = \
            getRipplePower(itiLFPData, lfpDeflections=itiStims_lfpIdx,
                           meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower)
        sesh.itiRipStartIdxsProbeStats_lfpIdx, sesh.itiRipLensProbeStats_lfpIdx, sesh.itiRipPeakIdxsProbeStats_lfpIdx, \
            sesh.itiRipPeakAmpsProbeStats, sesh.itiRipCrossThreshIdxsProbeStats_lfpIdx = \
            detectRipples(itiRipplePowerProbeStats)
        if len(sesh.itiRipStartIdxsProbeStats_lfpIdx) > 0:
            sesh.itiRipStartsProbeStats_ts = lfp_ts[
                sesh.itiRipStartIdxsProbeStats_lfpIdx + itiLfpStart_lfpIdx]
        else:
            sesh.itiRipStartsProbeStats_ts = np.array([])

        _, btRipplePowerProbeStats, _, _ = getRipplePower(
            btLFPData, lfpDeflections=btLfpArtifacts_lfpIdx, meanPower=sesh.probeMeanRipplePower,
            stdPower=sesh.probeStdRipplePower, showPlot=showPlot)
        sesh.btRipStartIdxsProbeStats_lfpIdx, sesh.btRipLensProbeStats_lfpIdx, sesh.btRipPeakIdxsProbeStats_lfpIdx, \
            sesh.btRipPeakAmpsProbeStats, sesh.btRipCrossThreshIdxsProbeStats_lfpIdx = \
            detectRipples(btRipplePowerProbeStats)
        if len(sesh.btRipStartIdxsProbeStats_lfpIdx) > 0:
            sesh.btRipStartsProbeStats_ts = lfp_ts[
                sesh.btRipStartIdxsProbeStats_lfpIdx + btLfpStart_lfpIdx]
        else:
            sesh.btRipStartsProbeStats_ts = np.array([])

        if sesh.btLfpBaselineFname is not None:
            # With baseline tetrode, calculated the way activelink does it
            lfpData = MountainViewIO.loadLFP(
                data_file=sesh.btLfpBaselineFname)
            baselfpV = lfpData[1]['voltage']
            # baselfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

            btRipPower, _, _, _ = getRipplePower(
                btLFPData, lfpDeflections=btLfpArtifacts_lfpIdx, meanPower=sesh.probeMeanRipplePower,
                stdPower=sesh.probeStdRipplePower, showPlot=showPlot)
            probeRipPower, _, _, _ = getRipplePower(
                probeLFPData, omitArtifacts=False)

            baselineProbeLFPData = baselfpV[probeLfpStart_lfpIdx:probeLfpEnd_lfpIdx]
            probeBaselinePower, _, baselineProbeMeanRipplePower, baselineProbeStdRipplePower = getRipplePower(
                baselineProbeLFPData, omitArtifacts=False)
            btBaselineLFPData = baselfpV[btLfpStart_lfpIdx:btLfpEnd_lfpIdx]
            btBaselineRipplePower, _, _, _ = getRipplePower(
                btBaselineLFPData, lfpDeflections=btLfpArtifacts_lfpIdx, meanPower=baselineProbeMeanRipplePower,
                stdPower=baselineProbeStdRipplePower,
                showPlot=showPlot)

            probeRawPowerDiff = probeRipPower - probeBaselinePower
            zmean = np.nanmean(probeRawPowerDiff)
            zstd = np.nanstd(probeRawPowerDiff)

            rawPowerDiff = btRipPower - btBaselineRipplePower
            zPowerDiff = (rawPowerDiff - zmean) / zstd

            sesh.btWithBaseRipStartIdx_lfpIdx, sesh.btWithBaseRipLens_lfpIdx, sesh.btWithBaseRipPeakIdx_lfpIdx, \
                sesh.btWithBaseRipPeakAmps, sesh.btWithBaseRipCrossThreshIdxs_lfpIdx = \
                detectRipples(zPowerDiff)
            if len(sesh.btWithBaseRipStartIdx_lfpIdx) > 0:
                sesh.btWithBaseRipStarts_ts = lfp_ts[
                    sesh.btWithBaseRipStartIdx_lfpIdx + btLfpStart_lfpIdx]
            else:
                sesh.btWithBaseRipStarts_ts = np.array([])

        print("\t{} ripples found during task".format(
            len(sesh.btRipStartsProbeStats_ts)))

    elif sesh.probePerformed:
        print("\tProbe performed but LFP in a separate file for session", sesh.name)


def runSanityChecks(sesh: BTSession, lfpData, baselineLfpData, showPlots=False, overrideNotes=True):
    # print(f"Running sanity checks for {sesh.name}")
    if lfpData is None:
        print("\tNo LFP data to look at")
    else:
        # lfpV = lfpData[0][1]['voltage']
        lfp_ts = lfpData[0][0]['time']
        # C = sesh.importOptions["consts"]

        # lfpDeflections = signal.find_peaks(np.abs(np.diff(
        #     lfpV, prepend=lfpV[0])), height=C["DEFLECTION_THRESHOLD_HI"], distance=C["MIN_ARTIFACT_DISTANCE"])
        # interruptions_lfpIdx = lfpDeflections[0]
        # interruption_ts = lfp_ts[interruptions_lfpIdx]
        # btInterruption_posIdx = np.searchsorted(sesh.btPos_ts, interruption_ts)
        # btInterruption_posIdx = sesh.btInterruption_posIdx[btInterruption_posIdx < len(
        #     sesh.btPos_ts)]

        numInterruptions = len(sesh.btInterruption_posIdx)
        print("\t{} interruptions detected".format(numInterruptions))
        if numInterruptions < 50:
            if sesh.isRippleInterruption and numInterruptions > 0:
                print("WARNING: FEWER THAN 50 STIMS DETECTED ON AN INTERRUPTION SESSION")
            elif numInterruptions == 0:
                print("WARNING: IGNORING BEHAVIOR NOTES FILE BECAUSE SEEING "
                      "0 INTERRUPTIONS, CALLING THIS A CONTROL SESSION")
                if overrideNotes:
                    sesh.isRippleInterruption = False
            else:
                print(
                    "WARNING: very few interruptions. This was a delay control but is basically a no-stim control")
        elif numInterruptions < 100:
            print("\t50-100 interruptions: not overriding label")

        dt = np.diff(lfp_ts)
        gapThresh = 2.0 * float(TRODES_SAMPLING_RATE / LFP_SAMPLING_RATE)
        isBigGap = dt > gapThresh

        if not any(isBigGap):
            print("\tNo gaps in LFP!")
        else:
            totalTime = (lfp_ts[-1] - lfp_ts[0]) / TRODES_SAMPLING_RATE
            totalGapTime = np.sum(dt[isBigGap])
            print(f"\t{totalGapTime}/{totalTime} ({int(100*totalGapTime/totalTime)}%) of lfp signal missing")

            maxGapIdx = np.argmax(dt)
            maxGapLen = dt[maxGapIdx] / TRODES_SAMPLING_RATE
            maxGapT1 = (lfp_ts[maxGapIdx] - lfp_ts[0]) / TRODES_SAMPLING_RATE
            maxGapT2 = (lfp_ts[maxGapIdx + 1] - lfp_ts[0]) / TRODES_SAMPLING_RATE
            print(f"\tBiggest gap: {maxGapLen}s long ({maxGapT1} - {maxGapT2})")

    # dummySesh = BTSession()
    # dummyDict = dummySesh.__dict__
    # for k in sesh.__dict__:
    #     v = sesh.__dict__[k]
    #     if k not in dummyDict:
    #         print(f"Found attribute {k} with value type {type(v)}")
    #     # elif not isinstance(v, type(dummyDict[k])) and dummyDict[k] is not None:
    #     #     print(f"Attribute {k} changed type:\t{type(dummyDict[k])}\t{type(v)}!")
    #     elif isinstance(v, (list, np.ndarray)):
    #         # print(k, v)
    #         same = np.array(v) == np.array(dummyDict[k])
    #         if isinstance(same, bool) and same:
    #             print(f"Attribute {k} was never changed")
    #     elif v == dummyDict[k]:
    #         print(f"Attribute {k} was never changed")


@TimeThisFunction
def posCalcVelocity(sesh):
    btVel = np.sqrt(np.power(np.diff(sesh.btPosXs), 2) +
                    np.power(np.diff(sesh.btPosYs), 2))

    oldSettings = np.seterr(invalid="ignore")
    sesh.btVelCmPerS = np.divide(btVel, np.diff(sesh.btPos_ts) /
                                 TRODES_SAMPLING_RATE) * CM_PER_FT
    np.seterr(**oldSettings)

    btIsMv = sesh.btVelCmPerS > sesh.importOptions["consts"]["VEL_THRESH"]
    if len(btIsMv) > 0:
        btIsMv = np.append(btIsMv, np.array(btIsMv[-1]))
    sesh.btIsMv = btIsMv

    if sesh.probePerformed:
        probeVel = np.sqrt(np.power(np.diff(sesh.probePosXs), 2) +
                           np.power(np.diff(sesh.probePosYs), 2))
        oldSettings = np.seterr(invalid="ignore")
        sesh.probeVelCmPerS = np.divide(probeVel, np.diff(sesh.probePos_ts) /
                                        TRODES_SAMPLING_RATE) * CM_PER_FT
        np.seterr(**oldSettings)
        probeIsMv = sesh.probeVelCmPerS > sesh.importOptions["consts"]["VEL_THRESH"]
        if len(probeIsMv) > 0:
            probeIsMv = np.append(probeIsMv, np.array(probeIsMv[-1]))
        sesh.probeIsMv = probeIsMv


@TimeThisFunction
def posCalcEntryExitTimes(sesh):
    # ===================================
    # Well and quadrant entry and exit times
    # ===================================
    sesh.btNearestWells = getNearestWell(sesh.btPosXs, sesh.btPosYs)

    sesh.btQuadrants = np.array(
        [quadrantOfWell(wi) for wi in sesh.btNearestWells])

    sesh.btWellEntryTimes_posIdx, sesh.btWellExitTimes_posIdx, \
        sesh.btWellEntryTimes_ts, sesh.btWellExitTimes_ts = \
        getWellEntryAndExitTimes(sesh.btNearestWells, sesh.btPos_ts)

    # ninc stands for neighbors included
    sesh.btWellEntryTimesNinc_posIdx, sesh.btWellExitTimesNinc_posIdx, \
        sesh.btWellEntryTimesNinc_ts, sesh.btWellExitTimesNinc_ts = \
        getWellEntryAndExitTimes(sesh.btNearestWells, sesh.btPos_ts, includeNeighbors=True)

    sesh.btQuadrantEntryTimes_posIdx, sesh.btQuadrantExitTimes_posIdx, \
        sesh.btQuadrantEntryTimes_ts, sesh.btQuadrantExitTimes_ts = \
        getWellEntryAndExitTimes(
            sesh.btQuadrants, sesh.btPos_ts, quads=True)

    for i in allWellNames:
        assert len(sesh.btWellEntryTimes_ts[i]) == len(sesh.btWellExitTimes_ts[i])

    # same for during probe
    if sesh.probePerformed:
        sesh.probeNearestWells = getNearestWell(sesh.probePosXs, sesh.probePosYs)

        sesh.probeWellEntryTimes_posIdx, sesh.probeWellExitTimes_posIdx, \
            sesh.probeWellEntryTimes_ts, sesh.probeWellExitTimes_ts = getWellEntryAndExitTimes(
                sesh.probeNearestWells, sesh.probePos_ts)
        sesh.probeWellEntryTimesNinc_posIdx, sesh.probeWellExitTimesNinc_posIdx, \
            sesh.probeWellEntryTimesNinc_ts, sesh.probeWellExitTimesNinc_ts = getWellEntryAndExitTimes(
                sesh.probeNearestWells, sesh.probePos_ts, includeNeighbors=True)

        sesh.probeQuadrants = np.array(
            [quadrantOfWell(wi) for wi in sesh.probeNearestWells])

        sesh.probeQuadrantEntryTimes_posIdx, sesh.probeQuadrantExitTimes_posIdx, \
            sesh.probeQuadrantEntryTimes_ts, sesh.probeQuadrantExitTimes_ts = getWellEntryAndExitTimes(
                sesh.probeQuadrants, sesh.probePos_ts, quads=True)

        for i in allWellNames:
            assert len(sesh.probeWellEntryTimes_ts[i]) == len(sesh.probeWellExitTimes_ts[i])


def getCurvature(x, y, H):
    # Knot-path-curvature as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000638
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
        # print(pi, end="\t")
        # forward
        ii = pi
        dxfi = 0.0
        dyfi = 0.0
        while ii < len(dx):
            dxfi += dx[ii]
            dyfi += dy[ii]
            magfi = dxfi * dxfi + dyfi * dyfi
            if magfi >= H2:
                # print(ii - pi)
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
            magbi = dxbi * dxbi + dybi * dybi
            if magbi >= H2:
                break
            ii -= 1

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

        dxf[pi] = dxfi
        dyf[pi] = dyfi
        dxb[pi] = dxbi
        dyb[pi] = dybi
        magf[pi] = magfi
        magb[pi] = magbi

    oldSettings = np.seterr(divide="ignore", invalid="ignore")

    uxf = dxf / np.sqrt(magf)
    uyf = dyf / np.sqrt(magf)
    uxb = dxb / np.sqrt(magb)
    uyb = dyb / np.sqrt(magb)
    dotprod = uxf * uxb + uyf * uyb
    curvature = np.arccos(dotprod)

    np.seterr(**oldSettings)

    return curvature, i1, i2, dxf, dyf, dxb, dyb


@TimeThisFunction
def posCalcCurvature(sesh):
    KNOT_H_POS = sesh.importOptions["consts"]["KNOT_H_CM"] / CM_PER_FT

    sesh.btCurvature, sesh.btCurvatureI1, sesh.btCurvatureI2, sesh.btCurvatureDxf, sesh.btCurvatureDyf, \
        sesh.btCurvatureDxb, sesh.btCurvatureDyb = getCurvature(
            sesh.btPosXs, sesh.btPosYs, KNOT_H_POS)

    if sesh.probePerformed:
        sesh.probeCurvature, sesh.probeCurvatureI1, sesh.probeCurvatureI2, sesh.probeCurvatureDxf, \
            sesh.probeCurvatureDyf, sesh.probeCurvatureDxb, sesh.probeCurvatureDyb = getCurvature(
                sesh.probePosXs, sesh.probePosYs, KNOT_H_POS)


def getExplorationCategories(ts, vel, nearestWells, consts, forcePauseIntervals=None):
    POS_FRAME_RATE = stats.mode(np.diff(ts), keepdims=True)[0] / float(TRODES_SAMPLING_RATE)
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
    ts = np.array(ts)
    exploreBoutLensSecs = (ts[exploreBoutEnds] - ts[exploreBoutStarts]) / TRODES_SAMPLING_RATE

    # add a category at each behavior time point for easy reference later:
    boutCategory = np.zeros_like(ts)
    lastStop = 0
    for bst, ben in zip(exploreBoutStarts, exploreBoutEnds):
        boutCategory[lastStop:bst] = 1
        lastStop = ben
    boutCategory[lastStop:] = 1
    if forcePauseIntervals is not None:
        for i1, i2 in forcePauseIntervals:
            pidx1 = np.searchsorted(ts[0:-1], i1)
            pidx2 = np.searchsorted(ts[0:-1], i2)
            boutCategory[pidx1:pidx2] = 2

    boutLabel = np.zeros_like(ts)
    for bi, (bst, ben) in enumerate(zip(exploreBoutStarts, exploreBoutEnds)):
        boutLabel[bst:ben] = bi + 1

    return smoothVel, exploreBoutStarts, exploreBoutEnds, \
        exploreBoutLensSecs, boutCategory, boutLabel


@TimeThisFunction
def posCalcExplorationBouts(sesh):
    sesh.btSmoothVel, sesh.btExploreBoutStart_posIdx, sesh.btExploreBoutEnd_posIdx, \
        sesh.btExploreBoutLensSecs, sesh.btBoutCategory, sesh.btBoutLabel = \
        getExplorationCategories(sesh.btPos_ts, sesh.btVelCmPerS, sesh.btNearestWells, sesh.importOptions["consts"],
                                 forcePauseIntervals=(list(zip(sesh.homeRewardEnter_ts, sesh.homeRewardExit_ts)) +
                                                      list(zip(sesh.awayRewardEnter_ts, sesh.awayRewardExit_ts))))

    if sesh.probePerformed:
        sesh.probeSmoothVel, sesh.probeExploreBoutStart_posIdx, sesh.probeExploreBoutEnd_posIdx, \
            sesh.probeExploreBoutLensSecs, sesh.probeBoutCategory, sesh.probeBoutLabel = \
            getExplorationCategories(sesh.probePos_ts, sesh.probeVelCmPerS,
                                     sesh.probeNearestWells, sesh.importOptions["consts"])


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


@TimeThisFunction
def posCalcExcursions(sesh):
    sesh.btExcursionCategory, sesh.btExcursionStart_posIdx, sesh.btExcursionEnd_posIdx, \
        sesh.btExcursionLensSecs = getExcursions(sesh.btNearestWells, sesh.btPos_ts)

    if sesh.probePerformed:
        sesh.probeExcursionCategory, sesh.probeExcursionStart_posIdx, sesh.probeExcursionEnd_posIdx, \
            sesh.probeExcursionLensSecs = getExcursions(
                sesh.probeNearestWells, sesh.probePos_ts)


def runPositionAnalyses(sesh):
    if sesh.lastAwayWell is None:
        sesh.numAwayFound = 0
    else:
        sesh.numAwayFound = next((i for i in range(
            len(sesh.awayWells)) if sesh.awayWells[i] == sesh.lastAwayWell), -1) + 1
    sesh.visitedAwayWells = sesh.awayWells[0:sesh.numAwayFound]
    # print(sesh.lastAwayWell)
    sesh.numHomeFound = sesh.numAwayFound
    if sesh.endedOnHome:
        sesh.numHomeFound += 1

    if not sesh.hasPositionData:
        print("\tCan't run analyses without any data!")
        return

    if sesh.hasRewardFindTimes:
        assert sesh.numAwayFound == len(sesh.awayRewardEnter_ts)
        assert sesh.numHomeFound == len(sesh.homeRewardEnter_ts)

    posCalcVelocity(sesh)
    posCalcEntryExitTimes(sesh)

    # ===================================
    # truncate path when rat was recalled (as he's running to be picked up) based on visit time of marked well
    # Note need to check sessions that ended near 7
    # ===================================
    # btRecall_posIdx
    if sesh.btEndedAtWell is None:
        assert sesh.lastAwayWell == sesh.awayWells[-1] and sesh.endedOnHome
        sesh.btEndedAtWell = sesh.homeWell

    sesh.btRecall_posIdx = sesh.btWellExitTimes_posIdx[sesh.btEndedAtWell][-1]

    if sesh.probePerformed:
        sesh.probeRecall_posIdx = sesh.probeWellExitTimes_posIdx[sesh.probeEndedAtWell][-1]

    posCalcCurvature(sesh)

    posCalcExplorationBouts(sesh)
    posCalcExcursions(sesh)
    # posCalcRewardCategories(sesh)


def extractAndSave(animalName: str, importOptions: dict):
    numExcluded = 0
    numExtracted = 0

    print("=========================================================")
    print(f"Extracting data for animal {animalName}")
    print("=========================================================")
    animalInfo = getInfoForAnimal(animalName)
    print(
        f"\tdataDir = {animalInfo.data_dir}\n\toutputDir = {animalInfo.output_dir}")
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
    infoProblemSessions = []
    for seshi, (seshDir, prevSeshDir) in enumerate(zip(sessionDirs, prevSessionDirs)):
        if importOptions["debug"]["debugMode"] and \
            importOptions["debug"]["maxNumSessions"] is not None and \
                seshi > importOptions["debug"]["maxNumSessions"]:
            break
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
        sesh.importOptions = importOptions.copy()
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

        if sesh.importOptions["confirmAfterEachSession"]:
            res = input("Press enter to continue, type s to skip, q to quit")
            if res == "q":
                return
            elif res == "s":
                continue

        print("Parsing info files")
        try:
            parseInfoFiles(sesh)
        except Exception as e:
            print("Error parsing info file, skipping session", sesh.name)
            infoProblemSessions.append((sesh.name, e))
            continue
        print("Loading position data")
        loadPositionData(sesh)
        if sesh.isNoInterruption:
            sesh.importOptions["skipLFP"] = True
        if sesh.importOptions["skipLFP"]:
            print("!!!!!\tSKIPPING LFP\t!!!!!")
            lfpData = None
            baselineLfpData = None
        else:
            print("Loading LFP")
            lfpData, baselineLfpData = loadLFPData(sesh)

        if sesh.importOptions["justExtractData"]:
            print("Just extracting the data, continuing to next session...\n")
            numExtracted += 1
            continue

        if not sesh.importOptions["skipLFP"]:
            print("Analyzing LFP")
            runLFPAnalyses(sesh, lfpData, baselineLfpData)
        print("Analyzing position")
        runPositionAnalyses(sesh)

        print("Running sanity checks")
        runSanityChecks(sesh, lfpData, baselineLfpData)

        print(f"Done with session {sesh.name}")
        dataObj.allSessions.append(sesh)

    if importOptions["justExtractData"]:
        print(
            f"Extracted data from {numExtracted} sessions, excluded {numExcluded}. Not analyzing or saving")
        if len(infoProblemSessions) > 0:
            print("Had problems with the following sessions:")
            for name, e in infoProblemSessions:
                print(name, e)
        return

    if not any([s.conditionGroup is not None for s in dataObj.allSessions]):
        for si, sesh in enumerate(dataObj.allSessions):
            sesh.conditionGroup = sesh.animalName + "-" + str(si)

    # save all sessions to disk
    if importOptions["debug"]["debugMode"] and importOptions["debug"]["dontsave"]:
        print("Not saving because in debug mode")
        print("Would've saved the following sessions:")
        for sesh in dataObj.allSessions:
            print(sesh.name)
    else:
        outputFname = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        print("Saving to file: {}".format(outputFname))
        dataObj.saveToFile(outputFname)
        print("Saved sessions:")
        for sesh in dataObj.allSessions:
            print(sesh.name)

    if len(infoProblemSessions) > 0:
        print("Had problems with the following sessions:")
        for name, e in infoProblemSessions:
            print(name, e)

    print(f"\tposCalcVelocity: {posCalcVelocity.totalTime}")
    print(f"\tposCalcEntryExitTimes: {posCalcEntryExitTimes.totalTime}")
    print(f"\tposCalcCurvature: {posCalcCurvature.totalTime}")
    print(f"\tposCalcExplorationBouts: {posCalcExplorationBouts.totalTime}")
    print(f"\tposCalcExcursions: {posCalcExcursions.totalTime}")


if __name__ == "__main__":
    animalNames = parseCmdLineAnimalNames(default=["B18"])
    importOptions = {
        "skipLFP": False,
        "skipUSB": False,
        "skipPrevSession": True,
        "forceAllTrackingAuto": False,
        "skipConfirmTrackingFileExtension": True,
        "skipCurvature": False,
        "runJustSpecified": False,
        "specifiedDays": [],
        "specifiedRuns": [],
        "justExtractData": False,
        "runInteractiveExtraction": True,
        "confirmAfterEachSession": False,
        "consts": {
            "VEL_THRESH": 10,  # cm/s

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
        },
        "debug": {
            "debugMode": False,
            "maxNumSessions": None,
            "dontSave": True
        }
    }

    for animalName in animalNames:
        importOptions["skipUSB"] = animalName == "Martin"
        extractAndSave(animalName, importOptions)
