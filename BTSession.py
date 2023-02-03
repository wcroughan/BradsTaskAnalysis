from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from numpy.typing import ArrayLike
from dataclasses import dataclass, field

from consts import TRODES_SAMPLING_RATE, allWellNames, CM_PER_FT, LFP_SAMPLING_RATE
from UtilFunctions import LoadInfo, getWellPosCoordinates, Ripple, ImportOptions


# A few design decisions I'm implementing now:
#
# - No more wellIdx. Whenever you refer to a well, use an int or str that is what I call the well
#   i.e. well 2 always refers to the well in the bottom left corner
#
# - Position is always normalized and corrected. During import, this correction is done according to the
#   manually annotated well locations. These well locations never need to be accessed again, and should
#   probably be destroyed
#
# - exit idxs are exclusive. Thus the exit idx from one well is equal to the entrance idx of the
#   next well, and the final exit idx is the length of the pos vector
#
# - quantities like curvature for which a calculation should be made over the whole behavior and then
#   a subset is referenced later get computed in ImportData for efficiency. However, simple calculations
#   like latency are now moved to here as functions
#
# Unit guide:
# *_ts - trodes timestamps
# *_posIdx - index in probe_pos* or bt_pos*
# *_lfpIdx - index in lfp data

@dataclass
class BTSession:
    """
    Contains all data for a session on Brad's task with probe
    Also has references to previous and next session.
    """

    # Static constants
    MOVE_FLAG_ALL = 0
    MOVE_FLAG_MOVING = 1
    MOVE_FLAG_STILL = 2

    AVG_FLAG_OVER_TIME = 0
    AVG_FLAG_OVER_VISITS = 1

    BOUT_STATE_EXPLORE = 0
    BOUT_STATE_REST = 1
    BOUT_STATE_REWARD = 2

    EXCURSION_STATE_OFF_WALL = 0
    EXCURSION_STATE_ON_WALL = 1

    CONDITION_INTERRUPTION = 0
    CONDITION_DELAY = 1
    CONDITION_NO_STIM = 2

    # Member vars
    loadInfo: LoadInfo = None
    importOptions: ImportOptions = None
    animalName: str = ""
    # ==================================
    # Info about the session
    # ==================================
    # The previous session chronologically, if one exists
    prevSession: Optional[BTSession] = None
    prevSessionDir: Optional[str] = None
    # The next session chronologically, if one exists
    nextSession: Optional[BTSession] = None
    # date object representing the day of this recording
    date: Optional[datetime] = None

    # Just the date part of session filename (i.e. "20200112")
    dateStr: str = ""
    # Just the time part of session filename (i.e. "140259")
    timeStr: str = ""
    # Data string in general. May modify in future by appending "S1" for first session if doing multiple in one day
    name: str = ""
    # name of raw data folder in which brad's task part of session was recorded
    btDir: str = ""
    # name of raw data folder in which ITI part of session was recorded. May be missing (empty string).
    # May be same as btDir
    itiDir: str = ""
    # name of raw data folder in which probe part of session was recorded. May be missing (empty string).
    # May be same as btDir
    probeDir: str = ""
    fileStartString: str = ""

    infoFileName: str = ""
    seshIdx: int = -1
    conditionGroup: str = ""

    prevInfoFileName: str = ""
    prevSessionInfoParsed: bool = False
    prevSessionHome: int = -1
    prevSessionAways: List[int] = field(default_factory=list)
    prevSessionLastAwayWell: int = -1
    prevSessionEndedOnHome: bool = False
    prevSessionItiStimOn: bool = False
    prevSessionProbeStimOn: bool = False
    prevSessionRippleDetectionThreshold: float = 0.0
    prevSessionItiStimOn: bool = False
    prevSessionIdx: int = -1

    hasActivelinkLog: bool = False
    activelinkLogFileName: str = ""
    loggedDetections_ts: np.ndarray = np.array([])
    loggedStats: List[Tuple[str, float, float]] = field(
        default_factory=list)  # contains all the stats listed in log file
    rpowmLog: float = 0.0  # The entry from above log list that was active during the session
    rpowsLog: float = 0.0  # The entry from above log list that was active during the session

    # Some flags indicated whether ITI was recorded and whether ITI and probe are in the same rec file or not
    separateItiFile: bool = False
    recordedIti: bool = False
    separateProbeFile: bool = False

    # more flags from info file
    rippleDetectionThreshold: float = 0.0
    lastAwayWell: float = 0
    endedOnHome: bool = False
    ItiStimOn: bool = False
    probeStimOn: bool = False

    btEndedAtWell: Optional[int] = None
    probeEndedAtWell: Optional[int] = None
    probePerformed: Optional[bool] = None

    # Any other notes that are stored in the info file are added here. Each list entry is one line from that file
    notes: List[str] = field(default_factory=list)

    # Well number of home well
    homeWell: int = 0
    # Well number of away wells
    awayWells: List[int] = field(default_factory=list)
    numAwayFound: int = 0
    numHomeFound: int = 0
    foundWells: Optional[List[int]] = None

    # Flags indicating stim condition
    rippleDetectionTetrodes: List[int] = field(default_factory=list)
    rippleBaselineTetrode: Optional[int] = None
    condition: Optional[int] = None
    prevSessionCondition: Optional[int] = None

    probeFillTime: Optional[int] = None

    # indicates that any well at all was found
    foundFirstHome: bool = False

    # For purposes of light on/off times, ignore this many seconds at start of video
    trodesLightsIgnoreSeconds: Optional[float] = None

    # ==================================
    # Raw data (ish ... position data is fisheye corrected and scaled)
    # ==================================
    hasPositionData: bool = False

    # Position data during brad's task
    btPos_ts: np.ndarray = np.array([])
    btPosXs: np.ndarray = np.array([])
    btPosYs: np.ndarray = np.array([])
    btRecall_posIdx: int = -1

    # Position data during probe
    probePos_ts: np.ndarray = np.array([])
    probePosXs: np.ndarray = np.array([])
    probePosYs: np.ndarray = np.array([])
    probeRecall_posIdx: int = -1

    # LFP data is huge, so only load on demand
    # noise has a lower threshold, and is more for cleaning up the signal
    # File names
    btLfpFnames: List[str] = field(default_factory=list)
    itiLfpFnames: List[str] = field(default_factory=list)
    probeLfpFnames: List[str] = field(default_factory=list)
    btLfpBaselineFname: Optional[str] = None

    # Borders between sections of session
    btLfpStart_lfpIdx: int = 0
    btLfpEnd_lfpIdx: int = 0
    itiLfpStart_ts: int = 0
    itiLfpEnd_ts: int = 0
    itiLfpStart_lfpIdx: int = 0
    itiLfpEnd_lfpIdx: int = 0
    probeLfpStart_lfpIdx: int = 0
    probeLfpEnd_lfpIdx: int = 0

    # IMPORTANT NOTE: The above variables are used in the full
    # LFP data from the file. However all the other lfpIdx variables below
    # are used to index to the specific region's lfp data, so
    # i.e. probeLFPBumps_lfpIdx value of 0 indicates a bump at the start of the probe

    # Bumps are detected large deflections, likely interruptions
    lfpBumps_lfpIdx: np.ndarray = np.array([])
    lfpBumps_ts: np.ndarray = np.array([])
    btLFPBumps_posIdx: np.ndarray = np.array([])
    itiLFPBumps_posIdx: np.ndarray = np.array([])
    probeLFPBumps_posIdx: np.ndarray = np.array([])
    btLFPBumps_lfpIdx: np.ndarray = np.array([])
    itiLFPBumps_lfpIdx: np.ndarray = np.array([])
    probeLFPBumps_lfpIdx: np.ndarray = np.array([])

    lfpNoise_lfpIdx: np.ndarray = np.array([])
    lfpNoise_ts: np.ndarray = np.array([])
    btLFPNoise_posIdx: np.ndarray = np.array([])
    itiLFPNoise_posIdx: np.ndarray = np.array([])
    probeLFPNoise_posIdx: np.ndarray = np.array([])
    btLFPNoise_lfpIdx: np.ndarray = np.array([])
    itiLFPNoise_lfpIdx: np.ndarray = np.array([])
    probeLFPNoise_lfpIdx: np.ndarray = np.array([])

    rpowmPreBt: np.float64 = np.float64(0)
    rpowsPreBt: np.float64 = np.float64(0)
    rpowmProbe: np.float64 = np.float64(0)
    rpowsProbe: np.float64 = np.float64(0)

    btRipsPreStats: List[Ripple] = field(default_factory=list)
    itiRipsPreStats: List[Ripple] = field(default_factory=list)
    probeRipsPreStats: List[Ripple] = field(default_factory=list)
    btRipsProbeStats: List[Ripple] = field(default_factory=list)
    itiRipsProbeStats: List[Ripple] = field(default_factory=list)
    probeRipsProbeStats: List[Ripple] = field(default_factory=list)
    btRipsLogStats: List[Ripple] = field(default_factory=list)
    itiRipsLogStats: List[Ripple] = field(default_factory=list)
    probeRipsLogStats: List[Ripple] = field(default_factory=list)

    # Lighting stuff
    frameTimes: np.ndarray = np.array([]).astype(np.uint32)
    trodesLightOffTime: int = 0
    trodesLightOnTime: int = 0
    usbVidFile: str = ""
    usbLightOffFrame: int = -1
    usbLightOnFrame: int = -1

    # ==================================
    # Analyzed data: Brad's task
    # ==================================
    hasRewardFindTimes: bool = False
    homeRewardEnter_ts: np.ndarray = np.array([])
    homeRewardEnter_posIdx: np.ndarray = np.array([])
    homeRewardExit_ts: np.ndarray = np.array([])
    homeRewardExit_posIdx: np.ndarray = np.array([])

    awayRewardEnter_ts: np.ndarray = np.array([])
    awayRewardEnter_posIdx: np.ndarray = np.array([])
    awayRewardExit_ts: np.ndarray = np.array([])
    awayRewardExit_posIdx: np.ndarray = np.array([])
    visitedAwayWells: List[int] = field(default_factory=list)

    # ==================================
    # Analyzed data: Probe
    # ==================================
    btVelCmPerSRaw: np.ndarray = np.array([])
    btVelCmPerS: np.ndarray = np.array([])
    btIsMv: np.ndarray = np.array([])
    probeVelCmPerSRaw: np.ndarray = np.array([])
    probeVelCmPerS: np.ndarray = np.array([])
    probeIsMv: np.ndarray = np.array([])

    probeNearestWells: np.ndarray = np.array([])
    probeWellEntryTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    probeWellExitTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    probeWellEntryTimes_ts: List[np.ndarray] = field(default_factory=list)
    probeWellExitTimes_ts: List[np.ndarray] = field(default_factory=list)

    btNearestWells: np.ndarray = np.array([])
    btWellEntryTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    btWellExitTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    btWellEntryTimes_ts: List[np.ndarray] = field(default_factory=list)
    btWellExitTimes_ts: List[np.ndarray] = field(default_factory=list)

    btQuadrants: np.ndarray = np.array([])
    btWellEntryTimesNinc_posIdx: List[np.ndarray] = field(default_factory=list)
    btWellExitTimesNinc_posIdx: List[np.ndarray] = field(default_factory=list)
    btWellEntryTimesNinc_ts: List[np.ndarray] = field(default_factory=list)
    btWellExitTimesNinc_ts: List[np.ndarray] = field(default_factory=list)
    btQuadrantEntryTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    btQuadrantExitTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    btQuadrantEntryTimes_ts: List[np.ndarray] = field(default_factory=list)
    btQuadrantExitTimes_ts: List[np.ndarray] = field(default_factory=list)

    probeQuadrants: np.ndarray = np.array([])
    probeWellEntryTimesNinc_posIdx: List[np.ndarray] = field(default_factory=list)
    probeWellExitTimesNinc_posIdx: List[np.ndarray] = field(default_factory=list)
    probeWellEntryTimesNinc_ts: List[np.ndarray] = field(default_factory=list)
    probeWellExitTimesNinc_ts: List[np.ndarray] = field(default_factory=list)
    probeQuadrantEntryTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    probeQuadrantExitTimes_posIdx: List[np.ndarray] = field(default_factory=list)
    probeQuadrantEntryTimes_ts: List[np.ndarray] = field(default_factory=list)
    probeQuadrantExitTimes_ts: List[np.ndarray] = field(default_factory=list)

    btCurvature: np.ndarray = np.array([])
    btCurvatureI1: np.ndarray = np.array([])
    btCurvatureI2: np.ndarray = np.array([])
    btCurvatureDxf: np.ndarray = np.array([])
    btCurvatureDyf: np.ndarray = np.array([])
    btCurvatureDxb: np.ndarray = np.array([])
    btCurvatureDyb: np.ndarray = np.array([])

    probeCurvature: np.ndarray = np.array([])
    probeCurvatureI1: np.ndarray = np.array([])
    probeCurvatureI2: np.ndarray = np.array([])
    probeCurvatureDxf: np.ndarray = np.array([])
    probeCurvatureDyf: np.ndarray = np.array([])
    probeCurvatureDxb: np.ndarray = np.array([])
    probeCurvatureDyb: np.ndarray = np.array([])

    btSmoothVel: np.ndarray = np.array([])
    btBoutCategory: np.ndarray = np.array([])
    btBoutLabel: np.ndarray = np.array([])
    btExploreBoutStart_posIdx: np.ndarray = np.array([])
    btExploreBoutEnd_posIdx: np.ndarray = np.array([])
    btExploreBoutLensSecs: np.ndarray = np.array([])

    probeSmoothVel: np.ndarray = np.array([])
    probeBoutCategory: np.ndarray = np.array([])
    probeBoutLabel: np.ndarray = np.array([])
    probeExploreBoutStart_posIdx: np.ndarray = np.array([])
    probeExploreBoutEnd_posIdx: np.ndarray = np.array([])
    probeExploreBoutLensSecs: np.ndarray = np.array([])

    btExcursionCategory: np.ndarray = np.array([])
    btExcursionStart_posIdx: np.ndarray = np.array([])
    btExcursionEnd_posIdx: np.ndarray = np.array([])
    btExcursionLensSecs: np.ndarray = np.array([])

    probeExcursionCategory: np.ndarray = np.array([])
    probeExcursionStart_posIdx: np.ndarray = np.array([])
    probeExcursionEnd_posIdx: np.ndarray = np.array([])
    probeExcursionLensSecs: np.ndarray = np.array([])

    positionFromDeepLabCut: Optional[bool] = None

    # def __eq__(self, __o: object) -> bool:
    #     if not isinstance(__o, BTSession):
    #         return False
    #     return __o.name == self.name

    # def __ne__(self, __o: object) -> bool:
    #     return not self.__eq__(__o)

    # def __lt__(self, other: BTSession) -> bool:
    #     return self.name < other.name

    @property
    def isRippleInterruption(self):
        return self.condition == BTSession.CONDITION_INTERRUPTION

    @property
    def isDelayedInterruption(self):
        return self.condition == BTSession.CONDITION_DELAY

    @property
    def isNoInterruption(self):
        return self.condition == BTSession.CONDITION_NO_STIM

    @property
    def btPos_secs(self):
        return (self.btPos_ts - self.btPos_ts[0]) / TRODES_SAMPLING_RATE

    @property
    def probePos_secs(self):
        return (self.probePos_ts - self.probePos_ts[0]) / TRODES_SAMPLING_RATE

    def conditionString(self, splitCtrls=False):
        if self.condition == BTSession.CONDITION_INTERRUPTION:
            return "SWR"
        elif not splitCtrls:
            return "Ctrl"
        elif self.condition == BTSession.CONDITION_DELAY:
            return "Delay"
        else:
            return "NoStim"

    def timeIntervalToPosIdx(self, ts: ArrayLike[int],
                             timeInterval: Optional[Tuple[int, int] | Tuple[int, int, str]],
                             noneVal=None) -> ArrayLike[int]:
        """
        Returns posIdx indices in ts (which has units timestamps) corresponding
        to given timeInterval. Units given as optional third element can be
        "secs", "ts", "posIdx", or None. If None or omitted, defaults to "secs"
        """
        if timeInterval is None:
            return noneVal

        if len(timeInterval) == 2 or (len(timeInterval) == 3 and timeInterval[2] is None):
            timeInterval = [*timeInterval, "secs"]

        if timeInterval[2] == "posIdx":
            return np.array([timeInterval[0], timeInterval[1]])
        elif timeInterval[2] == "ts":
            return np.searchsorted(ts, np.array([timeInterval[0], timeInterval[1]]))
        elif timeInterval[2] == "secs":
            return np.searchsorted((ts - ts[0]) / TRODES_SAMPLING_RATE,
                                   np.array([timeInterval[0], timeInterval[1]]))
        else:
            raise ValueError("Couldn't parse time interval")

    def avgDistToWell(self, inProbe: bool, wellName: int,
                      timeInterval=None, moveFlag=None, avgFunc=np.nanmean) -> float:
        """
        timeInterval is in seconds, where 0 == start of probe or task (as specified in inProbe flag)
        return units: cm
        """

        if moveFlag is None:
            moveFlag = BTSession.MOVE_FLAG_ALL

        wx, wy = getWellPosCoordinates(wellName)

        if inProbe:
            ts = self.probePos_ts
            mv = self.probeIsMv
            xs = self.probePosXs
            ys = self.probePosYs
        else:
            ts = self.btPos_ts
            mv = self.btIsMv
            xs = self.btPosXs
            ys = self.btPosYs

        if moveFlag == self.MOVE_FLAG_MOVING:
            keepMv = mv
        elif moveFlag == self.MOVE_FLAG_STILL:
            keepMv = ~ mv
        else:
            keepMv = np.ones_like(mv).astype(bool)

        durIdx = self.timeIntervalToPosIdx(ts, timeInterval, noneVal=(0, len(keepMv)))
        keepTimeInt = np.zeros_like(keepMv).astype(bool)
        keepTimeInt[durIdx[0]:durIdx[1]] = True
        # if timeInterval is None:
        #     keepTimeInt = np.ones_like(keepMv)
        # else:
        #     assert xs.shape == ts.shape
        #     keepTimeInt = np.zeros_like(keepMv).astype(bool)
        #     durIdx = np.searchsorted(ts, np.array(
        #         [ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE]))
        #     keepTimeInt[durIdx[0]:durIdx[1]] = True

        keepFlag = keepTimeInt & keepMv

        distToWell = np.sqrt(np.power(wx - xs[keepFlag], 2) +
                             np.power(wy - ys[keepFlag], 2))
        return avgFunc(distToWell * CM_PER_FT)

    def getAllTrialPosIdxs(self):
        ht1 = self.homeRewardEnter_posIdx
        ht0 = np.hstack(([0], self.awayRewardExit_posIdx))
        if not self.endedOnHome:
            ht0 = ht0[0:-1]
        assert len(ht1) == len(ht0)

        # away trials
        at1 = self.awayRewardEnter_posIdx
        at0 = self.homeRewardExit_posIdx
        if self.endedOnHome:
            at0 = at0[0:-1]
        assert len(at1) == len(at0)

        # print(f"{ ht0.shape = }")
        ret = np.empty((len(ht1)+len(at1), 2)).astype(int)
        if len(ht1) > 0:
            ret[::2, :] = np.vstack((ht0, ht1)).T
            if len(at1) > 1:
                ret[1::2, :] = np.vstack((at0, at1)).T

        return ret

    def getAllRewardPosIdxs(self):
        return np.append(self.getAllTrialPosIdxs().flat[1:],
                         self.homeRewardExit_posIdx[-1] if self.endedOnHome else
                         self.awayRewardExit_posIdx[-1]).reshape((-1, 2))

    def entryExitTimes(self, inProbe: bool, wellName: int | str,
                       timeInterval: Optional[Tuple | list] = None,
                       includeNeighbors: bool = False, excludeReward: bool = False,
                       returnIdxs: bool = False,
                       includeEdgeOverlap: str | Tuple[str, str] = "clip"):
        """
        return units: trodes timestamps, unless returnIdxs==True
        includeEdgeOverlap: if one str given, applies to start and end of timeInterval.
            If pair given, first element applies to start of time interval, second applied to end
            If only visit overlaps start and end of timeInterval and pair given and either is omit, whole visit
                will be omitted, otherwise each end is treated separately
            Values can be:
                "clip" (default) - If edge of a visit goes beyond timeInterval,
                    truncate start and/or end to be exactly the time interval edge
                "omit" - don't include any visits that overlap edge. Edges that
                    lie exactly on time interval are still included
                "full" - Include visits where the edge overlaps with timeInterval
                    and don't clip to be within timeInterval
        """
        # wellName == "QX" ==> look at quadrant X instead of a well
        if isinstance(wellName, str) and wellName[0] == 'Q':
            isQuad = True
            wellName = int(wellName[1])
        else:
            isQuad = False

        assert not (includeNeighbors and isQuad)
        assert (not excludeReward) or ((not isQuad) and (not includeNeighbors))

        if inProbe:
            varName1 = "probe"
        else:
            varName1 = "bt"

        if isQuad:
            varName1 += "Quadrant"
        else:
            varName1 += "Well"

        varName2 = "Times"
        if includeNeighbors:
            varName2 += "Ninc"

        if returnIdxs:
            varName2 += "_posIdx"
        else:
            varName2 += "_ts"

        entryVarName = f"{varName1}Entry{varName2}"
        exitVarName = f"{varName1}Exit{varName2}"

        ents = self.__getattribute__(entryVarName)[wellName]
        exts = self.__getattribute__(exitVarName)[wellName]

        if excludeReward and not inProbe and self.foundFirstHome:
            assert self.hasRewardFindTimes

            # Arrays of the times to remove
            if wellName == self.homeWell:
                rewardEnter = self.homeRewardEnter_posIdx if returnIdxs else self.homeRewardEnter_ts
                rewardExit = self.homeRewardExit_posIdx if returnIdxs else self.homeRewardExit_ts
            elif wellName in self.visitedAwayWells:
                awayIdx = np.argmax(self.visitedAwayWells == wellName)
                rewardEnter = np.array([self.awayRewardEnter_posIdx[awayIdx]
                                       if returnIdxs else self.awayRewardEnter_ts[awayIdx]])
                rewardExit = np.array([self.awayRewardExit_posIdx[awayIdx]
                                      if returnIdxs else self.awayRewardExit_ts[awayIdx]])
            else:
                rewardEnter = np.array([])
                rewardExit = np.array([])
            removeTimes = (rewardEnter + rewardExit) / 2

            for rt in removeTimes:
                whichPassFlags = (ents < rt) & (exts > rt)
                if np.count_nonzero(whichPassFlags) != 0:
                    raise Exception("Couldn't find correct reward time to exclude")

                todel = np.argmax(whichPassFlags)
                ents = np.delete(ents, todel)
                exts = np.delete(exts, todel)

        if timeInterval is not None:
            if inProbe:
                ts = self.probePos_ts
            else:
                ts = self.btPos_ts

            if isinstance(includeEdgeOverlap, str):
                includeEdgeOverlap = (includeEdgeOverlap, includeEdgeOverlap)

            timeInterval_posIdx = self.timeIntervalToPosIdx(ts, timeInterval)

            if returnIdxs:
                mint = timeInterval_posIdx[0]
                maxt = timeInterval_posIdx[1]
            else:
                mint = ts[timeInterval_posIdx[0]]
                maxt = ts[timeInterval_posIdx[1]]
            # mint = ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE
            # maxt = ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE
            # if returnIdxs:
                # mint = np.searchsorted(ts, mint)
                # maxt = np.searchsorted(ts, maxt)

            keepflag = np.logical_and(np.array(ents) < maxt, np.array(exts) > mint)
            ents = ents[keepflag]
            exts = exts[keepflag]
            if len(ents) > 0:
                if ents[0] < mint:
                    if includeEdgeOverlap[0] == "clip":
                        ents[0] = mint
                    elif includeEdgeOverlap[0] == "omit":
                        ents = np.delete(ents, 0)
                        exts = np.delete(exts, 0)
                    elif includeEdgeOverlap[0] == "full":
                        pass
                    else:
                        raise ValueError("Edge overlaps but invalid overlap rule given")
            if len(ents) > 0:
                if exts[-1] > maxt:
                    if includeEdgeOverlap[1] == "clip":
                        exts[-1] = maxt
                    elif includeEdgeOverlap[1] == "omit":
                        ents = np.delete(ents, -1)
                        exts = np.delete(exts, -1)
                    elif includeEdgeOverlap[1] == "full":
                        pass
                    else:
                        raise ValueError("Edge overlaps but invalid overlap rule given")

        return ents, exts

    def avgContinuousMeasureAtWell(self, inProbe, wellName, yvals,
                                   timeInterval=None, avgTypeFlag=None, avgFunc=np.nanmean, excludeReward=False,
                                   includeNeighbors=False, emptyVal=np.nan):
        if avgTypeFlag is None:
            avgTypeFlag = BTSession.AVG_FLAG_OVER_TIME

        wenis, wexis = self.entryExitTimes(
            inProbe, wellName, timeInterval=timeInterval, excludeReward=excludeReward,
            includeNeighbors=includeNeighbors, returnIdxs=True)

        if len(wenis) == 0:
            return emptyVal

        resAll = []
        for weni, wexi in zip(wenis, wexis):
            if wexi > yvals.size:
                continue
            resAll.append(yvals[weni:wexi])

        if avgTypeFlag == BTSession.AVG_FLAG_OVER_TIME:
            return avgFunc(np.concatenate(resAll))
        elif avgTypeFlag == BTSession.AVG_FLAG_OVER_VISITS:
            return avgFunc([avgFunc(x) for x in resAll])
        else:
            assert False

    def avgCurvatureAtWell(self, inProbe, wellName, **kwargs):
        if inProbe:
            yvals = self.probeCurvature
        else:
            yvals = self.btCurvature

        return self.avgContinuousMeasureAtWell(inProbe, wellName, yvals, **kwargs)

    def dwellTimes(self, inProbe, wellName, **kwargs):
        """
        return units: trodes timestamps
        """
        ents, exts = self.entryExitTimes(inProbe, wellName, returnIdxs=False, **kwargs)
        return exts - ents

    def numWellEntries(self, inProbe, wellName, **kwargs):
        ents, _ = self.entryExitTimes(
            inProbe, wellName, includeEdgeOverlap=("omit", "full"), **kwargs)
        return ents.size

    def totalDwellTime(self, inProbe, wellName, **kwargs):
        """
        return units: seconds
        """
        return np.sum(self.dwellTimes(inProbe, wellName, **kwargs) / TRODES_SAMPLING_RATE)

    def avgDwellTime(self, inProbe, wellName, avgFunc=np.nanmean, emptyVal=np.nan, **kwargs):
        """
        return units: seconds
        """
        ret = self.dwellTimes(inProbe, wellName, **kwargs)
        if len(ret) == 0:
            return emptyVal
        else:
            return avgFunc(ret / TRODES_SAMPLING_RATE)

    def numBouts(self, inProbe, timeInterval=None):
        if inProbe:
            lbls = self.probeBoutLabel
            ts = self.probePos_ts
        else:
            lbls = self.btBoutLabel
            ts = self.btPos_ts

        durIdx = self.timeIntervalToPosIdx(ts, timeInterval, noneVal=(0, len(lbls)))
        lbls = lbls[durIdx[0]:durIdx[1]]
        # if timeInterval is not None:
        #     imin = np.searchsorted(ts, ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE)
        #     imax = np.searchsorted(ts, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE)
        #     lbls = lbls[imin:imax]

        return len(set(lbls) - set([0]))

    def numBoutsWhereWellWasVisited(self, inProbe, wellName, **kwargs):
        ents, exts = self.entryExitTimes(inProbe, wellName, **kwargs, returnIdxs=True)

        if inProbe:
            lbls = self.probeBoutLabel
        else:
            lbls = self.btBoutLabel

        res = set([])
        for enti, exti in zip(ents, exts):
            res = res | set(lbls[enti:exti])

        res = res - set([0])
        return len(res)

    def pctBoutsWhereWellWasVisited(self, inProbe, wellName, timeInterval=None, boutsInterval=None,
                                    emptyVal=np.nan, **kwargs):
        # bouts interval is inclusive first, exclusive last
        if boutsInterval is not None:
            assert timeInterval is None
            if inProbe:
                boutStarts = self.probeExploreBoutStart_posIdx
                boutEnds = self.probeExploreBoutEnd_posIdx
            else:
                boutStarts = self.btExploreBoutStart_posIdx
                boutEnds = self.btExploreBoutEnd_posIdx

            if len(boutEnds) >= boutsInterval[1]:
                timeInterval = [boutStarts[boutsInterval[0]],
                                boutEnds[boutsInterval[1] - 1], "posIdx"]
            else:
                timeInterval = [boutStarts[boutsInterval[0]], boutEnds[len(boutEnds) - 1], "posIdx"]

        denom = self.numBouts(inProbe, timeInterval=timeInterval)
        if denom == 0:
            return emptyVal

        return self.numBoutsWhereWellWasVisited(inProbe, wellName, timeInterval=timeInterval, **kwargs) / denom

    def propTimeInBoutState(self, inProbe, boutState, timeInterval=None):
        if inProbe:
            cats = self.probeBoutCategory
            ts = self.probePos_ts
        else:
            cats = self.btBoutCategory
            ts = self.btPos_ts

        durIdx = self.timeIntervalToPosIdx(ts, timeInterval, noneVal=(0, len(ts)))
        # if timeInterval is None:
        #     imin = 0
        #     imax = len(ts)
        # else:
        #     imin = np.searchsorted(ts, ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE)
        #     imax = np.searchsorted(ts, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE)
        # cats = cats[imin:imax]

        cats = cats[durIdx[0]:durIdx[1]]
        if float(cats.size) == 0:
            print(len(self.btBoutCategory))
            print(inProbe, durIdx, timeInterval)
        return float(np.count_nonzero(cats == boutState)) / float(cats.size)

    def meanVel(self, inProbe, moveFlag=None, timeInterval=None):
        """
        return units: cm/s
        """
        if moveFlag is None:
            moveFlag = BTSession.MOVE_FLAG_ALL

        if inProbe:
            vel = self.probeVelCmPerS
            ts = self.probePos_ts
            mv = self.probeIsMv
        else:
            vel = self.btVelCmPerS
            ts = self.btPos_ts
            mv = self.btIsMv

        durIdx = self.timeIntervalToPosIdx(ts, timeInterval, noneVal=(0, len(mv)))
        inTime = np.zeros_like(mv).astype(bool)
        inTime[durIdx[0]:durIdx[1]] = True
        # if timeInterval is None:
        #     inTime = np.ones_like(mv).astype(bool)
        # else:
        #     imin = np.searchsorted(ts, ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE)
        #     imax = np.searchsorted(ts, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE)
        #     inTime = np.zeros_like(mv).astype(bool)
        #     inTime[imin:imax] = True

        if moveFlag == self.MOVE_FLAG_MOVING:
            keepMv = mv
        elif moveFlag == self.MOVE_FLAG_STILL:
            keepMv = ~ mv
        else:
            keepMv = np.ones_like(mv).astype(bool)

        keepFlag = keepMv & inTime
        return np.nanmean(vel[keepFlag])

    def pathOptimality(self, inProbe: bool,
                       timeInterval: Optional[Tuple[float, float]
                                              | Tuple[float, float, str]] = None,
                       wellName=None, emptyVal=np.nan):
        if (timeInterval is None) == (wellName is None):
            raise Exception("Gimme a time interval or a well name plz (but not both)")

        if inProbe:
            xs = self.probePosXs
            ys = self.probePosYs
            ts = self.probePos_ts
        else:
            xs = self.btPosXs
            ys = self.btPosYs
            ts = self.btPos_ts

        if wellName is not None:
            ents = self.entryExitTimes(inProbe, wellName, returnIdxs=True)[0]
            if len(ents) == 0:
                return emptyVal
            ei = ents[0]
            displacementX = xs[ei] - xs[0]
            displacementY = ys[ei] - ys[0]
            dx = np.diff(xs[0:ei])
            dy = np.diff(ys[0:ei])
        else:
            durIdx = self.timeIntervalToPosIdx(ts, timeInterval)
            assert durIdx is not None
            displacementX = xs[durIdx[1]] - xs[durIdx[0]]
            displacementY = ys[durIdx[1]] - ys[durIdx[0]]
            dx = np.diff(xs[durIdx[0]:durIdx[1]])
            dy = np.diff(ys[durIdx[0]:durIdx[1]])

        displacement = np.sqrt(displacementX * displacementX + displacementY * displacementY)
        distance = np.sum(np.sqrt(np.power(dx, 2) + np.power(dy, 2)))

        if distance / displacement < 1:
            print(f"{distance =}")
            print(f"{displacement =}")
            if timeInterval is not None:
                print(f"{durIdx = }")

        return distance / displacement

    def totalTimeNearWell(self, inProbe, wellName, radius=None, timeInterval=None, moveFlag=None):
        """
        timeInterval is in seconds, where 0 == start of probe or task (as specified in inProbe flag)
        radius is in cm
        return units: trode timestamps
        """

        if radius is None:
            # TODO just count total time in well by entry exit times
            # Note totalDwellTime as below doesn't work as is because moveFlag isn't accounted for
            # return self.totalDwellTime(inProbe, wellName, timeInterval=timeInterval):
            raise Exception("Unimplemented")

        if moveFlag is None:
            moveFlag = BTSession.MOVE_FLAG_ALL

        wx, wy = getWellPosCoordinates(wellName)

        if inProbe:
            ts = self.probePos_ts
            xs = self.probePosXs
            ys = self.probePosYs
            mv = self.probeIsMv
        else:
            ts = self.btPos_ts
            xs = self.btPosXs
            ys = self.btPosYs
            mv = self.btIsMv

        if moveFlag == self.MOVE_FLAG_MOVING:
            keepMv = mv
        elif moveFlag == self.MOVE_FLAG_STILL:
            keepMv = ~ mv
        else:
            keepMv = np.ones_like(mv).astype(bool)

        durIdx = self.timeIntervalToPosIdx(ts, timeInterval, noneVal=(0, len(mv)))
        inTime = np.zeros_like(mv).astype(bool)
        inTime[durIdx[0]:durIdx[1]] = True
        # if timeInterval is None:
        #     inTime = np.ones_like(mv).astype(bool)
        # else:
        #     imin = np.searchsorted(ts, ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE)
        #     imax = np.searchsorted(ts, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE)
        #     inTime = np.zeros_like(mv).astype(bool)
        #     inTime[imin:imax] = True

        distToWell = np.sqrt(np.power(wx - xs, 2) + np.power(wy - ys, 2)) * CM_PER_FT

        inRange = distToWell < radius
        keepFlag = inRange & inTime & keepMv

        tdiff = np.diff(ts, prepend=ts[0])
        return np.sum(tdiff[keepFlag])

    def getDelayFromSession(self, prevsession: BTSession, returnDateTimeDelta=False):
        """
        units: seconds (or dattimedelta if returnDateTimeDelta set to true)
        positive results indicates the session passed as an argument occured before this one
        """
        res = self.date - prevsession.date
        if returnDateTimeDelta:
            return res
        else:
            return res.total_seconds()

    def getLatencyToWell(self, inProbe, wellName, startTime=None, returnUnits: str = "ts", emptyVal=np.nan):
        """
        returnUnits in ["ts", "secs", "posIdx"]
        startTime should be in the same units expected for return
        """
        if returnUnits not in ["ts", "secs", "posIdx"]:
            raise ValueError("returnUnits incorrect")

        if startTime is not None:
            raise Exception("UNIMPLEMENTED")

        ents = self.entryExitTimes(inProbe, wellName, returnIdxs=(returnUnits == "posIdx"))[0]

        if len(ents) == 0:
            return emptyVal

        res = ents[0]
        if returnUnits == "posIdx":
            return res

        if inProbe:
            res -= self.probePos_ts[0]
        else:
            res -= self.btPos_ts[0]

        if returnUnits == "secs":
            res /= TRODES_SAMPLING_RATE

        return res

    def numStimsAtWell(self, wellName):
        """
        how many times did stim happen while rat was at the well during the task?
        """
        nw = np.array(self.btNearestWells)
        return np.count_nonzero(wellName == nw[self.btInterruption_posIdx])

    def numRipplesAtWell(self, inProbe, wellName):
        """
        how many times did ripple happen while rat was at the well?
        """
        if inProbe:
            ripPosIdxs = np.searchsorted(self.probePos_ts, np.array(
                [r.start_ts for r in self.probeRipsProbeStats]))
            nw = np.array(self.probeNearestWells)
        else:
            ripPosIdxs = np.searchsorted(self.btPos_ts, np.array(
                [r.start_ts for r in self.btRipsPreStats]))
            nw = np.array(self.btNearestWells)

        return np.count_nonzero(wellName == nw[ripPosIdxs])

    def gravityOfWell(self, inProbe, wellName, fromWells=allWellNames, emptyVal=np.nan, **kwargs):
        neighborWells = np.array([-9, -8, -7, -1, 0, 1, 7, 8, 9]) + wellName
        neighborWells = [w for w in neighborWells if (w in fromWells)]
        neighborExitIdxs = np.array([])
        neighborEntryIdxs = np.array([])
        for nw in neighborWells:
            ents, exts = self.entryExitTimes(inProbe, nw, returnIdxs=True, **kwargs)
            neighborEntryIdxs = np.concatenate((neighborEntryIdxs, ents))
            neighborExitIdxs = np.concatenate((neighborExitIdxs, exts))

        if len(neighborExitIdxs) == 0:
            return emptyVal

        keepEntryIdx = np.array([v not in neighborExitIdxs for v in neighborEntryIdxs])
        keepExitIdx = np.array([v not in neighborEntryIdxs for v in neighborExitIdxs])
        neighborEntryIdxs = sorted(neighborEntryIdxs[keepEntryIdx])
        neighborExitIdxs = sorted(neighborExitIdxs[keepExitIdx])

        wellEntryIdxs = self.entryExitTimes(inProbe, wellName, returnIdxs=True, **kwargs)[0]
        enteredHome = [any([wei <= v2 and wei >= v1 for wei in wellEntryIdxs])
                       for v1, v2 in zip(neighborEntryIdxs, neighborExitIdxs)]

        ret = np.count_nonzero(enteredHome) / len(neighborEntryIdxs)

        assert 0.0 <= ret <= 1.0

        return ret

    def getPasses(self, wellName: int, inProbe: bool, distance=1.5):
        if inProbe:
            x = self.probePosXs
            y = self.probePosYs
        else:
            x = self.btPosXs
            y = self.btPosYs

        wx, wy = getWellPosCoordinates(wellName)

        wellDist2 = np.power(x - np.array(wx), 2) + np.power(y - np.array(wy), 2)
        inRange = wellDist2 <= distance * distance
        borders = np.diff(inRange.astype(int), prepend=0, append=0)

        ents = np.nonzero(borders == 1)[0]
        exts = np.nonzero(borders == -1)[0]
        return ents, exts

    def getDotProductScore(self, inProbe: bool, wellName: int, timeInterval=None, moveFlag=None,
                           boutFlag=None, excludeTimesAtWell=True, binarySpotlight=False, spotlightAngle=70,
                           excursionFlag=None) -> float:
        if moveFlag is None:
            moveFlag = BTSession.MOVE_FLAG_ALL

        if inProbe:
            x = self.probePosXs
            y = self.probePosYs
            mv = self.probeIsMv[:-1]
            boutCats = self.probeBoutCategory[:-1]
            excursionCats = self.probeExcursionCategory[:-1]
            ts = self.probePos_ts
        else:
            x = self.btPosXs
            y = self.btPosYs
            mv = self.btIsMv[:-1]
            boutCats = self.btBoutCategory[:-1]
            excursionCats = self.btExcursionCategory[:-1]
            ts = self.btPos_ts
        wx, wy = getWellPosCoordinates(wellName)

        if moveFlag == self.MOVE_FLAG_MOVING:
            keepMv = mv
        elif moveFlag == self.MOVE_FLAG_STILL:
            keepMv = ~ mv
        else:
            keepMv = np.ones_like(mv).astype(bool)

        durIdx = self.timeIntervalToPosIdx(ts, timeInterval, noneVal=(0, len(keepMv)))
        keepTimeInt = np.zeros_like(keepMv).astype(bool)
        keepTimeInt[durIdx[0]:durIdx[1]] = True
        # if timeInterval is None:
        #     keepTimeInt = np.ones_like(keepMv)
        # else:
        #     assert x.shape == ts.shape
        #     keepTimeInt = np.zeros_like(keepMv).astype(bool)
        #     durIdx = np.searchsorted(ts, np.array(
        #         [ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE]))
        #     keepTimeInt[durIdx[0]:durIdx[1]] = True

        if boutFlag is None:
            keepBout = np.ones_like(boutCats).astype(bool)
        else:
            keepBout = boutCats == boutFlag

        if excursionFlag is None:
            keepExcursion = np.ones_like(excursionCats).astype(bool)
        else:
            keepExcursion = excursionCats == excursionFlag

        ents, exts = self.entryExitTimes(inProbe, wellName, returnIdxs=True)
        notAtWellFlag = np.ones_like(mv).astype(bool)
        if excludeTimesAtWell:
            for ent, ext in zip(ents, exts):
                notAtWellFlag[ent:ext+1] = False

        dx = np.diff(x)
        dy = np.diff(y)
        dwx = wx - np.array(x[1:])
        dwy = wy - np.array(y[1:])

        notStill = (dx != 0).astype(bool) | (dy != 0).astype(bool)
        keepFlag = keepBout & keepMv & keepTimeInt & notAtWellFlag & notStill & keepExcursion

        dx = dx[keepFlag]
        dy = dy[keepFlag]
        dwx = dwx[keepFlag]
        dwy = dwy[keepFlag]

        dots = dx * dwx + dy * dwy
        magp = np.sqrt(dx * dx + dy * dy)
        magw = np.sqrt(dwx * dwx + dwy * dwy)
        mags = magp * magw
        udots = dots / mags

        if binarySpotlight:
            angle = np.arccos(udots)
            allVals = (angle < np.deg2rad(spotlightAngle)).astype(float)
        else:
            # allVals = udots / magw
            allVals = udots

        return np.sum(allVals)

    def fillTimeCutoff(self) -> int:
        if self.probeFillTime is not None:
            return self.probeFillTime
        else:
            return 60*5
