import numpy as np
import scipy
from consts import TRODES_SAMPLING_RATE, allWellNames


class BTSession:
    """
    Contains all data for a session on Brad's task with probe
    Also has references to previous and next session.
    """

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

    PIXELS_PER_CM = 5.0

    def __init__(self):
        # ==================================
        # Info about the session
        # ==================================
        # The previous session chronologically, if one exists
        self.prevSession = None
        self.prevSessionDir = None
        # The next session chronologically, if one exists
        self.nextSession = None
        # date object representing the day of this recording
        self.date = None

        # Just the date part of session filename (i.e. "20200112")
        self.date_str = ""
        # Just the time part of session filename (i.e. "140259")
        self.time_str = ""
        # Data string in general. May modify in future by appending "S1" for first session if doing multiple in one day
        self.name = ""
        # name of raw data folder in which brad's task part of session was recorded
        self.bt_dir = ""
        # name of raw data folder in which ITI part of session was recorded. May be missing (empty string).
        # May be same as bt_dir
        self.iti_dir = ""
        # name of raw data folder in which probe part of session was recorded. May be missing (empty string).
        # May be same as bt_dir
        self.probe_dir = ""

        self.infoFileName = ""
        self.conditionGroup = -1

        # Some flags indicated whether ITI was recorded and whether ITI and probe are in the same rec file or not
        self.separate_iti_file = False
        self.recorded_iti = False
        self.separate_probe_file = False

        # more flags from info file
        self.ripple_detection_threshold = 0.0
        self.last_away_well = 0
        self.ended_on_home = False
        self.ITI_stim_on = False
        self.probe_stim_on = False

        self.bt_ended_at_well = None
        self.probe_ended_at_well = None
        self.probe_performed = None

        # Any other notes that are stored in the info file are added here. Each list entry is one line from that file
        self.notes = []

        # position coordinates of home well
        self.home_x = 0
        self.home_y = 0
        self.away_xs = []
        self.away_ys = []

        # Well number of home well
        self.home_well = 0
        self.home_well_idx_in_allwells = 0
        self.ctrl_home_well = 0
        self.ctrl_home_well_idx_in_allwells = 0
        # Well number of away wells
        self.away_wells = []
        self.num_away_found = 0
        self.num_home_found = 0
        self.foundWells = None

        # Flags indicating stim condition
        self.isRippleInterruption = False
        self.isDelayedInterruption = False
        self.isNoInterruption = False
        self.ripple_detection_tetrodes = []
        self.ripple_baseline_tetrode = None
        # self.ripple_detection_tetrodes = [37]

        # Rat weight if it was recorded
        self.rat_weight = 0

        # For purposes of light on/off times, ignore this many seconds at start of video
        self.trodesLightsIgnoreSeconds = None

        # ==================================
        # Raw data
        # ==================================
        # Position data during brad's task
        self.bt_pos_ts = []
        self.bt_pos_xs = []
        self.bt_pos_ys = []

        # Position data during probe
        self.probe_pos_ts = []
        self.probe_pos_xs = []
        self.probe_pos_ys = []

        # LFP data is huge, so only load on demand
        # brad's task
        self.bt_lfp_fnames = []
        self.bt_lfp_baseline_fname = None
        self.bt_lfp_start_ts = 0
        self.bt_lfp_end_ts = 0
        self.bt_lfp_start_idx = 0
        self.bt_lfp_end_idx = 0

        # ITI
        self.iti_lfp_fnames = []
        self.iti_lfp_start_ts = 0
        self.iti_lfp_end_ts = 0
        self.iti_lfp_start_idx = 0
        self.iti_lfp_end_idx = 0

        # probe
        self.probe_lfp_fnames = []
        self.probe_lfp_start_ts = 0
        self.probe_lfp_end_ts = 0
        self.probe_lfp_start_idx = 0
        self.probe_lfp_end_idx = 0

        self.interruption_timestamps = np.array([])
        self.artifact_timestamps = np.array([])

        # ==================================
        # Analyzed data: Brad's task
        # ==================================
        self.home_well_find_times = []
        self.home_well_find_pos_idxs = []
        self.home_well_leave_times = []
        self.home_well_leave_pos_idxs = []
        self.home_well_latencies = []
        self.home_well_displacements = []
        self.home_well_distances = []

        self.away_well_find_times = []
        self.away_well_find_pos_idxs = []
        self.away_well_leave_times = []
        self.away_well_leave_pos_idxs = []
        self.away_well_latencies = []
        self.away_well_displacements = []
        self.away_well_distances = []
        self.visited_away_wells = []

        # ==================================
        # Analyzed data: Probe
        # ==================================
        # analyzing paths, separating by velocity
        self.bt_vel_cm_s = []
        self.bt_is_mv = []
        self.bt_mv_xs = []
        self.bt_still_xs = []
        self.bt_mv_ys = []
        self.bt_still_ys = []
        self.probe_vel_cm_s = []
        self.probe_is_mv = []
        self.probe_mv_xs = []
        self.probe_still_xs = []
        self.probe_mv_ys = []
        self.probe_still_ys = []

        # avg dist to home and times at which rat entered home region
        self.probe_nearest_wells = []
        self.probe_well_entry_idxs = []
        self.probe_well_exit_idxs = []
        self.probe_well_entry_times = []
        self.probe_well_exit_times = []
        self.probe_home_well_entry_times = []
        self.probe_home_well_exit_times = []
        # self.probe_mean_dist_to_home_well = []
        # self.probe_mv_mean_dist_to_home_well = []
        # self.probe_still_mean_dist_to_home_well = []
        self.probe_ctrl_home_well_entry_times = []
        # self.probe_mean_dist_to_ctrl_home_well = []
        # self.probe_mv_mean_dist_to_ctrl_home_well = []
        # self.probe_still_mean_dist_to_ctrl_home_well = []

        self.bt_nearest_wells = []
        self.bt_well_entry_idxs = []
        self.bt_well_exit_idxs = []
        self.bt_well_entry_times = []
        self.bt_well_exit_times = []
        self.bt_home_well_entry_times = []
        self.bt_home_well_exit_times = []
        # self.bt_mean_dist_to_home_well = []
        # self.bt_mv_mean_dist_to_home_well = []
        # self.bt_still_mean_dist_to_home_well = []
        self.bt_ctrl_home_well_entry_times = []
        # self.bt_mean_dist_to_ctrl_home_well = []
        # self.bt_mv_mean_dist_to_ctrl_home_well = []
        # self.bt_still_mean_dist_to_ctrl_home_well = []

        self.bt_quadrants = []
        self.home_quadrant = None
        self.bt_well_entry_idxs_ninc = []
        self.bt_well_exit_idxs_ninc = []
        self.bt_well_entry_times_ninc = []
        self.bt_well_exit_times_ninc = []
        self.bt_quadrant_entry_idxs = []
        self.bt_quadrant_exit_idxs = []
        self.bt_quadrant_entry_times = []
        self.bt_quadrant_exit_times = []

        self.probe_quadrants = []
        self.probe_well_entry_idxs_ninc = []
        self.probe_well_exit_idxs_ninc = []
        self.probe_well_entry_times_ninc = []
        self.probe_well_exit_times_ninc = []
        self.probe_quadrant_entry_idxs = []
        self.probe_quadrant_exit_idxs = []
        self.probe_quadrant_entry_times = []
        self.probe_quadrant_exit_times = []

        self.well_coords_map = {}

        self.bt_ball_displacement = None
        self.bt_ballisticity = np.array([])
        self.probe_ball_displacement = None
        self.probe_ballisticity = None

        self.bt_curvature = np.array([])
        self.bt_curvature_i1 = np.array([])
        self.bt_curvature_i2 = np.array([])
        self.bt_curvature_dxf = np.array([])
        self.bt_curvature_dyf = np.array([])
        self.bt_curvature_dxb = np.array([])
        self.bt_curvature_dyb = np.array([])

        self.probe_curvature = np.array([])
        self.probe_curvature_i1 = np.array([])
        self.probe_curvature_i2 = np.array([])
        self.probe_curvature_dxf = np.array([])
        self.probe_curvature_dyf = np.array([])
        self.probe_curvature_dxb = np.array([])
        self.probe_curvature_dyb = np.array([])

        self.probe_latency_to_well = []

        self.bt_sm_vel = []
        self.bt_is_in_pause = []
        self.bt_is_in_explore = []
        self.bt_explore_bout_starts = []
        self.bt_explore_bout_ends = []
        self.bt_explore_bout_lens = []
        self.bt_bout_category = np.array([])
        self.bt_bout_label = np.array([])

        self.probe_sm_vel = []
        self.probe_is_in_pause = []
        self.probe_is_in_explore = []
        self.probe_explore_bout_starts = []
        self.probe_explore_bout_ends = []
        self.probe_explore_bout_lens = []
        self.probe_bout_category = np.array([])
        self.probe_bout_label = np.array([])

        self.bt_excursion_category = []
        self.bt_excursion_starts = []
        self.bt_excursion_ends = []
        self.bt_excursion_lens_secs = np.array([])

        self.probe_excursion_category = []
        self.probe_excursion_starts = []
        self.probe_excursion_ends = []
        self.probe_excursion_lens_secs = np.array([])

        self.well_sniff_times_entry = []
        self.well_sniff_times_exit = []
        self.bt_well_sniff_times_entry = []
        self.bt_well_sniff_times_exit = []
        self.probe_well_sniff_times_entry = []
        self.probe_well_sniff_times_exit = []

        self.sniff_pre_trial_light_off = None
        self.sniff_trial_start = None
        self.sniff_trial_stop = None
        self.sniff_probe_start = None
        self.sniff_probe_stop = None
        self.sniff_post_probe_light_on = None

        self.positionFromDeepLabCut = None

    def get_well_coordinates(self, wellName):
        return self.well_coords_map[str(wellName)]

    def avg_dist_to_well(self, inProbe, wellName, timeInterval=None, moveFlag=None, avgFunc=np.nanmean):
        """
        timeInterval is in seconds, where 0 == start of probe or task (as specified in inProbe flag)
        return units: cm
        """

        if moveFlag is None:
            moveFlag = BTSession.MOVE_FLAG_ALL

        wx, wy = self.get_well_coordinates(wellName)

        if inProbe:
            ts = np.array(self.probe_pos_ts)
            if moveFlag == self.MOVE_FLAG_MOVING:
                xs = np.array(self.probe_mv_xs)
                ys = np.array(self.probe_mv_ys)
            elif moveFlag == self.MOVE_FLAG_STILL:
                xs = np.array(self.probe_still_xs)
                ys = np.array(self.probe_still_ys)
            else:
                assert moveFlag == BTSession.MOVE_FLAG_ALL
                xs = np.array(self.probe_pos_xs)
                ys = np.array(self.probe_pos_ys)
        else:
            ts = np.array(self.bt_pos_ts)
            if moveFlag == self.MOVE_FLAG_MOVING:
                xs = np.array(self.bt_mv_xs)
                ys = np.array(self.bt_mv_ys)
            if moveFlag == self.MOVE_FLAG_STILL:
                xs = np.array(self.bt_still_xs)
                ys = np.array(self.bt_still_ys)
            else:
                assert moveFlag == BTSession.MOVE_FLAG_ALL
                xs = np.array(self.bt_pos_xs)
                ys = np.array(self.bt_pos_ys)

        # Note nan values are ignored. This is intentional, so caller
        # can just consider some time points by making all other values nan
        # If timeInterval is None, use all times points. Otherwise, take only timeInterval in seconds
        if timeInterval is not None:
            assert xs.shape == ts.shape
            dur_idx = np.searchsorted(ts, np.array(
                [ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE]))
            xs = xs[dur_idx[0]:dur_idx[1]]
            ys = ys[dur_idx[0]:dur_idx[1]]

        dist_to_well = np.sqrt(np.power(wx - np.array(xs), 2) +
                               np.power(wy - np.array(ys), 2))
        return avgFunc(dist_to_well / self.PIXELS_PER_CM)

    def avg_dist_to_home_well(self, inProbe, timeInterval=None, moveFlag=None, avgFunc=np.nanmean):
        """
        return units: cm
        """
        return self.avg_dist_to_well(inProbe, self.home_well, timeInterval=timeInterval,
                                     moveFlag=moveFlag, avgFunc=avgFunc)

    def entry_exit_times(self, inProbe, wellName, timeInterval=None, includeNeighbors=False, excludeReward=False,
                         returnIdxs=False, includeEdgeOverlap=True, subtractT0=False):
        """
        return units: trodes timestamps, unless returnIdxs==True
        """
        if subtractT0:
            raise Exception("Unimplemented")

        # wellName == "QX" ==> look at quadrant X instead of a well
        if isinstance(wellName, str) and wellName[0] == 'Q':
            isQuad = True
            quadName = int(wellName[1])
        else:
            isQuad = False
            wellIdx = np.argmax(allWellNames == wellName)

        assert not (includeNeighbors and isQuad)
        assert (not excludeReward) or ((not isQuad) and (not includeNeighbors))

        if inProbe:
            if isQuad:
                if returnIdxs:
                    ents = self.probe_quadrant_entry_idxs[quadName]
                    exts = self.probe_quadrant_entry_idxs[quadName]
                else:
                    ents = self.probe_quadrant_entry_times[quadName]
                    exts = self.probe_quadrant_exit_times[quadName]
            else:
                if includeNeighbors:
                    if returnIdxs:
                        ents = self.probe_well_entry_idxs_ninc[wellIdx]
                        exts = self.probe_well_exit_idxs_ninc[wellIdx]
                    else:
                        ents = self.probe_well_entry_times_ninc[wellIdx]
                        exts = self.probe_well_exit_times_ninc[wellIdx]
                else:
                    if returnIdxs:
                        ents = self.probe_well_entry_idxs[wellIdx]
                        exts = self.probe_well_exit_idxs[wellIdx]
                    else:
                        ents = self.probe_well_entry_times[wellIdx]
                        exts = self.probe_well_exit_times[wellIdx]
        else:
            if isQuad:
                if returnIdxs:
                    ents = self.bt_quadrant_entry_idxs[quadName]
                    ents = self.bt_quadrant_entry_idxs[quadName]
                else:
                    ents = self.bt_quadrant_entry_times[quadName]
                    exts = self.bt_quadrant_exit_times[quadName]
            else:
                if includeNeighbors:
                    if returnIdxs:
                        ents = self.bt_well_entry_idxs_ninc[wellIdx]
                        exts = self.bt_well_exit_idxs_ninc[wellIdx]
                    else:
                        ents = self.bt_well_entry_times_ninc[wellIdx]
                        exts = self.bt_well_exit_times_ninc[wellIdx]
                else:
                    if returnIdxs:
                        ents = self.bt_well_entry_idxs[wellIdx]
                        exts = self.bt_well_exit_idxs[wellIdx]
                    else:
                        ents = self.bt_well_entry_times[wellIdx]
                        exts = self.bt_well_exit_times[wellIdx]

        if excludeReward:
            if wellName == self.home_well:
                if len(self.home_well_find_times) == 0:
                    # don't know when reward was delivered
                    raise Exception("missing home well find times")

                for ft, lt in zip(self.home_well_find_times, self.home_well_leave_times):
                    wt = (ft + lt) / 2.0
                    found = False
                    todel = -1
                    for ei, (ent, ext) in enumerate(zip(ents, exts)):
                        if wt >= ent and wt <= ext:
                            found = True
                            todel = ei
                            break

                    if not found:
                        raise Exception("couldn't find home well find time")

                    ents = np.delete(ents, todel)
                    exts = np.delete(exts, todel)

            elif wellName in self.visited_away_wells:
                if len(self.away_well_find_times) == 0:
                    # don't know when rward was deliveered
                    raise Exception("missing away well find times")

                found = False
                for ei, awi in enumerate(self.visited_away_wells):
                    if awi == wellName:
                        wt = (self.away_well_find_times[ei] + self.away_well_leave_times[ei]) / 2.0
                        found = True
                        break

                if not found:
                    raise Exception("can't find this away well find time")

                found = False
                todel = -1
                for ei, (ent, ext) in enumerate(zip(ents, exts)):
                    if wt >= ent and wt <= ext:
                        found = True
                        todel = ei
                        break

                if not found:
                    raise Exception("can't find this away well find time")

                ents = np.delete(ents, todel)
                exts = np.delete(exts, todel)

        if timeInterval is not None:
            if inProbe:
                ts = self.probe_pos_ts
            else:
                ts = self.bt_pos_ts

            mint = ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE
            maxt = ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE
            if returnIdxs:
                mint = np.searchsorted(ts, mint)
                maxt = np.searchsorted(ts, maxt)

            keepflag = np.logical_and(np.array(ents) < maxt, np.array(exts) > mint)
            ents = ents[keepflag]
            exts = exts[keepflag]
            if len(ents) > 0:
                if ents[0] < mint:
                    if includeEdgeOverlap:
                        ents[0] = mint
                    else:
                        ents = np.delete(ents, 0)
                        exts = np.delete(exts, 0)
            if len(ents) > 0:
                if exts[-1] > maxt:
                    if includeEdgeOverlap:
                        exts[-1] = maxt
                    else:
                        ents = np.delete(ents, -1)
                        exts = np.delete(exts, -1)

        return ents, exts

    def sniff_entry_exit_times(self, inProbe, wellName, timeInterval=None, excludeReward=False,
                               includeEdgeOverlap=True, subtractT0=False):
        """
        return units: milliseconds
        """
        if subtractT0 or excludeReward:
            raise Exception("Unimplemented")

        wellIdx = np.argmax(allWellNames == wellName)

        if inProbe:
            ents = np.array(self.probe_well_sniff_times_entry[wellIdx])
            exts = np.array(self.probe_well_sniff_times_exit[wellIdx])
        else:
            ents = np.array(self.bt_well_sniff_times_entry[wellIdx])
            exts = np.array(self.bt_well_sniff_times_exit[wellIdx])

        if excludeReward:
            if wellName == self.home_well:
                if len(self.home_well_find_times) == 0:
                    # don't know when reward was delivered
                    raise Exception("missing home well find times")

                for ft, lt in zip(self.home_well_find_times, self.home_well_leave_times):
                    wt = (ft + lt) / 2.0
                    found = False
                    todel = -1
                    for ei, (ent, ext) in enumerate(zip(ents, exts)):
                        if wt >= ent and wt <= ext:
                            found = True
                            todel = ei
                            break

                    if not found:
                        raise Exception("couldn't find home well find time")

                    ents = np.delete(ents, todel)
                    exts = np.delete(exts, todel)

            elif wellName in self.visited_away_wells:
                if len(self.away_well_find_times) == 0:
                    # don't know when rward was deliveered
                    raise Exception("missing away well find times")

                found = False
                for ei, awi in enumerate(self.visited_away_wells):
                    if awi == wellName:
                        wt = (self.away_well_find_times[ei] + self.away_well_leave_times[ei]) / 2.0
                        found = True
                        break

                if not found:
                    raise Exception("can't find this away well find time")

                found = False
                todel = -1
                for ei, (ent, ext) in enumerate(zip(ents, exts)):
                    if wt >= ent and wt <= ext:
                        found = True
                        todel = ei
                        break

                if not found:
                    raise Exception("can't find this away well find time")

                ents = np.delete(ents, todel)
                exts = np.delete(exts, todel)

        if timeInterval is not None:
            if inProbe:
                mint = self.sniff_probe_start + timeInterval[0] * 1000
                maxt = self.sniff_probe_start + timeInterval[1] * 1000
            else:
                mint = self.sniff_bt_start + timeInterval[0] * 1000
                maxt = self.sniff_bt_start + timeInterval[1] * 1000

            keepflag = np.logical_and(np.array(ents) < maxt, np.array(exts) > mint)
            ents = ents[keepflag]
            exts = exts[keepflag]
            if len(ents) > 0:
                if ents[0] < mint:
                    if includeEdgeOverlap:
                        ents[0] = mint
                    else:
                        ents = np.delete(ents, 0)
                        exts = np.delete(exts, 0)
            if len(ents) > 0:
                if exts[-1] > maxt:
                    if includeEdgeOverlap:
                        exts[-1] = maxt
                    else:
                        ents = np.delete(ents, -1)
                        exts = np.delete(exts, -1)

        return ents, exts

    def avg_continuous_measure_at_well(self, inProbe, wellName, yvals,
                                       timeInterval=None, avgTypeFlag=None, avgFunc=np.nanmean, excludeReward=False,
                                       includeNeighbors=False):
        if avgTypeFlag is None:
            avgTypeFlag = BTSession.AVG_FLAG_OVER_TIME

        wenis, wexis = self.entry_exit_times(
            inProbe, wellName, timeInterval=timeInterval, excludeReward=excludeReward,
            includeNeighbors=includeNeighbors, returnIdxs=True)

        if len(wenis) == 0:
            return np.nan

        res_all = []
        for weni, wexi in zip(wenis, wexis):
            if wexi > yvals.size:
                continue
            res_all.append(yvals[weni:wexi])

        if avgTypeFlag == BTSession.AVG_FLAG_OVER_TIME:
            return avgFunc(np.concatenate(res_all))
        elif avgTypeFlag == BTSession.AVG_FLAG_OVER_VISITS:
            return avgFunc([avgFunc(x) for x in res_all])
        else:
            assert False

    def avg_ballisticity_at_well(self, inProbe, wellName, timeInterval=None, avgTypeFlag=None, avgFunc=np.nanmean,
                                 includeNeighbors=False):
        if inProbe:
            yvals = self.probe_ballisticity
        else:
            yvals = self.bt_ballisticity
        # note this method is slightly different from original (but better!)
        # original would count entire visit to well as long as entry was before cutoff point

        return self.avg_continuous_measure_at_well(inProbe, wellName, yvals, timeInterval=timeInterval,
                                                   avgTypeFlag=avgTypeFlag,
                                                   avgFunc=avgFunc, includeNeighbors=includeNeighbors)

    def avg_ballisticity_at_home_well(self, inProbe, timeInterval=None, avgTypeFlag=None, avgFunc=np.nanmean,
                                      includeNeighbors=False):
        return self.avg_ballisticity_at_well(inProbe, self.home_well, timeInterval=timeInterval,
                                             avgTypeFlag=avgTypeFlag, avgFunc=avgFunc,
                                             includeNeighbors=includeNeighbors)

    def avg_curvature_at_well(self, inProbe, wellName, timeInterval=None, avgTypeFlag=None, avgFunc=np.nanmean,
                              includeNeighbors=False):
        if inProbe:
            yvals = self.probe_curvature
        else:
            yvals = self.bt_curvature
        # note this method is slightly different from original (but better!)
        # original would count entire visit to well as long as entry was before cutoff point

        return self.avg_continuous_measure_at_well(inProbe, wellName, yvals, timeInterval=timeInterval,
                                                   avgTypeFlag=avgTypeFlag, avgFunc=avgFunc,
                                                   includeNeighbors=includeNeighbors)

    def avg_curvature_at_home_well(self, inProbe, timeInterval=None, avgTypeFlag=None, avgFunc=np.nanmean,
                                   includeNeighbors=False):
        return self.avg_curvature_at_well(inProbe, self.home_well, timeInterval=timeInterval, avgTypeFlag=avgTypeFlag,
                                          avgFunc=avgFunc, includeNeighbors=includeNeighbors)

    def dwell_times(self, inProbe, wellName, timeInterval=None, excludeReward=False, includeNeighbors=False):
        """
        return units: trodes timestamps
        """
        ents, exts = self.entry_exit_times(
            inProbe, wellName, timeInterval=timeInterval, includeNeighbors=includeNeighbors,
            excludeReward=excludeReward, returnIdxs=False)

        return np.array(exts) - np.array(ents)

    def num_well_entries(self, inProbe, wellName, timeInterval=None, excludeReward=False, includeNeighbors=False):
        ents, _ = self.entry_exit_times(
            inProbe, wellName, timeInterval=None, includeNeighbors=includeNeighbors, excludeReward=excludeReward,
            returnIdxs=False)

        if inProbe:
            t0 = self.probe_pos_ts[0]
        else:
            t0 = self.bt_pos_ts[0]

        # now we have ents, exts, t0
        # should filter for timeInterval
        if timeInterval is not None:
            mint = t0 + timeInterval[0] * TRODES_SAMPLING_RATE
            maxt = t0 + timeInterval[1] * TRODES_SAMPLING_RATE
            ents = ents[np.logical_and(ents > mint, ents < maxt)]

        return ents.size

    def total_dwell_time(self, inProbe, wellName, timeInterval=None, excludeReward=False, includeNeighbors=False):
        """
        return units: seconds
        """
        return np.sum(self.dwell_times(inProbe, wellName, timeInterval=timeInterval, excludeReward=excludeReward,
                                       includeNeighbors=includeNeighbors) / TRODES_SAMPLING_RATE)

    def total_converted_dwell_time(self, inProbe, wellName):
        """
        return units: seconds
        """
        if not inProbe:
            raise Exception("Unimplemented")

        m = scipy.stats.mode(np.diff(self.sniffClassificationT))
        k = m[0][0] / 1000.0
        return np.count_nonzero(self.sniffClassificationNearestWell == wellName) * k

    def avg_dwell_time(self, inProbe, wellName, timeInterval=None, avgFunc=np.nanmean, excludeReward=False,
                       includeNeighbors=False, emptyVal=np.nan):
        """
        return units: seconds
        """
        ret = self.dwell_times(inProbe, wellName, timeInterval=timeInterval,
                               excludeReward=excludeReward, includeNeighbors=includeNeighbors)
        if len(ret) == 0:
            # print("returning emptyval for well {}".format(wellName))
            return emptyVal
        else:
            # print("ret is {} for well {}".format(ret, wellName))
            return avgFunc(ret / TRODES_SAMPLING_RATE)

    def num_sniffs(self, inProbe, wellName, timeInterval=None, excludeReward=False):
        # Just for now, hacking in excluded rewards
        ents, _ = self.sniff_entry_exit_times(
            inProbe, wellName, timeInterval=timeInterval)
        # inProbe, wellName, timeInterval=timeInterval, excludeReward=excludeReward)

        ret = ents.size

        if excludeReward and not inProbe:
            if wellName == self.home_well:
                ret -= self.num_home_found
            elif wellName in self.visited_away_wells:
                ret -= 1

        return ret

    def sniff_times(self, inProbe, wellName, timeInterval=None, excludeReward=False):
        """
        return units: milliseconds
        """
        ents, exts = self.sniff_entry_exit_times(
            inProbe, wellName, timeInterval=timeInterval, excludeReward=excludeReward)

        return exts - ents

    def avg_sniff_time(self, inProbe, wellName, timeInterval=None, avgFunc=np.nanmean, excludeReward=False,
                       emptyVal=None):
        """
        return units: seconds
        """
        ret = self.sniff_times(inProbe, wellName, timeInterval=timeInterval,
                               excludeReward=excludeReward)
        if emptyVal is not None and len(ret) == 0:
            return emptyVal
        else:
            return avgFunc(ret / 1000.0)

    def total_sniff_time(self, inProbe, wellName, timeInterval=None, excludeReward=False):
        """
        return units: seconds
        """
        return np.sum(self.sniff_times(inProbe, wellName, timeInterval=timeInterval, excludeReward=excludeReward)
                      / 1000.0)

    def num_bouts(self, inProbe, timeInterval=None):
        if inProbe:
            lbls = self.probe_bout_label
            ts = self.probe_pos_ts
        else:
            lbls = self.bt_bout_label
            ts = self.bt_pos_ts

        if timeInterval is not None:
            imin = np.searchsorted(ts, ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE)
            imax = np.searchsorted(ts, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE)
            lbls = lbls[imin:imax]

        return len(set(lbls) - set([0]))

    def num_bouts_where_well_was_visited(self, inProbe, wellName, timeInterval=None, excludeReward=False,
                                         includeNeighbors=False):
        ents, exts = self.entry_exit_times(
            inProbe, wellName, timeInterval=timeInterval, includeNeighbors=includeNeighbors,
            excludeReward=excludeReward, returnIdxs=True)

        if inProbe:
            lbls = self.probe_bout_label
        else:
            lbls = self.bt_bout_label

        res = set([])
        for enti, exti in zip(ents, exts):
            res = res | set(lbls[enti:exti])

        res = res - set([0])
        return len(res)

    def pct_bouts_where_well_was_visited(self, inProbe, wellName, timeInterval=None, excludeReward=False,
                                         includeNeighbors=False, boutsInterval=None):
        # bouts interval is inclusive first, exclusive last
        if boutsInterval is not None:
            assert timeInterval is None
            if inProbe:
                bout_starts = self.probe_explore_bout_starts
                bout_ends = self.probe_explore_bout_ends
            else:
                bout_starts = self.bt_explore_bout_starts
                bout_ends = self.bt_explore_bout_ends

            if len(bout_ends) >= boutsInterval[1]:
                timeInterval = [bout_starts[boutsInterval[0]], bout_ends[boutsInterval[1] - 1]]
            else:
                timeInterval = [bout_starts[boutsInterval[0]], bout_ends[len(bout_ends) - 1]]

        denom = self.num_bouts(inProbe, timeInterval=timeInterval)
        if denom == 0:
            return np.nan

        return self.num_bouts_where_well_was_visited(inProbe, wellName, timeInterval=timeInterval,
                                                     excludeReward=excludeReward, includeNeighbors=includeNeighbors) / \
            denom

    def prop_time_in_bout_state(self, inProbe, boutState, timeInterval=None):
        if inProbe:
            cats = self.probe_bout_category
            ts = self.probe_pos_ts
        else:
            cats = self.bt_bout_category
            ts = self.bt_pos_ts

        if timeInterval is None:
            imin = 0
            imax = len(ts)
        else:
            imin = np.searchsorted(ts, ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE)
            imax = np.searchsorted(ts, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE)

        cats = cats[imin:imax]
        if float(cats.size) == 0:
            print(len(self.bt_bout_category))
            print(inProbe, imin, imax, timeInterval)
        return float(np.count_nonzero(cats == boutState)) / float(cats.size)

    def mean_vel(self, inProbe, onlyMoving=False, timeInterval=None):
        """
        return units: cm/s
        """
        if inProbe:
            vel = self.probe_vel_cm_s
            ts = self.probe_pos_ts
            is_mv = self.probe_is_mv
        else:
            vel = self.bt_vel_cm_s
            ts = self.bt_pos_ts
            is_mv = self.bt_is_mv

        if timeInterval is None:
            imin = 0
            imax = len(ts)
        else:
            imin = np.searchsorted(ts, ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE)
            imax = np.searchsorted(ts, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE)

        vel = vel[imin:imax]
        if onlyMoving:
            vel = vel[is_mv[imin:imax]]

        return np.nanmean(vel)

    def ctrl_well_for_well(self, wellName):
        return 49 - wellName

    def path_optimality(self, inProbe, timeInterval=None, wellName=None, emptyVal=np.nan):
        if (timeInterval is None) == (wellName is None):
            raise Exception("Gimme a time interval or a well name plz (but not both)")

        if inProbe:
            xs = self.probe_pos_xs
            ys = self.probe_pos_ys
            ts = self.probe_pos_ts
        else:
            xs = self.bt_pos_xs
            ys = self.bt_pos_ys
            ts = self.bt_pos_ts

        if wellName is not None:
            ents = self.entry_exit_times(inProbe, wellName, returnIdxs=True)[0]
            if len(ents) == 0:
                return emptyVal
            ei = ents[0]
            displacement_x = xs[ei] - xs[0]
            displacement_y = ys[ei] - ys[0]
            dx = np.diff(xs[0:ei])
            dy = np.diff(ys[0:ei])

        if timeInterval is not None:
            idxs = np.searchsorted(ts, np.array(timeInterval) * TRODES_SAMPLING_RATE + ts[0])
            displacement_x = xs[idxs[1]] - xs[idxs[0]]
            displacement_y = ys[idxs[1]] - ys[idxs[0]]
            dx = np.diff(xs[idxs[0]:idxs[1]])
            dy = np.diff(ys[idxs[0]:idxs[1]])

        displacement = np.sqrt(displacement_x * displacement_x + displacement_y * displacement_y)
        distance = np.sum(np.sqrt(np.power(dx, 2) + np.power(dy, 2)))

        return distance / displacement

    def total_time_near_well(self, inProbe, wellName, radius=None, timeInterval=None, moveFlag=None):
        """
        timeInterval is in seconds, where 0 == start of probe or task (as specified in inProbe flag)
        radius is in cm
        return units: trode timestamps
        """

        if radius is None:
            # TODO just count total time in well by entry exit times
            raise Exception("Unimplemented")

        if moveFlag is None:
            moveFlag = BTSession.MOVE_FLAG_ALL

        wx, wy = self.get_well_coordinates(wellName)

        if inProbe:
            ts = np.array(self.probe_pos_ts)
            if moveFlag == self.MOVE_FLAG_MOVING:
                xs = np.array(self.probe_mv_xs)
                ys = np.array(self.probe_mv_ys)
            elif moveFlag == self.MOVE_FLAG_STILL:
                xs = np.array(self.probe_still_xs)
                ys = np.array(self.probe_still_ys)
            else:
                assert moveFlag == BTSession.MOVE_FLAG_ALL
                xs = np.array(self.probe_pos_xs)
                ys = np.array(self.probe_pos_ys)
        else:
            ts = np.array(self.bt_pos_ts)
            if moveFlag == self.MOVE_FLAG_MOVING:
                xs = np.array(self.bt_mv_xs)
                ys = np.array(self.bt_mv_ys)
            if moveFlag == self.MOVE_FLAG_STILL:
                xs = np.array(self.bt_still_xs)
                ys = np.array(self.bt_still_ys)
            else:
                assert moveFlag == BTSession.MOVE_FLAG_ALL
                xs = np.array(self.bt_pos_xs)
                ys = np.array(self.bt_pos_ys)

        # Note nan values are ignored. This is intentional, so caller
        # can just consider some time points by making all other values nan
        # If timeInterval is None, use all times points. Otherwise, take only timeInterval in seconds
        if timeInterval is not None:
            assert xs.shape == ts.shape
            dur_idx = np.searchsorted(ts, np.array(
                [ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE]))
            xs = xs[dur_idx[0]:dur_idx[1]]
            ys = ys[dur_idx[0]:dur_idx[1]]
            ts = ts[dur_idx[0]:dur_idx[1]]

        dist_to_well = np.sqrt(np.power(wx - np.array(xs), 2) +
                               np.power(wy - np.array(ys), 2)) / self.PIXELS_PER_CM

        count_point = dist_to_well < radius
        tdiff = np.diff(ts, prepend=ts[0])
        return np.sum(tdiff[count_point])

    def getDelayFromSession(self, prevsession, returnDateTimeDelta=False):
        """
        units: seconds (or dattimedelta if returnDateTimeDelta set to true)
        positive results indicates the session passed as an argument occured before this one
        """
        # res = "this ({}) to other ({}) = {}".format(
        # self.date_str, prevsession.date_str, self.date-prevsession.date)
        # return res
        res = self.date - prevsession.date
        if returnDateTimeDelta:
            return res
        else:
            return res.total_seconds()

    def getLatencyToWell(self, inProbe, wellName, startTime=None, returnIdxs=False, returnSeconds=False,
                         emptyVal=np.nan):
        """
        units are trodes timestamps or idxs or seconds
        startTime should be in the same units expected for return
        """
        assert not (returnIdxs and returnSeconds)

        ts = self.entry_exit_times(inProbe, wellName, returnIdxs=returnIdxs)[0]

        if len(ts) == 0:
            return emptyVal

        res = ts[0]
        if returnIdxs:
            return res

        if inProbe:
            res -= self.probe_pos_ts[0]
        else:
            res -= self.bt_pos_ts[0]

        if returnSeconds:
            res /= TRODES_SAMPLING_RATE

        return res

    def numStimsAtWell(self, wellName):
        """
        how many times did stim happen while rat was at the well during the task?
        """
        nw = np.array(self.bt_nearest_wells)
        return np.count_nonzero(wellName == nw[self.bt_interruption_pos_idxs])

    def numRipplesAtWell(self, inProbe, wellName):
        """
        how many times did ripple happen while rat was at the well?
        """
        if inProbe:
            ripPosIdxs = np.searchsorted(self.probe_pos_ts, self.probeRipStartTimestamps)
            nw = np.array(self.probe_nearest_wells)
            return np.count_nonzero(wellName == nw[ripPosIdxs])
        else:
            ripPosIdxs = np.searchsorted(self.bt_pos_ts, self.btRipStartTimestampsPreStats)
            nw = np.array(self.bt_nearest_wells)
            return np.count_nonzero(wellName == nw[ripPosIdxs])

    def gravityOfWell_old(self, inProbe, wellName, timeInterval=None, fromWells=allWellNames, emptyVal=np.nan):
        neighborWells = np.array([-9, -8, -7, -1, 1, 7, 8, 9]) + wellName
        neighborWells = [w for w in neighborWells if (w in fromWells)]
        neighborExitIdxs = []
        neighborEntryIdxs = []
        # numNonZero = 0
        for nw in neighborWells:
            ents, exts = self.entry_exit_times(inProbe, nw,
                                               timeInterval=timeInterval, returnIdxs=True)
            neighborEntryIdxs += list(ents)
            neighborExitIdxs += list(exts)
            # if len(ents) > 0:
            #     numNonZero += 1

        if len(neighborExitIdxs) == 0:
            # print("emptyval ret, neighbor wells was {}".format(neighborWells))
            return emptyVal

        wellEntryIdxs = self.entry_exit_times(
            inProbe, wellName, timeInterval=timeInterval, returnIdxs=True)[0] - 1

        # print(wellEntryIdxs)
        ret = len([nwei for nwei in neighborExitIdxs if nwei in wellEntryIdxs]) / \
            len([nwei for nwei in neighborExitIdxs if nwei + 1 not in neighborEntryIdxs])
        # print(timeInterval, ret)

        return ret

    def gravityOfWell(self, inProbe, wellName, timeInterval=None, fromWells=allWellNames, emptyVal=np.nan):
        neighborWells = np.array([-9, -8, -7, -1, 0, 1, 7, 8, 9]) + wellName
        neighborWells = [w for w in neighborWells if (w in fromWells)]
        neighborExitIdxs = []
        neighborEntryIdxs = []
        # numNonZero = 0
        for nw in neighborWells:
            ents, exts = self.entry_exit_times(inProbe, nw,
                                               timeInterval=timeInterval, returnIdxs=True)
            neighborEntryIdxs += list(ents)
            neighborExitIdxs += list(exts)
            # if len(ents) > 0:
            #     numNonZero += 1

        if len(neighborExitIdxs) == 0:
            # print("emptyval ret, neighbor wells was {}".format(neighborWells))
            return emptyVal

        neighborEntryIdxs_all = np.array(neighborEntryIdxs)
        neighborExitIdxs_all = np.array(neighborExitIdxs)
        keepEntryIdx = np.array([v-1 not in neighborExitIdxs_all for v in neighborEntryIdxs_all])
        keepExitIdx = np.array([v+1 not in neighborEntryIdxs_all for v in neighborExitIdxs_all])
        neighborEntryIdxs = sorted(neighborEntryIdxs_all[keepEntryIdx])
        neighborExitIdxs = sorted(neighborExitIdxs_all[keepExitIdx])

        wellEntryIdxs = self.entry_exit_times(
            inProbe, wellName, timeInterval=timeInterval, returnIdxs=True)[0]
        enteredHome = [any([wei < v2 and wei > v1 for wei in wellEntryIdxs])
                       for v1, v2 in zip(neighborEntryIdxs, neighborExitIdxs)]

        # print(wellEntryIdxs)
        # ret = len([nwei for nwei in neighborExitIdxs if nwei + 1 in wellEntryIdxs]) / \
        #     len([nwei for nwei in neighborExitIdxs if nwei + 1 not in neighborEntryIdxs])
        # print(timeInterval, ret)
        ret = np.count_nonzero(enteredHome) / len(neighborEntryIdxs)

        if ret < 0 or ret > 1:
            print(neighborEntryIdxs_all)
            print(neighborExitIdxs_all)
            print(neighborEntryIdxs)
            print(neighborExitIdxs)
            print(wellEntryIdxs)
            print(enteredHome)
            print(ret)
            exit()

        return ret

    def getPasses(self, w: int, inProbe: bool, distance=200.0):
        if inProbe:
            x = self.probe_pos_xs
            y = self.probe_pos_ys
        else:
            x = self.bt_pos_xs
            y = self.bt_pos_ys
        wellCoords = self.well_coords_map[str(w)]
        wellX, wellY = wellCoords

        wellDist2 = np.power(x - np.array(wellX), 2) + np.power(y - np.array(wellY), 2)
        inRange = wellDist2 <= distance * distance
        borders = np.diff(inRange.astype(int), prepend=0, append=0)

        ents = np.nonzero(borders == 1)[0]
        exts = np.nonzero(borders == -1)[0]
        return ents, exts, wellCoords

    def getSpotlightScore(self, inProbe: bool, w: int, angleCutoff=70.0, timeInterval=None, test=False):
        if test:
            w11x, w11y = self.well_coords_map["11"]
            w28x, w28y = self.well_coords_map["28"]
            x = np.linspace(w11x, w28x, 100)
            y = np.linspace(w11y, w28y, 100)
            mv = np.ones((99,)).astype(bool)
        elif inProbe:
            x = self.probe_pos_xs
            y = self.probe_pos_ys
            # mv = self.probe_is_mv[:-1]
            mv = self.probe_bout_category[:-1] == self.BOUT_STATE_EXPLORE
            ts = self.probe_pos_ts
        else:
            x = self.bt_pos_xs
            y = self.bt_pos_ys
            # mv = self.bt_is_mv[:-1]
            mv = self.bt_bout_category[:-1] == self.BOUT_STATE_EXPLORE
            ts = self.bt_pos_ts
        wellCoords = self.well_coords_map[str(w)]
        wellX, wellY = wellCoords

        if timeInterval is not None:
            assert len(x) == len(ts)
            dur_idx = np.searchsorted(ts, np.array(
                [ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE]))
            x = x[dur_idx[0]:dur_idx[1]]
            y = y[dur_idx[0]:dur_idx[1]]
            mv = mv[dur_idx[0]:dur_idx[1]-1]

        dx = np.diff(x)
        dy = np.diff(y)
        dwx = wellX - np.array(x[1:])
        dwy = wellY - np.array(y[1:])

        dots = dx * dwx + dy * dwy
        magp = np.sqrt(np.multiply(dx, dx) + np.multiply(dy, dy))
        magw = np.sqrt(np.multiply(dwx, dwx) + np.multiply(dwy, dwy))
        mags = np.multiply(magp, magw)
        angle = np.arccos(np.divide(dots, mags))

        inSpotlight = (angle < np.deg2rad(angleCutoff)).astype(float)
        return np.mean(inSpotlight[mv])

    def getDotProductScore(self, inProbe: bool, w: int,
                           timeInterval: None | list | tuple = None,
                           excludeTimesAtWell=True,
                           test=False):
        wellIdx = np.argmax(allWellNames == w)
        if test:
            w11x, w11y = self.well_coords_map["11"]
            w29x, w29y = self.well_coords_map["29"]
            x = np.linspace(w11x, w29x, 100)
            y = np.linspace(w11y, w29y, 100)
            mv = np.ones((99,)).astype(bool)
            ents = [80]
            exts = [99]
        elif inProbe:
            x = self.probe_pos_xs
            y = self.probe_pos_ys
            mv = self.probe_bout_category[:-1] == self.BOUT_STATE_EXPLORE
            ts = self.probe_pos_ts
            ents = self.probe_well_entry_idxs[wellIdx]
            exts = self.probe_well_exit_idxs[wellIdx]
        else:
            x = self.bt_pos_xs
            y = self.bt_pos_ys
            mv = self.bt_bout_category[:-1] == self.BOUT_STATE_EXPLORE
            ts = self.bt_pos_ts
            ents = self.bt_well_entry_idxs[wellIdx]
            exts = self.bt_well_exit_idxs[wellIdx]
        wellCoords = self.well_coords_map[str(w)]
        wellX, wellY = wellCoords

        if timeInterval is not None:
            assert len(x) == len(ts)
            dur_idx = np.searchsorted(ts, np.array(
                [ts[0] + timeInterval[0] * TRODES_SAMPLING_RATE, ts[0] + timeInterval[1] * TRODES_SAMPLING_RATE]))
            # x = x[dur_idx[0]:dur_idx[1]]
            # y = y[dur_idx[0]:dur_idx[1]]
            # mv = mv[dur_idx[0]:dur_idx[1]-1]
            inTime = np.zeros_like(mv).astype(bool)
            inTime[dur_idx[0]:dur_idx[1]-1] = 1
        else:
            inTime = np.ones_like(mv).astype(bool)

        notAtWell = np.ones_like(mv).astype(bool)
        if excludeTimesAtWell:
            for ent, ext in zip(ents, exts):
                notAtWell[ent:ext+1] = False

        dx = np.diff(x)
        dy = np.diff(y)
        dwx = wellX - np.array(x[1:])
        dwy = wellY - np.array(y[1:])

        notStill = (dx != 0).astype(bool) | (dy != 0).astype(bool)

        magp = np.sqrt(dx * dx + dy * dy)
        magw = np.sqrt(dwx * dwx + dwy * dwy)

        ux = dx / magp
        uy = dy / magp
        uwx = dwx / magw
        uwy = dwy / magw

        dots = ux * uwx + uy * uwy

        keepFlag = mv & inTime & notAtWell & notStill

        return np.sum(dots[keepFlag])

    def fillTimeCutoff(self):
        if hasattr(self, "probe_fill_time"):
            return self.probe_fill_time
        else:
            return 60*5
