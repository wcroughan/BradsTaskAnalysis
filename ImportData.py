from BTData import BTData
from BTSession import BTSession
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
from PyQt5.QtWidgets import QApplication

from consts import all_well_names, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE
from UtilFunctions import readWellCoordsFile, readRawPositionData, readClipData, \
    processPosData, getWellCoordinates, getNearestWell, getWellEntryAndExitTimes, \
    quadrantOfWell, getListOfVisitedWells, onWall, getRipplePower, detectRipples, \
    getInfoForAnimal, AnimalInfo, generateFoundWells, getUSBVideoFile, processPosData_coords, \
    parseCmdLineAnimalNames
from ClipsMaker import runPositionAnnotator, AnnotatorWindow
from TrodesCameraExtrator import getTrodesLightTimes, processRawTrodesVideo, playFrames, processUSBVideoData


INSPECT_ALL = False
INSPECT_IN_DETAIL = []
# INSPECT_IN_DETAIL = ["20200526"]
INSPECT_NANVALS = False
INSPECT_PROBE_BEHAVIOR_PLOT = False
INSPECT_PLOT_WELL_OCCUPANCIES = False
ENFORCE_DIFFERENT_WELL_COLORS = False
RUN_JUST_SPECIFIED = False
SPECIFIED_DAYS = ["20210903"]
SPECIFIED_RUNS = []
INSPECT_BOUTS = True
SAVE_DONT_SHOW = True
SHOW_CURVATURE_VIDEO = False
SKIP_LFP = False
SKIP_PREV_SESSION = True
SKIP_SNIFF = True
JUST_EXTRACT_TRODES_DATA = False
RUN_INTERACTIVE = True
MAKE_ALL_TRACKING_AUTO = False
SKIP_CURVATURE = True

numExtracted = 0
numExcluded = 0

all_quadrant_idxs = [0, 1, 2, 3]
well_name_to_idx = np.empty((np.max(all_well_names) + 1))
well_name_to_idx[:] = np.nan
for widx, wname in enumerate(all_well_names):
    well_name_to_idx[wname] = widx

VEL_THRESH = 10  # cm/s

# Typical observed amplitude of LFP deflection on stimulation
DEFLECTION_THRESHOLD_HI = 6000.0
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

# well_coords_file = '/home/wcroughan/repos/BradsTaskAnalysis/well_locations.csv'

dataob = BTData()

parent_app = QApplication(sys.argv)
foundConditionGroups = False

for session_idx, session_dir in enumerate(filtered_data_dirs):
    # ===================================
    # Sniff times marked by hand from USB camera (Not available for Martin)
    # ===================================
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
            delta[:, i, 0] = delta[:, i - 1, 0] + dx[i:i + d1]
            delta[:, i, 1] = delta[:, i - 1, 1] + dy[i:i + d1]

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
        beta = m(y * x) / m(np.power(x, 2))
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
                delta[:, i, 0] = delta[:, i - 1, 0] + dx[i:i + d1]
                delta[:, i, 1] = delta[:, i - 1, 1] + dy[i:i + d1]

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
            beta = m(y * x) / m(np.power(x, 2))
            session.probe_ballisticity = beta
            assert np.sum(np.isnan(session.probe_ballisticity)) == 0

        # ===================================
        # Knot-path-curvature as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000638
        # ===================================

        if SKIP_CURVATURE:
            print("WARNING: Skipping curvature calculation")
        else:
            KNOT_H_POS = KNOT_H_CM * PIXELS_PER_CM
            dx = np.diff(session.bt_pos_xs)
            dy = np.diff(session.bt_pos_ys)

            if SHOW_CURVATURE_VIDEO:
                cmap = plt.cm.get_cmap('coolwarm')
                fig = plt.figure()
                plt.ion()

            session.bt_curvature = np.empty((dx.size + 1))
            session.bt_curvature_i1 = np.empty((dx.size + 1))
            session.bt_curvature_i2 = np.empty((dx.size + 1))
            session.bt_curvature_dxf = np.empty((dx.size + 1))
            session.bt_curvature_dyf = np.empty((dx.size + 1))
            session.bt_curvature_dxb = np.empty((dx.size + 1))
            session.bt_curvature_dyb = np.empty((dx.size + 1))
            for pi in range(dx.size + 1):
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

                session.probe_curvature = np.empty((dx.size + 1))
                session.probe_curvature_i1 = np.empty((dx.size + 1))
                session.probe_curvature_i2 = np.empty((dx.size + 1))
                session.probe_curvature_dxf = np.empty((dx.size + 1))
                session.probe_curvature_dyf = np.empty((dx.size + 1))
                session.probe_curvature_dxb = np.empty((dx.size + 1))
                session.probe_curvature_dyb = np.empty((dx.size + 1))
                for pi in range(dx.size + 1):
                    x0 = session.probe_pos_xs[pi]
                    y0 = session.probe_pos_ys[pi]
                    ii = pi
                    dxf = 0.0
                    dyf = 0.0

                    # cdx = np.cumsum(dx[pi:])
                    # cdy = np.cumsum(dy[pi:])
                    # cmagf = np.sqrt(cdx * cdx + cdy + cdy)
                    # firstPass = np.asarray(cmagf >= KNOT_H_POS).nonzero()[0][0]
                    # ii = firstPass + pi

                    while ii < dx.size:
                        dxf += dx[ii]
                        dyf += dy[ii]
                        magf = np.sqrt(dxf * dxf + dyf * dyf)
                        if magf >= KNOT_H_POS:
                            # print(numiters)
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
                        if session.probe_pos_ts[weni] <= session.probe_pos_ts[0] + 60 * TRODES_SAMPLING_RATE:
                            session.probe_well_curvatures_1min[i].append(
                                session.probe_curvature[weni:wexi])
                        if session.probe_pos_ts[weni] <= session.probe_pos_ts[0] + 30 * TRODES_SAMPLING_RATE:
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
                plt.plot([x[ben - 1], x[ben - 1]], [mny, mxy], 'r')
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
                        session.probe_pos_ts[ni], session.probe_pos_ts[ni + 1]
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
                    well_colors[i - 1, :] = cmap(random.uniform(0, 1))

                make_colors = False

                if ENFORCE_DIFFERENT_WELL_COLORS:
                    for i in all_well_names:
                        neighs = [i - 8, i - 1, i + 8, i + 1]
                        for n in neighs:
                            if n in all_well_names and np.all(well_colors[i - 1, :] == well_colors[n - 1, :]):
                                make_colors = True
                                print("gotta remake the colors!")
                                break

                        if make_colors:
                            break

            # print(session.bt_well_entry_idxs)
            # print(session.bt_well_exit_idxs)

            plt.clf()
            for i, wi in enumerate(all_well_names):
                color = well_colors[wi - 1, :]
                for j in range(len(session.bt_well_entry_times[i])):
                    i1 = session.bt_well_entry_idxs[i][j]
                    try:
                        i2 = session.bt_well_exit_idxs[i][j]
                        plt.plot(
                            session.bt_pos_xs[i1:i2], session.bt_pos_ys[i1:i2], color=color)
                    except Exception:
                        print("well {} had {} entries, {} exits".format(
                            wi, len(session.bt_well_entry_idxs[i]), len(session.bt_well_exit_idxs[i])))

            plt.show()


if JUST_EXTRACT_TRODES_DATA:
    print("extracted: {}\nexcluded: {}".format(numExtracted, numExcluded))
else:
    if not foundConditionGroups:
        for si, sesh in enumerate(dataob.allSessions):
            sesh.conditionGroup = si

    # save all sessions to disk
    print("Saving to file: {}".format(animalInfo.out_filename))
    dataob.saveToFile(os.path.join(animalInfo.output_dir, animalInfo.out_filename))
    print("Saved sessions:")
    for sesh in dataob.allSessions:
        print(sesh.name)

# Returns a list of directories that correspond to runs for analysis. Unless RUN_JUST_SPECIFIED is True,
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

        if importOptions.runJustSpecified and date_str not in importOptions.specifiedDays and session_dir not in importOptions.specifiedRuns:
            prevSession = session_dir
            continue

        if date_str in animalInfo.excluded_dates:
            print(f"skipping, excluded date {date_str}")
            prevSession = session_dir
            continue

        if animalInfo.minimum_date is not None and date_str < animalInfo.minimum_date:
            print("skipping date {}, is before minimum date {}".format(date_str, animalInfo.minimum_date))
            prevSession = session_dir
            continue

        filtered_data_dirs.append(session_dir)
        prevSessionDirs.append(prevSession)

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
    behavior_notes_dir = os.path.join(animalInfo.data_dir, 'behavior_notes')
    sesh.infoFileName = os.path.join(behavior_notes_dir, date_str + ".txt")
    if not os.path.exists(sesh.infoFileName):
        # Switched numbering scheme once started doing multiple sessions a day
        sesh.infoFileName = os.path.join(behaviorNotesDir, f"{date_str}_{sessionNumber}.txt")
    sesh.seshIdx = sessionNumber

    dir_split = prevSeshDir.split('_')
    prevSeshDateStr = dir_split[0][-8:]
    sesh.prevInfoFileName = os.path.join(behavior_notes_dir, prevSeshDateStr + ".txt")
    if not os.path.exists(sesh.prevInfoFileName ):
        # Switched numbering scheme once started doing multiple sessions a day
        sesh.prevInfoFileName = os.path.join(behaviorNotesDir, f"{prevSeshDateStr}_{prevSessionNumber}.txt")
    sesh.prevSeshIdx = prevSessionNumber

    sesh.fileStartString = os.path.join(animalInfo.data_dir, seshDir, seshDir)
    sesh.animalInfo = animalInfo


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

            # if JUST_EXTRACT_TRODES_DATA and fieldName.lower() not in ["reference", "ref", "baseline", "home", "aways", "last away", "last well", "probe performed"]:
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
                        type_in, info_file))
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
                    ended_on = fieldVal
                    if 'H' in fieldVal:
                        sesh.ended_on_home = True
                    elif 'A' in fieldVal:
                        sesh.ended_on_home = False
                    else:
                        print("Couldn't recognize last well {} in file {}".format(
                            fieldVal, info_file))
            elif fieldName.lower() == "iti stim on":
                if 'Y' in fieldVal:
                    sesh.ITI_stim_on = True
                elif 'N' in fieldVal:
                    sesh.ITI_stim_on = False
                else:
                    print("Couldn't recognize ITI Stim condition {} in file {}".format(
                        fieldVal, info_file))
            elif fieldName.lower() == "probe stim on":
                if 'Y' in fieldVal:
                    sesh.probe_stim_on = True
                elif 'N' in fieldVal:
                    sesh.probe_stim_on = False
                else:
                    print("Couldn't recognize Probe Stim condition {} in file {}".format(
                        fieldVal, info_file))
            elif fieldName.lower() == "probe performed":
                if 'Y' in fieldVal:
                    sesh.probe_performed = True
                elif 'N' in fieldVal:
                    sesh.probe_performed = False
                    if animal_name == "Martin":
                        raise Exception("I thought all martin runs had a probe")
                else:
                    print("Couldn't recognize Probe performed val {} in file {}".format(
                        fieldVal, info_file))
            elif fieldName.lower() == "task ended at":
                sesh.bt_ended_at_well = int(fieldVal)
            elif fieldName.lower() == "probe ended at":
                sesh.probe_ended_at_well = int(fieldVal)
            elif fieldName.lower() == "weight":
                sesh.weight = float(fieldVal)
            elif fieldName.lower() == "conditiongroup":
                sesh.conditionGroup = int(fieldVal)
                foundConditionGroups = True
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
                    type_in = field_val
                    if 'Ripple' in type_in or 'Interruption' in type_in:
                        sesh.prevSessionIsRippleInterruption = True
                    elif 'None' in type_in:
                        sesh.prevSessionIsNoInterruption = True
                    elif 'Delay' in type_in:
                        sesh.prevSessionIsDelayedInterruption = True
                    else:
                        print("Couldn't recognize Condition {} in file {}".format(
                            type_in, sesh.prevInfoFileName))
                elif field_name.lower() == "thresh":
                    if "Low" in field_val:
                        sesh.prevSession_ripple_detection_threshold = 2.5
                    elif "High" in field_val:
                        sesh.prevSession_ripple_detection_threshold = 4
                    else:
                        sesh.prevSession_ripple_detection_threshold = float(
                            field_val)
                elif field_name.lower() == "last away":
                    sesh.prevSession_last_away_well = float(field_val)
                    # print(field_val)
                elif field_name.lower() == "last well":
                    ended_on = field_val
                    if 'H' in field_val:
                        sesh.prevSession_ended_on_home = True
                    elif 'A' in field_val:
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

    if not JUST_EXTRACT_TRODES_DATA:
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
            position_data_metadata, position_data = readRawPositionData(trackingFile)

            if MAKE_ALL_TRACKING_AUTO and ("source" not in position_data_metadata or "trodescameraextractor" not in position_data_metadata["source"]):
                # print(position_data_metadata)
                position_data = None
                os.rename(sesh.fileStartString + '.1.videoPositionTracking', sesh.fileStartString +
                          '.1.videoPositionTracking.manualOutput')
            else:
                print("Got position tracking")
                sesh.frameTimes = position_data['timestamp']
                xs, ys, ts = processPosData(position_data, xLim=(
                    animalInfo.X_START, animalInfo.X_FINISH), yLim=(animalInfo.Y_START, animalInfo.Y_FINISH))
                sesh.hasPositionData = True
        else:
            position_data = None

        if position_data is None:
            print("running trodes camera extractor!")
            processRawTrodesVideo(sesh.fileStartString + '.1.h264')
            position_data_metadata, position_data = readRawPositionData( trackingFile)
            sesh.frameTimes = position_data['timestamp']
            xs, ys, ts = processPosData(position_data, xLim=(
                animalInfo.X_START, animalInfo.X_FINISH), yLim=(animalInfo.Y_START, animalInfo.Y_FINISH))
            sesh.hasPositionData = True


    if sesh.missingEndedAtWell:
        raise Exception(
            f"{sesh.name} has position but no last well marked in behavior notes and didn't end on last home")
    if sesh.missingProbeEndedAtWell:
        raise Exception("Didn't mark the probe end well for session {}".format(sesh.name))


    if sesh.positionFromDeepLabCut:
        well_coords_file_name = sesh.fileStartString + '.1.wellLocations_dlc.csv'
        if not os.path.exists(well_coords_file_name):
            well_coords_file_name = os.path.join(animalInfo.data_dir, 'well_locations_dlc.csv')
            print("Specific well locations not found, falling back to file {}".format(well_coords_file_name))
    else:
        well_coords_file_name = sesh.fileStartString + '.1.wellLocations.csv'
        if not os.path.exists(well_coords_file_name):
            well_coords_file_name = os.path.join(animalInfo.data_dir, 'well_locations.csv')
            print("Specific well locations not found, falling back to file {}".format(well_coords_file_name))
    sesh.well_coords_map = readWellCoordsFile(well_coords_file_name)
    sesh.home_x, sesh.home_y = getWellCoordinates(
        sesh.home_well, sesh.well_coords_map)

    if not sesh.hasPositionData :
        print(f"WARNING: {sesh.name} has no position data")
        return

    if sesh.importOptions["skipUSB"]:
        print("WARNING: Skipping USB ")
    else:
        if "lightonframe" in position_data_metadata:
            sesh.trodesLightOnFrame = int(position_data_metadata['lightonframe'])
            sesh.trodesLightOffFrame = int(position_data_metadata['lightoffframe'])
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
                lightInfo = np.fromfile(positionMetadataFile, sep=",").astype(int)
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
            "/media/WDC6/videos/B16-20/trimmed/{}/".format(animal_name),
            "/media/WDC7/videos/B16-20/trimmed/{}/".format(animal_name),
            "/media/WDC8/videos/B16-20/trimmed/{}/".format(animal_name),
            "/media/WDC6/{}/".format(animal_name),
            "/media/WDC7/{}/".format(animal_name),
            "/media/WDC8/{}/".format(animal_name),
        ]
        sesh.usbVidFile = getUSBVideoFile(
            sesh.name, possibleDirectories, seshIdx=sesh.seshIdx, useSeshIdxDirectly=(animal_name == "B18"))
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
                parent_app.exec()

        if not sesh.importOptions["justExtractData"]:
            if sesh.separate_probe_file:
                probe_file_str = os.path.join(
                    sesh.probe_dir, os.path.basename(sesh.probe_dir))
                bt_time_clips = readClipData(sesh.fileStartString + '.1.clips')[0]
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
                        animalInfo.X_START, animalInfo.X_FINISH), yLim=(animalInfo.Y_START, animalInfo.Y_FINISH))

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
        if session.num_home_found > 0:
            print("Well find times not marked for session {}".format(session.name))
            sesh.hasWellFindTimes = False
        else:
            sesh.hasWellFindTimes = True
    else:
        sesh.hasWellFindTimes = True
        well_visit_times = readClipData(rewardClipsFile)
        assert session.num_away_found + \
            session.num_home_found == np.shape(well_visit_times)[0]
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


    if not sesh.importOptions["skipSniff"]:
        gl = animalInfo.data_dir + sesh.name + '/*.rgs'
        dir_list = glob.glob(gl)
        if len(dir_list) == 0:
            print("Couldn't find sniff times files")
            sesh.hasSniffTimes = False
        elif SKIP_SNIFF:
            print("Warning: skipping sniff stuff")
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
                    sesh.bt_well_sniff_times_entry = [[] for _ in all_well_names]
                    sesh.bt_well_sniff_times_exit = [[] for _ in all_well_names]
                    sesh.probe_well_sniff_times_entry = [[] for _ in all_well_names]
                    sesh.probe_well_sniff_times_exit = [[] for _ in all_well_names]

                    sesh.sniff_pre_trial_light_off = int(sniffData[0][0])
                    sesh.sniff_trial_start = int(sniffData[0][1])
                    sesh.sniff_trial_stop = int(sniffData[0][2])
                    sesh.sniff_probe_start = int(sniffData[1][0])
                    sesh.sniff_probe_stop = int(sniffData[1][1])
                    sesh.sniff_post_probe_light_on = int(sniffData[1][2])

                    for i in sniffData[2:]:
                        w = int(well_name_to_idx[int(i[2])])
                        entry_time = int(i[0])
                        exit_time = int(i[1])
                        sesh.well_sniff_times_entry[w].append(entry_time)
                        sesh.well_sniff_times_exit[w].append(exit_time)

                        if exit_time < entry_time:
                            print("mismatched interval: {} - {}".format(entry_time, exit_time))
                            assert False
                        if exit_time < sesh.sniff_trial_stop:
                            sesh.bt_well_sniff_times_entry[w].append(entry_time)
                            sesh.bt_well_sniff_times_exit[w].append(exit_time)
                        else:
                            sesh.probe_well_sniff_times_entry[w].append(entry_time)
                            sesh.probe_well_sniff_times_exit[w].append(exit_time)



def loadLFPData(sesh):
    lfpData = []

    if len(sesh.ripple_detection_tetrodes) == 0:
        sesh.ripple_detection_tetrodes = [animalInfo.DEFAULT_RIP_DET_TET]

    for i in range(len(sesh.ripple_detection_tetrodes)):
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print(lfpdir, "doesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + sesh.fileStartString + ".rec"
            elif os.path.exists("/home/wcroughan/Software/Trodes21/linux/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes_2-2-3_Ubuntu1804/exportLFP -rec " + sesh.fileStartString + ".rec"
            print(syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + sesh.name + ".LFP_nt" + \
            str(sesh.ripple_detection_tetrodes[i]) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.bt_lfp_fnames.append(lfpfilename)
        lfpData.append(MountainViewIO.loadLFP(data_file=sesh.bt_lfp_fnames[-1]))

    if sesh.ripple_baseline_tetrode is None:
        sesh.ripple_baseline_tetrode = animalInfo.DEFAULT_RIP_BAS_TET

    # I think Martin didn't have this baseline tetrode? Need to check
    if sesh.ripple_baseline_tetrode is not None:
        lfpdir = sesh.fileStartString + ".LFP"
        if not os.path.exists(lfpdir):
            print(lfpdir, "doesn't exists, gonna try and extract the LFP")
            # syscmd = "/home/wcroughan/SpikeGadgets/Trodes_1_8_1/exportLFP -rec " + sesh.fileStartString + ".rec"
            if os.path.exists("/home/wcroughan/Software/Trodes21/exportLFP"):
                syscmd = "/home/wcroughan/Software/Trodes21/exportLFP -rec " + sesh.fileStartString + ".rec"
            else:
                syscmd = "/home/wcroughan/Software/Trodes21/linux/exportLFP -rec " + sesh.fileStartString + ".rec"
            print(syscmd)
            os.system(syscmd)

        gl = lfpdir + "/" + sesh.name + ".LFP_nt" + \
            str(sesh.ripple_baseline_tetrode) + "ch*.dat"
        lfpfilelist = glob.glob(gl)
        lfpfilename = lfpfilelist[0]
        sesh.bt_lfp_baseline_fname = lfpfilename
        baselineLfpData = MountainViewIO.loadLFP(data_file=sesh.bt_lfp_baseline_fname)


def runLFPAnalyses(sesh, lfpData, baselineLfpData):
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

    sesh.bt_interruption_pos_idxs = np.searchsorted( sesh.bt_pos_ts, sesh.interruption_timestamps)
    sesh.bt_interruption_pos_idxs = sesh.bt_interruption_pos_idxs[ sesh.bt_interruption_pos_idxs < len(sesh.bt_pos_ts)]

    # ========
    # dealing with weird Martin sessions:
    showPlot = False
    print("{} interruptions detected".format(len(sesh.bt_interruption_pos_idxs)))
    if len(sesh.bt_interruption_pos_idxs) < 50:
        if sesh.isRippleInterruption and len(sesh.bt_interruption_pos_idxs) > 0:
            # print( "WARNING: IGNORING BEHAVIOR NOTES FILE BECAUSE SEEING FEWER THAN 50 INTERRUPTIONS, CALLING THIS A CONTROL SESSION")
            # raise Exception("SAW FEWER THAN 50 INTERRUPTIONS ON AN INTERRUPTION SESSION: {} on session {}".format(
            # len(session.bt_interruption_pos_idxs), session.name))
            print("WARNING: FEWER THAN 50 STIMS DETECTED ON AN INTERRUPTION SESSION")
        elif len(sesh.bt_interruption_pos_idxs) == 0:
            print(
                "WARNING: IGNORING BEHAVIOR NOTES FILE BECAUSE SEEING 0 INTERRUPTIONS, CALLING THIS A CONTROL SESSION")
        else:
            print(
                "WARNING: very few interruptions. This was a delay control but is basically a no-stim control")
        # showPlot = True
        sesh.isRippleInterruption = False
    elif len(sesh.bt_interruption_pos_idxs) < 100:
        print("50-100 interruptions: not overriding label")
        # showPlot = True

    print("Condition - {}".format("SWR" if sesh.isRippleInterruption else "Ctrl"))
    lfp_deflections = signal.find_peaks(np.abs(
        np.diff(lfpV, prepend=lfpV[0])), height=DEFLECTION_THRESHOLD_LO, distance=MIN_ARTIFACT_DISTANCE)
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

    pre_bt_interruption_idxs = interruption_idxs[interruption_idxs < bt_lfp_start_idx]
    pre_bt_interruption_idxs_first_half = interruption_idxs[interruption_idxs < int(
        bt_lfp_start_idx / 2)]

    _, _, sesh.prebtMeanRipplePower, sesh.prebtStdRipplePower = getRipplePower(
        lfpV[0:bt_lfp_start_idx], omit_artifacts=False)
    _, ripple_power, _, _ = getRipplePower(
        btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=sesh.prebtMeanRipplePower, stdPower=sesh.prebtStdRipplePower, showPlot=showPlot)
    sesh.btRipStartIdxsPreStats, sesh.btRipLensPreStats, sesh.btRipPeakIdxsPreStats, sesh.btRipPeakAmpsPreStats, sesh.btRipCrossThreshIdxsPreStats = \
        detectRipples(ripple_power)
    # print(bt_lfp_start_idx, sesh.btRipStartIdxsPreStats)
    if len(sesh.btRipStartIdxsPreStats) == 0:
        sesh.btRipStartTimestampsPreStats = np.array([])
    else:
        sesh.btRipStartTimestampsPreStats = lfpTimestamps[sesh.btRipStartIdxsPreStats + bt_lfp_start_idx]

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
            lfpData = MountainViewIO.loadLFP(data_file=sesh.bt_lfp_baseline_fname)
            baselfpV = lfpData[1]['voltage']
            baselfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

            btRipPower, _, _, _ = getRipplePower(
                btLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=sesh.probeMeanRipplePower, stdPower=sesh.probeStdRipplePower, showPlot=showPlot)
            probeRipPower, _, _, _ = getRipplePower(probeLFPData, omit_artifacts=False)

            baselineProbeLFPData = baselfpV[probeLfpStart_idx:probeLfpEnd_idx]
            probeBaselinePower, _, baselineProbeMeanRipplePower, baselineProbeStdRipplePower = getRipplePower(
                baselineProbeLFPData, omit_artifacts=False)
            btBaselineLFPData = baselfpV[bt_lfp_start_idx:bt_lfp_end_idx]
            btBaselineRipplePower, _, _, _ = getRipplePower(
                btBaselineLFPData, lfp_deflections=bt_lfp_artifact_idxs, meanPower=baselineProbeMeanRipplePower, stdPower=baselineProbeStdRipplePower, showPlot=showPlot)

            probeRawPowerDiff = probeRipPower - probeBaselinePower
            zmean = np.nanmean(probeRawPowerDiff)
            zstd = np.nanstd(probeRawPowerDiff)

            rawPowerDiff = btRipPower - btBaselineRipplePower
            zPowerDiff = (rawPowerDiff - zmean) / zstd

            sesh.btWithBaseRipStartIdx, sesh.btWithBaseRipLens, sesh.btWithBaseRipPeakIdx, sesh.btWithBaseRipPeakAmps, sesh.btWithBaseRipCrossThreshIdxs = detectRipples(
                zPowerDiff)
            if len(sesh.btWithBaseRipStartIdx) > 0:
                sesh.btWithBaseRipStartTimestamps = lfpTimestamps[
                    sesh.btWithBaseRipStartIdx + bt_lfp_start_idx]
            else:
                sesh.btWithBaseRipStartTimestamps = np.array([])

        print("{} ripples found during task".format(
            len(sesh.btRipStartTimestampsProbeStats)))

    elif sesh.probe_performed:
        print("Probe performed but LFP in a separate file for session", sesh.name)

def runSanityChecks(sesh, lfpData, baselineLfpData):
    # TODO:
    # Check num stims in LFP, large gaps in signal
    pass

def posCalcVelocity(sesh):
    bt_vel = np.sqrt(np.power(np.diff(sesh.bt_pos_xs), 2) +
                        np.power(np.diff(sesh.bt_pos_ys), 2))
    sesh.bt_vel_cm_s = np.divide(bt_vel, np.diff(sesh.bt_pos_ts) /
                                    TRODES_SAMPLING_RATE) / PIXELS_PER_CM
    bt_is_mv = sesh.bt_vel_cm_s > VEL_THRESH
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
                                            TRODES_SAMPLING_RATE) / PIXELS_PER_CM
        probe_is_mv = sesh.probe_vel_cm_s > VEL_THRESH
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
                np.append([sesh.bt_pos_ts[0]], sesh.away_well_leave_times[0:-1])
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
        probe_ended_at_well_idx = np.argmax(all_well_names == sesh.probe_ended_at_well)
        sesh.probe_recall_pos_idx = sesh.probe_well_exit_idxs[probe_ended_at_well_idx][-1]


    


def extractAndSave(animalName, importOptions):
    numExcluded = 0
    numExtracted = 0

    print("=========================================================")
    print(f"Extracting data for animal {animalName}")
    print("=========================================================")
    animalInfo = getInfoForAnimal(animal_name)
    print(f"\tdata_dir = {animalInfo.data_dir}\n\toutput_dir = {animalInfo.output_dir}")
    if not os.path.exists(animalInfo.output_dir):
        os.mkdir(animalInfo.output_dir)

    sessionDirs, prevSessionDirs = getSessionDirs(animalInfo, importOptions)
    dirListStr = "".join([f"\t{s}\t{ss}\n"  for s, ss in zip(sessionDirs, prevSessionDirs)])
    print("Session dirs:", dirListStr)

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


        sesh = makeSessionObj(seshDir, prevSeshDir, sessionNumber, prevSessionNumber, animalInfo)
        sesh.animalName = animalName
        sesh.importOptions = importOptions
        print(sesh.infoFileName)
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

        runSanityChecks(sesh, lfpData, baselineLfpData)

        if importOptions["justExtractData"]:
            numExtracted += 1
            continue

        if not importOptions["skipLFP"]:
            runLFPAnalyses(sesh, lfpData, baselineLfpData)
        runPositionAnalyses(sesh)

        dataObj.allSessions.append(sesh)

    if importOptions["justExtractData"]:
        print(f"Extracted data from {numExtracted} sessions, excluded {numExcluded}. Not analyzing or saving")
        return

    if not any([s.conditionGroup is not None for s in dataObj.allSessions]):
        for si, sesh in enumerate(dataObj.allSessions):
            sesh.conditionGroup = si

    # save all sessions to disk
    print("Saving to file: {}".format(animalInfo.out_filename))
    dataob.saveToFile(os.path.join(animalInfo.output_dir, animalInfo.out_filename))
    print("Saved sessions:")
    for sesh in dataob.allSessions:
        print(sesh.name)



if __name__ == "__main__":
    animalNames = parseCmdLineAnimalNames(default=["B18"])
    importOptions = {
            "skipLFP" : False,
            "skipUSB" : False,
            "skipPrevSession" : True,
            "skipSniff" : True,
            "forceAllTrackingAuto" : False,
            "skipCurvature" : False,
            "runJustSpecified" : False,
            "specifiedDays" : [],
            "specifiedRuns" : [],
            "justExtractData": False,
            "runInteractiveExtraction": True,
            "consts": {
                "VEL_THRESH" : 10,  # cm/s
                "PIXELS_PER_CM" : None,

                # Typical observed amplitude of LFP deflection on stimulation
                "DEFLECTION_THRESHOLD_HI" : 6000.0,
                "DEFLECTION_THRESHOLD_LO" : 2000.0,
                "MIN_ARTIFACT_DISTANCE" : int(0.05 * LFP_SAMPLING_RATE),

                # constants for exploration bout analysis
                "BOUT_VEL_SM_SIGMA_SECS" : 1.5,
                "PAUSE_MAX_SPEED_CM_S" : 8.0,
                "MIN_PAUSE_TIME_BETWEEN_BOUTS_SECS" : 2.5,
                "MIN_EXPLORE_TIME_SECS" : 3.0,
                "MIN_EXPLORE_NUM_WELLS" : 4,

                # constants for ballisticity
                "BALL_TIME_INTERVALS" : list(range(1, 24)),
                "KNOT_H_CM" : 8.0
            }
    }

    for animalName in animalNames:
        importOptions["skipUSB"] = animalName == "Martin"
        extractAndSave(animalName, importOptions)