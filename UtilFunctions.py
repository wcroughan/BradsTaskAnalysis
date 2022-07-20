import numpy as np
from consts import allWellNames, TRODES_SAMPLING_RATE, LFP_SAMPLING_RATE
import csv
import glob
from scipy import stats, signal
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from itertools import groupby
from datetime import date
import sys
import os


def findDataDir(possibleDataDirs=["/media/WDC6/", "/media/fosterlab/WDC6/", "/home/wcroughan/data/"]):
    for dd in possibleDataDirs:
        if os.path.exists(dd):
            return dd

    return None


def parseCmdLineAnimalNames(default=None):
    if len(sys.argv) >= 2:
        return sys.argv[1:]
    else:
        return default


def readWellCoordsFile(well_coords_file):
    # For some reason json saving and loading turns the keys into strings, just going to change that here so it's consistent
    with open(well_coords_file, 'r') as wcf:
        well_coords_map = {}
        csv_reader = csv.reader(wcf)
        for data_row in csv_reader:
            try:
                well_coords_map[str(int(data_row[0]))] = (
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
            lineText = ""
            max_iter = 8
            iter = 0
            settings = {}
            while lineText != b'<end settings>\n':
                lineText = datafile.readline().lower()
                if lineText != b'<end settings>\n' and lineText != b'<start settings>\n':
                    ss = str(lineText).split(":")
                    if len(ss) == 2:
                        settings[str(ss[0])[2:].strip()] = str(ss[1])[0:-3].strip()
                # print(lineText)
                iter += 1
                if iter > max_iter:
                    raise Exception
            return settings, np.fromfile(datafile, dtype=dt)
    except Exception as err:
        print(err)
        return None, None


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


def processPosData(position_data, maxJumpDistance=50, nCleaningReps=2,
                   xLim=(100, 1050), yLim=(20, 900)):
    x_pos = np.array(position_data['x1'], dtype=float)
    y_pos = np.array(position_data['y1'], dtype=float)

    # Interpolate the position data into evenly sampled time points
    x = np.linspace(position_data['timestamp'][0],
                    position_data['timestamp'][-1], position_data.shape[0])

    xp = position_data['timestamp']
    x_pos = np.interp(x, xp, position_data['x1'])
    y_pos = np.interp(x, xp, position_data['y1'])
    # position_sampling_frequency = TRODES_SAMPLING_RATE / np.diff(x)[0]
    # Interpolated Timestamps:
    # position_data['timestamp'] = x

    # Remove large jumps in position (tracking errors)
    for _ in range(nCleaningReps):
        jump_distance = np.sqrt(np.square(np.diff(x_pos, prepend=x_pos[0])) +
                                np.square(np.diff(y_pos, prepend=y_pos[0])))
        # print(jump_distance)
        points_in_range = (x_pos > xLim[0]) & (x_pos < xLim[1]) &\
            (y_pos > yLim[0]) & (y_pos < yLim[1])
        clean_points = jump_distance < maxJumpDistance

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


def getWellCoordinates(well_num, well_coords_map):
    return well_coords_map[str(well_num)]


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
    for wi in allWellNames:
        wx, wy = getWellCoordinates(wi, well_coords)
        res.append(getMeanDistToWell(np.array(xs), np.array(
            ys), wx, wy, duration=duration, ts=np.array(ts)))

    return res


def getMedianDistToWells(xs, ys, well_coords, duration=-1, ts=np.array([])):
    res = []
    for wi in allWellNames:
        wx, wy = getWellCoordinates(wi, well_coords)
        res.append(getMedianDistToWell(np.array(xs), np.array(
            ys), wx, wy, duration=duration, ts=np.array(ts)))

    return res


# switchWellFactor of 0.8 means transition from well a -> b requires rat dist to b to be 0.8 * dist to a
def getNearestWell(xs, ys, well_coords, well_idxs=allWellNames, switchWellFactor=0.8):
    well_coords = np.array(
        [getWellCoordinates(i, well_coords) for i in well_idxs])
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


# def detectStimArtifacts(lfp_data):
#     deflection_metrics = signal.find_peaks(np.abs(np.diff(lfp_data,
#                                                           prepend=lfp_data[0])), height=DEFLECTION_THRESHOLD_LO,
#                                            distance=MIN_ARTIFACT_DISTANCE)

#     return deflection_metrics[0]


def getRipplePower(lfp_data, omit_artifacts=True, causal_smoothing=False,
                   lfp_deflections=None, meanPower=None, stdPower=None,
                   showPlot=False, rippleFilterBand=(150, 250), rippleFilterOrder=4,
                   skipTimePointsFoward=int(0.075 * LFP_SAMPLING_RATE),
                   skipTimePointsBackward=int(0.02 * LFP_SAMPLING_RATE)):
    """
    Get ripple power in LFP
    """
    lfp_data_copy = lfp_data.copy().astype(float)

    if (meanPower is None) != (stdPower is None):
        raise Exception("meanPower and stdPower must both be provided or None")

    if lfp_deflections is None:
        if omit_artifacts:
            raise Exception("this hasn't been updated")
            # # Remove all the artifacts in the raw ripple amplitude data
            # deflection_metrics = signal.find_peaks(np.abs(np.diff(lfp_data,
            #                                                       prepend=lfp_data[0])), height=DEFLECTION_THRESHOLD_LO,
            #                                        distance=MIN_ARTIFACT_DISTANCE)
            # lfp_deflections = deflection_metrics[0]

    # After this preprocessing, clean up the data if needed.
    lfp_mask = np.zeros_like(lfp_data_copy)
    if lfp_deflections is not None:
        for artifact_idx in lfp_deflections:
            if artifact_idx < 0 or artifact_idx > len(lfp_data):
                continue
            cleanup_start = max(0, artifact_idx - skipTimePointsBackward)
            cleanup_finish = min(len(lfp_data), artifact_idx +
                                 skipTimePointsFoward)
            # lfp_data_copy[cleanup_start:cleanup_finish] = np.nan
            lfp_mask[cleanup_start:cleanup_finish] = 1

        print("LFP mask letting {} of signal through".format(
            1 - (np.count_nonzero(lfp_mask) / len(lfp_mask))))

    nyq_freq = LFP_SAMPLING_RATE * 0.5
    lo_cutoff = rippleFilterBand[0] / nyq_freq
    hi_cutoff = rippleFilterBand[1] / nyq_freq
    pl, ph = signal.butter(rippleFilterOrder, [lo_cutoff, hi_cutoff], btype='band')
    if causal_smoothing:
        ripple_amplitude = signal.lfilter(pl, ph, lfp_data_copy)
    else:
        ripple_amplitude = signal.filtfilt(pl, ph, lfp_data_copy)

    ripple_amplitude[lfp_mask == 1] = np.nan

    # Smooth this data and get ripple power
    # smoothing_window_length = RIPPLE_POWER_SMOOTHING_WINDOW * LFP_SAMPLING_RATE
    # smoothing_weights = np.ones(int(smoothing_window_length))/smoothing_window_length
    # ripple_power = np.convolve(np.abs(ripple_amplitude), smoothing_weights, mode='same')

    # Use a Gaussian kernel for filtering - Make the Kernel Causal bu keeping only one half of the values
    smoothing_window_length = 10
    if causal_smoothing:
        # In order to no have NaN values affect the filter output, create a copy with the artifacts
        ripple_amplitude_copy = ripple_amplitude.copy()

        half_smoothing_signal = \
            np.exp(-np.square(np.linspace(0, -4 * smoothing_window_length, 4 *
                                          smoothing_window_length)) / (2 * smoothing_window_length * smoothing_window_length))
        smoothing_signal = np.concatenate(
            (np.zeros_like(half_smoothing_signal), half_smoothing_signal), axis=0)
        ripple_power = signal.convolve(np.abs(ripple_amplitude_copy),
                                       smoothing_signal, mode='same') / np.sum(smoothing_signal)
        ripple_power[np.isnan(ripple_amplitude)] = np.nan
    else:
        ripple_power = gaussian_filter(np.abs(ripple_amplitude), smoothing_window_length)

    # Get the mean/standard deviation for ripple power and adjust for those
    if meanPower is None:
        meanPower = np.nanmean(ripple_power)
        stdPower = np.nanstd(ripple_power)
    zpower = (ripple_power - meanPower) / stdPower

    if showPlot:
        lc = lfp_data.copy()
        lc = lc / np.nanmax(np.abs(lc)) * 10
        rc = np.array([min(10, p) for p in zpower])
        ts = np.linspace(0, len(lc) / 1500, len(lc))
        plt.plot(ts, rc, c="orange", zorder=0)
        plt.plot(ts, lc, c="blue", zorder=1)
        # plt.plot(np.diff(lc), c="red")
        # plt.plot([0, len(lc)], [3, 3], color="red", zorder=-1)
        if lfp_deflections is not None:
            plt.scatter(lfp_deflections / 1500, [0] * len(lfp_deflections), zorder=2, c="red")

        plt.show()

    return ripple_power, zpower, meanPower, stdPower


def detectRipples(ripplePower, minHeight=3.0, minLen=0.05, maxLen=0.3, edgeThresh=0.0):
    pks, peakInfo = signal.find_peaks(ripplePower, height=minHeight)

    ripStartIdxs = []
    ripLens = []
    ripPeakIdxs = []
    ripPeakAmps = []
    ripCrossThreshIdx = []

    i = 0
    while i < len(pks):
        pkidx = pks[i]
        ii = pkidx
        while ii >= 0 and ripplePower[ii] > edgeThresh:
            ii -= 1
        ii += 1
        ripStart = ii

        length = 0
        pkAmp = 0
        pkAmpI = 0
        crossI = 0
        crossed = False
        while ii < len(ripplePower) and ripplePower[ii] > edgeThresh:
            if ripplePower[ii] > pkAmp:
                pkAmp = ripplePower[ii]
                pkAmpI = ii

            if not crossed and ripplePower[ii] > minHeight:
                crossed = True
                crossI = ii

            ii += 1
            length += 1

        assert crossed

        lensec = float(length) / LFP_SAMPLING_RATE
        if lensec >= minLen and lensec <= maxLen:
            ripStartIdxs.append(ripStart)
            ripLens.append(length)
            ripPeakAmps.append(pkAmp)
            ripPeakIdxs.append(pkAmpI)
            ripCrossThreshIdx.append(crossI)

        while i < len(pks) and pks[i] < ii:
            i += 1

    return ripStartIdxs, ripLens, ripPeakIdxs, ripPeakAmps, ripCrossThreshIdx


def getWellEntryAndExitTimes(nearest_wells, ts, well_idxs=allWellNames, include_neighbors=False):
    entry_times = []
    exit_times = []
    entry_idxs = []
    exit_idxs = []

    ts = np.array(ts)
    for wi in well_idxs:
        # last data point should count as an exit, so appending a false
        # same for first point should count as entry, prepending
        if include_neighbors:
            neighbors = list({wi, wi - 1, wi + 1, wi - 7, wi - 8, wi - 9,
                              wi + 7, wi + 8, wi + 9}.intersection(allWellNames))
            near_well = np.concatenate(
                ([False], np.isin(nearest_wells, neighbors), [False]))
        else:
            near_well = np.concatenate(([False], nearest_wells == wi, [False]))
        idx = np.argwhere(np.diff(np.array(near_well, dtype=float)) == 1)
        idx2 = np.argwhere(np.diff(np.array(near_well, dtype=float)) == -1) - 1
        entry_idxs.append(idx.T[0])
        exit_idxs.append(idx2.T[0])
        entry_times.append(ts[idx.T[0]])
        exit_times.append(ts[idx2.T[0]])

    return entry_idxs, exit_idxs, entry_times, exit_times


def getSingleWellEntryAndExitTimes(xs, ys, ts, wellx, welly, radius=50):
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

    near_well = dist_to_well < radius
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


def getListOfVisitedWells(nearestWells, countFirstVisitOnly):
    if countFirstVisitOnly:
        return list(set(nearestWells))
    else:
        return [k for k, g in groupby(nearestWells)]


def onWall(well):
    return well < 9 or well > 40 or well % 8 in [2, 7]

def offWall(well):
    return not onWall(well)


class AnimalInfo:
    def __init__(self):
        self.X_START = None
        self.X_FINISH = None
        self.Y_START = None
        self.Y_FINISH = None
        self.data_dir = ""
        self.output_dir = ""
        self.fig_output_dir = ""
        self.out_filename = ""

        self.excluded_dates = []
        self.excluded_sessions = []
        self.minimum_date = None

        self.DEFAULT_RIP_DET_TET = None
        self.DEFAULT_RIP_BAS_TET = None


def getInfoForAnimal(animalName):
    ret = AnimalInfo
    if animalName == "Martin":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = '/media/WDC1/martindata/bradtask/'
        ret.output_dir = '/media/WDC6/Martin/processed_data/'
        ret.fig_output_dir = ret.output_dir
        ret.out_filename = "martin_bradtask.dat"

        ret.excluded_dates = ["20200528", "20200630", "20200702", "20200703"]
        ret.excluded_dates += ["20200531", "20200603", "20200602",
                               "20200606", "20200605", "20200601"]
        ret.excluded_dates += ["20200526"]
        ret.excluded_sessions = ["20200624_1", "20200624_2", "20200628_2"]
        ret.minimum_date = None

        ret.excluded_dates += ["20200527"]
        ret.excluded_dates += ["20200604"]
        ret.excluded_dates += ["20200608"]
        ret.excluded_dates += ["20200609"]

        # Too few interruptions to really call this an interruption session probably:
        ret.excluded_dates += ["20200607"]

        ret.DEFAULT_RIP_DET_TET = 37
        ret.DEFAULT_RIP_BAS_TET = None

    elif animalName == "B12":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_bradtask.dat"

        ret.excluded_dates = []
        ret.minimum_date = None
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 7

    elif animalName == "B12_goodpos":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_goodpos_bradtask.dat"

        ret.excluded_dates = ["20210816", "20210817", "20210818", "20210819"]
        ret.minimum_date = None
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 7

    elif animalName == "B12_no19":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_no19_bradtask.dat"

        ret.excluded_dates = ["20210819"]
        ret.minimum_date = None
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 7

    elif animalName == "B12_highthresh":
        ret.X_START = 200
        ret.X_FINISH = 1175
        ret.Y_START = 20
        ret.Y_FINISH = 1275
        ret.data_dir = "/media/WDC7/B12/bradtasksessions/"
        ret.output_dir = "/media/WDC7/B12/processed_data/"
        ret.fig_output_dir = "/media/WDC7/B12/processed_data/"
        ret.out_filename = "B12_highthresh_bradtask.dat"

        ret.excluded_dates = []
        ret.minimum_date = "20210916"
        ret.excluded_sessions = ["20210917_1", "20210923_1",
                                 "20211004_1", "20211005_2", "20211006_1"]
        ret.DEFAULT_RIP_DET_TET = 8

    elif animalName == "B13":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        ret.data_dir = "/media/WDC6/B13/bradtasksessions/"
        ret.output_dir = "/media/WDC6/B13/processed_data/"
        ret.fig_output_dir = "/media/WDC6/B13/processed_data/"
        ret.out_filename = "B13_bradtask.dat"

        ret.excluded_dates = ["20220209"]
        # minimum_date = "20211209"  # had one run on the 8th with probe but used high ripple threshold and a different reference tetrode
        ret.minimum_date = None
        # high ripple thresh on 12/08-1, forgot to turn stim on til after first home on 12/16-2
        ret.excluded_sessions = ["20211208_1", "20211216_2"]
        ret.DEFAULT_RIP_DET_TET = 7
        ret.DEFAULT_RIP_BAS_TET = 2

        # Messed up away well order, marked down 20 when he got 12. Ended up giving him reward at 12 twice
        ret.excluded_sessions += ["20220131_2"]
        # Made a custom foundwells field in the behaviornotes for this guy, but would need to update the rest of the import code
        # (i.e. clips loading assumes alternation with home)
        ret.excluded_sessions += ["20220222_2"]
        # video skips
        ret.excluded_sessions += ["20220304_2"]

        # THE FOLLOWING SHOULD BE SALVAGABLE JUST NEED TO CLEAN IT
        #  timing is off and/or tracking not working maybe
        ret.excluded_sessions += ["20220201_2"]
        # timing a bit off, tracking not working
        ret.excluded_sessions += ["20220305_2"]
        # tracking didn't work
        ret.excluded_sessions += ["20220306_2"]
        ret.excluded_sessions += ["20220309_1"]
        # timing off
        ret.excluded_sessions += ["20220307_1"]

        ret.rerun_trodes_videos = []
        ret.rerun_usb_videos = []

    elif animalName == "B14":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        ret.data_dir = "/media/WDC6/B14/bradtasksessions/"
        ret.output_dir = "/media/WDC6/B14/processed_data/"
        ret.fig_output_dir = "/media/WDC6/B14/processed_data/"
        ret.out_filename = "B14_bradtask.dat"

        ret.excluded_dates = []
        # minimum_date = "20211209"  # one run with high thresh and one 15 min run on the 8th
        ret.minimum_date = "20220220"  # Only after adjusting stim electrode to correct place!
        ret.excluded_sessions = []
        ret.DEFAULT_RIP_DET_TET = 3
        ret.DEFAULT_RIP_BAS_TET = 2
        # video skips
        ret.excluded_sessions += ["20220307_2"]
        # forgot to turn stim on until after first home well
        ret.excluded_sessions += ["20220311_1"]

        # THE FOLLOWING SHOULD BE SALVAGABLE JUST NEED TO CLEAN IT
        # timing off
        ret.excluded_sessions += ["20220306_2"]
        # tracking didn't work
        ret.excluded_sessions += ["20220310_1"]
        ret.excluded_sessions += ["20220310_2"]
        ret.excluded_sessions += ["20220311_2"]

        ret.rerun_usb_videos = []
        ret.rerun_trodes_videos = []
    elif animalName == "B18":
        ret.X_START = 100
        ret.X_FINISH = 1050
        ret.Y_START = 20
        ret.Y_FINISH = 900
        ret.data_dir = "/home/wcroughan/data/B18/bradtasksessions/"
        ret.output_dir = "/home/wcroughan/data/B18/processed_data/"
        ret.fig_output_dir = "/home/wcroughan/data/B18/processed_data/"
        ret.out_filename = "B18_bradtask.dat"

        ret.excluded_dates = []
        ret.minimum_date = None  
        ret.excluded_sessions = []


        # Amplitude at 100uA, probably don't need to exclude but should note
        # ret.excluded_sessions += ["20220620_2"]

        # stim wire was broken, shoulda been iterruption but was no-stim control
        # (both leads were disconnected, so no stim at all made it to brain)
        ret.excluded_sessions += ["20220624_2"]

        ret.DEFAULT_RIP_DET_TET = 5
        ret.DEFAULT_RIP_BAS_TET = 3
        ret.rerun_usb_videos = []
        ret.rerun_trodes_videos = []


        # ======================
        # Temporary
        ret.minimum_date = "20220621"
        ret.excluded_sessions = ["20220621_2", "20220622_2","20220623_2"]
        ret.excluded_sessions += ["2022062{}_1".format(v) for v in range(24, 29)]
        ret.excluded_sessions += ["2022062{}_2".format(v) for v in range(24, 29)]


    else:
        raise Exception("Unknown animal name")

    return ret


def generateFoundWells(home_well, away_wells, last_away_well, ended_on_home, found_first_home):
    if not found_first_home:
        return []
    elif last_away_well is None:
        return [home_well]

    foundWells = []
    for aw in away_wells:
        foundWells += [home_well, aw]
        if aw == last_away_well:
            break
    if ended_on_home:
        foundWells.append(home_well)
    return foundWells


def getUSBVideoFile(seshName, possibleDirectories):
    seshDate, seshTime = seshName.split("_")
    if len(seshTime) == 1:
        # seshTime is actually session idx
        seshIdx = int(seshTime) - 1

        usbDateStr = "-".join([seshDate[0:4], seshDate[4:6], seshDate[6:8]])
        possibleUSBVids = []
        for pd in possibleDirectories:
            gl = pd + "/" + usbDateStr + "*.mkv"
            possibleUSBVids += glob.glob(gl)

        if len(possibleUSBVids) == 0:
            return None

        possibleUSBVids = sorted(possibleUSBVids)
        return possibleUSBVids[seshIdx]
    else:
        seshTimeVal = float(seshTime[0:2]) * 3600 + \
            float(seshTime[2:4]) * 60 + float(seshTime[4:6])

        usbDateStr = "-".join([seshDate[0:4], seshDate[4:6], seshDate[6:8]])
        possibleUSBVids = []
        for pd in possibleDirectories:
            gl = pd + "/" + usbDateStr + "*.mkv"
            possibleUSBVids += glob.glob(gl)

        if len(possibleUSBVids) == 0:
            return None

        minDiff = 24 * 3600
        usbVidFile = None
        for uvi, uv in enumerate(sorted(possibleUSBVids)):
            fname = uv.split("/")[-1]
            if " " in fname:
                timeStr = fname.split(" ")[1].split(".")[0]
            else:
                timeStr = fname.split("_")[1].split(".")[0]
            timeVals = [float(v) for v in timeStr.split("-")]
            usbTime = timeVals[0] * 3600 + timeVals[1] * 60 + timeVals[0]

            diff = abs(usbTime - seshTimeVal)
            if diff < minDiff:
                minDiff = diff
                usbVidFile = uv
                # seshWithinDay = uvi

        return usbVidFile


def getTrodesVideoFile(seshInfoFileName, data_dir):
    seshDate, seshIdx = seshInfoFileName.split("_")

    gl = data_dir + "/" + seshDate + "_*/" + seshDate + "_*.1.h264"
    possibleTrodesVids = glob.glob(gl)

    if len(possibleTrodesVids) == 0:
        return None

    seshIdx = int(seshIdx) - 1
    possibleTrodesVids = sorted(possibleTrodesVids)
    return possibleTrodesVids[seshIdx]


def numWellsVisited(nearestWells, countReturns=False, wellSubset=None):
    g = groupby(nearestWells)
    if wellSubset is None:
        wellSubset = allWellNames
    if countReturns:
        return len([k for k, _ in g if k in wellSubset])
    else:
        return len(set([k for k, _ in g if k in wellSubset]))


def weekIdxForDateStr(datestr, d0=date(2016, 1, 4)):
    d = date(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:8]))
    return (d - d0).days // 7


def fillCounts(dest, src, t0, t1, windowSize):
    """
    t0, t1, src all in trodes timestamp units
    windowSize in seconds
    """
    ts = np.array(src)
    ts = ts[(ts > t0) & (ts < t1)] - t0
    ts /= TRODES_SAMPLING_RATE
    bins = np.arange(0, (t1 - t0) / TRODES_SAMPLING_RATE + windowSize, windowSize)
    h = np.histogram(ts, bins=bins)
    dest[0:len(bins) - 1] = h[0]
    dest[len(bins) - 1:] = np.nan
