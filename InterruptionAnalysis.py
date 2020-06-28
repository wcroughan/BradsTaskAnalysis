"""
Analyze interruption events in LFP and spike data
"""

# System imports
import sys
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import rc as pltParms

# PyQt5 imports
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QDialog, QFileDialog, QMessageBox

# Local Imports
import MountainViewIO
import QtHelperUtils

MODULE_IDENTIFIER = "[InterruptionAnalysis] "

DEFLECTION_THRESHOLD = 2000.0
SAMPLING_RATE = 30000.0
ACCEPTED_RIPPLE_SIZE = int(0.20 * SAMPLING_RATE)
LFP_SAMPLING_RATE = 1500.0
MIN_ARTIFACT_DISTANCE = int(0.05 * LFP_SAMPLING_RATE)
MIN_RIPPLE_DISTANCE = int(0.05 * LFP_SAMPLING_RATE)

RIPPLE_FILTER_BAND = [150, 250]
RIPPLE_FILTER_ORDER = 4
RIPPLE_POWER_SMOOTHING_WINDOW = 0.050
RIPPLE_POWER_THRESHOLD = 3.0
RIPPLE_STD_LOW_CUTOFF = 1.0
POPULATION_SPIKE_SMOOTHING_WINDOW = 0.005
DISPLAY_TIME_AROUND_ARTIFACT = 0.150

# Display parameters
N_SINGLE_SIDED_POP_SAMPLES = int(DISPLAY_TIME_AROUND_ARTIFACT/POPULATION_SPIKE_SMOOTHING_WINDOW)
N_SAMPLES_AROUND_POP = (N_SINGLE_SIDED_POP_SAMPLES * 2) + 1
N_SINGLE_SIDED_SAMPLES = int(DISPLAY_TIME_AROUND_ARTIFACT * SAMPLING_RATE)
N_SAMPLES_AROUND_ARTIFACT = (N_SINGLE_SIDED_SAMPLES * 2) + 1

# Parameters for skipping artifacts
SKIP_TPTS_FORWARD = int(0.075 * LFP_SAMPLING_RATE)
SKIP_TPTS_BACKWARD = int(0.005 * LFP_SAMPLING_RATE)
MAX_LATENCY = int(0.5 * LFP_SAMPLING_RATE)


def get_population_firing_rates(spike_data):
    """
    Convert spike events into population firing rates
    """

    n_units = len(spike_data)
    population_spikes = np.sort(np.concatenate(spike_data[:]))
    spike_bin_size = SAMPLING_RATE * POPULATION_SPIKE_SMOOTHING_WINDOW
    n_spike_bins = int((population_spikes[-1] - population_spikes[0])/spike_bin_size)
    population_fr = [np.zeros(n_spike_bins, dtype=float), np.zeros(n_spike_bins, dtype=float)]
    for st in range(n_spike_bins):
        t_bin_start = st * spike_bin_size
        t_bin_end = (st+1) * spike_bin_size
        data_indices = np.searchsorted(population_spikes, [t_bin_start, t_bin_end])
        population_fr[0][st] = (t_bin_start + t_bin_end) * 0.5
        population_fr[1][st] = data_indices[1] - data_indices[0]

    # Smooth the population firing rate
    # smoothing_window_length = 20
    # smoothing_weights = np.ones(smoothing_window_length)/smoothing_window_length
    # population_fr[1] = np.convolve(population_fr[1], smoothing_weights, mode='same')/n_units
    smoothing_window_length = 1
    population_fr[1] = gaussian_filter(
        population_fr[1], sigma=smoothing_window_length)/smoothing_window_length

    # Account for mean and standard deviation
    population_fr[1] = population_fr[1]/np.std(population_fr[1])
    return population_fr


def get_disruption_latency(lfp_data, lfp_artifacts=None, ripple_power=None):
    # Go to each LFP artifact, start looking backwards to identify the
    # time at which ripple power drops to 1 STD.
    if lfp_artifacts is None:
        ripple_power, lfp_artifacts = get_ripple_power(lfp_data, causal_smoothing=True)
    elif ripple_power is None:
        ripple_power, _ = get_ripple_power(
            lfp_data, causal_smoothing=True, lfp_deflections=lfp_artifacts)

    latency = np.zeros_like(lfp_artifacts, dtype='float')
    ripple_start = np.zeros_like(lfp_artifacts)
    for a_idx in range(len(lfp_artifacts)):
        search_start = max(0, lfp_artifacts[a_idx] - SKIP_TPTS_BACKWARD)
        search_end = max(0, lfp_artifacts[a_idx] - MAX_LATENCY)

        search_lfp = np.flip(ripple_power[search_end:search_start])
        latency[a_idx] = SKIP_TPTS_BACKWARD + \
            np.argmax(np.logical_and(search_lfp < RIPPLE_STD_LOW_CUTOFF,
                                     np.logical_not(np.isnan(search_lfp))))
        ripple_start[a_idx] = lfp_artifacts[a_idx] - latency[a_idx]
    latency = latency / LFP_SAMPLING_RATE
    # print("Mean stimulation latency: %.3f"%np.nanmean(latency))
    return latency, ripple_start


def get_ripple_stats(power_data, ripple_peaks):
    """
    Look at ripple power data and supplied ripple peaks to analyze stats for
    the ripples. For the time being, these stats include width and height of
    the ripple.
    """
    peak_ripple_power = np.zeros_like(ripple_peaks, dtype='float')
    ripple_width = np.zeros_like(ripple_peaks, dtype='float')
    ripple_start_mark = np.zeros_like(ripple_peaks, dtype='int')
    ripple_end_mark = np.zeros_like(ripple_peaks, dtype='int')

    for a_idx in range(len(ripple_peaks)):
        search_start = max(0, ripple_peaks[a_idx] - MAX_LATENCY)
        search_end = min(len(power_data), ripple_peaks[a_idx] + MAX_LATENCY)

        peak_ripple_power[a_idx] = power_data[ripple_peaks[a_idx]]
        tmp = power_data[ripple_peaks[a_idx]:search_end]
        ripple_end_mark[a_idx] = ripple_peaks[a_idx] + \
            np.argmax(np.less(tmp, RIPPLE_STD_LOW_CUTOFF, where=np.isnan(tmp) == False))
        tmp = power_data[search_start:ripple_peaks[a_idx]]
        ripple_start_mark[a_idx] = ripple_peaks[a_idx] - np.argmax(np.flip(np.less(tmp, RIPPLE_STD_LOW_CUTOFF,
                                                                                   where=np.isnan(tmp) == False)))
        ripple_width[a_idx] = (ripple_end_mark[a_idx] - ripple_start_mark[a_idx])/LFP_SAMPLING_RATE
    return peak_ripple_power, ripple_width, ripple_start_mark, ripple_end_mark


def get_ripple_peaks(power_data):
    ripple_peaks = signal.find_peaks(power_data,
                                     height=RIPPLE_POWER_THRESHOLD,
                                     distance=MIN_RIPPLE_DISTANCE)
    return ripple_peaks[0]


def get_ripple_power(lfp_data, omit_artifacts=True, causal_smoothing=False, lfp_deflections=None):
    """
    Get ripple power in LFP
    """

    nyq_freq = LFP_SAMPLING_RATE * 0.5
    lo_cutoff = RIPPLE_FILTER_BAND[0]/nyq_freq
    hi_cutoff = RIPPLE_FILTER_BAND[1]/nyq_freq
    pl, ph = signal.butter(RIPPLE_FILTER_ORDER, [lo_cutoff, hi_cutoff], btype='band')
    if causal_smoothing:
        ripple_amplitude = signal.lfilter(pl, ph, lfp_data)
    else:
        ripple_amplitude = signal.filtfilt(pl, ph, lfp_data)

    # In order to no have NaN values affect the filter output, create a copy with the artifacts
    ripple_amplitude_copy = ripple_amplitude.copy()

    if lfp_deflections is None:
        if omit_artifacts:
            # Remove all the artifacts in the raw ripple amplitude data
            deflection_metrics = signal.find_peaks(np.abs(np.diff(lfp_data,
                                                                  prepend=lfp_data[0])), height=DEFLECTION_THRESHOLD,
                                                   distance=MIN_ARTIFACT_DISTANCE)
            lfp_deflections = deflection_metrics[0]

    # After this preprocessing, clean up the data if needed.
    if lfp_deflections is not None:
        for artifact_idx in range(len(lfp_deflections)):
            cleanup_start = max(0, lfp_deflections[artifact_idx] - SKIP_TPTS_BACKWARD)
            cleanup_finish = min(len(lfp_data)-1, lfp_deflections[artifact_idx] +
                                 SKIP_TPTS_FORWARD)
            ripple_amplitude[cleanup_start:cleanup_finish] = np.nan

    # Smooth this data and get ripple power
    # smoothing_window_length = RIPPLE_POWER_SMOOTHING_WINDOW * LFP_SAMPLING_RATE
    # smoothing_weights = np.ones(int(smoothing_window_length))/smoothing_window_length
    # ripple_power = np.convolve(np.abs(ripple_amplitude), smoothing_weights, mode='same')

    # Use a Gaussian kernel for filtering - Make the Kernel Causal bu keeping only one half of the values
    smoothing_window_length = 10
    if causal_smoothing:
        half_smoothing_signal = \
            np.exp(-np.square(np.linspace(0, -4*smoothing_window_length, 4 *
                                          smoothing_window_length))/(2*smoothing_window_length * smoothing_window_length))
        smoothing_signal = np.concatenate(
            (np.zeros_like(half_smoothing_signal), half_smoothing_signal), axis=0)
        ripple_power = signal.convolve(np.abs(ripple_amplitude_copy),
                                       smoothing_signal, mode='same') / np.sum(smoothing_signal)
        ripple_power[np.isnan(ripple_amplitude)] = np.nan
    else:
        ripple_power = gaussian_filter(np.abs(ripple_amplitude), smoothing_window_length)

    # Get the mean/standard deviation for ripple power and adjust for those
    mean_ripple_power = np.nanmean(ripple_power)
    std_ripple_power = np.nanstd(ripple_power)
    return (ripple_power-mean_ripple_power)/std_ripple_power, lfp_deflections


if __name__ == "__main__":
    # Start a fake application
    # qt_args = list()
    # qt_args.append('-style')
    # qt_args.append('Windows')
    # print(MODULE_IDENTIFIER + "Qt Arguments: " + str(qt_args))
    # app = QApplication(qt_args)
    print(MODULE_IDENTIFIER + "Parsing Input Arguments: " + str(sys.argv))

    # Hard-coded file locations on my lab machine
    """
    firings_file_name = 'firings-1.curated.mda'
    spike_data_loc = '/run/media/ag/ProcessedEllisData/Ellis20190213_NOMASK.mnt/preprocessing'
    lfp_data_file = '/run/media/ag/ProcessedEllisData/Ellis20190213_NOMASK.mnt/lfp/full_config20190213_161714__ripple_interruption.LFP_nt2ch2.dat'
    """

    # Session 1 has a lot of disruption but it doesn't have a good baseline for comparison.
    # firings_file_name = 'firings-2.curated.mda'
    firings_file_name = 'firings-2.curated.mda'
    spike_data_loc = '/run/media/ag/ProcessedEllisData/Ellis20190213_NOMASK.mnt/preprocessing'
    lfp_data_file = '/run/media/ag/ProcessedEllisData/Ellis20190213_NOMASK.mnt/lfp/full_config20190213_165752__ripple_interruption.LFP_nt2ch2.dat'

    # Get the file using a dialog-box
    # spike_data_loc = QtHelperUtils.get_directory(message="Select spike-data directory.")
    # lfp_data_file = QtHelperUtils.get_open_file_name(message="Select LFP data file.")

    find_artifacts = False

    # Configuration
    plt.ioff()
    show_entire_session = True
    plot_individual_interruptions = False

    # Load LFP data
    lfp_data = MountainViewIO.loadLFP(data_file=lfp_data_file)
    spike_data = MountainViewIO.loadClusteredData(
        data_location=spike_data_loc, firings_file=firings_file_name)

    # Pick out the time chunk that we are interested in
    interruption_interval = (500.0, 2850.0)

    # Get ripple power
    ripple_power = get_ripple_power(lfp_data[1]['voltage'])

    # Find all the LFP deflections. This helps us identify the interruption times
    if find_artifacts:
        lfp_deflections = signal.find_peaks(-lfp_data[1]['voltage'], height=DEFLECTION_THRESHOLD,
                                            distance=MIN_ARTIFACT_DISTANCE)
    else:
        # Settle for high ripple power events (3 standard deviations)
        lfp_deflections = signal.find_peaks(ripple_power, height=RIPPLE_POWER_THRESHOLD,
                                            distance=MIN_ARTIFACT_DISTANCE)

    # Get population firing rate
    valid_spikes = [unit for unit in spike_data if unit is not None]
    population_fr = get_population_firing_rates(valid_spikes)

    n_units = len(valid_spikes)
    if show_entire_session:
        plt.figure()
        # Show the raw data - spike raster and LFP deflections
        for unit, spikes in enumerate(valid_spikes):
            tmp = np.true_divide(spikes, SAMPLING_RATE, where=np.isnan(spikes) == False)
            plt.scatter(tmp, unit * np.ones_like(spikes), s=32, alpha=0.9)

        # Show LFP and the deflection peaks
        # plt.plot(lfp_data[0]['time']/SAMPLING_RATE, 10.0 * ripple_power)
        # plt.plot(population_fr[0]/SAMPLING_RATE, 10.0 * population_fr[1])
        plt.plot(lfp_data[0]['time']/SAMPLING_RATE,  (1.2 * n_units) + n_units * np.array(lfp_data[1]['voltage'],
                                                                                          dtype=float)/DEFLECTION_THRESHOLD)
        plt.scatter(lfp_data[0]['time'][lfp_deflections[0]]/SAMPLING_RATE, n_units *
                    np.ones_like(lfp_deflections[0]), marker='v', s=32, color='black', alpha=0.5)
        plt.scatter(lfp_data[0]['time'][lfp_deflections[0]]/SAMPLING_RATE, np.zeros_like(lfp_deflections[0]),
                    marker='^', s=32, color='black')
        plt.ylim([-2, 2.0*n_units])
        plt.xlabel('Time (s)')
        plt.ylabel('Unit')
        plt.grid(True)
        plt.show()

    # Only consider lfp_deflections between some time intervals
    interruption_times = np.array(
        lfp_data[0]['time'][lfp_deflections[0]], dtype=float)/SAMPLING_RATE
    in_range_interruptions = np.logical_and(interruption_interval[0] < interruption_times,
                                            interruption_times < interruption_interval[1])

    # Real data
    interruption_events = lfp_deflections[0][in_range_interruptions]
    n_interruption_events = len(interruption_events)

    # Control
    # interruption_events = np.random.randint(interruption_events[0], high=interruption_events[-1], \
    #         size=n_interruption_events)

    plot_individual_interruptions = n_interruption_events < 20
    print(interruption_events)

    # Extract LFP, ripple power and spikes in the neighborhood of interruption events

    # it_X is short for interruption_X in the following variable names
    it_lfp = np.zeros((n_interruption_events, N_SAMPLES_AROUND_ARTIFACT), dtype=float)
    it_ripple_power = np.zeros((n_interruption_events, N_SAMPLES_AROUND_ARTIFACT), dtype=float)

    # For population, we need to separately find out the matching indices
    it_population_activity = np.zeros((n_interruption_events, N_SAMPLES_AROUND_POP), dtype=float)
    it_pop_indices = np.searchsorted(population_fr[0], lfp_data[0]['time'][interruption_events])

    first_valid_lfp_idx = 0
    first_valid_pop_idx = 0
    last_valid_lfp_idx = n_interruption_events
    last_valid_pop_idx = n_interruption_events

    for d_idx, deflection_timestamp in enumerate(interruption_events):
        if (deflection_timestamp-N_SINGLE_SIDED_SAMPLES < 0) or \
                (it_pop_indices[d_idx]-N_SINGLE_SIDED_POP_SAMPLES < 0):
            first_valid_lfp_idx = d_idx + 1
            first_valid_pop_idx = d_idx + 1
            continue

        if (deflection_timestamp+N_SINGLE_SIDED_SAMPLES > len(lfp_data[0])) or \
                (it_pop_indices[d_idx]+N_SINGLE_SIDED_POP_SAMPLES > len(population_fr[1])):
            last_valid_lfp_idx = d_idx-1
            last_valid_pop_idx = d_idx-1
            break

        it_lfp[d_idx, :] = lfp_data[1]['voltage'][deflection_timestamp -
                                                  N_SINGLE_SIDED_SAMPLES-1:deflection_timestamp+N_SINGLE_SIDED_SAMPLES]
        it_ripple_power[d_idx, :] = ripple_power[deflection_timestamp -
                                                 N_SINGLE_SIDED_SAMPLES-1:deflection_timestamp+N_SINGLE_SIDED_SAMPLES]
        it_population_activity[d_idx, :] = population_fr[1][it_pop_indices[d_idx] -
                                                            N_SINGLE_SIDED_POP_SAMPLES-1:it_pop_indices[d_idx]+N_SINGLE_SIDED_POP_SAMPLES]

    # Slice  the data to the appropriate, valid  size
    it_lfp = it_lfp[first_valid_lfp_idx:last_valid_lfp_idx, :]
    it_ripple_power = it_ripple_power[first_valid_lfp_idx:last_valid_lfp_idx, :]
    it_population_activity = it_population_activity[first_valid_pop_idx:last_valid_pop_idx, :]

    # Get the mean and standard deviation for all these quantities and plot it
    mean_lfp = np.mean(it_lfp, axis=0)
    mean_ripple_power = np.mean(it_ripple_power, axis=0)
    mean_pop_activity = np.mean(it_population_activity, axis=0)
    tmp = np.std(it_population_activity, axis=0)
    std_pop_activity = np.true_divide(tmp, np.sqrt(
        n_interruption_events), where=np.isnan(tmp) == False)

    n_valid_events = np.shape(it_population_activity)[0]
    if plot_individual_interruptions:
        for inter_idx in range(n_valid_events):
            plt.figure()
            plt.plot(np.linspace(-DISPLAY_TIME_AROUND_ARTIFACT, DISPLAY_TIME_AROUND_ARTIFACT,
                                 N_SAMPLES_AROUND_ARTIFACT), -it_lfp[inter_idx, :]/min(it_lfp[inter_idx, :]))
            plt.plot(np.linspace(-DISPLAY_TIME_AROUND_ARTIFACT, DISPLAY_TIME_AROUND_ARTIFACT,
                                 N_SAMPLES_AROUND_ARTIFACT), it_ripple_power[inter_idx, :])
            plt.plot(np.linspace(-DISPLAY_TIME_AROUND_ARTIFACT, DISPLAY_TIME_AROUND_ARTIFACT,
                                 N_SAMPLES_AROUND_POP), it_population_activity[inter_idx, :])
            plt.xlabel('Time (s)')
            plt.grid(True)
            plt.show()

    # Plot the mean activity
    peak_pop_activity = max(mean_pop_activity)
    plt.figure()
    # plt.plot(np.linspace(-DISPLAY_TIME_AROUND_ARTIFACT, DISPLAY_TIME_AROUND_ARTIFACT, N_SAMPLES_AROUND_ARTIFACT), -mean_lfp/min(mean_lfp))
    plt.plot(np.linspace(-DISPLAY_TIME_AROUND_ARTIFACT, DISPLAY_TIME_AROUND_ARTIFACT, N_SAMPLES_AROUND_ARTIFACT), peak_pop_activity * mean_ripple_power/max(mean_ripple_power),
             label='LFP Deflection (STD)', linewidth=2.0)
    plt.errorbar(np.linspace(-DISPLAY_TIME_AROUND_ARTIFACT, DISPLAY_TIME_AROUND_ARTIFACT, N_SAMPLES_AROUND_POP), mean_pop_activity, std_pop_activity, marker='s',
                 markerfacecolor='black', markeredgecolor='green', markersize=2, markeredgewidth=2, alpha=0.7, label='Population FR', linewidth=2.0)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Population Firing Rate (Hz)')
    plt.grid(True)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 32}

    pltParms('font', **font)
    plt.show()
    x = input('Press any key to Quit!')
