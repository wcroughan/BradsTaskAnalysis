from BTRestSession import BTRestSession
from datetime import datetime
import MountainViewIO
from scipy import signal

animal_name = "B13"

if animal_name == "B13":
    run_data_file = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
    restDataDir = "/media/WDC7/B13/postTaskRests/"
    output_dir = "/media/WDC7/B13/processed_data/rest_figs"
    DEFAULT_RIP_DET_TET = 7
elif animal_name == "B14":
    run_data_file = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
    restDataDir = "/media/WDC7/B14/postTaskRests/"
    output_dir = "/media/WDC7/B14/processed_data/rest_figs"
    DEFAULT_RIP_DET_TET = 3
else:
    raise Exception("Unknown rat")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

allRestDirs = sorted(os.listdir(restDataDir), key=lambda s: (
    s.split('_')[0], s.split('_')[1]))
restSessionNotesDir = os.path.join(restDataDir, 'rest_session_notes')

LFP_SAMPLING_RATE = 1500.0
# Typical duration of a Sharp-Wave Ripple
ACCEPTED_RIPPLE_LENGTH = int(0.1 * LFP_SAMPLING_RATE)
RIPPLE_FILTER_BAND = [150, 250]
RIPPLE_FILTER_ORDER = 4


def get_ripple_power(lfp_data, omit_artifacts=True, causal_smoothing=False, lfp_deflections=None):
    """
    Get ripple power in LFP
    """
    lfp_data_copy = lfp_data.copy()

    if lfp_deflections is None:
        if omit_artifacts:
            raise Exception("this hasn't been updated")
            # Remove all the artifacts in the raw ripple amplitude data
            deflection_metrics = signal.find_peaks(np.abs(np.diff(lfp_data,
                                                                  prepend=lfp_data[0])), height=DEFLECTION_THRESHOLD_LO,
                                                   distance=MIN_ARTIFACT_DISTANCE)
            lfp_deflections = deflection_metrics[0]

    # After this preprocessing, clean up the data if needed.
    if lfp_deflections is not None:
        for artifact_idx in range(len(lfp_deflections)):
            cleanup_start = max(0, lfp_deflections[artifact_idx] - SKIP_TPTS_BACKWARD)
            cleanup_finish = min(len(lfp_data)-1, lfp_deflections[artifact_idx] +
                                 SKIP_TPTS_FORWARD)
            lfp_data_copy[cleanup_start:cleanup_finish] = np.nan

    nyq_freq = LFP_SAMPLING_RATE * 0.5
    lo_cutoff = RIPPLE_FILTER_BAND[0]/nyq_freq
    hi_cutoff = RIPPLE_FILTER_BAND[1]/nyq_freq
    pl, ph = signal.butter(RIPPLE_FILTER_ORDER, [lo_cutoff, hi_cutoff], btype='band')
    if causal_smoothing:
        ripple_amplitude = signal.lfilter(pl, ph, lfp_data_copy)
    else:
        ripple_amplitude = signal.filtfilt(pl, ph, lfp_data_copy)

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
    return (ripple_power-mean_ripple_power)/std_ripple_power


def detectRipples(ripplePower, minHeight=3.0, minLen=0.5, edgeThresh=0.0):
    pks, peakInfo = signal.find_peaks(ripplePower, height=minHeight)
    ripIdxs = []
    ripLens = []
    ripPeakAmps = []
    return ripIdxs, ripLens, ripPeakAmps


if __name__ == "__main__":
    runData = BTData()
    runData.loadFromFile(run_data_file)

    for restIdx, restDir in enumerate(allRestDirs):
        restSession = BTRestSession()

        dir_split = restDir.split('_')
        date_str = dir_split[0][-8:]
        time_str = dir_split[1]
        restSession.date_str = date_str
        restSession.time_str = time_str
        restSession.name = restDir
        s = "{}_{}".format(date_str, time_str)
        print(s)
        restSession.date = datetime.strptime(s, "%Y%m%d_%H%M%S")

        infoFile = os.path.join(restSessionNotesDir, restSession.name + ".txt")
        with open(infoFile, "r") as infoFile:
            lines = infoFile.readlines()
            assert lines[0].split(':')[0] == "session name"
            restSession.btwpSessionName = lines[0].split(':')[1].strip()

        restSession.btwpSession = runData.getSessions(
            lambda s: s.name == restSession.btwpSessionName)[0]

        restSession.ripple_detection_tetrodes = [DEFAULT_RIP_DET_TET]

        all_lfp_data = []
        for i in range(len(restSession.ripple_detection_tetrodes)):
            file_str = os.path.join(restDataDir, restDir, restDir)
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

            gl = lfpdir + "/" + restDir + ".LFP_nt" + \
                str(restSession.ripple_detection_tetrodes[i]) + "ch*.dat"
            lfpfilelist = glob.glob(gl)
            lfpfilename = lfpfilelist[0]
            restSession.lfp_fnames.append(lfpfilename)
            all_lfp_data.append(MountainViewIO.loadLFP(data_file=restSession.lfp_fnames[-1]))

        lfp_data = all_lfp_data[0][1]['voltage']
        lfp_timestamps = all_lfp_data[0][0]['time']

        ripple_power = get_ripple_power(lfp_data, omit_artifacts=False)
        restSession.ripIdxs, restSession.ripLens, restSession.ripPeakAmps = detectRipples(
            ripple_power)

        runData.allRestSessions.append(restSession)

    runData.saveToFile(runData.filename)
