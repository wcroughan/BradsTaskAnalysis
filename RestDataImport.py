from datetime import datetime
import MountainViewIO
from scipy import signal
import os
import glob
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import sys

from BTRestSession import BTRestSession
from BTSession import BTSession
from BTData import BTData
from ImportData import get_ripple_power, detectRipples


if len(sys.argv) == 2:
    animal_name = sys.argv[1]
else:
    animal_name = 'B13'

print("Importing data for animal ", animal_name)

if animal_name == "B13":
    run_data_file = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
    restDataDir = "/media/WDC7/B13/postTaskRests/"
    output_dir = "/media/WDC7/B13/processed_data/rest_figs"
    DEFAULT_RIP_DET_TET = 7

    # the run before this is excluded, so excluding this rest. Note if want to figure out a cleaner way, this could actually be included. Session was just excluded b/c ITI stim was on
    excludedSessions = ["20211216_192019"]
elif animal_name == "B14":
    run_data_file = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
    restDataDir = "/media/WDC7/B14/postTaskRests/"
    output_dir = "/media/WDC7/B14/processed_data/rest_figs"
    DEFAULT_RIP_DET_TET = 3

    excludedSessions = []
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

if __name__ == "__main__":
    runData = BTData()
    runData.loadFromFile(run_data_file)

    runData.allRestSessions = []

    for restIdx, restDir in enumerate(allRestDirs):
        if restDir in excludedSessions:
            print("Excluding restDir {}", restDir)
            continue

        if restDir == "rest_session_notes":
            continue

        restSession = BTRestSession()

        dir_split = restDir.split('_')
        date_str = dir_split[0][-8:]
        timeStr = dir_split[1]
        restSession.date_str = date_str
        restSession.timeStr = timeStr
        restSession.name = restDir
        s = "{}_{}".format(date_str, timeStr)
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

        ripple_power, _, _ = get_ripple_power(lfp_data, omit_artifacts=False)
        restSession.ripStartIdxs, restSession.ripLens, restSession.ripPeakIdxs, restSession.ripPeakAmps = \
            detectRipples(ripple_power)

        restSession.restDuration = (
            lfp_timestamps[-1] - lfp_timestamps[0]) / BTSession.TRODES_SAMPLING_RATE
        print(lfp_timestamps[0], lfp_timestamps[-1], restSession.restDuration / 60 / 60)

        runData.allRestSessions.append(restSession)

    runData.saveToFile(runData.filename)
