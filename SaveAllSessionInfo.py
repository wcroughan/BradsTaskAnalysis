import sys
import os
import pandas as pd
import numpy as np

from BTSession import BTSession
from BTData import BTData

if len(sys.argv) == 2:
    animalName = sys.argv[1]
else:
    animalName = 'B13'
print("Saving data for animal ", animalName)


possibleDataDirs = ["/media/WDC7/", "/media/fosterlab/WDC7/", "/home/wcroughan/data/"]
dataDir = None
for dd in possibleDataDirs:
    if os.path.exists(dd):
        dataDir = dd
        break
if dataDir == None:
    print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
    exit()

outputDir = os.path.join(dataDir, animalName)

if animalName == "B13":
    dataFilename = os.path.join(dataDir, "B13/processed_data/B13_bradtask.dat")
elif animalName == "B14":
    dataFilename = os.path.join(dataDir, "B14/processed_data/B14_bradtask.dat")
elif animalName == "Martin":
    dataFilename = os.path.join(dataDir, "Martin/processed_data/martin_bradtask.dat")
else:
    raise Exception("Unknown rat " + animalName)
ratData = BTData()
ratData.loadFromFile(dataFilename)
sessions = ratData.getSessions()

df = pd.DataFrame({})
df['name'] = [s.name for s in sessions]
df['date'] = [s.date for s in sessions]
df['condition'] = ["SWR" if s.isRippleInterruption else "Ctrl" for s in sessions]
df['home well'] = [s.home_well for s in sessions]
df['away wells found'] = [",".join([str(w) for w in s.visited_away_wells]) for s in sessions]
df['task duration'] = [(s.bt_pos_ts[-1] - s.bt_pos_ts[0]) /
                       BTSession.TRODES_SAMPLING_RATE for s in sessions]
df['num wells found'] = [s.num_home_found + s.num_away_found for s in sessions]
df['probe performed?'] = [s.probe_performed for s in sessions]
df['Visited home in 90sec?'] = [s.probe_performed and s.num_well_entries(
    True, s.home_well, timeInterval=[0, 90]) > 0 for s in sessions]
df['curvature around home'] = [s.avg_curvature_at_well(
    True, s.home_well) if s.probe_performed else np.nan for s in sessions]
df['curvature around home 90sec'] = [s.avg_curvature_at_well(
    True, s.home_well, timeInterval=[0, 90]) if s.probe_performed else np.nan for s in sessions]
df['avg dwell time around home'] = [s.avg_dwell_time(
    True, s.home_well) if s.probe_performed else np.nan for s in sessions]
df['avg dwell time around home 90sec'] = [s.avg_dwell_time(
    True, s.home_well, timeInterval=[0, 90]) if s.probe_performed else np.nan for s in sessions]
df['num stims'] = [np.count_nonzero((s.interruption_timestamps < s.bt_pos_ts[-1])
                                    & (s.interruption_timestamps > s.bt_pos_ts[0])) for s in sessions]
df['stim rate'] = df['num stims'] / df['task duration']
df['num ripples in task'] = [len(s.btRipStartTimestampsPreStats) for s in sessions]
df['num ripples in iti'] = [len(s.ITIRipStartTimestampsProbeStats)
                            if s.probe_performed else np.nan for s in sessions]
df['num ripples in probe'] = [len(s.probeRipStartTimestamps)
                              if s.probe_performed else np.nan for s in sessions]

print(df)
outFileName = os.path.join(dataDir, animalName, "processed_data", "session_info.csv")
print("writing to " + outFileName)
df.to_csv(outFileName)
