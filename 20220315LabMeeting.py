import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
import sys


possibleDataDirs = ["/media/WDC7/", "/media/fosterlab/WDC7/", "/home/wcroughan/data/"]
dataDir = None
for dd in possibleDataDirs:
    if os.path.exists(dd):
        dataDir = dd
        break

if dataDir == None:
    print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
    exit()

globalOutputDir = os.path.join(dataDir, "figures", "20220315_labmeeting")


if len(sys.argv) >= 2:
    animalNames = sys.argv[1:]
else:
    animalNames = ['B13']
print("Plotting data for animals ", animalNames)

allSessionsByRat = {}
allSessionsWithProbeByRat = {}

for an in animalNames:
    if an == "B13":
        dataFilename = os.path.join(dataDir, "B13/processed_data/B13_bradtask.dat")
    elif an == "B14":
        dataFilename = os.path.join(dataDir, "B14/processed_data/B14_bradtask.dat")
    elif an == "Martin":
        dataFilename = os.path.join(dataDir, "Martin/processed_data/martin_bradtask.dat")
    else:
        raise Exception("Unknown rat " + an)
    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    allSessionsByRat[an] = ratData.getSessions()
    allSessionsWithProbeByRat[an] = ratData.getSessions(lambda s: s.probe_performed)


pp = PlotCtx(globalOutputDir)
for ratName in animalNames:
    print("Running rat", ratName)
    pp.setOutputDir(os.path.join(globalOutputDir, ratName))
    for sesh in allSessionsByRat[ratName]:
        print("Running session", sesh.name)
        pp.setOutputDir(os.path.join(globalOutputDir, ratName, sesh.name))

        lfpFName = sesh.bt_lfp_fnames[-1]
        lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
        lfpV = lfpData[1]['voltage']
        lfpT = np.array(lfpData[0]['time']) / BTSession.TRODES_SAMPLING_RATE

        amps = sesh.btRipPeakAmpsPreStats

        for i in range(10):
            # session.btRipStartIdxsPreStats, session.btRipLensPreStats, session.btRipPeakIdxsPreStats, session.btRipPeakAmpsPreStats = \
            maxRipIdx = np.argmax(amps)

            ripStartIdx = sesh.btRipStartIdxsPreStats[maxRipIdx]
            ripLen = sesh.btRipLensPreStats[maxRipIdx]
            ripPk = sesh.btRipPeakIdxsPreStats[maxRipIdx]
            margin = int(0.15 * 1500)
            i1 = max(0, ripStartIdx - margin)
            i2 = min(ripStartIdx + ripLen + margin, len(lfpV))
            x = lfpT[i1:i2] - lfpT[ripPk]
            y = lfpV[i1:i2]

            xStart = lfpT[ripStartIdx] - lfpT[ripPk]
            xEnd = lfpT[ripStartIdx + ripLen] - lfpT[ripPk]
            ymin = np.min(y)
            ymax = np.max(y)

            with pp.newFig("rawRip_task_{}".format(i)) as ax:
                ax.plot(x, y, zorder=1)
                ax.plot([xStart, xStart], [ymin, ymax], c="red", zorder=0)
                ax.plot([0, 0], [ymin, ymax], c="red", zorder=0)
                ax.plot([xEnd, xEnd], [ymin, ymax], c="red", zorder=0)

            amps[maxRipIdx] = 0
