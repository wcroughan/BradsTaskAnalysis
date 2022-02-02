import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
import seaborn as sns
import os
from scipy.stats import pearsonr, ttest_ind_from_stats
import statsmodels.api as sm
from matplotlib.lines import Line2D
from datetime import datetime

from BTData import BTData
from BTSession import BTSession
from BTRestSession import BTRestSession

SAVE_OUTPUT_PLOTS = True
SHOW_OUTPUT_PLOTS = False

possibleDataDirs = ["/media/WDC7/", "/media/fosterlab/WDC7/", "/home/wcroughan/data/"]
dataDir = None
for dd in possibleDataDirs:
    if os.path.exists(dd):
        dataDir = dd
        break

if dataDir == None:
    print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
    exit()

outputDir = os.path.join(dataDir, "figures", "thesisCommittee20220208")
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

allWellNames = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])

# animalNames = ["B13", "B14", "Martin"]
animalNames = ["B13"]
allSessions = []
allSessionsByRat = {}
allSessionsWithProbe = []
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
    allSessions += ratData.getSessions()
    allSessionsWithProbe += ratData.getSessions(lambda s: s.probe_performed)


def setupBehaviorTracePlot(axs, sesh, showAllWells=True, showHome=True, showAways=True, zorder=2):
    if not isinstance(axs, list):
        axs = [axs]
    x1 = np.min(sesh.bt_pos_xs)
    x2 = np.max(sesh.bt_pos_xs)
    y1 = np.min(sesh.bt_pos_ys)
    y2 = np.max(sesh.bt_pos_ys)
    for ax in axs:
        if showAllWells:
            for w in allWellNames:
                wx, wy = sesh.well_coords_map[str(w)]
                ax.scatter(wx, wy, c="black", zorder=2)
        if showAways:
            for w in sesh.visited_away_wells:
                wx, wy = sesh.well_coords_map[str(w)]
                ax.scatter(wx, wy, c="blue", zorder=zorder)
        if showHome:
            wx, wy = sesh.well_coords_map[str(sesh.home_well)]
            ax.scatter(wx, wy, c="red", zorder=zorder)

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.tick_params(axis="both", which="both", label1On=False,
                       label2On=False, tick1On=False, tick2On=False)


def saveOrShow(fname, clearAxes=True):
    if SAVE_OUTPUT_PLOTS:
        plt.savefig(os.path.join(outputDir, fname), dpi=800)
    if SHOW_OUTPUT_PLOTS:
        plt.show()

    if clearAxes:
        plt.cla()


fig = plt.figure()
axs = fig.subplots()
for sesh in allSessions:
    print(sesh.name)

    # =============
    # Raw behavioral trace (task)
    axs.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, c="#deac7f")
    setupBehaviorTracePlot(axs, sesh)
    saveOrShow("raw_task_{}".format(sesh.name))



for sesh in allSessionsWithProbe:
    print(sesh.name)

    # =============
    # Raw behavioral trace (probe)
    axs.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, c="#deac7f")
    setupBehaviorTracePlot(axs, sesh)
    saveOrShow("raw_probe_{}".format(sesh.name))


# if isinstance(allData, list):
#     self.all_sessions = []
#     self.all_rest_sessions = []
#     for ad in allData:
#         self.all_sessions += ad.getSessions()
#         self.all_rest_sessions += ad.getRestSessions()
# else:
#     self.all_sessions = allData.getSessions()
#     self.all_rest_sessions = allData.getRestSessions()
# self.all_sessions_with_probe = [s for s in self.all_sessions if s.probe_performed]
# self.tlbls = [self.trial_label(sesh) for sesh in self.all_sessions]
# self.tlbls_with_probe = [self.trial_label(sesh) for sesh in self.all_sessions_with_probe]

#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         statsFileName = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S_stats.txt"))
#         self.statsFile = open(statsFileName, "w")
#         idxKeyFileName = os.path.join(
#             output_dir, datetime.now().strftime("%Y%m%d_%H%M%S_idx_key.txt"))
#         with open(idxKeyFileName, "w") as idxKeyFile:
#             idxKeyFile.write("All sessions:\n")
#             for si, s in enumerate(self.all_sessions):
#                 idxKeyFile.write("{}: {} - {} (home {}, {} stims)\n".format(si, s.name,
#                                  "SWR " if s.isRippleInterruption else "CTRL", s.home_well, len(s.bt_interruption_pos_idxs)))

#             idxKeyFile.write("All sessions with probe:\n")
#             for si, s in enumerate(self.all_sessions_with_probe):
#                 idxKeyFile.write("{}: {} - {} (home {}, {} stims)\n".format(si, s.name,
#                                  "SWR " if s.isRippleInterruption else "CTRL", s.home_well, len(s.bt_interruption_pos_idxs)))

#             idxKeyFile.write("All rest sessions\n")
#             for si, s in enumerate(self.all_rest_sessions):
#                 idxKeyFile.write(str(si) + ": " + s.name + "\n")
