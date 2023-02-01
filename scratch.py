# import os
from BTData import BTData
from BTSession import BTSession, ImportOptions
import numpy as np
from UtilFunctions import Ripple
# import matplotlib.pyplot as plt
# from UtilFunctions import getRipplePower
import json
from dataclasses import is_dataclass, asdict

# r = Ripple(0, 1, 2, 3, 4, 5, 6, 7, 8)
# print(type(r))
# print(is_dataclass(r))
# print(str(r))
# d = asdict(r)
# print(d)
# rr = Ripple(**d)
# print(rr)

# exit()


i = ImportOptions()
i2 = ImportOptions(skipLFP=True)
r = Ripple(0, 1, 2, 3, 4, 5, 6, 7, 4)
r2 = Ripple(10, 11, 21, 31, 41, 51, 61, 71, 4)
d = BTData()
s1 = BTSession()
s1.name = "Hello i am s1"
s1.loggedDetections_ts = [1, 2, 3, 4]
s1.awayRewardEnter_posIdx = np.arange(9).reshape((3, 3))
s1.importOptions = i
s1.animalName = r
s2 = BTSession()
s2.name = "I'm s2!"
s2.loggedDetections_ts = np.arange(0, 12, 2)
s2.awayRewardEnter_posIdx = np.arange(9).reshape((3, 3))
s2.importOptions = i2
s2.animalName = r2
d.allSessions = [s1, s2]
d.saveToFile_new("./testSave.rat")
d.loadFromFile_new("./testSave.rat.npz")
print(d.allSessions)

# dd = {}
# dd["a"] = np.array([1, 2, 5, 5])
# dd["b"] = np.arange(9).reshape((3, 3))
# dd["c"] = "hello"
# dd["d"] = 2
# dd["e"] = {"subtest": "oiwejfoiewjf"}
# dd["e"] = json.dumps(dd["e"])
# dd["notarrays"] = ["c", "d", "e"]
# np.savez_compressed("./test.npz", **dd)

# od = {}
# l = np.load("./test.npz")
# print(l)
# print(l.files)
# for f in l.files:
#     if f == "notarrays":
#         continue
#     k = f
#     v = l[f]
#     print(f"{ f = }")
#     print(f"{ k = }")
#     print(f"{ v = }")
#     print(f"{ v.shape = }")
#     if k in l["notarrays"]:
#         v = v.item()
#     print(f"{ v = }")
#     print(f"{ type(v) = }")
#     if isinstance(v, str):
#         try:
#             v = json.loads(v)
#         except:
#             pass
#     print(f"{ v = }")
#     print(f"{ type(v) = }")

#     od[k] = v

# print(od)


# tlen = 1
# fs = 1500
# x = np.linspace(0, tlen, fs*tlen)
# ramp = np.linspace(0, 1, len(x))
# s1 = np.sin(x * 200 * 2 * np.pi)
# s2 = np.sin(x * 75 * 2 * np.pi)
# s3 = s1 * ramp + s2 * (1 - ramp)
# s4 = s1.copy()
# s4[(len(x) // 2):] = s2[(len(x) // 2):]

# sig = s4

# ps, _, _, _ = getRipplePower(sig, method="standard")
# pc, _, _, _ = getRipplePower(sig, method="causal")
# # pa, _, _, _ = getRipplePower(sig, method="activelink")

# assert len(sig) == len(ps) == len(pc)


# plt.plot(x, sig, label="signal")
# plt.plot(x, ps, label="standard")
# plt.plot(x, pc, label="causal")
# plt.legend()
# plt.show()


# locations = np.array([
#     [256, 124],
#     [367, 109],
#     [489, 93],
#     [621, 80],
#     [758, 79],
#     [892, 79],
#     [905, 199],
#     [348, 221],
#     [232, 232],
#     [208, 354],
#     [329, 346],
#     [461, 340],
#     [451, 483],
#     [314, 485],
#     [189, 483],
#     [176, 625],
#     [300, 633],
#     [443, 644],
#     [438, 800],
#     [293, 783],
#     [168, 765],
#     [597, 808],
#     [765, 811],
#     [932, 806],
#     [933, 644],
#     [770, 648],
#     [604, 646],
#     [607, 485],
#     [612, 332],
#     [452, 485],
#     [452, 485],
#     [929, 484],
#     [920, 332],
#     [906, 196],
#     [906, 196],
#     [906, 196],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [612, 330],
#     [766, 331],
#     [761, 199],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [617, 198],
#     [476, 211],
#     [463, 340],
#     [463, 340],
#     [463, 340],
#     [463, 340]
# ])

# plt.scatter(locations[:, 0], locations[:, 1])
# plt.show()


# from UtilFunctions import getInfoForAnimal
# from consts import TRODES_SAMPLING_RATE
# from PlotUtil import PlotCtx
# import MountainViewIO

# lfpFName = "/home/wcroughan/20220511_142645/20220511_142645.LFP/20220511_142645.LFP_nt6ch1.dat"
# print("LFP data from file {}".format(lfpFName))
# lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
# lfpV = lfpData[1]['voltage']
# lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

# pp = PlotCtx()
# with pp.newFig("lfp", showPlot=True, savePlot=False) as ax:
#     ax.plot(lfpT, lfpV)


# lfpFName = "/home/wcroughan/20220511_142645/20220511_142645.LFP/20220511_142645.LFP_nt3ch1.dat"
# print("LFP data from file {}".format(lfpFName))
# lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
# lfpV = lfpData[1]['voltage']
# lfpT = np.array(lfpData[0]['time']) / TRODES_SAMPLING_RATE

# pp = PlotCtx()
# with pp.newFig("lfp", showPlot=True, savePlot=False) as ax:
#     ax.plot(lfpT, lfpV)


# possibleDataDirs = ["/media/WDC6/", "/media/fosterlab/WDC6/", "/home/wcroughan/data/"]
# dataDir = None
# for dd in possibleDataDirs:
#     if os.path.exists(dd):
#         dataDir = dd
#         break

# if dataDir == None:
#     print("Couldnt' find data directory among any of these: {}".format(possibleDataDirs))
#     exit()

# globalOutputDir = os.path.join(dataDir, "figures", "20220315_labmeeting")


# if len(sys.argv) >= 2:
#     animalNames = sys.argv[1:]
# else:
#     animalNames = ['B13', 'B14', 'Martin']
# print("Plotting data for animals ", animalNames)

# for an in animalNames:
#     animalInfo = getInfoForAnimal(an)
#     dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
#     ratData = BTData()
#     ratData.loadFromFile(dataFilename)
#     swp = ratData.getSessions()

#     print("==========================")
#     print(an)
#     lastDate = None
#     for s in swp:
#         # d = s.name.split("_")[0]
#         # if d != lastDate:
#         #     print()
#         # lastDate = d
#         print("{}\t{}\t{}\t{}\t{}".format(s.name, "SWR" if s.isRippleInterruption else "Ctrl",
#               s.infoFileName.split("_")[-1], s.conditionGroup, s.probe_performed))


# data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
# data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
# alldata = BTData()
# alldata.loadFromFile(data_filename)

# TRODES_SAMPLING_RATE = 30000

# for sesh in alldata.getSessions(lambda s: s.probe_performed):
#     print(sesh.name, sesh.home_well, sesh.isRippleInterruption)
# 1918 35, quick check
# 1315 13, good check, pause before sniff
# 1514 21, longer check at 13 just before (also pause b4 sniff again at 13 but not at 21)
# 1517 37, quick passing check
# 1615 36, longer than a quick check, but also searched a bunch at other wells first
# 1717 30, checked a few previous home wells for a good bit, but spent even longer at 30
# 2014 20, went straight for the well but then got a bit distracted while next to it maybe heard something.
#                   Gave it a quick sniff after that and then moved on
# 2018 30, clearly very long check
# 2114 12, long check, went pretty directly there
# 2117 29, never visited the home. Spent a while at 21

# for sesh in alldata.getSessions(lambda s: s.probe_performed):
#     lfpFName = sesh.bt_lfp_fnames[-1]
#     lfpData = MountainViewIO.loadLFP(data_file=lfpFName)
#     lfpV = lfpData[1]['voltage']
#     lfpT = lfpData[0]['time']

#     itiStimIdxs = sesh.interruptionIdxs - sesh.ITIRippleIdxOffset
#     zeroIdx = np.searchsorted(itiStimIdxs, 0)
#     itiStimIdxs = itiStimIdxs[zeroIdx:]

#     i1 = sesh.ITIRippleIdxOffset
#     i2 = np.searchsorted(lfpT, lfpT[i1] + 60 * TRODES_SAMPLING_RATE)
#     x = lfpT[i1:i2]
#     y = lfpV[i1:i2]
#     ripplePower, _, _ = get_ripple_power(
#         y, omit_artifacts=False, lfp_deflections=itiStimIdxs)

#     # ripplePower, _, _ = get_ripple_power(y, omit_artifacts=False)

#     plt.clf()
#     plt.plot(lfpT[i1:i2], lfpV[i1:i2])
#     plt.plot(lfpT[i1:i2], ripplePower * 1000)
#     plt.xlim(lfpT[i1], lfpT[i1] + TRODES_SAMPLING_RATE)
#     plt.show()

# home_find_times = [(s.name, (s.home_well_find_times -
#                     (np.hstack(([s.bt_pos_ts[0]],
# (s.away_well_leave_times if s.endedOnHome else s.away_well_leave_times[0:-1]))))) / BTSession.TRODES_SAMPLING_RATE,
#                     s.home_well) for s in alldata.getSessions()]

# # print(home_find_times)

# h1 = []
# for h in home_find_times:
#     if len(h[1]) > 1:
#         h1.append((h[0], h[1][1], h[2]))
# print(h1)


# numAwaysFound = np.array([2, 5, 2, 0, 7, 11, 11, 3, 11, 6, 12, 8, 12, 10, 9, 12, 12,
#                           7, 12, 12, 12, 12, 12, 11, 7, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
#                           12, 12, 10, 12, 2, 3, 2, 0, 0, 1, 0, 2, 4, 2, 0, 1, 3, 1, 0, 2, 5, 5, 5, 6, 5,
#                           5, 7, 6, 5, 6, 4, 12, 11, 6, 7, 3, 8, 0, 3, 4, 5, 2, 3, 3, 12, 12, 12, 12, 12, 7, 8])

# timeSpentText = [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 1056, 2000, 1550, 2000, 2000,
#                  1301, 1045, 2000, 2000, 1711, 1206, 1158, 1420, 2000, 2000, 1426, 1516, 1247, 1131, 1352,
#                  1602, 1100, 1259, 1038, 1400, 1156, 1555, 1331, 1628, 2000, 1348, 6000, 6000, 6000, 3000,
#                  4500, 4500, 6000, 6000, 3000, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 3000, 3000,
#                  6000, 1700, 3000, 2000, 2000, 1800, 2000, 2000, 1700, 2000, 2000, 2000, 2000, 2000, 2000,
#                  2000, 2000, 2000, 2000, 2000, 2000, 1924, 1447, 1035, 1259, 1300, 2000, 2000]
# timeSpentMinutes = [float(str(s)[0:2]) + (float(str(s)[2:4]) / 60.0) for s in timeSpentText]
# plt.plot(numAwaysFound)
# plt.plot(timeSpentMinutes)

# cableHookedUpIdx = 41
# firstNesquikSession = 56
# firstCrystalSession = 61
# backToNesquikSession = 70
# firstSmallerRewardSession = 78
# backToCrystalSession = 79
# lines = [cableHookedUpIdx, firstNesquikSession, firstCrystalSession,
#          backToNesquikSession, firstSmallerRewardSession, backToCrystalSession]
# for ll in lines:
#     plt.plot([ll - 0.5, ll - 0.5], [0, 60], 'k')
# plt.plot([0, len(numAwaysFound)], [12, 12], '--k')

# plt.legend(["num away wells found", "session duration (mins)"])
# plt.xlabel("Session")
# plt.show()
