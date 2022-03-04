class TestManager:
    def __init__(self):
        self.count = 0

    def __enter__(self):
        print("enter", self.count)
        return self

    def __exit__(self, *args):
        print("exit", self.count)
        self.count += 1


t = TestManager()

with t as tt:
    print("hello!")
    print(tt.count)
    print(t.count)

with t:
    print("hello again!", t.count)

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
    # 2014 20, went straight for the well but then got a bit distracted while next to it maybe heard something. Gave it a quick sniff after that and then moved on
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
#                     (np.hstack(([s.bt_pos_ts[0]], (s.away_well_leave_times if s.ended_on_home else s.away_well_leave_times[0:-1]))))) / BTSession.TRODES_SAMPLING_RATE,
#                     s.home_well) for s in alldata.getSessions()]

# # print(home_find_times)

# h1 = []
# for h in home_find_times:
#     if len(h[1]) > 1:
#         h1.append((h[0], h[1][1], h[2]))
# print(h1)
