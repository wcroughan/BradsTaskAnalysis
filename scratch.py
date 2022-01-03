import numpy as np
from BTData import BTData
from BTSession import BTSession

# data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
alldata = BTData()
alldata.loadFromFile(data_filename)


home_find_times = [(s.name, (s.home_well_find_times -
                    (np.hstack(([s.bt_pos_ts[0]], (s.away_well_leave_times if s.ended_on_home else s.away_well_leave_times[0:-1]))))) / BTSession.TRODES_SAMPLING_RATE,
                    s.home_well) for s in alldata.getSessions()]

# print(home_find_times)

h1 = []
for h in home_find_times:
    if len(h[1]) > 1:
        h1.append((h[0], h[1][1], h[2]))
print(h1)
