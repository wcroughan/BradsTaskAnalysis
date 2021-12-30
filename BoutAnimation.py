from BTData import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from itertools import groupby
import random

# data_filename = "/media/WDC4/martindata/bradtask/martin_bradtask.dat"
# output_dir = '/media/WDC4/martindata/processed_data/animations'
data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
# output_dir = "/media/WDC7/B13/processed_data/behavior_figures/animations/"

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

alldata = BTData()
alldata.loadFromFile(data_filename)
all_sessions = alldata.getSessions()

fig, ax = plt.subplots()
line_ex, = ax.plot([], [], lw=2)
line_pa, = ax.plot([], [], lw=2)
line_re, = ax.plot([], [], lw=2)
# scat, = ax.scatter([], [])
ax.grid()
xdata_ex,  ydata_ex = [], []
xdata_re,  ydata_re = [], []
xdata_pa,  ydata_pa = [], []

PLOT_BEHAVIOR_FRAMES = 30


def init_plot():
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1000)
    line_ex.set_data([], [])
    line_re.set_data([], [])
    line_pa.set_data([], [])
    return line_ex, line_pa, line_re


def animate(i):
    i1 = max(0, i - PLOT_BEHAVIOR_FRAMES)
    line_ex.set_data(xdata_ex[i1:i], ydata_ex[i1:i])
    line_re.set_data(xdata_re[i1:i], ydata_re[i1:i])
    line_pa.set_data(xdata_pa[i1:i], ydata_pa[i1:i])
    return line_ex, line_pa, line_re


def getListOfVisitedWells(nearestWells, countFirstVisitOnly):
    if countFirstVisitOnly:
        return list(set(nearestWells))
    else:
        return [k for k, g in groupby(nearestWells)]


random.shuffle(all_sessions)
for sesh in all_sessions:
    print(sesh.name)
    xdata_ex = np.copy(sesh.bt_pos_xs[0:-1])
    ydata_ex = np.copy(sesh.bt_pos_ys[0:-1])
    xdata_re = np.empty_like(xdata_ex)
    ydata_re = np.empty_like(ydata_ex)
    xdata_re[:] = np.nan
    ydata_re[:] = np.nan
    xdata_pa = np.copy(sesh.bt_pos_xs[0:-1])
    ydata_pa = np.copy(sesh.bt_pos_ys[0:-1])

    last_stop = 0
    for bst, ben in zip(sesh.bt_explore_bout_starts, sesh.bt_explore_bout_ends):
        xdata_ex[last_stop:bst] = np.nan
        ydata_ex[last_stop:bst] = np.nan
        xdata_pa[bst:ben] = np.nan
        ydata_pa[bst:ben] = np.nan
        last_stop = ben
    xdata_ex[last_stop:] = np.nan
    ydata_ex[last_stop:] = np.nan

    for wft, wlt in zip(sesh.home_well_find_times, sesh.home_well_leave_times):
        pidx1 = np.searchsorted(sesh.bt_pos_ts[0:-1], wft)
        pidx2 = np.searchsorted(sesh.bt_pos_ts[0:-1], wlt)
        xdata_re[pidx1:pidx2] = sesh.bt_pos_xs[pidx1:pidx2]
        ydata_re[pidx1:pidx2] = sesh.bt_pos_ys[pidx1:pidx2]
        xdata_ex[pidx1:pidx2] = np.nan
        ydata_ex[pidx1:pidx2] = np.nan
        xdata_pa[pidx1:pidx2] = np.nan
        ydata_pa[pidx1:pidx2] = np.nan

    # wells_visited = getListOfVisitedWells(sesh.bt_nearest_wells[bst:ben], True)
    # for w in wells_visited:
    #     wx, wy = get_well_coordinates(w, sesh.well_coords_map)
    #     plt.scatter(wx, wy, c='red')

    ani = anim.FuncAnimation(fig, animate, range(len(sesh.bt_pos_ts) - 1),
                             repeat=False, init_func=init_plot, interval=5)

    plt.show()
    fig, ax = plt.subplots()
    line_ex, = ax.plot([], [], lw=2)
    line_re, = ax.plot([], [], lw=2)
    line_pa, = ax.plot([], [], lw=2)
    # scat, = ax.scatter([], [])
    ax.grid()
