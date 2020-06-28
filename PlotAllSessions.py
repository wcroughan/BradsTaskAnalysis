from BTData import *
import os
import matplotlib.pyplot as plt

data_filename = "/media/WDC4/martindata/bradtask/martin_bradtask.dat"
alldata = BTData()
alldata.loadFromFile(data_filename)

output_dir = '/media/WDC4/martindata/processed_data/behavior_figures/'

X_START = 280
X_FINISH = 1175
Y_START = 50
Y_FINISH = 1275
VEL_THRESH = 10  # cm/s
PIXELS_PER_CM = 5.0
TRODES_SAMPLING_RATE = 30000

RUN_JUST_ONE_SESSION = False
SAVE_OUTPUT_FIGS = True
SHOW_OUTPUT_FIGS = False
PLOT_FULL_BEHAVIOR_PATH = False
PLOT_RUNNING_VS_STILL = False
PLOT_NEAR_HOME_VS_DISTANT = True


def saveOrShow(fname):
    if SAVE_OUTPUT_FIGS:
        plt.savefig(fname, dpi=800)
    if SHOW_OUTPUT_FIGS:
        plt.show()


for (seshidx, sesh) in enumerate(alldata.getSessions()):
    if RUN_JUST_ONE_SESSION and seshidx > 0:
        break

    print(sesh.home_x, sesh.home_y)
    if PLOT_FULL_BEHAVIOR_PATH:
        output_fname_bt = os.path.join(output_dir, sesh.name + "_bt_fullpath.png")
        output_fname_probe = os.path.join(output_dir, sesh.name + "_probe_fullpath.png")

        plt.clf()
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        plt.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, zorder=0)
        plt.grid('on')
        saveOrShow(output_fname_bt)

        plt.clf()
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        plt.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, zorder=0)
        plt.grid('on')
        saveOrShow(output_fname_probe)

    if PLOT_RUNNING_VS_STILL:
        plt.clf()
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        plt.plot(sesh.bt_mv_xs, sesh.bt_mv_ys, zorder=0)
        plt.plot(sesh.bt_still_xs, sesh.bt_still_ys, zorder=0)
        plt.grid('on')
        output_fname = os.path.join(output_dir, sesh.name + "_bt_fullpath_running.png")
        saveOrShow(output_fname)

        plt.clf()
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        plt.plot(sesh.probe_mv_xs, sesh.probe_mv_ys, zorder=0)
        plt.plot(sesh.probe_still_xs, sesh.probe_still_ys, zorder=0)
        plt.grid('on')
        output_fname = os.path.join(output_dir, sesh.name + "_probe_fullpath_running.png")
        saveOrShow(output_fname)

    if PLOT_NEAR_HOME_VS_DISTANT:
        plt.clf()
        # plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        entry_idxs = np.searchsorted(sesh.bt_pos_ts, sesh.bt_home_well_entry_times)
        exit_idxs = np.searchsorted(sesh.bt_pos_ts, sesh.bt_home_well_exit_times)
        last_exit = 0
        assert len(entry_idxs) == len(exit_idxs)
        for i in range(len(entry_idxs)):
            eni = entry_idxs[i]
            exi = exit_idxs[i]
            plt.plot(sesh.bt_pos_xs[last_exit:eni],
                     sesh.bt_pos_ys[last_exit:eni], color='blue', zorder=0)
            plt.plot(sesh.bt_pos_xs[eni:exi], sesh.bt_pos_ys[eni:exi], color='red', zorder=0)
            last_exit = exi
        plt.plot(sesh.bt_pos_xs[last_exit:],
                 sesh.bt_pos_ys[last_exit:], color='blue', zorder=0)
        plt.grid('on')
        output_fname = os.path.join(output_dir, sesh.name + "_bt_fullpath_nearhome.png")
        saveOrShow(output_fname)

        plt.clf()
        # plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        entry_idxs = np.searchsorted(sesh.probe_pos_ts, sesh.probe_home_well_entry_times)
        exit_idxs = np.searchsorted(sesh.probe_pos_ts, sesh.probe_home_well_exit_times)
        last_exit = 0
        assert len(entry_idxs) == len(exit_idxs)
        for i in range(len(entry_idxs)):
            eni = entry_idxs[i]
            exi = exit_idxs[i]
            plt.plot(sesh.probe_pos_xs[last_exit:eni],
                     sesh.probe_pos_ys[last_exit:eni], color='blue', zorder=0)
            plt.plot(sesh.probe_pos_xs[eni:exi], sesh.probe_pos_ys[eni:exi], color='red', zorder=0)
            last_exit = exi
        plt.plot(sesh.probe_pos_xs[last_exit:],
                 sesh.probe_pos_ys[last_exit:], color='blue', zorder=0)
        plt.grid('on')
        output_fname = os.path.join(output_dir, sesh.name + "_probe_fullpath_nearhome.png")
        saveOrShow(output_fname)
