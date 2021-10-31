from BTData import *
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpat
import random


animal_name = 'B12'

if animal_name == "Martin":
    data_filename = "/media/WDC4/martindata/bradtask/martin_bradtask.dat"
    output_dir = '/media/WDC4/martindata/processed_data/behavior_figures/'

elif animal_name == "B12":
    data_filename = "/media/WDC7/B12/processed_data/B12_bradtask.dat"
    output_dir = "/media/WDC7/B12/processed_data/behavior_figures/"

alldata = BTData()
alldata.loadFromFile(data_filename)


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
PLOT_FULL_BEHAVIOR_PATH = True
PLOT_FULL_BEHAVIOR_PATH_COMBO = True
PLOT_RUNNING_VS_STILL = False
PLOT_NEAR_HOME_VS_DISTANT = False
PLOT_HOME_AWAY_LATENCIES = False
PLOT_WELL_OCCUPANCY = False
ENFORCE_DIFFERENT_WELL_COLORS = False
PLOT_EACH_WELL_PATH = False
PLOT_COMBINED_WELL_PATH = False

RUN_THIS_SESSION = "20200608"


all_well_names = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])


def saveOrShow(fname):
    if SAVE_OUTPUT_FIGS:
        plt.savefig(fname, dpi=800)
    if SHOW_OUTPUT_FIGS:
        plt.show()


for (seshidx, sesh) in enumerate(alldata.getSessions()):
    if RUN_JUST_ONE_SESSION and RUN_THIS_SESSION not in sesh.name:
        print("skipping", sesh.name)
        continue

    print(sesh.name)
    if PLOT_FULL_BEHAVIOR_PATH:
        output_fname_bt = os.path.join(output_dir, sesh.name + "_bt_fullpath.png")
        output_fname_probe = os.path.join(output_dir, sesh.name + "_probe_fullpath.png")
        output_fname_probe_1min = os.path.join(output_dir, sesh.name + "_probe_1min.png")
        output_fname_probe_30sec = os.path.join(output_dir, sesh.name + "_probe_30sec.png")

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

        plt.clf()
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        t1 = sesh.probe_pos_ts[0]
        t2 = t1 + 60 * 30000
        i1 = np.searchsorted(sesh.probe_pos_ts, t1)
        i2 = np.searchsorted(sesh.probe_pos_ts, t2)
        plt.plot(sesh.probe_pos_xs[i1:i2], sesh.probe_pos_ys[i1:i2], zorder=0)
        plt.scatter(sesh.probe_pos_xs[i2], sesh.probe_pos_ys[i2], color='red', zorder=2)
        plt.grid('on')
        saveOrShow(output_fname_probe_1min)

        plt.clf()
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        t2 = t1 + 30 * 30000
        i2 = np.searchsorted(sesh.probe_pos_ts, t2)
        plt.plot(sesh.probe_pos_xs[i1:i2], sesh.probe_pos_ys[i1:i2], zorder=0)
        plt.scatter(sesh.probe_pos_xs[i2], sesh.probe_pos_ys[i2], color='red', zorder=2)
        plt.grid('on')
        saveOrShow(output_fname_probe_30sec)

    if PLOT_FULL_BEHAVIOR_PATH_COMBO:
        output_fname_combo = os.path.join(output_dir, sesh.name + "_combo.png")
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(9, 3)
        plt.subplot(131)
        plt.xlim(0, 1200)
        plt.ylim(0, 1000)
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        plt.plot(sesh.bt_pos_xs, sesh.bt_pos_ys, zorder=0)
        plt.grid('on')
        plt.subplot(132)
        plt.xlim(0, 1200)
        plt.ylim(0, 1000)
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        plt.plot(sesh.probe_pos_xs, sesh.probe_pos_ys, zorder=0)
        plt.grid('on')
        plt.subplot(133)
        plt.xlim(0, 1200)
        plt.ylim(0, 1000)
        plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
        t1 = sesh.probe_pos_ts[0]
        t2 = t1 + 60 * 30000
        i1 = np.searchsorted(sesh.probe_pos_ts, t1)
        i2 = np.searchsorted(sesh.probe_pos_ts, t2)
        plt.plot(sesh.probe_pos_xs[i1:i2], sesh.probe_pos_ys[i1:i2], zorder=0)
        plt.scatter(sesh.probe_pos_xs[i2], sesh.probe_pos_ys[i2], color='red', zorder=2)
        plt.grid('on')
        saveOrShow(output_fname_combo)

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

    if PLOT_HOME_AWAY_LATENCIES:
        plt.clf()

        lh = np.array(sesh.home_well_latencies) / 30000.0
        la = np.array(sesh.away_well_latencies) / 30000.0
        # print(lh, la)
        addpad = False
        if len(lh) > len(la):
            addpad = True
            la = np.append(la, [0])

        all_latencies = [val for pair in zip(lh, la) for val in pair]
        all_colors = [(1, 0, 0, 1), (0, 1, 1, 1)] * (len(all_latencies) // 2)
        if addpad:
            all_latencies.pop()
            all_colors.pop()
        xvals = list(range(len(all_latencies)))
        plt.bar(xvals, all_latencies, color=all_colors)
        plt.ylabel('Latency (s)')
        plt.legend(handles=[mpat.Patch(color='red', label='home'),
                            mpat.Patch(color='cyan', label='away')])

        output_fname = os.path.join(output_dir, sesh.name + "_home_away_latencies.png")
        saveOrShow(output_fname)

    if PLOT_WELL_OCCUPANCY:
        cmap = plt.get_cmap('Set3')
        make_colors = True
        while make_colors:
            well_colors = np.zeros((48, 4))
            for i in all_well_names:
                well_colors[i-1, :] = cmap(random.uniform(0, 1))

            make_colors = False

            if ENFORCE_DIFFERENT_WELL_COLORS:
                for i in all_well_names:
                    neighs = [i-8, i-1, i+8, i+1]
                    for n in neighs:
                        if n in all_well_names and np.all(well_colors[i-1, :] == well_colors[n-1, :]):
                            make_colors = True
                            print("gotta remake the colors!")
                            break

                    if make_colors:
                        break

        plt.clf()
        for i, wi in enumerate(all_well_names):
            color = well_colors[wi-1, :]
            for j in range(len(sesh.bt_well_entry_times[i])):
                i1 = sesh.bt_well_entry_idxs[i][j]
                try:
                    i2 = sesh.bt_well_exit_idxs[i][j]
                    plt.plot(
                        sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2], color=color)
                except:
                    print("well {} had {} entries, {} exits".format(
                        wi, len(sesh.bt_well_entry_idxs[i]), len(sesh.bt_well_exit_idxs[i])))

        output_fname = os.path.join(output_dir, sesh.name + "_well_occupancy.png")
        saveOrShow(output_fname)

    if PLOT_EACH_WELL_PATH:
        for pi, hwi in enumerate(sesh.home_well_find_pos_idxs):
            if pi == 0:
                i1 = 0
            else:
                i1 = sesh.away_well_leave_pos_idxs[pi-1]
            i2 = hwi

            plt.clf()
            plt.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2])
            plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
            plt.xlim(0, 1200)
            plt.ylim(0, 1000)

            output_fname = os.path.join(
                output_dir, 'each_well', sesh.name + "_bt_" + str(pi+1) + "H" + ".png")
            saveOrShow(output_fname)

        for pi, hwi in enumerate(sesh.away_well_find_pos_idxs):
            i1 = sesh.home_well_leave_pos_idxs[pi]
            i2 = hwi
            awx, awy = sesh.get_well_coordinates(sesh.away_wells[pi])

            plt.clf()
            plt.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2])
            plt.scatter(awx, awy, color='green', zorder=2)
            plt.xlim(0, 1200)
            plt.ylim(0, 1000)

            output_fname = os.path.join(
                output_dir, 'each_well', sesh.name + "_bt_" + str(pi+1) + "W" + ".png")
            saveOrShow(output_fname)

    if PLOT_COMBINED_WELL_PATH:
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(11, 3)
        for pi, hwi in enumerate(sesh.home_well_find_pos_idxs):
            plt.subplot(2, len(sesh.home_well_find_pos_idxs), pi+1)
            if pi == 0:
                i1 = 0
            else:
                i1 = sesh.away_well_leave_pos_idxs[pi-1]
            i2 = hwi

            plt.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2])
            plt.scatter(sesh.home_x, sesh.home_y, color='green', zorder=2)
            plt.xlim(0, 1200)
            plt.ylim(0, 1000)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                            left=False, right=False, labelleft=False)

            if pi == 0:
                plt.ylabel('Home')

            if len(sesh.away_well_find_pos_idxs) > pi:
                plt.subplot(2, len(sesh.home_well_find_pos_idxs),
                            pi+1 + len(sesh.home_well_find_pos_idxs))
                i1 = sesh.home_well_leave_pos_idxs[pi]
                i2 = sesh.away_well_find_pos_idxs[pi]
                awx, awy = sesh.get_well_coordinates(sesh.away_wells[pi])

                plt.plot(sesh.bt_pos_xs[i1:i2], sesh.bt_pos_ys[i1:i2])
                plt.scatter(awx, awy, color='green', zorder=2)
                plt.xlim(0, 1200)
                plt.ylim(0, 1000)
                plt.tick_params(axis='both', which='both', bottom=False,
                                top=False, labelbottom=False,
                                left=False, right=False, labelleft=False)
                if pi == 0:
                    plt.ylabel('Away')

        output_fname = os.path.join(
            output_dir, 'each_well', sesh.name + "_bt_all.png")
        saveOrShow(output_fname)
