# TODO
# ANOVA spherical assumption is not met because spread of other < spread of away < spread of home
#   Need to do correct statistical test
# Reversal measure: how often visited wells ABC vs ABA when A is home or etc
#
# Comparisons to do:
# dwell time during tasks excluding reward
# num home checks excluding reward (maybe over time? Some way to account for probably increasing checks over course of run?)
#
# Model start of probe as [pause] -> run -> search. Isolate search parth, draw all paths cetered around home, color by start->finish of search path
#
# Familiarity: For occupancy maps, use a GLM instead


from BTData import *
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, ttest_ind_from_stats
import statsmodels.api as sm
from MyPlottingFunctions import *

# animal_name = 'B12_goodpos'
# animal_name = 'B12'
# animal_name = 'B12_highthresh'
# animal_name = 'B12_blur'
animal_name = 'B14'

if animal_name == "Martin":
    data_filename = "/media/WDC4/martindata/bradtask/martin_bradtask.dat"
    output_dir = '/media/WDC4/martindata/processed_data/'

elif animal_name == "B12":
    data_filename = "/media/WDC7/B12/processed_data/B12_bradtask.dat"
    output_dir = "/media/WDC7/B12/processed_data/behavior_figures/"

elif animal_name == "B12_no19":
    data_filename = "/media/WDC7/B12/processed_data/B12_no19_bradtask.dat"
    output_dir = "/media/WDC7/B12/processed_data/behavior_figures/no19/"

elif animal_name == "B12_goodpos":
    data_filename = "/media/WDC7/B12/processed_data/B12_goodpos_bradtask.dat"
    output_dir = "/media/WDC7/B12/processed_data/behavior_figures/goodpos/"

elif animal_name == "B12_highthresh":
    data_filename = "/media/WDC7/B12/processed_data/B12_highthresh_bradtask.dat"
    output_dir = "/media/WDC7/B12/processed_data/behavior_figures/highthresh/"

elif animal_name == "B12_converted":
    data_filename = "/media/WDC6/B12/conversion/B12_conversion.dat"
    output_dir = "/media/WDC6/B12/conversion/behavior_figures/"

elif animal_name == "B12_blur":
    data_filename = "/media/WDC6/B12/conversion/B12_conversion_blur.dat"
    output_dir = "/media/WDC6/B12/conversion/behavior_figures/blur/"

elif animal_name == "B13":
    data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
    output_dir = "/media/WDC7/B13/processed_data/behavior_figures/"

elif animal_name == "B14":
    data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
    output_dir = "/media/WDC7/B14/processed_data/behavior_figures/"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

alldata = BTData()
alldata.loadFromFile(data_filename)
all_sessions = alldata.getSessions()


SHOW_OUTPUT_PLOTS = False
SAVE_OUTPUT_PLOTS = True

SKIP_BOX_PLOTS = False
SKIP_SCATTER_PLOTS = False
SKIP_SWARM_PLOTS = False
SKIP_PERSEVBIAS_PLOTS = True
SKIP_PERSEV_MEASURE_PLOTS = False
SKIP_PERSEV_BOX_PLOTS = False
SKIP_BINARY_PERSEVBIAS_PLOTS = True
SKIP_BOUT_PLOTS = False
SKIP_WAIT_PLOTS = False
SKIP_BALL_PLOTS = False
SKIP_CURVATURE_PLOTS = False
SKIP_HISTOGRAMS = False
SKIP_LINE_PLOTS = False
SKIP_AVG_SPEED_COMP_PLOTS = False
SKIP_AVG_SPEED_PLOTS = False
SKIP_PERSEV_QUAD_PLOTS = False
SKIP_BOUT_PROP_PLOTS = False
SKIP_HW_PLOT = False
SKIP_ORDER_PLOTS = False
SKIP_RIPPLE_PLACE_PLOTS = False
SKIP_EVERY_MINUTE_PLOTS = False
SKIP_PREV_SESSION_PLOTS = True
SKIP_PSEUDOPROBE_PLOTS = True
SKIP_BASIC_BEHAVIOR_COMPARISON = True
SKIP_CONSECUTIVE_SESSION_PLOTS = True
SKIP_PSEUDOPROBE_PATH_PLOTS = True
SKIP_OCCUPANCY_PLOTS = True
SKIP_WELL_COLOR_GRIDS = False
SKIP_SNIFF_TIMES = True
SKIP_SNIFF_WITH_POS_PLOTS = True
SKIP_TOP_HALF_PLOTS = True
SKIP_CONVERSION_PLOTS = True

PRINT_TRIAL_INFO = False
SKIP_TO_MY_LOU_DARLIN = False

P = MyPlottingFunctions(all_sessions, output_dir)

TRODES_SAMPLING_RATE = 30000

all_well_names = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])

plt.figure()
tlbls = [P.trial_label(sesh) for sesh in all_sessions]

if PRINT_TRIAL_INFO:
    for i, s in enumerate(all_sessions):
        print(i, s.name, tlbls[i])
    exit()


if SKIP_PREV_SESSION_PLOTS:
    print("Warning: Skipping prev session plots")
else:
    if not SKIP_TO_MY_LOU_DARLIN:
        sessions_with_prev = [sesh for sesh in all_sessions if sesh.prevSessionInfoParsed]
        tlbls_wprev = [P.trial_label(sesh) for sesh in sessions_with_prev]
        P.makeABoxPlot([sesh.avg_dwell_time(True, sesh.prevSessionHome, timeInterval=[0, 30]) for sesh in sessions_with_prev],
                       tlbls_wprev, ['Condition', 'probe_avg_dwell_time_30sec_prevsession_home'])
        P.makeABoxPlot([sesh.avg_dwell_time(True, sesh.prevSessionHome, timeInterval=[0, 300]) for sesh in sessions_with_prev],
                       tlbls_wprev, ['Condition', 'probe_avg_dwell_time_300sec_prevsession_home'])
        P.makeABoxPlot([sesh.avg_dwell_time(True, sesh.prevSessionHome, timeInterval=[0, 90]) for sesh in sessions_with_prev],
                       tlbls_wprev, ['Condition', 'probe_avg_dwell_time_90sec_prevsession_home'])

    P.makeAPlotOverDays(lambda s, w: s.avg_dwell_time(True, w, timeInterval=[
        0, 90], emptyVal=0), tlbls, "avg_dwell_over_days", plotPast=False)
    P.makeAPlotOverDays(lambda s, w: s.avg_dwell_time(True, w, timeInterval=[
        0, 90], emptyVal=0), tlbls, "avg_dwell_over_days_with_prev", plotPast=True)
    P.makeAPlotOverDays(lambda s, w: s.avg_dwell_time(False, w, excludeReward=True,
                                                      emptyVal=0), tlbls, "task_avg_dwell_over_days", plotPast=False)
    P.makeAPlotOverDays(lambda s, w: s.avg_dwell_time(False, w, excludeReward=True,
                                                      emptyVal=0), tlbls, "task_avg_dwell_over_days_with_prev", plotPast=True)
    P.makeAPlotOverDays(lambda s, w: s.avg_dwell_time(False, w, excludeReward=True, timeInterval=[0, 5*60],
                                                      emptyVal=0), tlbls, "task_avg_dwell_over_days_5mins", plotPast=False)
    P.makeAPlotOverDays(lambda s, w: s.avg_dwell_time(False, w, excludeReward=True, timeInterval=[0, 5*60],
                                                      emptyVal=0), tlbls, "task_avg_dwell_over_days_with_prev_5mins", plotPast=True)
    P.makeAPlotOverDays(lambda s, w: s.avg_dwell_time(False, w, excludeReward=True, timeInterval=[0, 90],
                                                      emptyVal=0), tlbls, "task_avg_dwell_over_days_90secs", plotPast=False)


def makeAConsecutiveSessionPlot(datafunc, pair_lbls, fname, pair_delays=None):
    colors = ["orange", "blue", "green", "black"]
    all_lbls = sorted(set(pair_lbls))
    print("Key: {}".format(list(zip(colors, all_lbls))))
    pair_color = []
    for lbl in pair_lbls:
        for i in range(len(all_lbls)):
            if all_lbls[i] == lbl:
                # print(li, i)
                pair_color.append(colors[i])
                break

    plt.clf()
    xvals = np.arange(len(pair_lbls)) + 1
    yvals = [datafunc(s[0], s[1]) for s in zip(all_sessions[:-1], all_sessions[1:])]
    plt.scatter(xvals, yvals, c=pair_color)
    saveOrShow(fname + "_by_session_idx")

    if pair_delays is not None:
        pair_delays = np.array(pair_delays)
        pair_delays -= np.min(pair_delays)
        pair_delays /= np.max(pair_delays)
        s = pd.Series([pair_lbls, yvals, pair_delays], index=["label", "yval", "delay"])
    else:
        s = pd.Series([pair_lbls, yvals], index=["label", "yval"])

    plt.clf()
    print(s)
    sns.boxplot(x="label", y="yval", data=s, palette="Set3")
    if pair_delays is not None:
        sns.swarmplot(x="label", y="yval", data=s, hue="delay", palette="ch:s=.25,rot=-.25")
    else:
        sns.swarmplot(x="label", y="yval", data=s, color="0.25")
    saveOrShow(fname + "_grouped")


def firstDwellTimeAtWell(sesh, well):
    ees = sesh.entry_exit_times(False, well)
    ents = ees[0]
    exts = ees[1]
    return (exts[0] - ents[0]) / TRODES_SAMPLING_RATE


if SKIP_CONSECUTIVE_SESSION_PLOTS:
    print("Warning: skipping consecutive session plots")
else:
    # print("\n".join(list(map(lambda s: "{}".format(s.date), all_sessions))))
    # print("\n".join(list(map(lambda s: str(s[1].getDelayFromSession(
    # s[0])), zip(all_sessions[:-1], all_sessions[1:])))))
    allDelays = list(map(lambda s: s[1].getDelayFromSession(s[0]),
                         zip(all_sessions[:-1], all_sessions[1:])))

    consecutive_lbls = list(map(lambda s: s[0]+"_"+s[1], zip(tlbls[:-1], tlbls[1:])))

    lat_to_ph = list(map(lambda s: s[1].getLatencyToWell(False, s[0].home_well) /
                         TRODES_SAMPLING_RATE, zip(all_sessions[:-1], all_sessions[1:])))
    opt_to_ph = list(map(lambda s: s[1].path_optimality(False, wellName=s[0].home_well),
                         zip(all_sessions[:-1], all_sessions[1:])))
    del_bw_ses = list(map(lambda x: x / 60.0 / 60.0, allDelays))
    dwell_at_ph = list(map(lambda s: firstDwellTimeAtWell(s[1], s[0].home_well), zip(
        all_sessions[:-1], all_sessions[1:])))
    if not SKIP_TO_MY_LOU_DARLIN:
        P.makeAConsecutiveSessionPlot(lambda s1, s2: s2.getLatencyToWell(False, s1.home_well) / TRODES_SAMPLING_RATE,
                                      consecutive_lbls, "latency_to_prev_home_withdelay", pair_delays=allDelays)
        P.makeAConsecutiveSessionPlot(lambda s1, s2: s2.path_optimality(
            False, wellName=s1.home_well), consecutive_lbls, "optimality_to_prev_home_withdelay", pair_delays=allDelays)
        P.makeAConsecutiveSessionPlot(lambda s1, s2: s2.getDelayFromSession(s1),
                                      consecutive_lbls, "Delay between sessions")
        P.makeAConsecutiveSessionPlot(lambda s1, s2: firstDwellTimeAtWell(
            s2, s1.home_well), consecutive_lbls, "first_dwell_time_at_prev_home_withdelay", pair_delays=allDelays)

        P.makeAConsecutiveSessionPlot(lambda s1, s2: s2.getLatencyToWell(False, s1.home_well) / TRODES_SAMPLING_RATE,
                                      consecutive_lbls, "latency_to_prev_home")
        P.makeAConsecutiveSessionPlot(lambda s1, s2: s2.path_optimality(
            False, wellName=s1.home_well), consecutive_lbls, "optimality_to_prev_home")
        P.makeAConsecutiveSessionPlot(lambda s1, s2: firstDwellTimeAtWell(
            s2, s1.home_well), consecutive_lbls, "first_dwell_time_at_prev_home")

        P.makeAScatterPlot(del_bw_ses, lat_to_ph,
                           ["Delay between sessions (hr)", "Latency to prev home well (sec)"], consecutive_lbls, makeLegend=True)
        P.makeAScatterPlot(del_bw_ses, dwell_at_ph,
                           ["Delay between sessions (hr)", "Dwell time at prev home well (sec)"], consecutive_lbls, makeLegend=True)
        P.makeAScatterPlot(del_bw_ses, opt_to_ph,
                           ["Delay between sessions (hr)", "Optimality to prev home well"], consecutive_lbls, makeLegend=True)
        P.makeAScatterPlot(lat_to_ph, dwell_at_ph,
                           ["Latency to prev home well (sec)", "Dwell time at prev home well (sec)"], consecutive_lbls, makeLegend=True)
        P.makeAScatterPlot(opt_to_ph, dwell_at_ph,
                           ["Optimality to prev home well", "Dwell time at prev home well (sec)"], consecutive_lbls, makeLegend=True)

    newhome = np.array(list(map(lambda s: s[1].home_well_find_times[0] - s[1].bt_pos_ts[0],
                                zip(all_sessions[:-1], all_sessions[1:]))))
    oldhome = np.array(list(map(lambda s: s[1].getLatencyToWell(False, s[0].home_well),
                                zip(all_sessions[:-1], all_sessions[1:]))))
    foundOldHomeBeforeNew = newhome > oldhome

    filt_lbls = [consecutive_lbls[i]
                 for i in range(len(consecutive_lbls)) if foundOldHomeBeforeNew[i]]

    P.makeAScatterPlot(np.array(del_bw_ses)[foundOldHomeBeforeNew], np.array(lat_to_ph)[foundOldHomeBeforeNew],   [
        "Just clean psuedoprobe Delay between sessions (hr)", "Latency to prev home well (sec)"], filt_lbls, makeLegend=True)
    P.makeAScatterPlot(np.array(del_bw_ses)[foundOldHomeBeforeNew], np.array(dwell_at_ph)[foundOldHomeBeforeNew], [
        "Just clean psuedoprobe Delay between sessions (hr)", "Dwell time at prev home well (sec)"], filt_lbls, makeLegend=True)
    P.makeAScatterPlot(np.array(del_bw_ses)[foundOldHomeBeforeNew], np.array(opt_to_ph)[foundOldHomeBeforeNew],   [
        "Just clean psuedoprobe Delay between sessions (hr)", "Optimality to prev home well"], filt_lbls, makeLegend=True)
    P.makeAScatterPlot(np.array(lat_to_ph)[foundOldHomeBeforeNew], np.array(dwell_at_ph)[foundOldHomeBeforeNew],  [
        "Just clean psuedoprobe Latency to prev home well (sec)", "Dwell time at prev home well (sec)"], filt_lbls, makeLegend=True)
    P.makeAScatterPlot(np.array(opt_to_ph)[foundOldHomeBeforeNew], np.array(dwell_at_ph)[foundOldHomeBeforeNew],  [
        "Just clean psuedoprobe Optimality to prev home well", "Dwell time at prev home well (sec)"], filt_lbls, makeLegend=True)


def makeAPathPlot(session, inProbe, title, fname, idxInterval=None, postProcess=None, ax=None):
    if inProbe:
        xvals = session.probe_pos_xs
        yvals = session.probe_pos_ys
    else:
        xvals = session.bt_pos_xs
        yvals = session.bt_pos_ys

    if idxInterval is not None:
        xvals = xvals[idxInterval[0]:idxInterval[1]]
        yvals = yvals[idxInterval[0]:idxInterval[1]]

    if ax is None:
        ax = plt

    ax.clf()
    ax.plot(xvals, yvals)
    ax.grid('on')
    ax.xlim(0, 1200)
    ax.ylim(0, 1000)
    ax.title(title)

    if postProcess is not None:
        postProcess(ax)

    saveOrShow(fname)


def pseudoprobePostProcess(s1, s2, ax):
    frame_rate = 15  # ...approximately at least
    tmarkInterval = 2 * frame_rate

    new_home_idx = s2.home_well_find_pos_idxs[0]
    old_home_idx = s2.getLatencyToWell(False, s1.home_well, returnIdxs=True)

    tmarkIdxs = np.concatenate(
        (np.arange(0, old_home_idx, tmarkInterval), np.array([old_home_idx])))
    tmarkTimes = np.array(s2.bt_pos_ts)[tmarkIdxs] - s2.bt_pos_ts[0]
    tmarkXs = np.array(s2.bt_pos_xs)[tmarkIdxs]
    tmarkYs = np.array(s2.bt_pos_ys)[tmarkIdxs]

    for ((x, y), t) in zip(zip(tmarkXs, tmarkYs), tmarkTimes):
        ax.text(x, y, str(float(int(t / TRODES_SAMPLING_RATE * 10)) / 10))

    if old_home_idx < new_home_idx:
        ax.plot(s2.bt_pos_xs[old_home_idx:new_home_idx],
                s2.bt_pos_ys[old_home_idx:new_home_idx], color='grey')

    ax.scatter(s2.bt_pos_xs[old_home_idx], s2.bt_pos_ys[old_home_idx], c='green')
    ax.scatter(s2.bt_pos_xs[new_home_idx], s2.bt_pos_ys[new_home_idx], c='red')

    ax.text(0, 50, "Dwell time: {}".format(round(firstDwellTimeAtWell(s2, s1.home_well), 2)))
    ax.text(600, 50, "Latency: {}".format(
        round(s2.getLatencyToWell(False, s1.home_well) / TRODES_SAMPLING_RATE, 2)))


if SKIP_PSEUDOPROBE_PATH_PLOTS:
    print("Warning: skipping pseudoprobe path plots")
else:
    for si, (s1, s2) in enumerate(zip(all_sessions[:-1], all_sessions[1:])):
        title = "{} ({})".format(s2.name, s2.getDelayFromSession(s1, returnDateTimeDelta=True))
        fname = s1.name + "_pseudoprobe_path"
        i2 = s2.getLatencyToWell(False, s1.home_well, returnIdxs=True)
        P.makeAPathPlot(s2, False,  title, fname, idxInterval=[
            0, i2], postProcess=lambda ax: pseudoprobePostProcess(s1, s2, ax))


def wellGridX(wellName):
    return (wellName - 2) % 8


def wellGridY(wellName):
    return wellName // 8


def makeAWellColorGrid(data, quadrants=False, ax=None, fname=None):
    if SKIP_WELL_COLOR_GRIDS:
        print("Warning, skipping well color grids")
        return

    if ax is None:
        ax = plt

    gridRes = data.shape[0]

    ax.imshow(data, origin='lower', extent=(0, gridRes, 0, gridRes))
    ax.set_xticks(np.arange(gridRes))
    ax.set_yticks(np.arange(gridRes))
    ax.grid(which='major')
    ax.tick_params(axis='both', which='both', bottom=False,
                   top=False, left=False, labelleft=False, labelbottom=False)

    if fname is not None:
        saveOrShow(fname)


def wellGridVals(datafunc, quadrants=False):
    if quadrants:
        gridRes = 2
    else:
        gridRes = 6

    vals = np.zeros((gridRes, gridRes))
    if quadrants:
        vals[0, 0] = datafunc(0)
        vals[0, 1] = datafunc(1)
        vals[1, 0] = datafunc(2)
        vals[1, 1] = datafunc(3)
    else:
        for w in all_well_names:
            vals[wellGridX(w), wellGridY(w)] = datafunc(w)

    return vals


def makeAnOccupancyPlot(session):
    plt.clf()
    _, axs = plt.subplots(2, 4)

    vAll = wellGridVals(lambda w: session.total_dwell_time(False, w))
    P.makeAWellColorGrid(vAll, ax=axs[0, 0])
    axs[0, 0].set_title("Full task", {'fontsize': 10})

    vNoRew = wellGridVals(lambda w: session.total_dwell_time(
        False, w, excludeReward=True))
    P.makeAWellColorGrid(vNoRew, ax=axs[0, 1])
    axs[0, 1].set_title("No reward", {'fontsize': 10})

    vRew = vAll - vNoRew
    P.makeAWellColorGrid(vRew, ax=axs[0, 2])
    axs[0, 2].set_title("Just reward", {'fontsize': 10})

    vProbe = wellGridVals(lambda w: session.total_dwell_time(
        True, w, timeInterval=[0, 90]))
    P.makeAWellColorGrid(vProbe, ax=axs[0, 3])
    axs[0, 3].set_title("Probe (90sec)", {'fontsize': 10})

    P.makeAScatterPlot(vAll.reshape(-1), vProbe.reshape(-1),
                       ["Full task", "Probe"], ax=axs[1, 0], bigDots=False)
    P.makeAScatterPlot(vNoRew.reshape(-1), vProbe.reshape(-1),
                       ["No reward", "Probe"], ax=axs[1, 1], bigDots=False)
    P.makeAScatterPlot(vRew.reshape(-1), vProbe.reshape(-1),
                       ["Just reward", "Probe"], ax=axs[1, 2], bigDots=False)

    axs[1, 1].set_ylabel("")
    axs[1, 2].set_ylabel("")
    axs[1, 1].set_yticks([])
    axs[1, 2].set_yticks([])
    axs[1, 3].axis('off')

    saveOrShow("occupancy_{}".format(session.name))

    return vAll, vNoRew, vRew, vProbe


if SKIP_OCCUPANCY_PLOTS:
    print("Warning: skipping occupancy plots")
else:
    occupancyRs = np.zeros((len(all_sessions), 1))
    # cumOccupancyRs = np.zeros((len(all_sessions), 1))
    rewardRs = np.zeros((len(all_sessions), 1))
    # cumTask = np.zeros((6, 6))
    # cumAll = np.zeros((6, 6))

    nw = len(all_well_names)
    ns = len(all_sessions)
    X = np.ones((ns * nw, 3))
    Y = np.zeros((ns * nw, 1))
    X_SWR = np.zeros((0, 3))
    Y_SWR = np.zeros((0, 1))
    X_CTL = np.zeros((0, 3))
    Y_CTL = np.zeros((0, 1))
    for si, sesh in enumerate(all_sessions):
        vAll, vNoRew, vRew, vProbe = P.makeAnOccupancyPlot(sesh)
        occupancyRs[si], _ = pearsonr(vProbe.reshape(-1), vAll.reshape(-1))
        rewardRs[si], _ = pearsonr(vProbe.reshape(-1), vRew.reshape(-1))

        X[si*nw:si*nw+nw, 1] = vAll.reshape(-1)
        # X[si*nw:si*nw+nw, 2] = vNoRew.reshape(-1)
        X[si*nw:si*nw+nw, 2] = vRew.reshape(-1)
        Y[si*nw:si*nw+nw] = vProbe.reshape((nw, 1))

        if tlbls[si] == "SWR":
            x21 = np.ones((nw, 1))
            x22 = vAll.reshape((nw, 1))
            x23 = vRew.reshape((nw, 1))
            x2 = np.hstack((x21, x22, x23))
            X_SWR = np.vstack((X_SWR, x2))
            Y_SWR = np.vstack((Y_SWR, vProbe.reshape((nw, 1))))
            print("SWR adding")
        if tlbls[si] == "CTRL":
            x21 = np.ones((nw, 1))
            x22 = vAll.reshape((nw, 1))
            x23 = vRew.reshape((nw, 1))
            x2 = np.hstack((x21, x22, x23))
            X_CTL = np.vstack((X_CTL, x2))
            Y_CTL = np.vstack((Y_CTL, vProbe.reshape((nw, 1))))
            print("CTRL adding")

        # cumTask += vAll
        # cumAll += vAll
        # cumOccupancyRs[si], _ = pearsonr(vProbe.reshape(-1), cumAll.reshape(-1))
        # cumAll += wellGridVals(lambda w: sesh.total_dwell_time(True, w))

    P.makeAScatterPlot(occupancyRs, rewardRs, [
        "Occupancy r", "Reward r"], categories=tlbls, output_filename="occupancy_vs_reward", midline=True, makeLegend=True)

    # P.makeAScatterPlot(list(np.arange(0, len(all_sessions))),
    #                  cumOccupancyRs,
    #                  ['Session idx', 'Cumulative Occupancy vs probe r'], tlbls, midline=False)

    glm_model = sm.GLM(Y, X, family=sm.families.Gamma())
    glm_res = glm_model.fit()
    print("=============================== FULL MODEL")
    print(glm_res.summary())

    glm_model = sm.GLM(Y_SWR, X_SWR, family=sm.families.Gamma())
    glm_res_swr = glm_model.fit()
    print("=============================== SWR MODEL")
    print(glm_res_swr.summary())

    glm_model = sm.GLM(Y_CTL, X_CTL, family=sm.families.Gamma())
    glm_res_ctl = glm_model.fit()
    print("=============================== CTRL MODEL")
    print(glm_res_ctl.summary())

    xp = np.array([0, 0.2, 1, 1.2, 2, 2.2])
    yp = np.hstack((glm_res.params[1:3], glm_res_swr.params[1:3], glm_res_ctl.params[1:3]))
    err = np.hstack((glm_res.bse[1:3], glm_res_swr.bse[1:3], glm_res_ctl.bse[1:3]))
    print(xp, yp)
    plt.clf()
    plt.errorbar(xp, yp, yerr=err, fmt='o')
    saveOrShow("Occupancy GLM")
    print("Groups on X axis: general model, swr, ctl. Within group: all occ coeff, just reward coeff. Bars are std err")


def makeAPseudoprobePlot(datafunc, duration, interval, fname, cumulative=True, jitter=True):
    """
    datafunc(s1(prev), s2(current), t(interval)) ==> v(scalar)
    """
    vals = np.zeros((len(all_sessions)-1, int(duration/interval)))
    dual_label = []
    for si, sesh in enumerate(all_sessions):
        if si == 0:
            continue

        vals[si-1, :] = np.array([datafunc(all_sessions[si-1], sesh, [t1, t1+interval])
                                  for t1 in np.arange(0, duration, interval)])
        dual_label.append(tlbls[si-1] + "_" + tlbls[si])

    if cumulative:
        vals = np.cumsum(vals, axis=1)

    if jitter:
        noise = np.random.uniform(low=-0.2, high=0.2, size=vals.shape)
        # print(np.std(noise))
        vals = vals + noise

    colors = ["orange", "blue", "green", "black"]
    all_lbls = sorted(set(dual_label))
    print("Key: {}".format(list(zip(colors, all_lbls))))
    sesh_color = []
    for li, lbl in enumerate(dual_label):
        for i in range(len(all_lbls)):
            if all_lbls[i] == lbl:
                # print(li, i)
                sesh_color.append(colors[i])
                break

    plt.clf()
    for i in range(len(dual_label)):
        plt.plot(np.arange(0, duration, interval), vals[i, :], c=sesh_color[i])

    saveOrShow(fname)


if SKIP_PSEUDOPROBE_PLOTS:
    print("Warning skipping pseudoprobe plots")
else:
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.num_well_entries(
        False, s1.home_well, timeInterval=t), 120, 15, "pseudo_numentries_15sec")
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.num_well_entries(
        False, s1.home_well, timeInterval=t), 15*60, 60, "pseudo_numentries_60sec")
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.total_dwell_time(
        False, s1.home_well, timeInterval=t, excludeReward=True), 120, 15, "pseudo_total_dwell_15sec")
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.total_dwell_time(
        False, s1.home_well, timeInterval=t, excludeReward=True), 15*60, 60, "pseudo_total_dwell_60sec")
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.total_dwell_time(
        False, s1.home_well, timeInterval=t, excludeReward=False), 120, 15, "pseudo_total_dwell_15sec_withreward")
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.total_dwell_time(
        False, s1.home_well, timeInterval=t, excludeReward=False), 15*60, 60, "pseudo_total_dwell_60sec_withreward")
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.num_well_entries(
        False, s1.home_well, timeInterval=t), 15*60, 1, "pseudo_numentries_continuous", jitter=False)
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.total_dwell_time(
        False, s1.home_well, timeInterval=t, excludeReward=True), 15*60, 1, "pseudo_total_dwell_continuous", jitter=False)
    P.makeAPseudoprobePlot(lambda s1, s2, t: s2.total_dwell_time(
        False, s1.home_well, timeInterval=t, excludeReward=False), 15*60, 1, "pseudo_total_dwell_continuous_withreward", jitter=False)

if SKIP_BASIC_BEHAVIOR_COMPARISON:
    print("Warning skipping basic behavior comparison plots")
else:
    P.makeAScatterPlot(list(np.arange(0, len(all_sessions))),
                       [sesh.num_well_entries(False, sesh.home_well, excludeReward=False) - sesh.num_well_entries(
                           False, sesh.home_well, excludeReward=True) for sesh in all_sessions],
                       ['Session idx', 'Num home rewards'], tlbls, midline=False)

# Quick graphs made during lab meeting checking how task behavior and perseveration relate:
# Todo organize these in somewhere
if False:
    P.makeAScatterPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       [sesh.total_dwell_time(False, sesh.home_well)
                        for sesh in all_sessions],
                       ['Probe avg home dwell time 1min', 'Task total home well dwell time'], tlbls, midline=False)
    P.makeAScatterPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       [sesh.num_well_entries(False, sesh.home_well) for sesh in all_sessions],
                       ['Probe avg home dwell time 1min', 'Task num home well entries'], tlbls, midline=False)
    P.makeAScatterPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       [len(sesh.bt_pos_ts) for sesh in all_sessions],
                       ['Probe avg home dwell time 1min', 'Task length'], tlbls, midline=False)

    exit()


if not SKIP_TO_MY_LOU_DARLIN:
    P.makeAScatterPlot([sesh.total_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(False)
                        for sesh in all_sessions],
                       ['Probe total home dwell time 1min', 'Task Curvature at Home'], tlbls, midline=False)
    P.makeAScatterPlot([sesh.avg_curvature_at_home_well(True,  timeInterval=[0, 60])
                        for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(False)
                        for sesh in all_sessions],
                       ['Probe curvature at Home 1min', 'Task Curvature at Home'], tlbls, midline=False)

    P.makeABoxPlot([sesh.probe_home_well_entry_times.size for sesh in all_sessions],
                   tlbls, ['Condition', 'NumHomeEntries'], title="Probe num home entries")
    P.makeABoxPlot([sesh.probe_ctrl_home_well_entry_times.size for sesh in all_sessions],
                   tlbls, ['Condition', 'NumCtrlHomeEntries'], title="Probe num ctrl home entries")

    P.makeAScatterPlot([sesh.bt_home_well_entry_times.size for sesh in all_sessions],
                       [sesh.probe_home_well_entry_times.size for sesh in all_sessions],
                       ['BT num home well entries', 'Probe num home well entries'], tlbls)
    P.makeAScatterPlot([sesh.bt_ctrl_home_well_entry_times.size for sesh in all_sessions],
                       [sesh.probe_ctrl_home_well_entry_times.size for sesh in all_sessions],
                       ['BT num ctrl home well entries', 'Probe num ctrl home well entries'], tlbls)

    P.makeAScatterPlot([sesh.probe_ctrl_home_well_entry_times.size for sesh in all_sessions],
                       [sesh.probe_home_well_entry_times.size for sesh in all_sessions],
                       ['Probe num ctrl home well entries', 'Probe num home well entries'], tlbls, midline=True)

    P.makeAScatterPlot([sesh.total_dwell_time(True, sesh.home_well, [0, 60])
                        for sesh in all_sessions],
                       [sesh.total_dwell_time(False, sesh.home_well)
                        for sesh in all_sessions],
                       ['Probe total home well dwell time', 'Task total home well dwell time'], tlbls, midline=True)
    # P.makeAScatterPlot([sesh.total_dwell_time(True, sesh.home_well, [0, 60])
    # P.makeABoxPlot([sesh.probe_mean_dist_to_home_well for sesh in all_sessions],
    P.makeABoxPlot([sesh.avg_dist_to_home_well(True) for sesh in all_sessions],
                   tlbls, ['Condition', 'MeanDistToHomeWell'], title="Probe Mean Dist to Home")
    # P.makeABoxPlot([sesh.probe_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    P.makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well) for sesh in all_sessions],
                   tlbls, ['Condition', 'MeanDistToCtrlHomeWell'], title="Probe Mean Dist to Ctrl Home")

    # P.makeABoxPlot([sesh.probe_mv_mean_dist_to_home_well for sesh in all_sessions],
    P.makeABoxPlot([sesh.avg_dist_to_home_well(True, moveFlag=BTSession.MOVE_FLAG_MOVING) for sesh in all_sessions],
                   tlbls, ['Condition', 'MoveMeanDistToHomeWell'], title="Probe (move) Mean Dist to Home")
    # P.makeABoxPlot([sesh.probe_mv_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    P.makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well, moveFlag=BTSession.MOVE_FLAG_MOVING) for sesh in all_sessions],
                   tlbls, ['Condition', 'MoveMeanDistToCtrlHomeWell'], title="Probe (move) Mean Dist to Ctrl Home")

    # P.makeABoxPlot([sesh.probe_still_mean_dist_to_home_well for sesh in all_sessions],
    P.makeABoxPlot([sesh.avg_dist_to_home_well(True, moveFlag=BTSession.MOVE_FLAG_STILL) for sesh in all_sessions],
                   tlbls, ['Condition', 'StillMeanDistToHomeWell'], title="Probe (still) Mean Dist to Home")
    # P.makeABoxPlot([sesh.probe_still_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    P.makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well, moveFlag=BTSession.MOVE_FLAG_STILL) for sesh in all_sessions],
                   tlbls, ['Condition', 'StillMeanDistToCtrlHomeWell'], title="Probe (still) Mean Dist to Ctrl Home")
    # P.makeAScatterPlot([sesh.bt_mean_dist_to_home_well for sesh in all_sessions],
    P.makeAScatterPlot([sesh.avg_dist_to_home_well(False) for sesh in all_sessions],
                       #  [sesh.probe_mean_dist_to_home_well for sesh in all_sessions],
                       [sesh.avg_dist_to_home_well(True) for sesh in all_sessions],
                       ['BT Mean Dist to Home', 'Probe Mean Dist to Home'], tlbls)
    # P.makeAScatterPlot([sesh.bt_mean_dist_to_ctrl_home_well for sesh in all_sessions],
    P.makeAScatterPlot([sesh.avg_dist_to_well(False, sesh.ctrl_home_well) for sesh in all_sessions],
                       #  [sesh.probe_mean_dist_to_ctrl_home_well for sesh in all_sessions],
                       [sesh.avg_dist_to_well(True, sesh.ctrl_home_well) for sesh in all_sessions],
                       ['BT Mean Dist to ctrl Home', 'Probe Mean Dist to ctrl Home'], tlbls)

    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 10sec mean vel', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 30sec mean vel', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 60sec mean vel', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 120sec mean vel', 'probe avg home dwell time 90sec'], tlbls)

    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec moving mean vel', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec moving mean vel', 'probe avg home dwell time 90sec'], tlbls)

    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec explore proportion', 'probe avg home dwell time 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec explore proportion', 'probe avg home dwell time 90sec'], tlbls)

    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 10sec mean vel', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 30sec mean vel', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 60sec mean vel', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec mean vel', 'probe avg home curvature 90sec'], tlbls)

    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 10], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 10sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 30sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 60], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 60sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec moving mean vel', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.mean_vel(True, timeInterval=[0, 120], onlyMoving=True) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 120sec moving mean vel', 'probe avg home curvature 90sec'], tlbls)

    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 10sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 10]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 10sec explore proportion', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 30sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 30]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 30sec explore proportion', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 60sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 60]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 60sec explore proportion', 'probe avg home curvature 90sec'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Probe 120sec explore proportion', 'probe avg home curvature 1min'], tlbls)
    P.makeAScatterPlot([sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[0, 120]) for sesh in all_sessions],
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Probe 120sec explore proportion', 'probe avg home curvature 90sec'], tlbls)


# ===================================
# Analysis: active exploration vs chilling, dwell times, etc
# ===================================
if SKIP_SWARM_PLOTS:
    print("Warning, skipping swarm plots!")
else:
    for sesh in all_sessions:
        well_idxs = []
        well_dwell_times = []
        well_category = []
        for i, wi in enumerate(all_well_names):
            for j in range(len(sesh.probe_well_entry_times[i])):
                well_idxs.append(wi)
                dwell_time = (sesh.probe_well_exit_times[i][j] -
                              sesh.probe_well_entry_times[i][j]) / TRODES_SAMPLING_RATE
                # if dwell_time <= 0:
                #     print(i, wi, j, sesh.probe_well_entry_times[i][j], sesh.probe_well_exit_times[i][j],
                #           sesh.probe_well_entry_idxs[i][j], sesh.probe_well_exit_idxs[i][j])
                well_dwell_times.append(dwell_time)
                if wi == sesh.home_well:
                    well_category.append("home")
                elif wi in sesh.visited_away_wells:
                    well_category.append("away")
                else:
                    well_category.append("other")

        axesNames = ['Well number', 'Dwell Time']
        title = 'Dwell time by well, Probe - ' + sesh.name
        fname = "Probe_dwelltime_" + sesh.name
        P.makeASwarmPlot(well_idxs, well_dwell_times, axesNames,
                         well_category, output_filename=fname, title=title)

    all_quadrants = [0, 1, 2, 3]
    for sesh in all_sessions:
        quadrant_idxs = []
        quadrant_dwell_times = []
        quadrant_category = []
        for i, wi in enumerate(all_quadrants):
            for j in range(len(sesh.probe_quadrant_entry_times[i])):
                quadrant_idxs.append(wi)
                dwell_time = (sesh.probe_quadrant_exit_times[i][j] -
                              sesh.probe_quadrant_entry_times[i][j]) / TRODES_SAMPLING_RATE
                # if dwell_time <= 0:
                #     print(i, wi, j, sesh.probe_well_entry_times[i][j], sesh.probe_well_exit_times[i][j],
                #           sesh.probe_well_entry_idxs[i][j], sesh.probe_well_exit_idxs[i][j])
                quadrant_dwell_times.append(dwell_time)
                if wi == sesh.home_quadrant:
                    quadrant_category.append("home")
                else:
                    quadrant_category.append("other")

        axesNames = ['Quadrant number', 'Dwell Time']
        title = 'Dwell time by Quadrant, Probe - ' + sesh.name
        fname = "Probe_quad_dwelltime_" + sesh.name
        P.makeASwarmPlot(quadrant_idxs, quadrant_dwell_times, axesNames,
                         quadrant_category, output_filename=fname, title=title)

    for sesh in all_sessions:
        well_idxs = []
        well_dwell_times = []
        well_category = []
        for i, wi in enumerate(all_well_names):
            for j in range(len(sesh.bt_well_entry_times[i])):
                well_idxs.append(wi)
                dwell_time = (sesh.bt_well_exit_times[i][j] -
                              sesh.bt_well_entry_times[i][j]) / TRODES_SAMPLING_RATE
                # if dwell_time <= 0:
                #     print(i, wi, j, sesh.bt_well_entry_times[i][j], sesh.bt_well_exit_times[i][j],
                #           sesh.bt_well_entry_idxs[i][j], sesh.bt_well_exit_idxs[i][j])
                well_dwell_times.append(dwell_time)
                if wi == sesh.home_well:
                    well_category.append("home")
                elif wi in sesh.visited_away_wells:
                    well_category.append("away")
                else:
                    well_category.append("other")

        axesNames = ['Well number', 'Dwell Time']
        title = 'Dwell time by well, BT - ' + sesh.name
        fname = "BT_dwelltime_" + sesh.name
        P.makeASwarmPlot(well_idxs, well_dwell_times, axesNames,
                         well_category, output_filename=fname, title=title)

# ===================================
# Perseveration bias measures
# ===================================
if SKIP_PERSEVBIAS_PLOTS:
    print("Warning skipping Persev bias plots!")
else:
    if not SKIP_TO_MY_LOU_DARLIN:
        P.makeAPersevMeasurePlot("bt_persev_bias_mean_dist_to_well",
                                 lambda s, w: s.avg_dist_to_well(False, w) - s.avg_dist_to_well(False, s.ctrl_well_for_well(w)))
        P.makeAPersevMeasurePlot("bt_persev_bias_num_entries_to_well",
                                 lambda s, w: s.num_well_entries(False, w) - s.num_well_entries(False, s.ctrl_well_for_well(w)))
        P.makeAPersevMeasurePlot("bt_persev_bias_total_dwell_time",
                                 lambda s, w: s.total_dwell_time(False, w) - s.total_dwell_time(False, s.ctrl_well_for_well(w)))
        P.makeAPersevMeasurePlot("bt_persev_bias_avg_dwell_time",
                                 lambda s, w: s.avg_dwell_time(False, w) - s.avg_dwell_time(False, s.ctrl_well_for_well(w)))
        P.makeAPersevMeasurePlot("bt_persev_bias_total_dwell_time_excluding_reward",
                                 lambda s, w: s.total_dwell_time(False, w, excludeReward=True) - s.total_dwell_time(False, s.ctrl_well_for_well(w), excludeReward=True))
        P.makeAPersevMeasurePlot("bt_persev_bias_avg_dwell_time_excluding_reward",
                                 lambda s, w: s.avg_dwell_time(False, w, excludeReward=True) - s.avg_dwell_time(False, s.ctrl_well_for_well(w), excludeReward=True))
        P.makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well",
                                 lambda s, w: s.avg_dist_to_well(True, w) - s.avg_dist_to_well(True, s.ctrl_well_for_well(w)))
        P.makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well",
                                 lambda s, w: s.num_well_entries(True, w) - s.num_well_entries(True, s.ctrl_well_for_well(w)))
        P.makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time",
                                 lambda s, w: s.total_dwell_time(True, w) - s.total_dwell_time(True, s.ctrl_well_for_well(w)))
        P.makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time",
                                 lambda s, w: s.avg_dwell_time(True, w) - s.avg_dwell_time(True, s.ctrl_well_for_well(w)))

    P.makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well_1min",
                             lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 60]) - s.avg_dist_to_well(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well_1min",
                             lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]) - s.num_well_entries(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time_1min",
                             lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]) - s.total_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time_1min",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]) - s.avg_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_persev_bias_mean_dist_to_well_30sec",
                             lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 30]) - s.avg_dist_to_well(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))
    P.makeAPersevMeasurePlot("probe_persev_bias_num_entries_to_well_30sec",
                             lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30]) - s.num_well_entries(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))
    P.makeAPersevMeasurePlot("probe_persev_bias_total_dwell_time_30sec",
                             lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30]) - s.total_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))
    P.makeAPersevMeasurePlot("probe_persev_bias_avg_dwell_time_30sec",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30]) - s.avg_dwell_time(True, s.ctrl_well_for_well(w), timeInterval=[0, 30]))

# ===================================
# Perseveration measures
# ===================================
if SKIP_PERSEV_MEASURE_PLOTS:
    print("Warning skipping Persev measure plots!")
else:

    if not SKIP_TO_MY_LOU_DARLIN:
        P.makeAPersevMeasurePlot("probe_num_entries_to_well",
                                 lambda s, w: s.num_well_entries(True, w))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well_1min",
                                 lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
        # P.makeAPersevMeasurePlot("probe_well_total_dwell_times_1min", scaleValue=1.0 /
        #    float(TRODES_SAMPLING_RATE), yAxisLabel="Probe 1st Min Total Dwell Time (sec)")
        # P.makeAPersevMeasurePlot("probe_well_avg_dwell_times_1min", scaleValue=1.0 /
        #    float(TRODES_SAMPLING_RATE), yAxisLabel="Probe 1st Min Average Dwell Time (sec)")
        P.makeAPersevMeasurePlot("bt_mean_dist_to_wells",
                                 lambda s, w: s.avg_dist_to_well(False, w))
        P.makeAPersevMeasurePlot("bt_median_dist_to_wells",
                                 lambda s, w: s.avg_dist_to_well(False, w, avgFunc=np.nanmedian))
        P.makeAPersevMeasurePlot("bt_num_entries_to_well",
                                 lambda s, w: s.num_well_entries(False, w))
        P.makeAPersevMeasurePlot("bt_total_dwell_time",
                                 lambda s, w: s.total_dwell_time(False, w))
        P.makeAPersevMeasurePlot("bt_avg_dwell_time",
                                 lambda s, w: s.avg_dwell_time(False, w))
        try:
            P.makeAPersevMeasurePlot("bt_total_dwell_time_excluding_reward",
                                     lambda s, w: s.total_dwell_time(False, w, excludeReward=True))
            P.makeAPersevMeasurePlot("bt_avg_dwell_time_excluding_reward",
                                     lambda s, w: s.avg_dwell_time(False, w, excludeReward=True))
        except:
            print("Missing well find times? Not making plots excluding rewards")
        P.makeAPersevMeasurePlot("probe_mean_dist_to_well",
                                 lambda s, w: s.avg_dist_to_well(True, w))
        P.makeAPersevMeasurePlot("probe_median_dist_to_well",
                                 lambda s, w: s.avg_dist_to_well(True, w, avgFunc=np.nanmedian))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well",
                                 lambda s, w: s.num_well_entries(True, w))
        P.makeAPersevMeasurePlot("probe_total_dwell_time",
                                 lambda s, w: s.total_dwell_time(True, w))
        P.makeAPersevMeasurePlot("probe_avg_dwell_time",
                                 lambda s, w: s.avg_dwell_time(True, w))
        P.makeAPersevMeasurePlot("probe_mean_dist_to_well_1min",
                                 lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 60]))
        P.makeAPersevMeasurePlot("probe_median_dist_to_well_1min",
                                 lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 60], avgFunc=np.nanmedian))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well_1min",
                                 lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
        P.makeAPersevMeasurePlot("probe_total_dwell_time_1min",
                                 lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
        P.makeAPersevMeasurePlot("probe_avg_dwell_time_1min",
                                 lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))
        P.makeAPersevMeasurePlot("probe_mean_dist_to_well_30sec",
                                 lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 30]))
        P.makeAPersevMeasurePlot("probe_median_dist_to_well_30sec",
                                 lambda s, w: s.avg_dist_to_well(True, w, timeInterval=[0, 30], avgFunc=np.nanmedian))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well_30sec",
                                 lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30]))
        P.makeAPersevMeasurePlot("probe_total_dwell_time_30sec",
                                 lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30]))
        P.makeAPersevMeasurePlot("probe_avg_dwell_time_30sec",
                                 lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30]))

        P.makeAPersevMeasurePlot("bt_num_entries_to_well_ninc",
                                 lambda s, w: s.num_well_entries(False, w, includeNeighbors=True))
        P.makeAPersevMeasurePlot("bt_total_dwell_time_ninc",
                                 lambda s, w: s.total_dwell_time(False, w, includeNeighbors=True))
        P.makeAPersevMeasurePlot("bt_avg_dwell_time_ninc",
                                 lambda s, w: s.avg_dwell_time(False, w, includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well_ninc",
                                 lambda s, w: s.num_well_entries(True, w, includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_total_dwell_time_ninc",
                                 lambda s, w: s.total_dwell_time(True, w, includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_avg_dwell_time_ninc",
                                 lambda s, w: s.avg_dwell_time(True, w, includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well_1min_ninc",
                                 lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60], includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well_30sec_ninc",
                                 lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30], includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_total_dwell_time_30sec_ninc",
                                 lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30], includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_avg_dwell_time_30sec_ninc",
                                 lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30], includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_avg_dwell_time_1min_ninc",
                                 lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60], includeNeighbors=True))

        P.makeAPersevMeasurePlot("probe_avg_dwell_time_90sec",
                                 lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 90]))
        P.makeAPersevMeasurePlot("probe_num_entries_to_well_90sec",
                                 lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 90]))

        P.makeAPersevMeasurePlot("time_within_20_cm_of_well",
                                 lambda s, w: s.total_time_near_well(True, w, radius=20, timeInterval=[0, 60]) / TRODES_SAMPLING_RATE)

        P.makeAPersevMeasurePlot("probe_total_dwell_time_1min",
                                 lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
        P.makeAPersevMeasurePlot("probe_avg_dwell_time_60sec",
                                 lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))

        P.makeAPersevMeasurePlot("probe_total_dwell_time_1min_ninc",
                                 lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60], includeNeighbors=True))
        P.makeAPersevMeasurePlot("probe_avg_curve_1min_ninc",
                                 lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 60], includeNeighbors=True))

if SKIP_PERSEV_BOX_PLOTS:
    print("Warning skipping Persev measure plots!")
else:
    P.makeABoxPlot([sesh.num_well_entries(True, sesh.home_well) for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_num_entries_to_well'])

    try:
        P.makeABoxPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60]) for sesh in all_sessions],
                       tlbls, ['Condition', 'probe_avg_dwell_time_60sec'])
    except:
        pass
    P.makeABoxPlot([sesh.total_dwell_time(True, sesh.home_well, timeInterval=[0, 60]) for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_total_dwell_time_60sec'])
    P.makeABoxPlot([sesh.num_well_entries(True, sesh.home_well, timeInterval=[0, 60]) for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_num_entries_to_well_60sec'])

    P.makeABoxPlot([sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90]) for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_avg_dwell_time_90sec'])
    P.makeABoxPlot([sesh.total_dwell_time(True, sesh.home_well, timeInterval=[0, 90]) for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_total_dwell_time_90sec'])
    P.makeABoxPlot([sesh.num_well_entries(True, sesh.home_well, timeInterval=[0, 90]) for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_num_entries_to_well_90sec'])

    P.makeABoxPlot([(sesh.entry_exit_ts(True, sesh.home_well)[0][0] - sesh.probe_pos_ts[0]) / TRODES_SAMPLING_RATE for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_latency_to_home'])
    P.makeABoxPlot([sesh.path_optimality(True, wellName=sesh.home_well) for sesh in all_sessions],
                   tlbls, ['Condition', 'probe_optimality_to_home'])

    P.makeABoxPlot([sesh.avg_dist_to_home_well(True) for sesh in all_sessions],
                   tlbls, ['Condition', 'mean_dist_to_home'])
    P.makeABoxPlot([sesh.avg_dist_to_home_well(True, timeInterval=[0, 60]) for sesh in all_sessions],
                   tlbls, ['Condition', 'mean_dist_to_home_1min'])

    P.makeABoxPlot([sesh.total_time_near_well(True, sesh.home_well, radius=20) / TRODES_SAMPLING_RATE for sesh in all_sessions],
                   tlbls, ['Condition', 'total_time_within_20_cm_home'])
    P.makeABoxPlot([sesh.total_time_near_well(True, sesh.home_well, radius=20, timeInterval=[0, 60]) / TRODES_SAMPLING_RATE for sesh in all_sessions],
                   tlbls, ['Condition', 'total_time_within_20_cm_home_1min'])

if SKIP_BOUT_PLOTS:
    print("Warning skipping exploration bout plots!")
else:
    P.makeAPersevMeasurePlot("probe_bout_count_by_well",
                             lambda s, w: s.num_bouts_where_well_was_visited(True, w))
    P.makeAPersevMeasurePlot("probe_bout_count_by_well_1min",
                             lambda s, w: s.num_bouts_where_well_was_visited(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_bout_count_by_well_30sec",
                             lambda s, w: s.num_bouts_where_well_was_visited(True, w, timeInterval=[0, 30]))
    P.makeAPersevMeasurePlot("bt_bout_count_by_well",
                             lambda s, w: s.num_bouts_where_well_was_visited(False, w))

    P.makeAPersevMeasurePlot("probe_bout_pct_by_well",
                             lambda s, w: s.pct_bouts_where_well_was_visited(True, w))
    P.makeAPersevMeasurePlot("probe_bout_pct_by_well_1min",
                             lambda s, w: s.pct_bouts_where_well_was_visited(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_bout_pct_by_well_30sec",
                             lambda s, w: s.pct_bouts_where_well_was_visited(True, w, timeInterval=[0, 30]))
    P.makeAPersevMeasurePlot("bt_bout_pct_by_well",
                             lambda s, w: s.pct_bouts_where_well_was_visited(False, w))

    num_bouts_1min = [sesh.num_bouts(True, timeInterval=[0, 60]) for sesh in all_sessions]
    num_bouts_10sec = [sesh.num_bouts(True, timeInterval=[0, 10]) for sesh in all_sessions]
    P.makeAHistogram(num_bouts_1min, categories=["all"]
                     * len(num_bouts_1min), title="num bouts 1min")
    P.makeAHistogram(num_bouts_10sec, categories=["all"]
                     * len(num_bouts_10sec), title="num bouts 10sec")
    P.makeAPersevMeasurePlot("pct_first_1_bouts",
                             lambda s, w: s.pct_bouts_where_well_was_visited(True, w, boutsInterval=[0, 1]))
    P.makeAPersevMeasurePlot("pct_first_3_bouts",
                             lambda s, w: s.pct_bouts_where_well_was_visited(True, w, boutsInterval=[0, 3]))

if SKIP_WAIT_PLOTS:
    print("Warning skipping exploration wait plots!")
else:
    # P.makeAPersevMeasurePlot("probe_wait_count_by_well")
    # P.makeAPersevMeasurePlot("probe_wait_count_by_well_1min")
    # P.makeAPersevMeasurePlot("probe_wait_count_by_well_30sec")
    # P.makeAPersevMeasurePlot("bt_wait_count_by_well")

    # P.makeAPersevMeasurePlot("probe_wait_pct_by_well")
    # P.makeAPersevMeasurePlot("probe_wait_pct_by_well_1min")
    # P.makeAPersevMeasurePlot("probe_wait_pct_by_well_30sec")
    # P.makeAPersevMeasurePlot("bt_wait_pct_by_well")
    pass


if SKIP_BALL_PLOTS:
    print("Warning skipping ballisticity by well plots!")
else:
    # P.makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_time")
    # P.makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_visits")
    # P.makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_time_1min")
    # P.makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_visits_1min")
    # P.makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_time_30sec")
    # P.makeAPersevMeasurePlot("probe_well_avg_ballisticity_over_visits_30sec")
    # P.makeAPersevMeasurePlot("bt_well_avg_ballisticity_over_time")
    # P.makeAPersevMeasurePlot("bt_well_avg_ballisticity_over_visits")
    pass

if SKIP_CURVATURE_PLOTS:
    print("Warning skipping curvature by well plots!")
else:
    if not SKIP_TO_MY_LOU_DARLIN:
        P.makeAPersevMeasurePlot("probe_well_avg_curvature_over_time",
                                 lambda s, w: s.avg_curvature_at_well(True, w))
        P.makeAPersevMeasurePlot("probe_well_avg_curvature_over_visits",
                                 lambda s, w: s.avg_curvature_at_well(True, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS))
        P.makeAPersevMeasurePlot("probe_well_avg_curvature_over_time_1min",
                                 lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 60]))
        P.makeAPersevMeasurePlot("probe_well_avg_curvature_over_visits_1min",
                                 lambda s, w: s.avg_curvature_at_well(True, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS, timeInterval=[0, 60]))
        P.makeAPersevMeasurePlot("probe_well_avg_curvature_over_time_30sec",
                                 lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 30]))
        P.makeAPersevMeasurePlot("probe_well_avg_curvature_over_visits_30sec",
                                 lambda s, w: s.avg_curvature_at_well(True, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS, timeInterval=[0, 30]))
        P.makeAPersevMeasurePlot("bt_well_avg_curvature_over_time",
                                 lambda s, w: s.avg_curvature_at_well(False, w))
        P.makeAPersevMeasurePlot("bt_well_avg_curvature_over_visits",
                                 lambda s, w: s.avg_curvature_at_well(False, w, avgTypeFlag=BTSession.AVG_FLAG_OVER_VISITS))

    P.makeAPersevMeasurePlot("probe_well_avg_curvature_over_time_90sec",
                             lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 90]))


if SKIP_PERSEV_QUAD_PLOTS:
    print("Warning skipping quad plots!")
else:
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries",
                                     lambda s, w: s.num_well_entries(True, w))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times",
                                     lambda s, w: s.total_dwell_time(True, w))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times",
                                     lambda s, w: s.avg_dwell_time(True, w))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries_1min",
                                     lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times_1min",
                                     lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times_1min",
                                     lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_num_entries_30sec",
                                     lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 30]))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_total_dwell_times_30sec",
                                     lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 30]))
    P.makeAQuadrantPersevMeasurePlot("probe_quadrant_avg_dwell_times_30sec",
                                     lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 30]))

# ===================================
# What do dwell times look like?
# ===================================
if not SKIP_TO_MY_LOU_DARLIN:
    dt_vals = [
        dt for sesh in all_sessions for dtlist in [sesh.dwell_times(True, w) for w in all_well_names] for dt in dtlist]
    P.makeAHistogram(dt_vals, categories=["all"] * len(dt_vals), title="dwell times")

# ===================================
# Avg speed
# ===================================


def calcAvgSpeed(sesh, start, stop):
    if start <= 3 and stop <= 3:
        # This version uses 0-1 to symbolize times in task, 2-3 for times in probe
        # Just calls recursively using actual times
        if start <= 1:
            assert start >= 0 and stop <= 1 and stop >= 0
            xp = [0, 1]
            fp = [sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]]
        else:
            assert start >= 2 and start <= 3 and stop >= 2 and stop <= 3
            xp = [2, 3]
            fp = [sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]]
        start_t = np.interp(start, xp, fp)
        stop_t = np.interp(stop, xp, fp)
        return calcAvgSpeed(sesh, start_t, stop_t)
    if start < sesh.probe_pos_ts[0]:
        tref = sesh.bt_pos_ts[0: -1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        return np.mean(sesh.bt_vel_cm_s[start_i: stop_i])
    else:
        tref = sesh.probe_pos_ts[0: -1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        return np.mean(sesh.probe_vel_cm_s[start_i: stop_i])


def avg_speed_plot(interval, output_filename=""):
    if SKIP_AVG_SPEED_PLOTS:
        print("Warning: skipping avg speed comp plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]
                         for sesh in all_sessions])
    max_probe_dur = np.max(
        [sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
    istep = interval * TRODES_SAMPLING_RATE
    bt_num_speeds = int(np.floor(max_bt_dur / istep))
    probe_num_speeds = int(np.floor(max_probe_dur / istep))

    all_bt_speeds = np.empty((len(all_sessions), bt_num_speeds))
    all_probe_speeds = np.empty((len(all_sessions), probe_num_speeds))
    all_bt_speeds[:] = np.nan
    all_probe_speeds[:] = np.nan

    for si, sesh in enumerate(all_sessions):
        bt_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                     for i1 in np.arange(sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]-istep, istep)]
        probe_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                        for i1 in np.arange(sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]-istep, istep)]

        all_bt_speeds[si, 0: len(bt_speeds)] = bt_speeds
        all_probe_speeds[si, 0: len(probe_speeds)] = probe_speeds

    bt_mvals = np.nanmean(all_bt_speeds, axis=0)
    probe_mvals = np.nanmean(all_probe_speeds, axis=0)
    bt_sems = np.nanstd(all_bt_speeds, axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_bt_speeds)), axis=0))
    probe_sems = np.nanstd(all_probe_speeds, axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_probe_speeds)), axis=0))

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * bt_num_speeds,
                             bt_num_speeds), bt_mvals, bt_sems)
    plt.title("BT Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "bt_avg_speed_{}".format(interval))
        else:
            output_filename = os.path.join(output_dir, output_filename + "_bt")
        plt.savefig(output_filename, dpi=800)

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * probe_num_speeds,
                             probe_num_speeds), probe_mvals, probe_sems)
    plt.title("Probe Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "probe_avg_speed_{}".format(interval))
        else:
            output_filename = os.path.join(
                output_dir, output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


if not SKIP_TO_MY_LOU_DARLIN and False:
    avg_speed_plot(30)
    avg_speed_plot(60)
    avg_speed_plot(300)


def avg_speed_plot_separate_conditions(interval, output_filename=""):
    if SKIP_AVG_SPEED_COMP_PLOTS:
        print("Warning: skipping avg speed comp plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]
                         for sesh in all_sessions])
    max_probe_dur = np.max(
        [sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
    istep = interval * TRODES_SAMPLING_RATE
    bt_num_speeds = int(np.floor(max_bt_dur / istep))
    probe_num_speeds = int(np.floor(max_probe_dur / istep))

    all_bt_speeds = np.empty((len(all_sessions), bt_num_speeds))
    all_probe_speeds = np.empty((len(all_sessions), probe_num_speeds))
    all_bt_speeds[:] = np.nan
    all_probe_speeds[:] = np.nan
    session_type = np.empty((len(all_sessions), ))

    for si, sesh in enumerate(all_sessions):
        bt_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                     for i1 in np.arange(sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]-istep, istep)]
        probe_speeds = [calcAvgSpeed(sesh, i1, i1+istep)
                        for i1 in np.arange(sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]-istep, istep)]

        all_bt_speeds[si, 0:len(bt_speeds)] = bt_speeds
        all_probe_speeds[si, 0:len(probe_speeds)] = probe_speeds
        if sesh.isRippleInterruption:
            session_type[si] = 1
        else:
            session_type[si] = 0

    # print(all_bt_speeds)
    # print(all_bt_speeds[np.where(session_type == 1), :])
    # print(session_type)
    bt_mvals_swr = np.nanmean(
        all_bt_speeds[np.where(session_type == 1), :][0], axis=0)
    bt_mvals_ctrl = np.nanmean(
        all_bt_speeds[np.where(session_type == 0), :][0], axis=0)
    probe_mvals_swr = np.nanmean(
        all_probe_speeds[np.where(session_type == 1), :][0], axis=0)
    probe_mvals_ctrl = np.nanmean(
        all_probe_speeds[np.where(session_type == 0), :][0], axis=0)
    bt_sems_swr = np.nanstd(all_bt_speeds[np.where(session_type == 1), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_bt_speeds[np.where(session_type == 0), :][0])), axis=0))
    bt_sems_ctrl = np.nanstd(all_bt_speeds[np.where(session_type == 0), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_bt_speeds[np.where(session_type == 0), :][0])), axis=0))
    probe_sems_swr = np.nanstd(all_probe_speeds[np.where(session_type == 1), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_probe_speeds[np.where(session_type == 0), :][0])), axis=0))
    probe_sems_ctrl = np.nanstd(all_probe_speeds[np.where(session_type == 0), :][0], axis=0) / \
        np.sqrt(np.count_nonzero(np.logical_not(
            np.isnan(all_probe_speeds[np.where(session_type == 0), :][0])), axis=0))

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * bt_num_speeds,
                             bt_num_speeds), bt_mvals_swr, bt_sems_swr)
    plt.errorbar(np.linspace(interval, interval * bt_num_speeds,
                             bt_num_speeds), bt_mvals_ctrl, bt_sems_ctrl)
    plt.title("BT Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    plt.legend(["SWR", "CTRL"])
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "bt_avg_speed_{}_comparison".format(interval))
        else:
            output_filename = os.path.join(output_dir, output_filename + "_bt")
        plt.savefig(output_filename, dpi=800)

    plt.clf()
    plt.errorbar(np.linspace(interval, interval * probe_num_speeds,
                             probe_num_speeds), probe_mvals_swr, probe_sems_swr)
    plt.errorbar(np.linspace(interval, interval * probe_num_speeds,
                             probe_num_speeds), probe_mvals_ctrl, probe_sems_ctrl)
    plt.title("Probe Avg speed, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Avg Speed (cm/s)")
    plt.legend(["SWR", "CTRL"])
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(output_filename) > 0:
        if len(output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "probe_avg_speed_{}_comparison".format(interval))
        else:
            output_filename = os.path.join(
                output_dir, output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


if not SKIP_TO_MY_LOU_DARLIN and False:
    avg_speed_plot_separate_conditions(30)
    avg_speed_plot_separate_conditions(60)
    avg_speed_plot_separate_conditions(150)
    avg_speed_plot_separate_conditions(300)

# for sesh in all_sessions:
#     if SKIP_LINE_PLOTS:
#         print("Warning, skipping line plots")
#         break
#     plt.clf()

#     x = sesh.bt_ball_by_dist_to_home_xvals
#     y = sesh.bt_ball_by_dist_to_home
#     tse = sesh.bt_ball_by_dist_to_home_sem
#     # session.probe_ball_by_dist_to_ctrl_home
#     # session.probe_ball_by_dist_to_ctrl_home_sem

#     plt.plot(x, y)
#     # plt.fill_between(x, y-tse, y+tse)
#     plt.show()


def calcBoutProps(sesh, start, stop):
    if start <= 3 and stop <= 3:
        # This version uses 0-1 to symbolize times in task, 2-3 for times in probe
        # Just calls recursively using actual times
        if start <= 1:
            assert start >= 0 and stop <= 1 and stop >= 0
            xp = [0, 1]
            fp = [sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]]
        else:
            assert start >= 2 and start <= 3 and stop >= 2 and stop <= 3
            xp = [2, 3]
            fp = [sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]]
        start_t = np.interp(start, xp, fp)
        stop_t = np.interp(stop, xp, fp)
        return calcBoutProps(sesh, start_t, stop_t)
    if start < sesh.probe_pos_ts[0]:
        tref = sesh.bt_pos_ts[0:-1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        cats = sesh.bt_bout_category[start_i:stop_i]
        c0 = float(np.count_nonzero(cats == 0)) / float(cats.size)
        c1 = float(np.count_nonzero(cats == 1)) / float(cats.size)
        c2 = float(np.count_nonzero(cats == 2)) / float(cats.size)
        return c0, c1, c2
    else:
        tref = sesh.probe_pos_ts[0:-1]
        start_i = np.searchsorted(tref, start)
        stop_i = np.searchsorted(tref, stop)
        cats = sesh.probe_bout_category[start_i:stop_i]
        c0 = float(np.count_nonzero(cats == 0)) / float(cats.size)
        c1 = float(np.count_nonzero(cats == 1)) / float(cats.size)
        c2 = float(np.count_nonzero(cats == 2)) / float(cats.size)
        return c0, c1, c2


def boutProportionPlot(interval, given_output_filename=""):
    if SKIP_BOUT_PROP_PLOTS:
        print("Warning: skipping bout prop plots")
        return
    max_bt_dur = np.max([sesh.bt_pos_ts[-1] - sesh.bt_pos_ts[0]
                         for sesh in all_sessions])
    max_probe_dur = np.max(
        [sesh.probe_pos_ts[-1] - sesh.probe_pos_ts[0] for sesh in all_sessions])
    istep = interval * TRODES_SAMPLING_RATE
    bt_num_cats = int(np.floor(max_bt_dur / istep))
    probe_num_cats = int(np.floor(max_probe_dur / istep))

    all_bt_cats = np.empty((len(all_sessions), bt_num_cats, 3))
    all_probe_cats = np.empty((len(all_sessions), probe_num_cats, 3))
    all_bt_cats[:] = np.nan
    all_probe_cats[:] = np.nan

    for si, sesh in enumerate(all_sessions):
        bt_cats = np.array([calcBoutProps(sesh, i1, i1+istep)
                            for i1 in np.arange(sesh.bt_pos_ts[0], sesh.bt_pos_ts[-1]-istep, istep)])
        probe_cats = np.array([calcBoutProps(sesh, i1, i1+istep)
                               for i1 in np.arange(sesh.probe_pos_ts[0], sesh.probe_pos_ts[-1]-istep, istep)])

        all_bt_cats[si, 0:bt_cats.shape[0], :] = bt_cats
        all_probe_cats[si, 0:probe_cats.shape[0], :] = probe_cats

    bt_mvals = np.nanmean(all_bt_cats, axis=0)
    probe_mvals = np.nanmean(all_probe_cats, axis=0)
    bt_sems = np.divide(np.nanstd(all_bt_cats, axis=0),
                        np.sqrt(np.count_nonzero(np.logical_not(np.isnan(all_bt_cats[:, :, 0])), axis=0))[:, None])
    probe_sems = np.divide(np.nanstd(all_probe_cats, axis=0),
                           np.sqrt(np.count_nonzero(np.logical_not(np.isnan(all_probe_cats[:, :, 0])), axis=0))[:, None])

    plt.clf()
    x = np.linspace(interval, interval * bt_num_cats, bt_num_cats)
    widths = (x[1]-x[0]) * 0.8
    p1 = plt.bar(x, bt_mvals[:, 0], widths, yerr=bt_sems[:, 0])
    p2 = plt.bar(x, bt_mvals[:, 1], widths,
                 yerr=bt_sems[:, 1], bottom=bt_mvals[:, 0])
    p3 = plt.bar(x, bt_mvals[:, 2], widths, yerr=bt_sems[:, 2],
                 bottom=bt_mvals[:, 0]+bt_mvals[:, 1])
    plt.title("BT bout props, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Proportion in each category")
    plt.legend((p1[0], p2[0], p3[0]), ('explore', 'rest', 'reward'))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(given_output_filename) > 0:
        if len(given_output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "bt_bout_proportions_{}".format(interval))
        else:
            output_filename = os.path.join(
                output_dir, given_output_filename + "_bt")
        plt.savefig(output_filename, dpi=800)

    plt.clf()
    x = np.linspace(interval, interval * probe_num_cats, probe_num_cats)
    widths = (x[1]-x[0]) * 0.8
    p1 = plt.bar(x, probe_mvals[:, 0], widths, yerr=probe_sems[:, 0])
    p2 = plt.bar(x, probe_mvals[:, 1], widths,
                 yerr=probe_sems[:, 1], bottom=probe_mvals[:, 0])
    p3 = plt.bar(x, probe_mvals[:, 2], widths, yerr=probe_sems[:, 2],
                 bottom=probe_mvals[:, 0]+probe_mvals[:, 1])
    plt.title("probe bout props, interval={}".format(interval))
    plt.xlabel("Time (sec)")
    plt.ylabel("Proportion in each category")
    plt.legend((p1[0], p2[0], p3[0]), ('explore', 'rest', 'reward'))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS or len(given_output_filename) > 0:
        if len(given_output_filename) == 0:
            output_filename = os.path.join(
                output_dir, "probe_bout_proportions_{}".format(interval))
        else:
            output_filename = os.path.join(
                output_dir, given_output_filename + "_probe")
        plt.savefig(output_filename, dpi=800)


boutProportionPlot(60)
boutProportionPlot(30)
boutProportionPlot(15)

if SKIP_HW_PLOT:
    print("Warning, skipping home well plot")
else:
    session_type = [0 if P.trial_label(
        sesh) == "SWR" else 1 for sesh in all_sessions]
    xvals = [sesh.home_x for sesh in all_sessions]
    yvals = [sesh.home_y for sesh in all_sessions]
    plt.clf()
    plt.xlim(0, 1200)
    plt.ylim(0, 1000)
    plt.scatter(xvals, yvals, c=session_type, zorder=2)
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "all_home_wells"), dpi=800)

if SKIP_ORDER_PLOTS:
    print("Warning, skipping trial order plots")
else:
    if all_sessions[0].hasSniffTimes:
        P.makeAScatterPlot(list(range(len(all_sessions))),
                           [sesh.num_sniffs(False, sesh.home_well, excludeReward=True)
                            for sesh in all_sessions],
                           ['Session idx', 'Num Home Sniffs'], tlbls)
    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.num_home_found for sesh in all_sessions],
                       ['Session idx', 'Num Homes Found'], tlbls)

    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Session idx', 'Home avg dwell time probe 1min'], tlbls)
    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 60])
                        for sesh in all_sessions],
                       ['Session idx', 'home avg curvature over time 1min'], tlbls)

    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.prop_time_in_bout_state(True, BTSession.BOUT_STATE_EXPLORE, timeInterval=[
                           0, 60]) for sesh in all_sessions],
                       ['Session idx', 'Probe 60sec explore proportion'], tlbls)
    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.mean_vel(True, timeInterval=[0, 30])
                        for sesh in all_sessions],
                       ['Session idx', 'Probe 30sec mean vel'], tlbls)
    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.mean_vel(True, timeInterval=[0, 30], onlyMoving=True)
                        for sesh in all_sessions],
                       ['Session idx', 'Probe 30sec moving mean vel'], tlbls)

    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.avg_dwell_time(True, sesh.home_well, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Session idx', 'Home avg dwell time probe 90sec'], tlbls)
    P.makeAScatterPlot(list(range(len(all_sessions))),
                       [sesh.avg_curvature_at_home_well(True, timeInterval=[0, 90])
                        for sesh in all_sessions],
                       ['Session idx', 'home avg curvature over time 90sec'], tlbls)


if SKIP_RIPPLE_PLACE_PLOTS:
    print("Warning, skipping ripple place plots")
else:
    ripoutdir = os.path.join(output_dir, 'ripple_place')
    all_rip_speeds = np.array([])
    all_speeds = np.array([])
    for sesh in all_sessions:
        # print(sesh.home_well_find_pos_idxs)
        well_find_pos_idxs = np.concatenate(
            (sesh.home_well_find_pos_idxs, sesh.away_well_find_pos_idxs))
        well_find_pos_idxs.sort()
        # print(well_find_pos_idxs)

        posx = np.array(sesh.bt_pos_xs)
        posy = np.array(sesh.bt_pos_ys)
        post = np.array(sesh.bt_pos_ts)
        rip_ts = np.array(sesh.interruption_timestamps)

        if not SKIP_TO_MY_LOU_DARLIN:
            last_find_idx = 0
            for wi, next_find_idx in enumerate(well_find_pos_idxs):

                t1 = post[last_find_idx]
                t2 = post[next_find_idx]
                rts = rip_ts[np.logical_and(rip_ts >= t1, rip_ts < t2)]
                rip_idxs = np.searchsorted(post[last_find_idx:next_find_idx], rts)

                plt.clf()
                plt.plot(posx[last_find_idx:next_find_idx], posy[last_find_idx:next_find_idx])
                plt.scatter(posx[last_find_idx + rip_idxs],
                            posy[last_find_idx + rip_idxs], c=[[1, 0, 0, 1]], zorder=32)
                # print(wi, len(rts), len(rip_ts))
                # print(t1, t2, rip_ts[0], rip_ts[-1])
                plt.scatter(posx[last_find_idx], posy[last_find_idx], c=[[0, 1, 0, 1]],
                            s=plt.rcParams['lines.markersize'] ** 2 * 2, zorder=31)
                plt.gca().set_xlim(200, 1100)
                plt.gca().set_ylim(0, 1000)

                if SHOW_OUTPUT_PLOTS:
                    plt.show()
                if SAVE_OUTPUT_PLOTS:
                    output_filename = os.path.join(ripoutdir, sesh.name + "_" + str(wi))
                    plt.savefig(output_filename, dpi=800)

                last_find_idx = next_find_idx

        # speed
        rip_idxs = np.searchsorted(post, rip_ts[rip_ts < post[-1]])
        if rip_idxs[-1] == len(sesh.bt_vel_cm_s):
            rip_idxs = np.delete(rip_idxs, -1)
        rip_speeds = sesh.bt_vel_cm_s[rip_idxs]
        all_rip_speeds = np.concatenate((all_rip_speeds, rip_speeds))
        all_speeds = np.concatenate((all_speeds, sesh.bt_vel_cm_s))

    plt.clf()
    # sns.distplot(speeds)
    # bin_max = max(np.max(all_speeds), np.max(all_rip_speeds))
    bin_max = 100
    x, bins, p = plt.hist((all_speeds, all_rip_speeds),
                          bins=np.arange(0, bin_max, 5), label=['all', 'stim'])
    plt.legend()
    # print(x, bins, p)
    # x, bins, p = plt.hist(all_speeds, bins=np.arange(0, np.max(all_speeds), 5))
    for item in p[0]:
        item.set_height(item.get_height()/np.sum(x[0]))
    for item in p[1]:
        item.set_height(item.get_height()/np.sum(x[1]))
    # x, bins, p = plt.hist(all_rip_speeds, bins=np.arange(0, np.max(all_rip_speeds), 5))
    # for item in p:
        # item.set_height(item.get_height()/sum(x))
    prop_txt = str(np.count_nonzero(all_rip_speeds < 5) / np.size(all_rip_speeds))
    plt.text(1, 1, prop_txt, horizontalalignment='right',
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.gca().set_ylim((0, 0.5))
    if SHOW_OUTPUT_PLOTS:
        plt.show()
    if SAVE_OUTPUT_PLOTS:
        output_filename = os.path.join(output_dir, "all_speeds_at_interruption")
        plt.savefig(output_filename, dpi=800)

    # distance to home well
    # in exploration bout?

if SKIP_EVERY_MINUTE_PLOTS:
    print("Warning, skipping plots that look at each minute of probe")
else:
    P.makeAPersevMeasurePlot("probe_curvature_full",
                             lambda s, w: s.avg_curvature_at_well(True, w))
    P.makeAPersevMeasurePlot("probe_curvature_0_60s",
                             lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_curvature_60_120s",
                             lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[60, 120]))
    P.makeAPersevMeasurePlot("probe_curvature_120_180s",
                             lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[120, 180]))
    P.makeAPersevMeasurePlot("probe_curvature_120_180s",
                             lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[120, 180]))
    P.makeAPersevMeasurePlot("probe_curvature_180_240",
                             lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[180, 240]))
    P.makeAPersevMeasurePlot("probe_curvature_240_300",
                             lambda s, w: s.avg_curvature_at_well(True, w, timeInterval=[240, 300]))

    P.makeAPersevMeasurePlot("probe_avg_dwell_full",
                             lambda s, w: s.avg_dwell_time(True, w))
    P.makeAPersevMeasurePlot("probe_avg_dwell_0_60s",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_avg_dwell_60_120s",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[60, 120]))
    P.makeAPersevMeasurePlot("probe_avg_dwell_120_180s",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[120, 180]))
    P.makeAPersevMeasurePlot("probe_avg_dwell_180_240s",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[180, 240]))
    P.makeAPersevMeasurePlot("probe_avg_dwell_240_300s",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[240, 300]))

    P.makeAPersevMeasurePlot("probe_num_entries_full",
                             lambda s, w: s.num_well_entries(True, w))
    P.makeAPersevMeasurePlot("probe_num_entries_0_60s",
                             lambda s, w: s.num_well_entries(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_num_entries_60_120s",
                             lambda s, w: s.num_well_entries(True, w, timeInterval=[60, 120]))
    P.makeAPersevMeasurePlot("probe_num_entries_120_180s",
                             lambda s, w: s.num_well_entries(True, w, timeInterval=[120, 180]))
    P.makeAPersevMeasurePlot("probe_num_entries_180_240s",
                             lambda s, w: s.num_well_entries(True, w, timeInterval=[180, 240]))
    P.makeAPersevMeasurePlot("probe_num_entries_240_300s",
                             lambda s, w: s.num_well_entries(True, w, timeInterval=[240, 300]))

if SKIP_SNIFF_TIMES:
    print("Warning, skipping sniff time plots")
else:
    P.makeAPersevMeasurePlot("probe_num_sniffs_full", lambda s, w: s.num_sniffs(True, w))
    P.makeAPersevMeasurePlot("probe_avg_sniff_time_full", lambda s, w: s.avg_sniff_time(True, w))
    P.makeAPersevMeasurePlot("probe_median_sniff_time_full", lambda s,
                             w: s.avg_sniff_time(True, w, avgFunc=np.nanmedian))
    P.makeAPersevMeasurePlot("probe_total_sniff_time_full", lambda s,
                             w: s.total_sniff_time(True, w))

    P.makeAPersevMeasurePlot("probe_avg_sniff_time_full_capped", lambda s,
                             w: min(s.avg_sniff_time(True, w), 10))
    P.makeAPersevMeasurePlot("probe_median_sniff_time_full_capped", lambda s,
                             w: min(s.avg_sniff_time(True, w, avgFunc=np.nanmedian), 10))
    P.makeAPersevMeasurePlot("probe_total_sniff_time_full_capped", lambda s,
                             w: min(s.total_sniff_time(True, w), 10))

    P.makeAPersevMeasurePlot("probe_num_sniffs_30sec", lambda s,
                             w: s.num_sniffs(True, w, timeInterval=[0, 30]))
    P.makeAPersevMeasurePlot("probe_avg_sniff_time_30sec", lambda s,
                             w: s.avg_sniff_time(True, w, timeInterval=[0, 30]))
    P.makeAPersevMeasurePlot("probe_total_sniff_time_30sec", lambda s,
                             w: s.total_sniff_time(True, w, timeInterval=[0, 30]))

    P.makeAPersevMeasurePlot("probe_num_sniffs_90sec", lambda s,
                             w: s.num_sniffs(True, w, timeInterval=[0, 90]))
    P.makeAPersevMeasurePlot("probe_avg_sniff_time_90sec", lambda s,
                             w: s.avg_sniff_time(True, w, timeInterval=[0, 90]))
    P.makeAPersevMeasurePlot("probe_total_sniff_time_90sec", lambda s,
                             w: s.total_sniff_time(True, w, timeInterval=[0, 90]))

    # P.makeAPersevMeasurePlot("bt_num_sniffs_full", lambda s,
    #    w: s.num_sniffs(False, w, excludeReward=True))

    P.makeABoxPlot([sum([len(v) for v in sesh.probe_well_sniff_times_entry]) for sesh in all_sessions],
                   tlbls, ['Condition', 'num_probe_sniffs'], title="Probe_num_sniffs")
    # P.makeABoxPlot([sum([len(v) for v in sesh.bt_well_sniff_times_entry]) for sesh in all_sessions],
    #  tlbls, ['Condition', 'num_task_sniffs'], title="Task_num_sniffs")

    P.makeABoxPlot([sum([sesh.total_sniff_time(True, w) for w in all_well_names]) for sesh in all_sessions],
                   tlbls, ['Condition', 'total_sniff_time'], title="Probe_total_sniff_time_allwells")
    P.makeABoxPlot([(sesh.total_sniff_time(True, sesh.home_well) / sum([sesh.total_sniff_time(True, w) for w in all_well_names])) for sesh in all_sessions],
                   tlbls, ['Condition', 'frac_sniff_at_home'], title="Probe_frac_sniff_at_home")


if SKIP_SNIFF_WITH_POS_PLOTS:
    print("warning: skipping sniff plus pos plots")
else:
    P.makeAPersevMeasurePlot("probe_sniffs_per_pass", lambda s,
                             w: s.num_sniffs(True, w) / (1.0 + s.num_well_entries(True, w)))
    P.makeAPersevMeasurePlot("probe_sniffs_per_pass_ninc", lambda s,
                             w: s.num_sniffs(True, w) / (1.0 + s.num_well_entries(True, w, includeNeighbors=True)))

if SKIP_TOP_HALF_PLOTS:
    print("Skipping plots using just top half of pos")
else:
    sessions_in_top_half = [sesh for sesh in all_sessions if sesh.home_well > 32]
    tlbls_th = [P.trial_label(sesh) for sesh in sessions_in_top_half]
    print("\n".join([sesh.date_str for sesh in sessions_in_top_half]))
    P.makeABoxPlot([sesh.avg_dwell_time(True, sesh.home_well) for sesh in sessions_in_top_half],
                   tlbls_th, ['Condition', 'probe_avg_home_dwell_time_top_half'])
    P.makeABoxPlot([sesh.total_dwell_time(True, sesh.home_well) for sesh in sessions_in_top_half],
                   tlbls_th, ['Condition', 'probe_total_home_dwell_time_top_half'])
    P.makeABoxPlot([sesh.num_well_entries(True, sesh.home_well) for sesh in sessions_in_top_half],
                   tlbls_th, ['Condition', 'probe_num_home_well_entries_top_half'])

if SKIP_CONVERSION_PLOTS:
    print("Skipping conversion plots")
else:
    sessions_with_converted_wells = [sesh for sesh in all_sessions if sesh.date_str != "20211004"]
    tlbls_converted = [P.trial_label(sesh) for sesh in sessions_with_converted_wells]
    y = [sesh.total_converted_dwell_time(True, sesh.home_well)
         for sesh in sessions_with_converted_wells]
    outl = np.argmax(y)
    ynoout = y[0:outl] + y[outl+1:]
    lblnoout = tlbls_converted[0:outl] + tlbls_converted[outl+1:]
    P.makeABoxPlot(ynoout, lblnoout, ['Condition', 'probe_converted_total_dwell_nooutlier'])
    P.makeABoxPlot(y, tlbls_converted, ['Condition', 'probe_converted_total_dwell'])
    print(sessions_with_converted_wells[outl].name)

    P.makeAPersevMeasurePlot("probe_conv_total", lambda s,
                             w: s.total_converted_dwell_time(True, w))
    P.makeAPersevMeasurePlot("probe_conv_total_capped", lambda s,
                             w: min(20, s.total_converted_dwell_time(True, w)))

    P.makeAPersevMeasurePlot("probe_conv_total_bias", lambda s,
                             w: s.total_converted_dwell_time(True, w) - s.total_converted_dwell_time(True, s.ctrl_well_for_well(w)))
    P.makeAPersevMeasurePlot("probe_conv_total_capped_bias", lambda s,
                             w: min(20, s.total_converted_dwell_time(True, w)) - min(20, s.total_converted_dwell_time(True, s.ctrl_well_for_well(w))))
