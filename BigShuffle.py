from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt

from BTData import BTData
from BTSession import BTSession

animals = []
animals += ["B13"]
# animals += ["B14"]
# animals += ["Martin"]

ONLY_AWAYS_OFF_WALL = False
NUM_SHUFFLES = 1000
# NUM_SHUFFLES = 40

rng = default_rng()


def getData(animal_name):
    if animal_name == "B13":
        data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
    elif animal_name == "B14":
        data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
    elif animal_name == "Martin":
        data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
    alldata = BTData()
    print("LOading animal", animal_name)
    alldata.loadFromFile(data_filename)

    return alldata

    # print("Analyzing...")

    # if ONLY_AWAYS_OFF_WALL:
    #     ctrlWithProbe = alldata.getSessions(lambda s: (
    #         not s.isRippleInterruption) and s.probe_performed and any([not onWall(w) for w in s.visited_away_wells]))
    #     swrWithProbe = alldata.getSessions(lambda s: s.isRippleInterruption and s.probe_performed and any([
    #         not onWall(w) for w in s.visited_away_wells]))
    # else:
    #     ctrlWithProbe = alldata.getSessions(lambda s: (
    #         not s.isRippleInterruption) and s.probe_performed)
    #     swrWithProbe = alldata.getSessions(lambda s: s.isRippleInterruption and s.probe_performed)

    # ctrlHomeAvgDwell = [s.avg_dwell_time(True, s.home_well, timeInterval=[
    #                                      0, 90]) for s in ctrlWithProbe]
    # swrHomeAvgDwell = [s.avg_dwell_time(True, s.home_well, timeInterval=[
    #                                     0, 90]) for s in swrWithProbe]
    # ctrlHomeCurve = [s.avg_curvature_at_well(True, s.home_well, timeInterval=[
    #                                          0, 90]) for s in ctrlWithProbe]
    # swrHomeCurve = [s.avg_curvature_at_well(True, s.home_well, timeInterval=[
    #                                         0, 90]) for s in swrWithProbe]
    # ctrlAwayAvgDwell = [np.nanmean([s.avg_dwell_time(True, w, timeInterval=[
    #     0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in ctrlWithProbe]
    # swrAwayAvgDwell = [np.nanmean([s.avg_dwell_time(True, w, timeInterval=[
    #     0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in swrWithProbe]
    # ctrlAwayCurve = [np.nanmean([s.avg_curvature_at_well(True, w, timeInterval=[
    #     0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in ctrlWithProbe]
    # swrAwayCurve = [np.nanmean([s.avg_curvature_at_well(True, w, timeInterval=[
    #     0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in swrWithProbe]

    # return ctrlHomeAvgDwell, swrHomeAvgDwell, ctrlAwayAvgDwell, swrAwayAvgDwell, ctrlHomeCurve, swrHomeCurve, ctrlAwayCurve, swrAwayCurve


for animal in animals:
    dat = getData(animal)
    seshs = dat.getSessions(lambda s: s.probe_performed)

    homes = np.array([s.home_well for s in seshs])
    aways = np.array([s.visited_away_wells for s in seshs], dtype=object)
    isSWR = np.array([s.isRippleInterruption for s in seshs])
    isCtrl = np.array([not s.isRippleInterruption for s in seshs])

    shufValsCtrlHome = np.zeros((NUM_SHUFFLES))
    shufValsSWRHome = np.zeros((NUM_SHUFFLES))
    shufValsCtrlAway = np.zeros((NUM_SHUFFLES))
    shufValsSWRAway = np.zeros((NUM_SHUFFLES))

    for si in range(NUM_SHUFFLES):
        shuf_idx = rng.permutation(len(homes))
        shuf_homes = homes[shuf_idx]
        shuf_aways = aways[shuf_idx]
        shuf_isSWR = isSWR[shuf_idx]
        shuf_isCtrl = isCtrl[shuf_idx]

        homeDwells = np.array([s.avg_dwell_time(True, shuf_homes[ii], timeInterval=[
            0, 90], emptyVal=np.nan) for ii, s in enumerate(seshs)])
        awayDwells = np.array([np.nanmean(np.array([s.avg_dwell_time(True, shuf_aways[ii][ai], timeInterval=[
            0, 90], emptyVal=np.nan) for ai in range(len(shuf_aways[ii]))])) for ii, s in enumerate(seshs)])

        meanCtrlHome = np.nanmean(homeDwells[shuf_isCtrl])
        meanSWRHome = np.nanmean(homeDwells[shuf_isSWR])
        meanCtrlAway = np.nanmean(awayDwells[shuf_isCtrl])
        meanSWRAway = np.nanmean(awayDwells[shuf_isSWR])

        # print(si, shuf_idx, meanCtrlHome, meanSWRAway)

        shufValsCtrlHome[si] = meanCtrlHome
        shufValsSWRHome[si] = meanSWRHome
        shufValsCtrlAway[si] = meanCtrlAway
        shufValsSWRAway[si] = meanSWRAway

    homeDwells = np.array([s.avg_dwell_time(True, homes[ii], timeInterval=[
        0, 90], emptyVal=np.nan) for ii, s in enumerate(seshs)])
    awayDwells = np.array([np.nanmean(np.array([s.avg_dwell_time(True, aways[ii][ai], timeInterval=[
        0, 90], emptyVal=np.nan) for ai in range(len(aways[ii]))])) for ii, s in enumerate(seshs)])

    realMeanCtrlHome = np.nanmean(homeDwells[isCtrl])
    realMeanSWRHome = np.nanmean(homeDwells[isSWR])
    realMeanCtrlAway = np.nanmean(awayDwells[isCtrl])
    realMeanSWRAway = np.nanmean(awayDwells[isSWR])

    heurValsHomeDiff = shufValsCtrlHome - shufValsSWRHome
    heurValsHomeVsAwayDiff = shufValsCtrlHome - shufValsCtrlAway

    N = 2
    if N == 1:
        h, bs = np.histogram(heurValsHomeDiff[~np.isnan(heurValsHomeDiff)])
        print(realMeanCtrlHome - realMeanSWRHome)
    else:
        h, bs = np.histogram(heurValsHomeVsAwayDiff[~np.isnan(heurValsHomeVsAwayDiff)])
        print(realMeanCtrlHome - realMeanCtrlAway)

    # print(len(h), len(bs))
    plt.clf()
    plt.bar(bs[:-1], h)
    plt.show()
