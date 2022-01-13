from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np
import pandas as pd
from BTData import BTData
import matplotlib.pyplot as plt
import seaborn as sns

rngesus = np.random.default_rng()

SHOW_REAL_DATA = True
SHOW_EXAMPLE_SAMPLE = False

# MEASURE = "curvature"
MEASURE = "avgdwell"
ONE_N_PER_RAT = True
ONLY_AWAYS_OFF_WALL = False
# RAT_TO_USE = "all"
RAT_TO_USE = "Martin"

# test values:
# CTRL_HOME_MEAN = 5.0
# CTRL_HOME_STD = 1.0
# CTRL_AWAY_MEAN = 2.5
# CTRL_AWAY_STD = 1.0
# SWR_HOME_MEAN = 2.0
# SWR_HOME_STD = 1.0
# SWR_AWAY_MEAN = 3.0
# SWR_AWAY_STD = 1.0


# measured values:
def onWall(well):
    return well < 9 or well > 40 or well % 8 in [2, 7]


def wallCheck(well):
    if ONLY_AWAYS_OFF_WALL:
        return not onWall(well)
    else:
        return True


def getPointsForAnimal(animal_name):
    if animal_name == "B13":
        data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
    elif animal_name == "B14":
        data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
    elif animal_name == "Martin":
        data_filename = '/media/WDC7/Martin/processed_data/martin_bradtask.dat'
    alldata = BTData()
    print("LOading animal", animal_name)
    alldata.loadFromFile(data_filename)

    print("Analyzing...")

    ctrlWithProbe = alldata.getSessions(lambda s: (
        not s.isRippleInterruption) and s.probe_performed and any([not onWall(w) for w in s.visited_away_wells]))
    swrWithProbe = alldata.getSessions(lambda s: s.isRippleInterruption and s.probe_performed and any([
                                       not onWall(w) for w in s.visited_away_wells]))

    ctrlHomeAvgDwell = [s.avg_dwell_time(True, s.home_well, timeInterval=[
                                         0, 90]) for s in ctrlWithProbe]
    swrHomeAvgDwell = [s.avg_dwell_time(True, s.home_well, timeInterval=[
                                        0, 90]) for s in swrWithProbe]
    ctrlHomeCurve = [s.avg_curvature_at_well(True, s.home_well, timeInterval=[
                                             0, 90]) for s in ctrlWithProbe]
    swrHomeCurve = [s.avg_curvature_at_well(True, s.home_well, timeInterval=[
                                            0, 90]) for s in swrWithProbe]
    ctrlAwayAvgDwell = [np.nanmean([s.avg_dwell_time(True, w, timeInterval=[
        0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in ctrlWithProbe]
    swrAwayAvgDwell = [np.nanmean([s.avg_dwell_time(True, w, timeInterval=[
        0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in swrWithProbe]
    ctrlAwayCurve = [np.nanmean([s.avg_curvature_at_well(True, w, timeInterval=[
        0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in ctrlWithProbe]
    swrAwayCurve = [np.nanmean([s.avg_curvature_at_well(True, w, timeInterval=[
        0, 90]) for w in s.visited_away_wells if wallCheck(w)]) for s in swrWithProbe]

    return ctrlHomeAvgDwell, swrHomeAvgDwell, ctrlAwayAvgDwell, swrAwayAvgDwell, ctrlHomeCurve, swrHomeCurve, ctrlAwayCurve, swrAwayCurve


animals = ["B13", "B14", "Martin"]
b13ctrlHomeAvgDwell, b13swrHomeAvgDwell, b13ctrlAwayAvgDwell, b13swrAwayAvgDwell, b13ctrlHomeCurve, b13swrHomeCurve, b13ctrlAwayCurve, b13swrAwayCurve = getPointsForAnimal(
    "B13")
b14ctrlHomeAvgDwell, b14swrHomeAvgDwell, b14ctrlAwayAvgDwell, b14swrAwayAvgDwell, b14ctrlHomeCurve, b14swrHomeCurve, b14ctrlAwayCurve, b14swrAwayCurve = getPointsForAnimal(
    "B14")
marctrlHomeAvgDwell, marswrHomeAvgDwell, marctrlAwayAvgDwell, marswrAwayAvgDwell, marctrlHomeCurve, marswrHomeCurve, marctrlAwayCurve, marswrAwayCurve = getPointsForAnimal(
    "Martin")

if MEASURE == "avgdwell":
    if ONE_N_PER_RAT:
        ctrlHomePoints = np.array([np.nanmean(b13ctrlHomeAvgDwell), np.nanmean(
            b14ctrlHomeAvgDwell), np.nanmean(marctrlHomeAvgDwell)])
        ctrlAwayPoints = np.array([np.nanmean(b13ctrlAwayAvgDwell), np.nanmean(
            b14ctrlAwayAvgDwell), np.nanmean(marctrlAwayAvgDwell)])
        swrHomePoints = np.array([np.nanmean(b13swrHomeAvgDwell), np.nanmean(
            b14swrHomeAvgDwell), np.nanmean(marswrHomeAvgDwell)])
        swrAwayPoints = np.array([np.nanmean(b13swrAwayAvgDwell), np.nanmean(
            b14swrAwayAvgDwell), np.nanmean(marswrAwayAvgDwell)])
    elif RAT_TO_USE == "all":
        ctrlHomePoints = np.hstack((b13ctrlHomeAvgDwell, b14ctrlHomeAvgDwell, marctrlHomeAvgDwell))
        ctrlAwayPoints = np.hstack((b13ctrlAwayAvgDwell, b14ctrlAwayAvgDwell, marctrlAwayAvgDwell))
        swrHomePoints = np.hstack((b13swrHomeAvgDwell, b14swrHomeAvgDwell, marswrHomeAvgDwell))
        swrAwayPoints = np.hstack((b13swrAwayAvgDwell, b14swrAwayAvgDwell, marswrAwayAvgDwell))
    elif RAT_TO_USE == "B13":
        ctrlHomePoints = b13ctrlHomeAvgDwell
        ctrlAwayPoints = b13ctrlAwayAvgDwell
        swrHomePoints = b13swrHomeAvgDwell
        swrAwayPoints = b13swrAwayAvgDwell
    elif RAT_TO_USE == "B14":
        ctrlHomePoints = b14ctrlHomeAvgDwell
        ctrlAwayPoints = b14ctrlAwayAvgDwell
        swrHomePoints = b14swrHomeAvgDwell
        swrAwayPoints = b14swrAwayAvgDwell
    elif RAT_TO_USE == "Martin":
        ctrlHomePoints = marctrlHomeAvgDwell
        ctrlAwayPoints = marctrlAwayAvgDwell
        swrHomePoints = marswrHomeAvgDwell
        swrAwayPoints = marswrAwayAvgDwell
elif MEASURE == "curvature":
    if ONE_N_PER_RAT:
        ctrlHomePoints = np.array([np.nanmean(b13ctrlHomeCurve), np.nanmean(
            b14ctrlHomeCurve), np.nanmean(marctrlHomeCurve)])
        ctrlAwayPoints = np.array([np.nanmean(b13ctrlAwayCurve), np.nanmean(
            b14ctrlAwayCurve), np.nanmean(marctrlAwayCurve)])
        swrHomePoints = np.array([np.nanmean(b13swrHomeCurve), np.nanmean(
            b14swrHomeCurve), np.nanmean(marswrHomeCurve)])
        swrAwayPoints = np.array([np.nanmean(b13swrAwayCurve), np.nanmean(
            b14swrAwayCurve), np.nanmean(marswrAwayCurve)])
    elif RAT_TO_USE == "all":
        ctrlHomePoints = np.hstack((b13ctrlHomeCurve, b14ctrlHomeCurve, marctrlHomeCurve))
        ctrlAwayPoints = np.hstack((b13ctrlAwayCurve, b14ctrlAwayCurve, marctrlAwayCurve))
        swrHomePoints = np.hstack((b13swrHomeCurve, b14swrHomeCurve, marswrHomeCurve))
        swrAwayPoints = np.hstack((b13swrAwayCurve, b14swrAwayCurve, marswrAwayCurve))
    elif RAT_TO_USE == "B13":
        ctrlHomePoints = b13ctrlHomeCurve
        ctrlAwayPoints = b13ctrlAwayCurve
        swrHomePoints = b13swrHomeCurve
        swrAwayPoints = b13swrAwayCurve
    elif RAT_TO_USE == "B14":
        ctrlHomePoints = b14ctrlHomeCurve
        ctrlAwayPoints = b14ctrlAwayCurve
        swrHomePoints = b14swrHomeCurve
        swrAwayPoints = b14swrAwayCurve
    elif RAT_TO_USE == "Martin":
        ctrlHomePoints = marctrlHomeCurve
        ctrlAwayPoints = marctrlAwayCurve
        swrHomePoints = marswrHomeCurve
        swrAwayPoints = marswrAwayCurve

else:
    raise Exception("Unknown measure")


CTRL_HOME_MEAN = np.nanmean(ctrlHomePoints)
CTRL_HOME_STD = np.nanstd(ctrlHomePoints)
CTRL_AWAY_MEAN = np.nanmean(ctrlAwayPoints)
CTRL_AWAY_STD = np.nanstd(ctrlAwayPoints)
SWR_HOME_MEAN = np.nanmean(swrHomePoints)
SWR_HOME_STD = np.nanstd(swrHomePoints)
SWR_AWAY_MEAN = np.nanmean(swrAwayPoints)
SWR_AWAY_STD = np.nanstd(swrAwayPoints)

seriesIndex = ["welltype", "val", "condition"]
if SHOW_REAL_DATA:
    data = np.hstack((ctrlHomePoints, ctrlAwayPoints, swrHomePoints, swrAwayPoints))
    welltypes = ["home"] * len(ctrlHomePoints) + ["away"] * len(ctrlAwayPoints) + \
        ["home"] * len(swrHomePoints) + ["away"] * len(swrAwayPoints)
    condition = ["ctrl"] * (len(ctrlHomePoints) + len(ctrlAwayPoints)) + \
        ["swr"] * (len(swrHomePoints) + len(swrAwayPoints))
    s = pd.Series([welltypes, data, condition], index=seriesIndex)
    plt.clf()
    sns.boxplot(x="welltype", y="val", data=s,
                hue="condition", palette="Set3")
    sns.swarmplot(x="welltype", y="val", data=s,
                    color="0.25", hue="condition", dodge=True)
    anovaModel = ols(
        "val ~ C(welltype) + C(condition) + C(welltype):C(condition)", data=s).fit()
    anova_table = anova_lm(anovaModel, typ=2)
    print(anova_table)

    plt.show()


mu = np.array([CTRL_HOME_MEAN, CTRL_AWAY_MEAN, SWR_HOME_MEAN, SWR_AWAY_MEAN])
std = np.array([CTRL_HOME_STD, CTRL_AWAY_STD, SWR_HOME_STD, SWR_AWAY_STD])

print(mu, std)

# N_REPLICATES = 1000
N_REPLICATES = 1000
P_THRESH = 0.05

if ONE_N_PER_RAT:
    sampleSizes = [3, 5, 7, 10, 15]
elif RAT_TO_USE == "all":
    sampleSizes = np.round(np.exp(np.linspace(np.log(10), np.log(1000), 10))).astype(int)
else:
    sampleSizes = np.round(np.exp(np.linspace(np.log(5), np.log(40), 6))).astype(int)
# sampleSizes = [2, 3]
numSignificant = np.zeros_like(sampleSizes)
for ssi, sampleSize in enumerate(sampleSizes):
    print("testing sample size", str(sampleSize))

    sz = (sampleSize, 4, N_REPLICATES)

    mu = np.reshape(mu, (1, -1, 1))
    std = np.reshape(std, (1, -1, 1))
    data = rngesus.normal(loc=mu, scale=std, size=sz)

    for ri in range(N_REPLICATES):
        welltypes = ["home", "away"] * (2 * sampleSize)
        condition = ["ctrl", "ctrl", "swr", "swr"] * sampleSize
        s = pd.Series([welltypes, np.reshape(data[:, :, ri], (-1)), condition], index=seriesIndex)
        anovaModel = ols(
            "val ~ C(welltype) + C(condition) + C(welltype):C(condition)", data=s).fit()
        anova_table = anova_lm(anovaModel, typ=2)
        pval = anova_table.at["C(welltype):C(condition)", "PR(>F)"]
        if pval < P_THRESH:
            numSignificant[ssi] += 1

        if SHOW_EXAMPLE_SAMPLE and ri == 0:
            plt.clf()
            sns.boxplot(x="welltype", y="val", data=s,
                        hue="condition", palette="Set3")
            sns.swarmplot(x="welltype", y="val", data=s,
                          color="0.25", hue="condition", dodge=True)

            plt.show()


print(sampleSizes)
print(numSignificant)
