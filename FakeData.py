import pandas as pd
import numpy as np
import random
import seaborn as sns

from PlotUtil import PlotCtx


def generateRatData(rng):
    condition = []
    wellType = []
    dwellTime = []
    curvature = []
    conditionGroup = []

    ratFactor = rng.random() * 10
    homeFactor = rng.random() * 5
    conditionFactor = rng.random()
    baseDwell = 1

    numSessions = rng.integers(10, 30)
    # numSessions = rng.integers(2, 5)
    for si in range(numSessions):
        numAwaysFound = rng.integers(4, 10)
        c = 1 if rng.random() > 0.5 else 0

        # Add home data point
        v = baseDwell + ratFactor + homeFactor * (1 - c * conditionFactor)
        dwellTime.append(v + rng.random() * 0.5)
        curvature.append(v + rng.random() * 0.5)
        wellType.append("home")
        condition.append("SWR" if c == 1 else "Ctrl")
        conditionGroup.append(si // 4)

        # Add away data points
        for ai in range(numAwaysFound):
            v = baseDwell + ratFactor 
            dwellTime.append(v + rng.random() * 0.5)
            curvature.append(v + rng.random() * 0.5)
            wellType.append("away")
            condition.append("SWR" if c == 1 else "Ctrl")
            conditionGroup.append(si // 4)

    sortList = ["{}{}".format(a,0 if b == "home" else 1) for a,b in zip(condition, wellType)]
    dwellTime = [v for _,v in sorted(zip(sortList, dwellTime))]
    curvature = [v for _,v in sorted(zip(sortList, curvature))]
    wellType = [v for _,v in sorted(zip(sortList, wellType))]
    condition = [v for _,v in sorted(zip(sortList, condition))]
    conditionGroup = [v for _,v in sorted(zip(sortList, conditionGroup))]

    measureNames = ["Dwell Time", "Curvature", "Well Type", "Condition", "conditionGroup"]
    measureNamesNoSpaces = [a.replace(" ", "_") for a in measureNames]
    data = pd.Series([dwellTime, curvature, wellType, condition, conditionGroup], index=measureNamesNoSpaces)
    return data

def collapseAndCombineData(data, collapseAvgIndex, collapseCombineIndex, originName, usualSort=True):
    combinedIndices = list(data[list(data.keys())[0]].index) 
    combinedDataIndex = combinedIndices + [originName]
    combinedData = pd.Series(index=combinedDataIndex, dtype=object)
    for mn in combinedDataIndex:
        combinedData[mn] = []
    
    collapsedDataIndex = collapseAvgIndex + collapseCombineIndex + [originName]
    collapsedData = pd.Series(index=collapsedDataIndex, dtype=object)
    for mn in collapsedDataIndex:
        collapsedData[mn] = []
    
    for rat in data:
        print("rat:", rat)
        d = data[rat]
        for mn in combinedIndices:
            combinedData[mn] += d[mn]
            l = len(d[mn])
        combinedData[originName] += [rat] * l

        dd = {}
        for i in (collapseAvgIndex + collapseCombineIndex):
            dd[i] = d[i]
        df = pd.DataFrame(data=dd)
        ma = df.groupby(collapseCombineIndex).mean()
        for i in ma.index:
            for ii, idxname in enumerate(i):
                collapsedData[collapseCombineIndex[ii]].append(idxname)
            for ai in collapseAvgIndex:
                collapsedData[ai].append( ma.loc[i][ai])
            collapsedData[originName].append(rat)



    if usualSort:
        sortList = ["{}{}".format(a,0 if b == "home" else 1) for a,b in zip(collapsedData["Condition"], collapsedData["Well_Type"])]
        for i in collapsedData.index:
            collapsedData[i] = [v for _,v in sorted(zip(sortList, collapsedData[i]))]

    return collapsedData, combinedData

def dataToDf(data):
    dd = {}
    for i in data.index:
        dd[i] = data[i]
    return pd.DataFrame(data=dd)

def dfToData(df):
    data = pd.Series(index=df.columns, dtype=object)
    for c in df.columns:
        data[c] = list(df[c])

    return data

def shuffleMyData(df, shuffleColumns, groupColumns, rng):
    valSet = {}
    for gc in groupColumns:
        valSet[gc] = set(df[gc])

    comboGroups = [[]]
    for gc in groupColumns:
        newComboGroups = []
        for v in valSet[gc]:
            newComboGroups += [cg + [v] for cg in comboGroups]
        comboGroups = newComboGroups
    
    # for cg in comboGroups:
    #     sdf = df.copy()
    #     for gi, g in enumerate(cg):
    #         sdf = sdf[sdf[groupColumns[gi]] == g]
    #     for sc in shuffleColumns:
    #         sdf[sc] = sdf[sc].sample(frac=1, random_state=rng).reset_index(drop=True)

    for cg in comboGroups:
        idx = pd.Series([True] * len(df.index), dtype=bool)
        for gi, g in enumerate(cg):
            idx = (df[groupColumns[gi]] == g) & idx

        for sc in shuffleColumns:
            vals = list(df.loc[idx, sc]).copy()
            random.shuffle(vals)
            df.loc[idx,sc] = vals
        

def shufflePlot(pc, data, shufi=None, figName="statsFig"):
    pal = sns.color_palette(palette=["cyan", "orange"])
    with pc.newFig(figName + str(shufi) if shufi is not None else figName) as ax:
        sns.violinplot(ax=ax, hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, palette=pal, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=ax, hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, color="0.25", zorder=3, dodge=True, size=2)
        ax.set_title("fake data!")

def twoWayPlots(pc, data):
    pal = sns.color_palette(palette=["cyan", "orange"])
    pal2 = sns.color_palette(palette=["violet", "yellow"])
    with pc.newFig("FD_glo_cond") as ax:
        sns.violinplot(ax=ax, x="Condition", y="Dwell_Time", data=data,
            palette=pal, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=ax, x="Condition", y="Dwell_Time", data=data,
            color="0.25", zorder=3, dodge=True, size=2)
        ax.set_title("fake data!")

    with pc.newFig("FD_glo_well_type") as ax:
        sns.violinplot(ax=ax, x="Well_Type", y="Dwell_Time", data=data,
            palette=pal2, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=ax, x="Well_Type", y="Dwell_Time", data=data,
            color="0.25", zorder=3, dodge=True, size=2)
        ax.set_title("fake data!")



if __name__ == "__main__":
    globalOutputDir = "/home/wcroughan/data/figures/examples/"
    pc = PlotCtx(globalOutputDir)
    rats = ["ratA", "ratB", "ratC"]
    seed = 6160
    randomGen = np.random.default_rng(seed)
    data = {}
    for rat in rats:
        d = generateRatData(randomGen)
        data[rat] = d

    collapsedData, combinedData = collapseAndCombineData(data,  ["Curvature", "Dwell_Time"],["Condition", "Well_Type"], "Rat")
    shufflePlot(pc, collapsedData, figName="collapsed")
    shufflePlot(pc, combinedData, figName="combined")

    # df = dataToDf(combinedData)
    # for i in range(5):
    #     shuffleMyData(df, ["Condition"], ["Rat", "Well_Type"], randomGen)
    #     for rat in rats:
    #         d = dfToData(df.loc[df["Rat"] == rat])
    #         pc.setOutputSubDir(rat)
    #         shufflePlot(pc, d, shufi=i)
    #     pc.setOutputSubDir("")

    twoWayPlots(pc, data["ratB"])
