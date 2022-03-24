import numpy as np
import time
import os
import seaborn as sns
import pandas as pd

from PlotUtil import PlotCtx, conditionShuffle

def exampleFigure1(pc):
    with pc.newFig("fig1") as ax:
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.scatter([1, 2], [3, 4])

    print("fig1 saved!")

def exampleFigure2(pc):
    x = np.arange(3) 
    with pc.newFig("fig2") as ax:
        ax.plot(x, x * x)

    x = np.arange(5)
    with pc.continueFig("fig2_appended") as ax:
        ax.plot(x, -x * x)

def exampleFigure3(pc):
    with pc.newFig("fig3", subPlots=(1, 3)) as axs:
        for i in range(len(axs)):
            x = np.arange(i + 5)
            axs[i].plot(x, np.sin(x))

def exampleFigure4(pc):
    with pc.newFig("fig4_big", subPlots=(10, 10)) as axs:
        for i in range(10):
            for j in range(10):
                axs[i,j].scatter(np.random.uniform(size=(5,)), np.random.uniform(size=(5,)))

    with pc.newFig("fig4_small", subPlots=(10, 10), figScale=0.4) as axs:
        for i in range(10):
            for j in range(10):
                axs[i,j].scatter(np.random.uniform(size=(5,)), np.random.uniform(size=(5,)))

def exampleFigure5(pc):
    with pc.newFig("prio_longTime", subPlots=(10,10), priority=10) as axs:
        x = np.random.uniform(size=(10000,1))
        y = np.random.uniform(size=(10000,1))
        for i in range(10):
            for j in range(10):
                axs[i,j].scatter(x,y)

    with pc.newFig("prio_quick", priority=1) as ax:
        x = np.random.uniform(size=(10,1))
        y = np.random.uniform(size=(10,1))
        ax.scatter(x,y)


def exampleFigure6(pc, data, axesNames):
    pal = sns.color_palette(palette=["cyan", "orange"])
    with pc.newFig("statsFig", subPlots=(1,2)) as axs:
        sns.violinplot(ax=axs[0], hue=axesNames[3], y=axesNames[0], x=axesNames[2],
            data=data, palette=pal, linewidth=0.2)
        axs[0].set_title("fake data!")
        sns.violinplot(ax=axs[1], hue=axesNames[3], y=axesNames[1], x=axesNames[2],
            data=data, palette=pal, linewidth=0.2)


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
    return data, measureNamesNoSpaces

def exampleFigure7(pc, data, axesNames):
    pal = sns.color_palette(palette=["cyan", "orange"])
    with pc.newFig("statsFig", subPlots=(1,2), withStats=True) as (axs, yvals, categories, info):
        sns.violinplot(ax=axs[0], hue=axesNames[3], y=axesNames[0], x=axesNames[2],
            data=data, palette=pal, linewidth=0.2)
        axs[0].set_title("fake data!")
        sns.violinplot(ax=axs[1], hue=axesNames[3], y=axesNames[1], x=axesNames[2],
            data=data, palette=pal, linewidth=0.2)

        yvals["Dwell_Time"] = data["Dwell_Time"]
        yvals["Curvature"] = data["Curvature"]
        categories["Well_Type"] = data["Well_Type"]
        categories["Condition"] = data["Condition"]

def exampleFigure8(pc, data, axesNames):
    pal = sns.color_palette(palette=["cyan", "orange"])
    with pc.newFig("statsFig", subPlots=(1,2), withStats=True) as (axs, yvals, categories, info):
        sns.violinplot(ax=axs[0], hue=axesNames[3], y=axesNames[0], x=axesNames[2],
            data=data, palette=pal, linewidth=0.2)
        axs[0].set_title("fake data!")
        sns.violinplot(ax=axs[1], hue=axesNames[3], y=axesNames[1], x=axesNames[2],
            data=data, palette=pal, linewidth=0.2)

        yvals["Dwell_Time"] = data["Dwell_Time"]
        yvals["Curvature"] = data["Curvature"]
        categories["Well_Type"] = data["Well_Type"]
        categories["Condition"] = data["Condition"]

        info["conditionGroup"] = data["conditionGroup"]
        pc.setCustomShuffleFunction("Condition", conditionShuffle)



if __name__ == "__main__":
    globalOutputDir = "/home/wcroughan/data/figures/examples/"
    pc = PlotCtx(globalOutputDir)
    # exampleFigure1(pc)
    # exampleFigure2(pc)
    # exampleFigure3(pc)
    # exampleFigure4(pc)

    # pc.setPriorityLevel(5)
    # exampleFigure5(pc)

    rats = ["ratA", "ratB", "ratC"]
    # for rat in rats:
    #     pc.setOutputDir(os.path.join( globalOutputDir, rat))
    #     exampleFigure1(pc)


    # seed = int(time.perf_counter())
    # print("random seed =", seed)
    seed = 6160
    randomGen = np.random.default_rng(seed)
    data = {}
    for rat in rats:
        d, measureNames = generateRatData(randomGen)
        data[rat] = d
        
    for rat in rats:
        pc.setOutputDir(os.path.join(globalOutputDir, rat))
        exampleFigure6(pc, data[rat], measureNames)

    for rat in rats:
        pc.setOutputDir(os.path.join(globalOutputDir, rat))
        pc.setStatCategory("Rat", rat)
        exampleFigure8(pc, data[rat], measureNames)

    pc.runShuffles(numShuffles=50)

