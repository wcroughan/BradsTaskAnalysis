import numpy as np
import time
import os
import seaborn as sns
import pandas as pd

from PlotUtil import PlotCtx, conditionShuffle
from FakeData import collapseAndCombineData, generateRatData

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


def exampleFigure6(pc, data, figName="statsFig"):
    pal = sns.color_palette(palette=["cyan", "orange"])
    with pc.newFig(figName, subPlots=(2,1)) as axs:
        sns.violinplot(ax=axs[0], hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, palette=pal, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=axs[0], hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, color="0.25", zorder=3, dodge=True, size=2)
        axs[0].set_title("fake data!")
        sns.violinplot(ax=axs[1], hue="Condition", y="Curvature", x="Well_Type",
            data=data, palette=pal, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=axs[1], hue="Condition", y="Curvature", x="Well_Type",
            data=data, color="0.25", zorder=3, dodge=True, size=2)


def exampleFigure7(pc, data):
    pal = sns.color_palette(palette=["cyan", "orange"])
    with pc.newFig("statsFig", subPlots=(2,1), withStats=True) as (axs, yvals, categories, info):
        sns.violinplot(ax=axs[0], hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, palette=pal, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=axs[0], hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, color="0.25", zorder=3, dodge=True, size=2)
        axs[0].set_title("fake data!")
        sns.violinplot(ax=axs[1], hue="Condition", y="Curvature", x="Well_Type",
            data=data, palette=pal, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=axs[1], hue="Condition", y="Curvature", x="Well_Type",
            data=data, color="0.25", zorder=3, dodge=True, size=2)

        yvals["Dwell_Time"] = data["Dwell_Time"]
        yvals["Curvature"] = data["Curvature"]
        categories["Well_Type"] = data["Well_Type"]
        categories["Condition"] = data["Condition"]

def exampleFigure8(pc, data):
    pal = sns.color_palette(palette=["cyan", "orange"])
    with pc.newFig("statsFig", subPlots=(2,1), withStats=True) as (axs, yvals, categories, info):
        sns.violinplot(ax=axs[0], hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, palette=pal, linewidth=0.2, zorder=1)
        sns.swarmplot(ax=axs[0], hue="Condition", y="Dwell_Time", x="Well_Type",
            data=data, color="0.25", zorder=3, dodge=True, size=2)
        axs[0].set_title("fake data!")
        sns.violinplot(ax=axs[1], hue="Condition", y="Curvature", x="Well_Type",
            data=data, palette=pal, linewidth=0.2)
        sns.swarmplot(ax=axs[1], hue="Condition", y="Curvature", x="Well_Type",
            data=data, color="0.25", zorder=3, dodge=True, size=2)

        yvals["Dwell_Time"] = data["Dwell_Time"]
        yvals["Curvature"] = data["Curvature"]
        categories["Well_Type"] = data["Well_Type"]
        categories["Condition"] = data["Condition"]

        info["conditionGroup"] = data["conditionGroup"]
        pc.setCustomShuffleFunction("Condition", conditionShuffle)




if __name__ == "__main__":
    globalOutputDir = "/home/wcroughan/data/figures/examples/"
    pc = PlotCtx(globalOutputDir)
    exampleFigure1(pc)
    # exampleFigure2(pc)
    # exampleFigure3(pc)
    # exampleFigure4(pc)

    # pc.setPriorityLevel(5)
    # exampleFigure5(pc)

    # rats = ["ratA", "ratB", "ratC"]
    # for rat in rats:
    #     pc.setOutputSubDir(rat)
    #     exampleFigure1(pc)


    # # seed = int(time.perf_counter())
    # # print("random seed =", seed)
    # seed = 6160
    # randomGen = np.random.default_rng(seed)
    # data = {}
    # for rat in rats:
    #     d = generateRatData(randomGen)
    #     data[rat] = d

    # for rat in rats:
    #     pc.setOutputSubDir(rat)
    #     exampleFigure6(pc, data[rat])
    # pc.setOutputSubDir("")

    # for rat in rats:
    #     pc.setOutputSubDir(rat)
    #     pc.setStatCategory("Rat", rat)
    #     exampleFigure7(pc, data[rat])

    # for rat in rats:
    #     pc.setOutputSubDir(rat)
    #     pc.setStatCategory("Rat", rat)
    #     exampleFigure8(pc, data[rat])

    # pc.runShuffles(numShuffles=50)