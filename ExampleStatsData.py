import pandas as pd
import numpy as np
import os
from datetime import datetime

from PlotUtil import PlotManager, PlotContext, violinPlot
from UtilFunctions import findDataDir


def main():
    # Set up plotting manager
    dataDir = findDataDir()
    outputDir = "ExampleStatsData"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = 1234
    print("random seed =", rseed)
    pm = PlotManager(outputDir=globalOutputDir, randomSeed=rseed)

    # seed np random for reproducibility
    np.random.seed(rseed + 1)
    n = 100

    # # Main effect of cat1 on val1
    cat1 = np.random.choice(['cat1_A', 'cat1_B'], 100)
    val1 = np.random.normal(0, 1, 100)
    val1[cat1 == 'cat1_A'] += 2
    with pm.newFig("fig1") as pc:
        violinPlot(pc.ax, val1, cat1)
        pc.yvals["val1"] = val1
        pc.categories["cat1"] = cat1

    # Interaction of cat1 and cat2 on val2
    cat2 = np.random.choice(['cat2_A', 'cat2_B'], 100)
    val2 = np.random.normal(0, 1, 100)
    val2[(cat2 == 'cat2_A') & (cat1 == 'cat1_A')] += 2
    val2[(cat2 == 'cat2_B') & (cat1 == 'cat1_B')] += 2
    with pm.newFig("fig2") as pc:
        violinPlot(pc.ax, val2, cat1, categories2=cat2)
        pc.yvals["val2"] = val2
        pc.categories["cat1"] = cat1
        pc.categories["cat2"] = cat2

    # Across cat1, main effect of cat2 on val3
    val3 = np.random.normal(0, 1, 100)
    val3[cat2 == 'cat2_A'] += 2
    val3[cat1 == 'cat1_A'] += 50
    with pm.newFig("fig3") as pc:
        violinPlot(pc.ax, val3, cat1, categories2=cat2)
        pc.yvals["val3"] = val3
        pc.categories["cat1"] = cat1
        pc.categories["cat2"] = cat2

    # Within cat1 == cat1_A, main effect of cat2 on val4
    val4 = np.random.normal(0, 1, 100)
    val4[(cat2 == 'cat2_A') & (cat1 == 'cat1_A')] += 2
    with pm.newFig("fig4") as pc:
        violinPlot(pc.ax, val4, cat1, categories2=cat2)
        pc.yvals["val4"] = val4
        pc.categories["cat1"] = cat1
        pc.categories["cat2"] = cat2

    # Main effect of cat3 == cat3_A on val5
    cat3 = np.random.choice(['cat3_A', 'cat3_B', 'cat3_C', 'cat3_D'], 100)
    val5 = np.random.normal(0, 1, 100)
    val5[cat3 == 'cat3_A'] += 2
    with pm.newFig("fig5") as pc:
        violinPlot(pc.ax, val5, cat3)
        pc.yvals["val5"] = val5
        pc.categories["cat3"] = cat3

    pm.runShuffles(numShuffles=1000)


if __name__ == "__main__":
    main()
