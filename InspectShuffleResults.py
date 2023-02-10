import pandas as pd
import os
import numpy as np


def main():
    dir = "/media/WDC8/figures/202302_labmeeting"
    fileName = os.path.join(dir, "B17_20230209_141714.txtsignificantShuffles.h5")
    df = pd.read_hdf(fileName, key="significantShuffles")
    # mostSig = df["pval"] < 0.02
    # df = df[mostSig]
    idx = df["plot"].str.contains("probe")
    df = df[idx]
    # idx = df["plot"].str.contains("diff")
    # df = df[idx].sort_values(by="plot")
    idx = df["shuffle"].str.contains("WIT")
    df = df[~idx]
    df.reset_index(inplace=True, drop=True)
    # print(df.to_string(index=False))

    outDir = os.path.join(dir, "tmpSigFigs")
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    scores = {}
    for i, row in df.iterrows():
        plot = str(row["plot"])
        si = plot.index("_d")
        try:
            splitPoint = plot.index("_", si + 1)
            configuration = plot[0:splitPoint]
            measure = plot[splitPoint + 1:]
        except ValueError:
            configuration = plot
            measure = "nomeasure"

        if measure == "nomeasure" or measure.endswith("diff") or measure.endswith("cond"):
            isConditionMeasure = True
        else:
            isConditionMeasure = False
        if measure.endswith("cond") or ("ctrl_" in measure and "diff" not in measure):
            isCtrlMeasure = True
        else:
            isCtrlMeasure = False

        if configuration not in scores:
            scores[configuration] = {"cond": 0, "ctrl": 0}

        if isConditionMeasure:
            scores[configuration]["cond"] += 1
        if isCtrlMeasure:
            scores[configuration]["ctrl"] += 1

        # pval = row["pval"]
        # figFileName = os.path.join(dir, "B17", plot + ".png")
        # os.system(f"cp {figFileName} {outDir}/{pval:.3f}_{plot}.png")

    sdf = pd.DataFrame(scores).T
    sdf.sort_values(by="cond", inplace=True)
    print(sdf.to_string())
    sdf.sort_values(by="ctrl", inplace=True)
    print(sdf.to_string())
    # for configuration, score in scores.items():
    #     print(
    #         f"{configuration}: {score['cond']} condition measures, {score['ctrl']} control measures")


if __name__ == "__main__":
    main()
