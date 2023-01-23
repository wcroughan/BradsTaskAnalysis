import numpy as np
import pandas as pd

from PlotUtil import PlotCtx, ShuffSpec


def conditionShuffle(dataframe, colName, rng):
    def swapFunc(val): return "SWR" if val == "Ctrl" else "Ctrl"
    numGroups = dataframe["conditionGroup"].max() + 1
    swapBool = (np.random.uniform(size=(numGroups,)) < 0.5).astype(bool)
    return [swapFunc(val) if swapBool[cg] else val for cg, val in zip(dataframe["conditionGroup"], dataframe[colName])]


def testShuffles():
    pp = PlotCtx("/home/wcroughan/data/figures/test")

    # for pcat in range(3):
    # pp.setStatCategory("pcat", pcat)
    with pp.newFig("testStats", withStats=True) as (ax, yvals, cats, infoVals):
        cats["c1"] = []
        cats["c2"] = []
        cats["c3"] = []
        yvals["v1"] = []
        yvals["v2"] = []
        infoVals["i1"] = []
        infoVals["i2"] = []
        for reps in range(2):
            for c1 in range(2):
                for c2 in range(3):
                    for c3 in range(3):
                        cats["c1"].append("asdf" if c1 == 0 else "poiu")
                        cats["c2"].append(str(c2))
                        cats["c3"].append(c3)
                        val = c1 * c2 + 4 * c1 + 0 * c3 + np.random.uniform() * 0.2
                        yvals["v1"].append(val)
                        yvals["v2"].append(np.random.uniform())
                        infoVals["i1"].append(str(np.random.uniform()))
                        infoVals["i2"].append(np.random.uniform())

        print(cats, yvals)

    pp.runShuffles(numShuffles=4)
    # sizes = np.linspace(15, 500, 5).astype(int)
    # res = np.empty((len(sizes), 2))
    # res[:, 0] = sizes
    # ts = [time.perf_counter()]

    # for ns in sizes:
    #     pp.runShuffles(numShuffles=ns)
    #     ts.append(time.perf_counter())

    # res[:, 1] = np.diff(np.array(ts))
    # print(res)


def testIndividualShuffleSpecs():
    pp = PlotCtx("/home/wcroughan/data/figures/test")
    pp.numShuffles = 5

    df = pd.DataFrame(data={"c1": [0, 0, 0, 0, 1, 1, 1, 1],
                            "c2": [10, 10, 20, 20, 10, 10, 20, 20],
                            "y": np.linspace(0, 1, 8)})

    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    print("===============================\n")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.WITHIN, categoryName="c1", value=1),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=10), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.WITHIN, categoryName="c2", value=10),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=1), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.WITHIN, categoryName="c1", value=0),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.ACROSS, categoryName="c1", value=None),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.INTERACTION, categoryName="c1", value=0),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL, categoryName="c1", value=None),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=20), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    print("===============================\n")


def testAcross():
    pp = PlotCtx("/home/wcroughan/data/figures/test")
    pp.numShuffles = 500
    c1 = np.array([0] * 8 + [1] * 8)
    c2 = np.array(([10] * 2 + [20] * 2) * 4)
    y = c1 + 50 * c2 + np.random.uniform() * 0.2
    df = pd.DataFrame(data={"c1": c1, "c2": c2, "y": y})
    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    print("testing where c1 has small effect, c2 has big effect")
    print("===============================\n")
    print("First just global c1")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=0)]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))

    print("===============================\n")
    print("global c2")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c2", value=10)]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))

    print("===============================\n")
    print("Now across c2, global c1")
    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.ACROSS, categoryName="c2", value=None),
            ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=0), ]
    res = pp._doShuffleSpec(df, spec, valSet, "y")
    print("\n".join([str(r) for r in res]))
    r = res[0]
    print("p =", np.count_nonzero(r.shuffleDiffs <= r.diff) / len(r.shuffleDiffs))


def testCustomShuffleFunction():
    pp = PlotCtx("/home/wcroughan/data/figures/test", randomSeed=1)
    pp.numShuffles = 5
    c1 = np.array([0] * 8 + [1] * 8)
    c2 = np.array(([10] * 2 + [20] * 2) * 4)
    y = c1 + 50 * c2 + np.random.uniform() * 0.2
    info = (np.random.uniform(size=y.shape) > 0.5).astype(int)
    df = pd.DataFrame(data={"c1": c1, "c2": c2, "y": y, "i": info})
    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    def shufFunc(dataframe, colName, rng):
        # r = dataframe["c1"].sample(frac=1, random_state=rng).reset_index(drop=True)
        r = dataframe["i"]
        print(r)
        return r
    # def shufFunc(df, rng):
    #     print("hi")
    #     return [0] * len(c1)
    pp.customShuffleFunctions["c1"] = shufFunc

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="c1", value=0)]
    res = pp._doShuffleSpec(df, spec, valSet, ["y"])
    for r in res:
        print(r.getFullInfoString(linePfx="\t"))


def testConditionShuffle():
    pp = PlotCtx("/home/wcroughan/data/figures/test", randomSeed=1)
    pp.numShuffles = 5
    conditionGroup = np.array([0] * 8 + [1] * 8)
    condition = ["SWR", "Ctrl"] * 8
    y = np.linspace(0, 1, len(condition))
    df = pd.DataFrame(data={"conditionGroup": conditionGroup, "condition": condition, "y": y})
    valSet = {}
    for col in df.columns:
        valSet[col] = set(df[col])

    pp.customShuffleFunctions["condition"] = conditionShuffle

    spec = [ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName="condition", value="SWR")]
    res = pp._doShuffleSpec(df, spec, valSet, ["y"])
    for r in res:
        print(r.getFullInfoString(linePfx="\t"))


if __name__ == "__main__":
    testConditionShuffle()
