from __future__ import annotations
from enum import IntEnum, auto
from typing import List, Dict, Tuple, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import os
import pickle
import time
from datetime import datetime
from tqdm import tqdm
import warnings
from UtilFunctions import getPreferredCategoryOrder
from itertools import product
from scipy.stats import pearsonr


def notNanPvalFilter(plotName: str, shuffleResult: ShuffleResult, pval: float, measureName: str) -> bool:
    return not np.isnan(pval)


class ShuffSpec:
    """
    Specifies one shuffle to run
    if type is WITHIN, GLOBAL, or INTERACTION, value is a single value
    if type is ACROSS or RECURSIVE_ALL, value is None
    """
    class ShuffType(IntEnum):
        UNSPECIFIED = auto()
        GLOBAL = auto()
        WITHIN = auto()
        ACROSS = auto()
        INTERACTION = auto()
        RECURSIVE_ALL = auto()  # includes interaction, across, and within

        def __repr__(self) -> str:
            return self.name

        def __str__(self) -> str:
            return self.name[0:3]

    def __init__(self, shuffType=ShuffType.UNSPECIFIED, categoryName="", value=None):
        self.shuffType = shuffType
        self.categoryName = categoryName
        self.value = value

    def copy(self) -> ShuffSpec:
        ret = ShuffSpec()
        ret.shuffType = self.shuffType
        ret.categoryName = self.categoryName
        ret.value = self.value
        return ret

    def __str__(self) -> str:
        return "{} {} ({})".format(self.shuffType, self.categoryName, self.value)

    def __repr__(self) -> str:
        return "{} {} ({})".format(self.shuffType, self.categoryName, self.value)

    def __lt__(self, other) -> bool:
        if self.shuffType < other.shuffType:
            return True
        elif self.shuffType > other.shuffType:
            return False

        if self.categoryName < other.categoryName:
            return True
        elif self.categoryName > other.categoryName:
            return False

        if self.value is None or other.value is None:
            return False
        return self.value < other.value

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, ShuffSpec):
            return False
        return self.shuffType == __o.shuffType and self.categoryName == __o.categoryName and \
            (self.value is None) == (__o.value is None) and (
                self.value is None or self.value == __o.value)

    def __ne__(self, __o: object) -> bool:
        return not self == __o


class ShuffleResult:
    def __init__(self, specs: List[ShuffSpec] = [], diff=np.nan, shuffleDiffs=None, dataNames: List[str] = []) -> None:
        self.specs = specs
        self.diff = diff
        self.shuffleDiffs = shuffleDiffs
        self.dataNames = dataNames

    def copy(self) -> ShuffleResult:
        ret = ShuffleResult()
        ret.diff = self.diff
        if self.shuffleDiffs is not None:
            ret.shuffleDiffs = np.copy(self.shuffleDiffs)
        ret.specs = [s.copy() for s in self.specs]
        ret.dataNames = self.dataNames.copy()
        return ret

    def getPVals(self) -> np.ndarray:
        if np.all(np.isnan(self.shuffleDiffs)):
            return np.full(self.diff.shape, np.nan)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"invalid value encountered in divide")
            pvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
                                      axis=0) / np.count_nonzero(~np.isnan(self.shuffleDiffs), axis=0)
            pvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
                                      axis=0) / np.count_nonzero(~np.isnan(self.shuffleDiffs), axis=0)
            # oldpvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
            #                              axis=0) / self.shuffleDiffs.shape[0]
            # oldpvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
            #                              axis=0) / self.shuffleDiffs.shape[0]
            # if np.any(np.isnan(self.shuffleDiffs)):
            #     print("nans in shuffle diff: ", np.count_nonzero(np.isnan(self.shuffleDiffs),
            #           axis=0) / self.shuffleDiffs.shape[0], oldpvals1, pvals1, sep="\t")
        return (pvals1 + pvals2) / 2

    def getFullInfoString(self, linePfx="") -> str:
        sdmin = np.min(self.shuffleDiffs, axis=0)
        sdmax = np.max(self.shuffleDiffs, axis=0)
        pvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        pvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        ret = [f"{self.specs}:"] + [linePfx + f"{self.dataNames[i]}: {float(self.diff[i])} ({sdmin[i]}, {sdmax[i]}) "
                                    f"p1 = {pvals1[i]}\tp2 = {pvals2[i]}" for i in range(len(self.diff))]
        return "\n".join(ret)

    def __str__(self) -> str:
        if isinstance(self.diff, np.ndarray):
            diffStr = ", ".join([f"{d:.3f}" for d in self.diff.flat])
        else:
            diffStr = f"{self.diff:.3f}"
        if self.shuffleDiffs is None:
            return f"{self.specs}: {diffStr}"
        else:
            return f"{self.specs}: {diffStr} ({np.min(self.shuffleDiffs):.3f}, {np.max(self.shuffleDiffs):.3f})"

    def __repr__(self) -> str:
        if isinstance(self.diff, np.ndarray):
            diffStr = ", ".join([f"{d:.3f}" for d in self.diff.flat])
        else:
            diffStr = f"{self.diff:.3f}"
        if self.shuffleDiffs is None:
            return f"{self.specs}: {diffStr}"
        else:
            return f"{self.specs}: {diffStr} ({np.min(self.shuffleDiffs):.3f}, {np.max(self.shuffleDiffs):.3f})"


class Shuffler:
    def __init__(self, rng: np.random.Generator = None, numShuffles=100) -> None:
        self.rng = rng
        self.numShuffles = numShuffles
        self.customShuffleFunctions: Dict[str, Callable[[
            pd.DataFrame, str, np.random.Generator], pd.Series]] = {}

    def setCustomShuffleFunction(self, category: str, func: Callable[[pd.DataFrame, str, np.random.Generator], pd.Series]):
        self.customShuffleFunctions[category] = func

    def doShuffles(self, df: pd.DataFrame, specs: List[List[ShuffSpec]],
                   dataNames: List[str], numShuffles: List[int] = None) -> List[List[ShuffleResult]]:
        if numShuffles is not None:
            assert len(numShuffles) == len(specs)

        columnsToShuffle = set()
        for spec in specs:
            for s in spec:
                columnsToShuffle.add(s.categoryName)

        # print(df)
        valSet: Dict[str, set] = {}
        for col in columnsToShuffle:
            vs = set(df[col])
            if len(vs) <= 1:
                raise Exception("Only one value {} for category {}".format(vs, col))
            valSet[col] = vs
        # print("created valset:", valSet)

        ret: List[List[ShuffleResult]] = []
        for si, spec in enumerate(specs):
            # print(f'{spec}                 ', end='\r')
            if numShuffles is not None:
                self.numShuffles = numShuffles[si]
            res = self._doShuffleSpec(df, spec, valSet, dataNames)
            ret.append(res)

        return ret

    def getListOfStatsFiles(self, infoFileName: str) -> List[Tuple[str, str]]:
        with open(infoFileName, "r") as fid:
            lines = fid.readlines()
            statsFiles = []
            for line in lines:
                if line.startswith("statsFile:"):
                    statsFiles.append(
                        (line.split("__!__")[1].strip(), line.split("__!__")[2].strip()))
        return statsFiles

    def runImmediateShufflesAcrossPersistentCategories(
        self, infoFileNames: List[str], numShuffles=100,
            resultsFilter: Callable[[str, ShuffleResult, float, str], bool] = lambda *_: True) -> str:
        self.numShuffles = numShuffles

        savedStatsFiles = []
        for infoFileName in infoFileNames:
            savedStatsFiles += self.getListOfStatsFiles(infoFileName)
        filesForEachPlot = {}
        for plotName, statsFile in savedStatsFiles:
            if plotName not in filesForEachPlot:
                filesForEachPlot[plotName] = []
            filesForEachPlot[plotName].append(statsFile)

        todel = []
        for plotName in filesForEachPlot:
            # Get the immediate shuffles, and if there's multiple versions, run each separately
            immGroups = {}
            for fname in filesForEachPlot[plotName]:
                pickleFile = fname[:-3] + "_immediateShuffles.pkl"
                if not os.path.exists(pickleFile):
                    print("No immediate shuffles for {}".format(pickleFile))
                    continue
                with open(pickleFile, "rb") as fid:
                    key = str(pickle.load(fid))

                if key not in immGroups:
                    immGroups[key] = [fname]
                else:
                    immGroups[key].append(fname)

            if len(immGroups) > 1:
                for ki, key in enumerate(immGroups):
                    filesForEachPlot[f"{plotName}_immgroup{ki}"] = immGroups[key]
                todel.append(plotName)
        for d in todel:
            del filesForEachPlot[d]

        shuffResults: List[Tuple[str, List[List[ShuffleResult]], str]] = []
        for plotName in tqdm(filesForEachPlot, desc="Running shuffles", total=len(filesForEachPlot)):
            # for plotName in filesForEachPlot:
            # Get the immediate shuffles. Should be the same for all files
            fname = filesForEachPlot[plotName][0]
            pickleFile = fname[:-3] + "_immediateShuffles.pkl"
            if not os.path.exists(pickleFile):
                # print("No immediate shuffles for {}".format(pickleFile))
                continue
            with open(pickleFile, "rb") as fid:
                immediateShuffles = pickle.load(fid)

            df = pd.concat([pd.read_hdf(f, key="stats") for f in filesForEachPlot[plotName]])
            yvalNames = pd.read_hdf(filesForEachPlot[plotName][0], key="yvalNames")
            # categoryNames = pd.read_hdf(filesForEachPlot[plotName][0], key="categoryNames")
            persistentCategoryNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="persistentCategoryNames")
            # infoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="infoNames")
            # persistentInfoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="persistentInfoNames")
            measureName = os.path.dirname(filesForEachPlot[plotName][0]).split(os.path.sep)[-2]

            # print("Plot: {}, measure: {}".format(plotName, measureName))
            # print(df.to_string())

            nu = df.nunique(axis=0)
            catsToShuffle = list(persistentCategoryNames)
            # print("Categories to shuffle: {}".format(catsToShuffle))
            # print("Unique vals per category:\n{}".format(nu))
            todel = set()
            for cat in catsToShuffle:
                if nu[cat] <= 1:
                    todel.add(cat)
                    print("Category {} has only one val. Not including in shuffle for plot {}".format(
                        cat, plotName))
            for td in todel:
                catsToShuffle.remove(td)

            specs = []
            for ish in immediateShuffles:
                specs += self.getAllShuffleSpecsWithLeaf(
                    df, leaf=ish[0], columnsToShuffle=catsToShuffle)
            ss = [len(s) for s in specs]
            specs = [x for _, x in sorted(zip(ss, specs))]
            # print("All specs:")
            # print("\n".join(["\t" + str(s) for s in specs]))
            sr = self._doShuffles(df, specs, yvalNames)
            shuffResults.append((plotName, sr, measureName))

        filteredResults: List[Tuple[str, ShuffleResult, float, str]] = []
        for plotName, sr, measureName in shuffResults:
            for s in sr:
                for s2 in s:
                    pval = s2.getPVals().item()
                    if resultsFilter(plotName, s2, pval, measureName):
                        filteredResults.append(
                            (plotName, s2, pval if pval < 0.5 else 1 - pval,
                                measureName))

        sdf = pd.DataFrame([(pn, str(s.specs), pv, mn) for pn, s, pv, mn in filteredResults],
                           columns=["plot", "shuffle", "pval", "measure"])
        sdf.sort_values(by="pval", inplace=True)
        # print("immediateShufflesAcrossPersistentCategories")
        # print(sdf.to_string(index=False))

        outputFileName = os.path.join(os.path.dirname(
            infoFileNames[0]), datetime.now().strftime("%Y%m%d_%H%M%S_immediateShuffles.h5"))
        sdf.to_hdf(outputFileName, key="immediateShuffles")
        return outputFileName

    def getAllShuffleSpecsWithLeaf(self, df: pd.DataFrame, leaf: List[ShuffSpec],
                                   columnsToShuffle: Optional[List[str]] = None) -> List[List[ShuffSpec]]:
        if columnsToShuffle is None:
            raise Exception("Not implemented. leaf is a list...")
            columnsToShuffle = set(df.columns)
            columnsToShuffle.remove(leaf.categoryName)
            columnsToShuffle = list(columnsToShuffle)

        ret = [leaf]
        for col in columnsToShuffle:
            otherCols = [c for c in columnsToShuffle if c != col]
            rec = self.getAllShuffleSpecsWithLeaf(df, leaf, columnsToShuffle=otherCols)
            for r in rec:
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL,
                                      categoryName=col, value=None)] + r)
        return ret

    def runCorrelations(self, df: pd.DataFrame, xvar: str, yvar: str, catNames: List[str]):
        catVals = [getPreferredCategoryOrder(df[cn].values) for cn in catNames]
        catCombs = list(product(*catVals))
        ret = []

        for cc in catCombs:
            dfc = df
            for cn, cv in zip(catNames, cc):
                dfc = dfc[dfc[cn] == cv]

            x = dfc[xvar].values
            y = dfc[yvar].values
            # print(f"Doing correlation for {catStr}x={xvar} y={yvar} with {len(x)} values")
            # filter out inf and nan values
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if len(x) < 2:
                ret.append((cc, np.nan, np.nan))
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", r"An input array is constant; the correlation coefficient is not defined.")
                r, p = pearsonr(x, y)
            ret.append((cc, r, p))

        return ret

    def runAllShuffles(self,
                       infoFileNames: Optional[List[str]],
                       numShuffles: int,
                       significantThreshold: Optional[float] = 0.15,
                       shuffleResultsFilter: Callable[[
                           str, ShuffleResult, float, str], bool] = notNanPvalFilter,
                       savedStatsFiles: Optional[List[Tuple[str, str]]] = None,
                       outputFileName: Optional[str] = None,
                       justGlobal=False,
                       skipCorrelations=False) -> str:
        self.numShuffles = numShuffles

        if infoFileNames is None and savedStatsFiles is None:
            raise Exception("Either infoFileNames or savedStatsFiles must be specified")

        if savedStatsFiles is None:
            savedStatsFiles = []
            for infoFileName in infoFileNames:
                savedStatsFiles += self.getListOfStatsFiles(infoFileName)
        filesForEachPlot = {}
        for plotName, statsFile in savedStatsFiles:
            if plotName not in filesForEachPlot:
                filesForEachPlot[plotName] = []
            filesForEachPlot[plotName].append(statsFile)

        shuffResults: List[Tuple[str, List[List[ShuffleResult]], str]] = []
        correlationResults: List[Tuple[str, List[Tuple[Iterable[str], float, float]], str]] = []
        for plotName in tqdm(filesForEachPlot, desc="Running shuffles and correlations", total=len(filesForEachPlot), smoothing=0):
            df = pd.concat([pd.read_hdf(f, key="stats") for f in filesForEachPlot[plotName]])
            yvalNames = pd.read_hdf(filesForEachPlot[plotName][0], key="yvalNames")
            xvalNames = pd.read_hdf(filesForEachPlot[plotName][0], key="xvalNames")
            categoryNames = pd.read_hdf(filesForEachPlot[plotName][0], key="categoryNames")
            persistentCategoryNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="persistentCategoryNames")
            # infoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="infoNames")
            # persistentInfoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="persistentInfoNames")
            measureName = os.path.dirname(filesForEachPlot[plotName][0]).split(os.path.sep)[-2]

            # Shuffle categories
            catsToShuffle = list(categoryNames) + list(persistentCategoryNames)
            todel = set()
            for cat in catsToShuffle:
                vals = set(df[cat])
                if len(vals) <= 1:
                    todel.add(cat)
                    print("Category {} has only one val ({}). Not including in shuffle for plot {}".format(
                        cat, vals, plotName))
            for td in todel:
                catsToShuffle.remove(td)

            try:
                specs = self.getAllShuffleSpecs(
                    df, columnsToShuffle=catsToShuffle, justGlobal=justGlobal)
            except Exception as e:
                print("Error getting shuffle specs for plot {} of measure {}".format(plotName, measureName))
                print(e)
                raise e
            ss = [len(s) for s in specs]
            specs = [x for _, x in sorted(zip(ss, specs))]
            # print("\n".join([str(s) for s in specs]))
            sr = self._doShuffles(df, specs, yvalNames)
            shuffResults.append((plotName, sr, measureName))

            if not skipCorrelations:
                # Look for xval correlations
                xynamepairs = list(product(xvalNames, yvalNames))
                cr = []
                for xname, yname in xynamepairs:
                    cr.append(self.runCorrelations(df, xname, yname, catsToShuffle))

                correlationResults.append((plotName, cr, measureName))

        if significantThreshold is None:
            significantThreshold = 1.0

        significantResults: List[Tuple[str, str, float, str, str, bool, bool]] = []
        for plotName, sr, measureName in tqdm(shuffResults, desc="Filtering shuffles", total=len(shuffResults)):
            for s in sr:
                for s2 in s:
                    for pi, pval in enumerate(s2.getPVals()):
                        if pval < significantThreshold or pval > 1 - significantThreshold:
                            if shuffleResultsFilter(plotName, s2, pval, measureName):
                                # isCondShuf = any([s.categoryName == "condition" for s in s2.specs])

                                measurenameWithoutPrefix = "_".join(
                                    measureName.split("_")[1:])
                                plotSuffix = plotName[len(measurenameWithoutPrefix):]
                                isCondShuf = plotSuffix == "" or plotSuffix.endswith(
                                    "diff") or plotSuffix.endswith("cond")
                                isCtrlShuf = plotSuffix.endswith("cond") or (
                                    "ctrl_" in plotSuffix and "diff" not in plotSuffix)
                                isDiffShuf = plotSuffix.endswith("diff")
                                isNextSeshDiffShuf = plotSuffix.endswith("nextsession_diff")
                                significantResults.append(
                                    (plotName, str(s2), pval if pval < 0.5 else 1 - pval, pval < 0.5,
                                     measureName, plotSuffix, isCondShuf, isCtrlShuf, isDiffShuf,
                                     isNextSeshDiffShuf, pi))

        sdf = pd.DataFrame(significantResults,
                           columns=["plot", "shuffle", "pval", "direction", "measure", "plotSuffix", "isCondShuf",
                                    "isCtrlShuf", "isDiffShuf", "isNextSeshDiffShuf", "pvalIndex"])
        sdf.sort_values(by="pval", inplace=True)

        if outputFileName is None:
            outputFileName = os.path.join(os.path.dirname(
                infoFileNames[0]), datetime.now().strftime("%Y%m%d_%H%M%S_significantShuffles.h5"))
        sdf.to_hdf(outputFileName, key="significantShuffles")

        print(sdf)
        print(f"{ len(shuffResults)=  }")

        if not skipCorrelations:
            significantCorrelations = []
            for plotName, cr, measureName in tqdm(correlationResults, desc="Filtering correlations", total=len(correlationResults)):
                for catCombRes in cr:
                    for catNames, corr, pval in catCombRes:
                        if pval < significantThreshold or pval > 1 - significantThreshold:
                            if shuffleResultsFilter(plotName, None, pval, measureName):
                                plotSuffix = plotName[len(measureName)-3:]
                                significantCorrelations.append(
                                    (plotName, "_".join(catNames), corr, pval if pval < 0.5 else 1 - pval, pval < 0.5,
                                        measureName, plotSuffix))

            sdf = pd.DataFrame(significantCorrelations,
                               columns=["plot", "categories", "correlation", "pval", "direction", "measure", "plotSuffix"])
            sdf.sort_values(by="pval", inplace=True)
            sdf.to_hdf(outputFileName, key="significantCorrelations")

        return outputFileName

    def summarizeShuffleResults(self, hdfFile: str) -> None:
        df = pd.read_hdf(hdfFile, key="significantShuffles")
        # print("summarizeShuffleResults")
        # print(df.to_string(index=False))

        # dfWithoutShuffle = df.drop(columns=["shuffle"])
        # dfWithoutShuffle = dfWithoutShuffle[dfWithoutShuffle["isDiffShuf"]]
        # print(dfWithoutShuffle.to_string(index=False))

        minNextSeshPvals = df[df["isNextSeshDiffShuf"]].sort_values(
            by="pval", ascending=True).drop_duplicates(subset=["measure"])
        # print("minNextSeshPvals")
        # print(minNextSeshPvals.to_string(index=False))

        minNonNextSeshPvals = df[df["isDiffShuf"] & ~df["isNextSeshDiffShuf"]].sort_values(
            by="pval", ascending=True).drop_duplicates(subset=["measure"])
        # print("minNonNextSeshPvals")
        # print(minNonNextSeshPvals.to_string(index=False))

        # print("summarizeShuffleResults")
        # condCounts = df.groupby(["isCondShuf", "measure"])["pval"].count().xs(
        #     True, level="isCondShuf").sort_values(ascending=False).rename("CondShufCount").reset_index()
        # ctrlCounts = df.groupby(["isCtrlShuf", "measure"])["pval"].count().xs(
        #     True, level="isCtrlShuf").sort_values(ascending=False).rename("CtrlShufCount").reset_index()
        # df = pd.merge(condCounts, ctrlCounts, on="measure")
        # df.sort_values(by="CondShufCount", inplace=True, ascending=False)
        # print(df.to_string(index=False))

        outputFileName = os.path.join(os.path.dirname(
            hdfFile), datetime.now().strftime("%Y%m%d_%H%M%S_summary.txt"))
        with open(outputFileName, "w") as f:
            # f.write(df.to_string(index=False))
            f.write("Non next session shuffle results")
            f.write(minNonNextSeshPvals.to_string(index=False))
            f.write("Next session shuffle results")
            f.write(minNextSeshPvals.to_string(index=False))

        outputFileNameCSV_nextSesh = os.path.join(os.path.dirname(
            hdfFile), datetime.now().strftime("%Y%m%d_%H%M%S_summary_nextSesh.csv"))
        minNextSeshPvals.to_csv(outputFileNameCSV_nextSesh, index=False)

        outputFileNameCSV_nonNextSesh = os.path.join(os.path.dirname(
            hdfFile), datetime.now().strftime("%Y%m%d_%H%M%S_summary_nonNextSesh.csv"))
        minNonNextSeshPvals.to_csv(outputFileNameCSV_nonNextSesh, index=False)

    def _doShuffles(self, df: pd.DataFrame, specs: List[List[ShuffSpec]],
                    dataNames: List[str], numShuffles: List[int] = None) -> List[List[ShuffleResult]]:
        if numShuffles is not None:
            assert len(numShuffles) == len(specs)

        columnsToShuffle = set()
        for spec in specs:
            for s in spec:
                columnsToShuffle.add(s.categoryName)

        # print(df)
        valSet: Dict[str, set] = {}
        for col in columnsToShuffle:
            vs = set(df[col])
            if len(vs) <= 1:
                raise Exception("Only one value {} for category {}".format(vs, col))
            valSet[col] = vs
        # print("created valset:", valSet)

        ret: List[List[ShuffleResult]] = []
        for si, spec in enumerate(specs):
            # print(f'{spec}                 ', end='\r')
            if numShuffles is not None:
                self.numShuffles = numShuffles[si]
            res = self._doShuffleSpec(df, spec, valSet, dataNames)
            ret.append(res)

        return ret

    def getAllShuffleSpecs(self, df, columnsToShuffle=None, justGlobal=False):
        if columnsToShuffle is None:
            columnsToShuffle = list(df.columns)

        valSet = {}
        for col in columnsToShuffle:
            try:
                valSet[col] = sorted(list(set(df[col])))
            except TypeError as te:
                print("Error getting values for column", col)
                print("Values:", df[col].to_numpy())
                print(df.head())
                raise te

        ret = []
        for col in columnsToShuffle:
            for val in valSet[col]:
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName=col, value=val)])

            if not justGlobal:
                otherCols = [c for c in columnsToShuffle if c != col]
                rec = self.getAllShuffleSpecs(df, otherCols)
                for r in rec:
                    ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.RECURSIVE_ALL,
                                          categoryName=col, value=None)] + r)

        return ret

    def _doOneWayShuffle(self, df: pd.DataFrame, spec: ShuffSpec, dataNames: List[str]) -> ShuffleResult:
        assert isinstance(spec, ShuffSpec) and \
            spec.shuffType == ShuffSpec.ShuffType.GLOBAL

        if spec.categoryName in self.customShuffleFunctions:
            shufFunc = self.customShuffleFunctions[spec.categoryName]
        else:
            def shufFunc(dataframe: pd.DataFrame, colName: str, rng: np.random.Generator) -> pd.Series:
                return dataframe[colName].sample(frac=1, random_state=rng).reset_index(drop=True)

        ret = ShuffleResult([spec], dataNames=dataNames)

        withinIdx = df[spec.categoryName] == spec.value
        d = np.array(df[withinIdx][dataNames].mean() - df[~withinIdx][dataNames].mean())
        ret.diff = np.reshape(d, (-1, 1))

        ret.shuffleDiffs = np.empty((self.numShuffles, len(dataNames)))
        sdf = df.copy().reset_index(drop=True)
        for si in range(self.numShuffles):
            sdf[spec.categoryName] = shufFunc(sdf, spec.categoryName, self.rng)
            withinIdx = sdf[spec.categoryName] == spec.value
            ret.shuffleDiffs[si, :] = sdf[withinIdx][dataNames].mean() - \
                sdf[~withinIdx][dataNames].mean()

        return ret

    def _doShuffleSpec(self, df: pd.DataFrame, spec: List[ShuffSpec], valSet: Dict[str, set], dataNames: List[str]) -> List[ShuffleResult]:
        assert isinstance(spec, list)
        for si, s in enumerate(spec):
            assert isinstance(s, ShuffSpec)
            assert s.shuffType != ShuffSpec.ShuffType.UNSPECIFIED
            assert s.shuffType != ShuffSpec.ShuffType.GLOBAL or si == len(spec) - 1

        s = spec[0]
        if s.shuffType == ShuffSpec.ShuffType.GLOBAL:
            return [self._doOneWayShuffle(df, s, dataNames)]

        if s.shuffType == ShuffSpec.ShuffType.WITHIN:
            withinIdx = df[s.categoryName] == s.value
            recres = self._doShuffleSpec(df.loc[withinIdx], spec[1:], valSet, dataNames)
            for r in recres:
                r.specs = [s] + r.specs.copy()
            return recres

        if s.shuffType == ShuffSpec.ShuffType.INTERACTION:
            withinIdx = df[s.categoryName] == s.value
            withoutIdx = ~ withinIdx
            rin = self._doShuffleSpec(df.loc[withinIdx], spec[1:], valSet, dataNames)
            rout = self._doShuffleSpec(df.loc[withoutIdx], spec[1:], valSet, dataNames)
            ret = []
            for reti, (i, o) in enumerate(zip(rin, rout)):
                r = i.copy()
                r.specs = [s] + i.specs.copy()
                r.diff = i.diff - o.diff
                r.shuffleDiffs = i.shuffleDiffs - o.shuffleDiffs
                ret.append(r)
            return ret

        if s.shuffType == ShuffSpec.ShuffType.ACROSS:
            vals = valSet[s.categoryName]
            withinRes: List[List[ShuffleResult]] = []
            for val in vals:
                withinIdx = df[s.categoryName] == val
                withinRes.append(self._doShuffleSpec(
                    df.loc[withinIdx], spec[1:], valSet, dataNames))

            numEffects = len(withinRes[0])
            ret = []
            for ei in range(numEffects):
                r = withinRes[0][ei].copy()
                r.specs = [s] + r.specs.copy()
                r.diff = np.zeros((len(dataNames), 1))
                r.shuffleDiffs = np.zeros((self.numShuffles, len(dataNames)))
                for vi in range(len(vals)):
                    r.diff += withinRes[vi][ei].diff
                    r.shuffleDiffs += withinRes[vi][ei].shuffleDiffs
                ret.append(r)
            return ret

        if s.shuffType == ShuffSpec.ShuffType.RECURSIVE_ALL:
            # all the recursive calls needed
            vals = valSet[s.categoryName]
            withinRes = []
            withoutRes = []
            for val in vals:
                withinIdx = df[s.categoryName] == val
                withinRes.append(self._doShuffleSpec(
                    df.loc[withinIdx], spec[1:], valSet, dataNames))
                withoutIdx = ~ withinIdx
                withoutRes.append(self._doShuffleSpec(
                    df.loc[withoutIdx], spec[1:], valSet, dataNames))

            ret = []
            # within effects
            for vi, val in enumerate(vals):
                wr = withinRes[vi]
                for ei, effect in enumerate(wr):
                    r = effect.copy()
                    ss = s.copy()
                    ss.shuffType = ShuffSpec.ShuffType.WITHIN
                    ss.value = val
                    r.specs = [ss] + r.specs.copy()
                    ret.append(r)

            # across effect
            numEffects = len(withinRes[0])
            for ei in range(numEffects):
                r = withinRes[0][ei].copy()
                ss = s.copy()
                ss.shuffType = ShuffSpec.ShuffType.ACROSS
                r.specs = [ss] + r.specs.copy()
                r.diff = np.zeros((len(dataNames), 1))
                r.shuffleDiffs = np.zeros((self.numShuffles, len(dataNames)))
                for vi in range(len(vals)):
                    r.diff += withinRes[vi][ei].diff
                    r.shuffleDiffs += withinRes[vi][ei].shuffleDiffs
                ret.append(r)

            # interaction effects
            for vi, val in enumerate(vals):
                wr = withinRes[vi]
                wor = withoutRes[vi]
                for ei, (rin, rout) in enumerate(zip(wr, wor)):
                    ss = s.copy()
                    ss.shuffType = ShuffSpec.ShuffType.INTERACTION
                    ss.value = val
                    r = ShuffleResult([ss] + rin.specs.copy())
                    r.dataNames = dataNames
                    r.diff = rin.diff - rout.diff
                    r.shuffleDiffs = rin.shuffleDiffs - rout.shuffleDiffs
                    ret.append(r)

            return ret


if __name__ == "__main__":
    randomSeed = int(time.perf_counter())
    print(f"{ randomSeed = }")
    s = Shuffler(np.random.default_rng(randomSeed), 100)
    infoFileNames = []
    # infoFileNames.append("B17_20230210_171338.txt")
    dataDir = "/media/WDC8/figures/202302_labmeeting"
    # infoFileNames.append("B17_20230213_103244.txt")
    # infoFileNames.append("B17_20230213_133656.txt")
    # infoFileNames.append("B17_20230210_171338.txt")  # "simple" measures like latency
    # infoFileNames.append("B17_20230213_173328.txt")
    # infoFileNames = [os.path.join(dataDir, f) for f in infoFileNames]
    # outFile = s.runAllShuffles(infoFileNames, 100)
    # print(outFile)
    outFile = os.path.join(dataDir, "20230214_092451_significantShuffles.h5")
    # outFile = "/media/WDC8/figures/202302_labmeeting/20230213_162512_significantShuffles.h5"
    s.summarizeShuffleResults(outFile)
