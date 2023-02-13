from __future__ import annotations
from enum import IntEnum, auto
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle


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
        pvals1 = np.count_nonzero(self.diff.T < self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
        pvals2 = np.count_nonzero(self.diff.T <= self.shuffleDiffs,
                                  axis=0) / self.shuffleDiffs.shape[0]
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
        if self.shuffleDiffs is None:
            return "{}: {}".format(self.specs, self.diff)
        else:
            return "{}: {} ({}, {})".format(self.specs, self.diff, np.min(self.shuffleDiffs), np.max(self.shuffleDiffs))

    def __repr__(self) -> str:
        if self.shuffleDiffs is None:
            return "{}: {}".format(self.specs, self.diff)
        else:
            return "{}: {} ({}-{})".format(self.specs, self.diff, np.min(self.shuffleDiffs), np.max(self.shuffleDiffs))


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
        self, infoFileName: str, numShuffles=100,
            resultsFilter: Callable[[str, ShuffleResult, float, str], bool] = lambda *_: True) -> None:
        # TODO when adjusting this to combine possible multiple info file names, need to make sure
        # all the immediate shuffles are the same. Maybe split into groups

        self.numShuffles = numShuffles

        savedStatsFiles = self.getListOfStatsFiles(infoFileName)
        filesForEachPlot = {}
        for plotName, statsFile in savedStatsFiles:
            if plotName not in filesForEachPlot:
                filesForEachPlot[plotName] = []
            filesForEachPlot[plotName].append(statsFile)

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
        print("immediateShufflesAcrossPersistentCategories")
        print(sdf.to_string(index=False))

        sdf.to_hdf(infoFileName + "_immediateShuffles.h5", key="immediateShuffles")

    def getAllShuffleSpecsWithLeaf(self, df: pd.DataFrame, leaf: List[ShuffleResult],
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

    def runAllShuffles(self, infoFileName: str, numShuffles: int,
                       significantThreshold: Optional[float] = 0.15,
                       resultsFilter: Callable[[str, ShuffleResult, float, str], bool] = lambda *_: True) -> None:
        self.numShuffles = numShuffles

        savedStatsFiles = self.getListOfStatsFiles(infoFileName)
        filesForEachPlot = {}
        for plotName, statsFile in savedStatsFiles:
            if plotName not in filesForEachPlot:
                filesForEachPlot[plotName] = []
            filesForEachPlot[plotName].append(statsFile)

        shuffResults: List[Tuple[str, List[List[ShuffleResult]], str]] = []
        for plotName in tqdm(filesForEachPlot, desc="Running shuffles", total=len(filesForEachPlot)):
            df = pd.concat([pd.read_hdf(f, key="stats") for f in filesForEachPlot[plotName]])
            yvalNames = pd.read_hdf(filesForEachPlot[plotName][0], key="yvalNames")
            categoryNames = pd.read_hdf(filesForEachPlot[plotName][0], key="categoryNames")
            persistentCategoryNames = pd.read_hdf(
                filesForEachPlot[plotName][0], key="persistentCategoryNames")
            # infoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="infoNames")
            # persistentInfoNames = pd.read_hdf(filesForEachPlot[plotName][0], key="persistentInfoNames")
            measureName = os.path.dirname(filesForEachPlot[plotName][0]).split(os.path.sep)[-2]

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

            specs = self.getAllShuffleSpecs(df, columnsToShuffle=catsToShuffle)
            ss = [len(s) for s in specs]
            specs = [x for _, x in sorted(zip(ss, specs))]
            # print("\n".join([str(s) for s in specs]))
            sr = self._doShuffles(df, specs, yvalNames)
            shuffResults.append((plotName, sr, measureName))

        if significantThreshold is None:
            significantThreshold = 1.0

        significantResults: List[Tuple[str, ShuffleResult, float, str]] = []
        for plotName, sr, measureName in shuffResults:
            for s in sr:
                for s2 in s:
                    pval = s2.getPVals().item()
                    if pval < significantThreshold or pval > 1 - significantThreshold:
                        if resultsFilter(plotName, s2, pval, measureName):
                            significantResults.append(
                                (plotName, s2, pval if pval < 0.5 else 1 - pval,
                                 measureName))

        sdf = pd.DataFrame([(pn, str(s.specs), pv, mn) for pn, s, pv, mn in significantResults],
                           columns=["plot", "shuffle", "pval", "measure"])
        sdf.sort_values(by="pval", inplace=True)
        print("all shuffles")
        print(sdf.to_string(index=False))

        sdf.to_hdf(infoFileName + "_significantShuffles.h5", key="significantShuffles")

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

    def getAllShuffleSpecs(self, df, columnsToShuffle=None):
        if columnsToShuffle is None:
            columnsToShuffle = list(df.columns)

        valSet = {}
        for col in columnsToShuffle:
            valSet[col] = sorted(list(set(df[col])))

        ret = []
        for col in columnsToShuffle:
            for val in valSet[col]:
                ret.append([ShuffSpec(shuffType=ShuffSpec.ShuffType.GLOBAL, categoryName=col, value=val)])

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
