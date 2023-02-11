from __future__ import annotations
from enum import IntEnum, auto
from typing import List, Dict
import numpy as np
import pandas as pd


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

    def runAllShuffles(self, infoFileName: str) -> None:
        pass

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
