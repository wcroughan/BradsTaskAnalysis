from glob import glob
import os
from itertools import product
from pprint import pprint
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import time

from Shuffler import Shuffler
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from UtilFunctions import findDataDir, getLoadInfo
from tqdm import tqdm
from MeasureTypes import LocationMeasure
from BTData import BTData
from PlotUtil import PlotManager


def getNameFromParams(name, params, func, correlationName):
    if name == "measureFromFunc":
        bp = params["bp"]
        radius = params["radius"]
        smoothDist = params["smoothDist"]
        ret = f"LM_{func.__name__}_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}"
        if correlationName is not None:
            ret += f"_X_{correlationName}"
        return ret
    else:
        raise ValueError(f"Unrecognized name: {name}")


def findStatsDir(subDir: str, name: str):
    possibleDrives = [
        "/media/WDC11/",
        "/media/WDC10/",
        "/media/WDC9/",
        "/media/WDC4/",
    ]
    drivesToAppend = []
    for posd in possibleDrives:
        if os.path.exists(os.path.join(posd, "BY_PID")):
            gl = glob(os.path.join(posd, "BY_PID", "*"))
            drivesToAppend.extend(gl)
    possibleDrives.extend(drivesToAppend)
    possibleLocations = [f"{posd}{os.path.sep}{subDir}" for posd in possibleDrives]
    for pl in possibleLocations:
        checkFileName = os.path.join(pl, name, "processed.txt")
        if os.path.exists(checkFileName):
            return os.path.join(pl, name, "stats")
        checkFileName = os.path.join(pl, name, "corr_processed.txt")
        if os.path.exists(checkFileName):
            return os.path.join(pl, name, "stats")
        checkFileName = os.path.join(pl, name, "error.txt")
        if os.path.exists(checkFileName):
            print(f"Found error file for {name} at {checkFileName}")
            return None
    print(f"Could not find stats dir for {name}")
    # print(f"Checked {len(possibleLocations)} possible locations")
    # print(f"Possible locations:")
    # pprint(possibleLocations)
    # return None
    raise ValueError(f"Could not find stats dir for {name}")


def getParamsForName(specName, func):
    allConsideredBPs = [BP(probe=False),
                        BP(probe=False, inclusionFlags="explore"),
                        BP(probe=False, inclusionFlags="offWall"),
                        BP(probe=False, inclusionFlags="moving"),
                        BP(probe=False, inclusionFlags=["explore", "offWall"]),
                        BP(probe=False, inclusionFlags="homeTrial", erode=3),
                        BP(probe=False, inclusionFlags="awayTrial", erode=3),
                        BP(probe=False, trialInterval=(2, 7), erode=3),
                        BP(probe=False, trialInterval=(2, 7),
                        inclusionFlags="homeTrial", erode=3),
                        BP(probe=False, trialInterval=(2, 7),
                        inclusionFlags="awayTrial", erode=3),
                        BP(probe=False, trialInterval=(10, 15), erode=3),
                        BP(probe=False, trialInterval=(10, 15),
                        inclusionFlags="homeTrial", erode=3),
                        BP(probe=False, trialInterval=(10, 15),
                        inclusionFlags="awayTrial", erode=3),
                        BP(probe=False, trialInterval=(0, 1), erode=3),
                        BP(probe=False, trialInterval=(2, 3), erode=3),
                        BP(probe=False, trialInterval=(3, 4), erode=3),
                        BP(probe=False, inclusionFlags="homeTrial"),
                        BP(probe=False, inclusionFlags="awayTrial"),
                        BP(probe=False, trialInterval=(2, 7)),
                        BP(probe=False, trialInterval=(
                            2, 7), inclusionFlags="homeTrial"),
                        BP(probe=False, trialInterval=(
                            2, 7), inclusionFlags="awayTrial"),
                        BP(probe=False, trialInterval=(10, 15)),
                        BP(probe=False, trialInterval=(
                            10, 15), inclusionFlags="homeTrial"),
                        BP(probe=False, trialInterval=(
                            10, 15), inclusionFlags="awayTrial"),
                        BP(probe=False, trialInterval=(0, 1)),
                        BP(probe=False, trialInterval=(2, 3)),
                        BP(probe=False, trialInterval=(3, 4)),
                        BP(probe=True),
                        BP(probe=True, timeInterval=(0, 60)),
                        BP(probe=True, timeInterval=(0, 120)),
                        BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                        BP(probe=True, inclusionFlags="moving"),
                        BP(probe=True, inclusionFlags="offWall"),
                        BP(probe=True, inclusionFlags=["offWall", "moving"]),
                        BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                        inclusionFlags="moving"),
                        BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                        inclusionFlags="offWall"),
                        BP(probe=True, timeInterval=BTSession.fillTimeInterval,
                        inclusionFlags=["offWall", "moving"]),
                        ]
    allConsideredSmoothDists = [0, 0.5, 1]
    allConsideredRadii = [0.25, 0.5, 1, 1.5]
    basicParams = {
        "bp": allConsideredBPs,
        "smoothDist": allConsideredSmoothDists,
    }

    def isConvex(bp: BP):
        return bp.inclusionArray is None and bp.inclusionFlags is None and bp.moveThreshold is None
    allConvexBPs = [bp for bp in allConsideredBPs if isConvex(bp)]

    if specName == "measureFromFunc":
        return {
            "radius": allConsideredRadii,
            **basicParams
        }
    else:
        raise ValueError(f"Unrecognized specName: {specName}")


def lookAtShuffles(specName, func, testData=False, filters: Optional[Callable[[Dict], bool]] = None,
                   plotShuffles=True, plotCorrelations=True):
    specParams = getParamsForName(specName, func)
    dataDir = findDataDir()
    outputFileName = os.path.join(
        dataDir, f"{specName}{'' if func is None else '_' + func.__name__}{'_test' if testData else ''}.h5")

    # get all combinations of parameters in specParams
    paramValues = list(specParams.values())
    paramValueCombos = list(product(*paramValues))
    correlationNames = [
        "numStims",
        "numRipplesPreStats",
        "numRipplesProbeStats",
        "stimRate",
        "rippleRatePreStats",
        "rippleRateProbeStats",
    ]

    if os.path.exists(outputFileName):
        print(f"Output file {outputFileName} already exists!")
    else:
        print(f"Output file {outputFileName} does not exist, creating it now...")
        print(f"Gathering files for {len(paramValueCombos)} param combos")

        savedStatsNames = []

        epvc = enumerate(paramValueCombos)
        for ci, combo in tqdm(epvc, total=len(paramValueCombos), smoothing=0):
            if testData and ci > 10:
                break
            params = dict(zip(specParams.keys(), combo))
            # pprint(params)
            name = getNameFromParams(specName, params, func, None)
            noCorrStatsDir = findStatsDir(os.path.join(
                "figures", "GreatBigDatamining_datamined", "B17", specName), name)
            if noCorrStatsDir is None:
                continue

            # print(f"Found stats dir: {noCorrStatsDir}")
            statsFiles = glob(os.path.join(noCorrStatsDir, "*.h5"))
            for sf in statsFiles:
                savedStatsNames.append((f"{name}_{os.path.basename(sf)}", os.path.abspath(sf)))
            assert noCorrStatsDir.endswith("/stats")

            for correlationName in correlationNames:
                name = getNameFromParams(specName, params, func, correlationName)
                # print(f"Running {name}")
                likelyStatsDir = noCorrStatsDir[:-6] + f"_X_{correlationName}/stats"
                if not os.path.exists(likelyStatsDir):
                    statsDir = findStatsDir(os.path.join(
                        "figures", "GreatBigDatamining_datamined", "B17", specName), name)
                    # print(f"Found stats dir: {statsDir}")
                else:
                    statsDir = likelyStatsDir
                    # print(f"Found expected stats dir: {likelyStatsDir}")
                statsFiles = glob(os.path.join(statsDir, "*.h5"))
                for sf in statsFiles:
                    savedStatsNames.append((f"{name}_{os.path.basename(sf)}", os.path.abspath(sf)))

        # print a random sample of the saved stats names
        # print("Sample of saved stats names:")
        # for i in range(10):
        #     print(random.choice(savedStatsNames))

        shuffler = Shuffler()
        input("Press enter to start shuffling (output file: {})".format(outputFileName))

        numShuffles = 9 if testData else 100
        shuffler.runAllShuffles(None, numShuffles=numShuffles,
                                savedStatsFiles=savedStatsNames, outputFileName=outputFileName,
                                justGlobal=True)

    if plotShuffles:
        significantShuffles = pd.read_hdf(outputFileName, key="significantShuffles")
        significantShuffles = significantShuffles[["plot", "shuffle", "pval", "direction"]]
        significantShuffles = significantShuffles[significantShuffles["shuffle"].str.startswith(
            "[GLO") & ~ significantShuffles["plot"].str.contains("_X_")]
        # significantShuffles = significantShuffles[significantShuffles["plot"].str.endswith("diff.h5")]

        paramsByPlotData = []
        for combo in paramValueCombos:
            params = dict(zip(specParams.keys(), combo))
            name = getNameFromParams(specName, params, func, None)
            params["bp"] = params["bp"].conciseString()
            paramsByPlotData.append((name + "_measureByCondition.h5", *params.values(), "", ""))
            # away control
            paramsByPlotData.append((name + "_ctrl_away.h5", *params.values(), "away", ""))
            paramsByPlotData.append((name + "_ctrl_away_cond.h5", *params.values(), "away", "cond"))
            paramsByPlotData.append((name + "_ctrl_away_diff.h5", *params.values(), "away", "diff"))
            # later sessions control
            paramsByPlotData.append((name + "_ctrl_latersessions.h5",
                                    *params.values(), "later", ""))
            paramsByPlotData.append((name + "_ctrl_latersessions_cond.h5",
                                    *params.values(), "later", "cond"))
            paramsByPlotData.append((name + "_ctrl_latersessions_diff.h5",
                                    *params.values(), "later", "diff"))
            # next session control
            paramsByPlotData.append((name + "_ctrl_nextsession.h5", *params.values(), "next", ""))
            paramsByPlotData.append((name + "_ctrl_nextsession_cond.h5",
                                    *params.values(), "next", "cond"))
            paramsByPlotData.append((name + "_ctrl_nextsession_diff.h5",
                                    *params.values(), "next", "diff"))
            # other sessions control
            paramsByPlotData.append((name + "_ctrl_othersessions.h5",
                                    *params.values(), "other", ""))
            paramsByPlotData.append((name + "_ctrl_othersessions_cond.h5",
                                    *params.values(), "other", "cond"))
            paramsByPlotData.append((name + "_ctrl_othersessions_diff.h5",
                                    *params.values(), "other", "diff"))
            # symmetric control
            paramsByPlotData.append((name + "_ctrl_symmetric.h5", *
                                    params.values(), "symmetric", ""))
            paramsByPlotData.append((name + "_ctrl_symmetric_cond.h5", *
                                    params.values(), "symmetric", "cond"))
            paramsByPlotData.append((name + "_ctrl_symmetric_diff.h5", *
                                    params.values(), "symmetric", "diff"))

        paramsByPlot = pd.DataFrame(paramsByPlotData, columns=[
                                    "plot", *specParams.keys(), "ctrlName", "suffix"])

        # If ctrlName is empty, suffix should be empty too
        badFlags = (paramsByPlot["ctrlName"] == "") & (paramsByPlot["suffix"] != "")
        # print(f"Bad flags: {badFlags.sum()}")
        # print(paramsByPlot[badFlags])
        assert badFlags.sum() == 0
        # Now add these columns to the significant shuffles dataframe, merging on the plot name
        significantShuffles = significantShuffles.merge(paramsByPlot, on="plot")
        significantShuffles["shuffleCategory"] = significantShuffles["shuffle"].str.split(
            " ", expand=True)[1]
        significantShuffles["shuffleCategoryValue"] = significantShuffles["shuffle"].str.split(
            ")", expand=True)[0].str.split("(", expand=True)[1]

        # possibly interesting things to look for:
        # effect with no suffix or control with condition shuffle, value SWR
        significantShuffles["flag"] = (significantShuffles["suffix"] == "") & \
            (significantShuffles["ctrlName"] == "") & \
            (significantShuffles["shuffleCategory"] == "condition") & \
            (significantShuffles["shuffleCategoryValue"] == "SWR")

        # _diff plot shows effect of condition
        significantShuffles["flag"] = (significantShuffles["flag"]) | \
            (significantShuffles["suffix"] == "diff") & \
            (significantShuffles["shuffleCategory"] == "condition") & \
            (significantShuffles["shuffleCategoryValue"] == "SWR")
        # _ctrl plot shows effect of val vs ctrl. For consistency, shuffle value should be home or same
        significantShuffles["flag"] = (significantShuffles["flag"]) | \
            (significantShuffles["suffix"] == "") & \
            (significantShuffles["ctrlName"] != "") & \
            (significantShuffles["shuffleCategoryValue"].isin(
                ["home", "same"]))
        # _ctrl_cond plot shows effect of cond  -  Don't need ot look at interaction, already taken care of in diff plot
        significantShuffles["flag"] = (significantShuffles["flag"]) | \
            (significantShuffles["suffix"] == "cond") & \
            (significantShuffles["ctrlName"] != "") & \
            (significantShuffles["shuffleCategory"] == "condition") & \
            (significantShuffles["shuffleCategoryValue"] == "SWR")

        significantShuffles = significantShuffles[significantShuffles["flag"]]
        # can drop the flag column now
        significantShuffles = significantShuffles.drop(
            columns=["flag", "shuffleCategoryValue", "shuffleCategory"])
        print(significantShuffles.head(20))
        print(significantShuffles.shape)

        # All column names except for plot, shuffle, pval, and direction are parameters
        allParams = significantShuffles.drop(columns=["plot", "shuffle", "pval", "direction"])
        # Separate into numerical columns and categorical columns
        numericalParams = allParams.select_dtypes(include=np.number)
        categoricalParams = allParams.select_dtypes(exclude=np.number)
        # print these out to confirm
        print(numericalParams.columns)
        print(categoricalParams.columns)

        # For each combination of values for the categorical params, plot the effect of changing the numerical params on the pvalues
        categoricalCombinations = list(itertools.product(
            *[categoricalParams[col].unique() for col in categoricalParams.columns]))
        for combo in categoricalCombinations:
            if filters is not None:
                # If none of the filters are met, skip this combination
                found = False
                for filter in filters:
                    match = True
                    for k, v in filter.items():
                        if k == "bp":
                            v = v.conciseString()
                        if combo[categoricalParams.columns.get_loc(k)] != v:
                            match = False
                            break
                    if match:
                        found = True
                        break
                if not found:
                    continue

            # get a df that's just the rows with this combination of values
            comboDf = significantShuffles.copy()
            # If this is empty, skip it
            for i, col in enumerate(categoricalParams.columns):
                comboDf = comboDf[comboDf[col] == combo[i]]
            if comboDf.shape[0] == 0:
                continue

            fig, axs = plt.subplots(1, len(numericalParams.columns), figsize=(10, 3))
            for i, col in enumerate(numericalParams.columns):
                ax = axs[i]
                assert isinstance(ax, plt.Axes)
                ax.set_title(col)

                # Set the min and max of the y axis according to the min and max of the pvalues
                maxPval = significantShuffles["pval"].max()
                ax.set_ylim(-0.01, maxPval * 1.1)
                # Set the min and max of the x axis according to the min and max of the numerical param
                minX = significantShuffles[col].min()
                if minX > 0:
                    minX = minX * 0.9
                elif minX < 0:
                    minX = minX * 1.1
                else:
                    minX = -0.1
                maxX = significantShuffles[col].max()
                if maxX > 0:
                    maxX = maxX * 1.1
                elif maxX < 0:
                    maxX = maxX * 0.9
                else:
                    maxX = 0.1
                ax.set_xlim(minX, maxX)
                # Set the x ticks to be the unique values of the numerical param
                ax.set_xticks(significantShuffles[col].unique())

                # get all the unique values for all the other numerical params
                otherNumericalParams = numericalParams.columns.drop(col)
                otherNumericalParamValues = [numericalParams[col].unique()
                                             for col in otherNumericalParams]
                # get all the combinations of values for all the other numerical params
                otherNumericalParamCombinations = list(
                    itertools.product(*otherNumericalParamValues))
                # for each combination of values for all the other numerical params, plot the pvalue for this numerical param
                for onci, otherNumCombo in enumerate(otherNumericalParamCombinations):
                    # get a df that's just the rows with this combination of values
                    otherNumComboDf = comboDf.copy()
                    for j, otherNumCol in enumerate(otherNumericalParams):
                        otherNumComboDf = otherNumComboDf[otherNumComboDf[otherNumCol]
                                                          == otherNumCombo[j]]
                    # plot the pvalue for this numerical param, but add some jitter to the x axis
                    # so that the points don't overlap. Depending on the value of the direction column,
                    # the points will be circles or x's
                    # jitter = np.random.uniform(-1, 1, len(otherNumComboDf)) * 0.01
                    jitter = onci * (maxX - minX) * 0.01
                    xvals = (otherNumComboDf[col] + jitter).to_numpy()
                    yvals = (otherNumComboDf["pval"]).to_numpy()
                    direction = otherNumComboDf["direction"].to_numpy()

                    # Choose a color here so both plots are the same color
                    color = next(ax._get_lines.prop_cycler)["color"]

                    ax.plot(xvals[direction], yvals[direction], "o",
                            label=str(otherNumCombo), color=color)
                    ax.plot(xvals[~direction], yvals[~direction], "x", color=color)

                ax.legend()

            # Add a title to the whole figure
            fig.suptitle(
                "    ".join([f"{col}={val}" for col, val in zip(categoricalParams.columns, combo)]))

            plt.tight_layout()
            plt.show()

    if plotCorrelations:
        significantCorrelations = pd.read_hdf(outputFileName, key="significantCorrelations")
        #    columns=["plot", "categories", "correlation", "pval", "direction", "measure", "plotSuffix"]
        # Possibly interesting things to see:
        # - Correlation between measue and any SM: plot ends in "measure"
        # - Difference in correlation between condition groups: plot ends in "measureByCondition", and one is in table and not other, or both in table but direciton is different
        # - Same but for measureVsCtrl: plot ends in "ctrl_{ctrlName}"
        # - Correlation of diff measure and any SM: plot ends in "ctrl_{ctrlName}_diff"
        # - Correlation of diff measure in one conditoin and not another: plot ends in "ctrl_{ctrlName}_diff_byCond"
        print(significantCorrelations.head(10).to_string())

        paramsByPlotData = []
        for combo in paramValueCombos:
            for sm in correlationNames:
                params = dict(zip(specParams.keys(), combo))
                name = getNameFromParams(specName, params, func, sm)
                params["bp"] = params["bp"].conciseString()

                paramsByPlotData.append([(name + "_measure.h5", * params.values(), "", "", sm)])
                paramsByPlotData.append(
                    [(name + "_measureByCondition.h5", * params.values(), "", "", sm)])
                # TODO fill this in
                # Just copied from above
                # .....
                # .....
                # .....
                paramsByPlotData.append((name + "_ctrl_symmetric_diff.h5", *
                                        params.values(), "symmetric", "diff"))

        paramsByPlot = pd.DataFrame(paramsByPlotData, columns=[
                                    "plot", *specParams.keys(), "ctrlName", "suffix", "correlationName"])


def getChosenParams(specName, func) -> List[Dict[str, Any]]:
    if specName == "measureFromFunc" and func == BTSession.totalTimeAtPosition:
        chosenParams = [
            {
                "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(2, 7)),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(2, 7)),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags=["offWall", "moving"]),
                "radius": 0.25,
                "smoothDist": 0,
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags=["offWall", "moving"]),
                "radius": 0.5,
                "smoothDist": 0,
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                "radius": 0.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False),
                "radius": 1,
                "smoothDist": 0
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3),
                "radius": 1,
                "smoothDist": 0
            },
            {
                "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3),
                "radius": 1,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3, trialInterval=(2, 7)),
                "radius": 1,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3, trialInterval=(2, 7)),
                "radius": 1.5,
                "smoothDist": 1.0
            },
            {
                "bp": BP(probe=False, erode=3, trialInterval=(0, 1)),
                "radius": 0.5,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(10, 15)),
                "radius": 0.5,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(10, 15)),
                "radius": 0.5,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3, trialInterval=(10, 15)),
                "radius": 0.5,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="explore"),
                "radius": 0.5,
                "smoothDist": 0.5
            },
        ]
    else:
        raise Exception("No chosen params for this specName and func")

    return chosenParams


def generateChosenPlots(specName, func, chosenParams):
    ratName = "B17"
    ratInfo = getLoadInfo(ratName)
    dataFilename = os.path.join(ratInfo.output_dir, ratInfo.out_filename)
    print("loading from " + dataFilename)
    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    sessions = ratData.getSessions()

    infoFileName = datetime.now().strftime("chosenB17Plots_%Y%m%d_%H%M%S.txt")
    dataDir = findDataDir()
    outputDir = "chosenB17Plots"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=3)

    for params in chosenParams:
        if specName == "measureFromFunc":
            bp = params["bp"]
            radius = params["radius"]
            smoothDist = params["smoothDist"]
            lm = LocationMeasure(f"{func.__name__}_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: func(sesh, bp, pos, radius),
                                 ), sessions, smoothDist=smoothDist)
            lm.makeFigures(pp, everySessionBehaviorPeriod=bp, excludeFromCombo=True)


if __name__ == "__main__":
    specName = "measureFromFunc"
    filters = [
        # {
        #     "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(2, 7)),
        #     "ctrlName": "symmetric",
        #     "suffix": "diff"
        # },
    ]
    # filters = None
    # for func in [BTSession.numVisitsToPosition,
    #  BTSession.avgDwellTimeAtPosition,
    #  BTSession.avgCurvatureAtPosition]:
    # func = BTSession.totalTimeAtPosition
    # func = BTSession.numVisitsToPosition
    func = BTSession.avgDwellTimeAtPosition
    # func = BTSession.avgCurvatureAtPosition
    lookAtShuffles(specName, func, filters=filters, plotCorrelations=False, plotShuffles=False)

    # chosenParams = getChosenParams(specName, func)
    # generateChosenPlots(specName, func, chosenParams)
