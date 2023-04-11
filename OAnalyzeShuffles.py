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
from functools import partial

from Shuffler import Shuffler
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from UtilFunctions import findDataDir, getLoadInfo
from tqdm import tqdm
from MeasureTypes import LocationMeasure, SessionMeasure, TimeMeasure
from BTData import BTData
from PlotUtil import PlotManager
from GreatBigDatamining import avgNumWellsVisitedPerPeriod, speedFunc, fracExploreFunc, fracOffWallFunc, \
    durationFunc, pathOptimalityFunc, pathLengthFunc, avgDurationPerPeriod, avgPathLengthPerPeriod, numWellsVisitedFunc
from consts import allWellNames, offWallWellNames


def isConvex(bp: BP):
    return bp.inclusionArray is None and bp.inclusionFlags is None and bp.moveThreshold is None


def getNameFromParams(name, params, func, correlationName):
    if name == "measureFromFunc":
        bp = params["bp"]
        radius = params["radius"]
        smoothDist = params["smoothDist"]
        ret = f"LM_{func.__name__}_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}"
        if correlationName is not None:
            ret += f"_X_{correlationName}"
        return ret
    elif name == "makeNumWellsVisitedMeasure":
        mode = params["mode"]
        inProbe = params["inProbe"]
        countRepeats = params["countRepeats"]
        offWallOnly = params["offWallOnly"]
        probeStr = "probe" if inProbe else "bt"
        return f"SM_numWellsVisitedPer{mode.capitalize()}_{probeStr}_cr{countRepeats}_ow{offWallOnly}"
    elif name == "makeCoarseTimeMeasure":
        inProbe = params["inProbe"]
        probeStr = "probe" if inProbe else "bt"
        return f"TiM_{func.__name__}_{probeStr}"
    elif name == "makeTimePeriodsMeasure":
        timePeriodsGenerator = params["timePeriodsGenerator"]
        return f"TiM_{func.__name__}_{timePeriodsGenerator[1]}"
    elif name == "makePerPeriodMeasure":
        inProbe = params["inProbe"]
        probeStr = "probe" if inProbe else "bt"
        mode = params["mode"]
        return f"SM_{func.__name__}_{probeStr}_{mode}"
    elif name == "makeTotalDurationMeasure":
        inProbe = params["inProbe"]
        probeStr = "probe" if inProbe else "bt"
        return f"SM_totalDuration_{probeStr}"
    elif name == "makeNumWellsVisitedTimeMeasure":
        countReturns = params["countReturns"]
        offWallWells = params["offWallWells"]
        timePeriodsGenerator = params["timePeriodsGenerator"]
        return f"TiM_numWellsVisited_cr{countReturns}_ow{offWallWells}_{timePeriodsGenerator[1]}"
    elif name == "makeGravityMeasure":
        bp = params["bp"]
        passRadius = params["passRadius"]
        visitRadiusFactor = params["visitRadiusFactor"]
        passDenoiseFactor = params["passDenoiseFactor"]
        smoothDist = params["smoothDist"]
        return f"LM_gravity_{bp.filenameString()}_pr{passRadius:.2f}_vrf{visitRadiusFactor:.2f}_df{passDenoiseFactor:.2f}_sd{smoothDist:.2f}"
    elif name == "makeDotProdMeasure":
        bp = params["bp"]
        distanceWeight = params["distanceWeight"]
        normalize = params["normalize"]
        smoothDist = params["smoothDist"]
        onlyPositive = params["onlyPositive"]
        return f"LM_dotprod_{bp.filenameString()}_dw{distanceWeight:.2f}_n{normalize}_sd={smoothDist:.2f}_op{onlyPositive}"

    else:
        raise ValueError(f"Unrecognized name: {name}")


def findStatsDir(subDir: str, name: str):
    possibleDrives = [
        "/media/Harold/",
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
    validRets = []
    foundError = False
    for pl in possibleLocations:
        checkFileName = os.path.join(pl, name, "processed.txt")
        if os.path.exists(checkFileName):
            validRets.append(os.path.join(pl, name, "stats"))
        checkFileName = os.path.join(pl, name, "corr_processed.txt")
        if os.path.exists(checkFileName):
            validRets.append(os.path.join(pl, name, "stats"))
        checkFileName = os.path.join(pl, name, "error.txt")
        if os.path.exists(checkFileName):
            print(f"Found error file for {name} at {checkFileName}")
            foundError = True
    if len(validRets) > 0:
        # Return the directory with the most files
        maxLen = 0
        maxDir = None
        for vr in validRets:
            if len(os.listdir(vr)) > maxLen:
                maxLen = len(os.listdir(vr))
                maxDir = vr
        return maxDir
    if foundError:
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

    allConvexBPs = [bp for bp in allConsideredBPs if isConvex(bp)]
    allConsideredTimePeriods = {"timePeriodsGenerator":
                                [(TimeMeasure.trialTimePeriodsFunction(), "trial"),
                                 (TimeMeasure.trialTimePeriodsFunction((2, 6)), "trial26"),
                                 (TimeMeasure.trialTimePeriodsFunction((2, 4)), "trial24"),
                                 (TimeMeasure.earlyVsLateTrials(), "evl"),
                                 (TimeMeasure.earlyVsLateTrials((0, 1), (2, 3)), "evl0123")]}

    if specName == "measureFromFunc":
        return {
            "radius": allConsideredRadii,
            **basicParams
        }
    elif specName == "makeNumWellsVisitedMeasure":
        return {
            "countRepeats": [True, False],
            "offWallOnly": [True, False],
            "inProbe": [True, False],
            "mode": ["bout", "excursion"]
        }
    elif specName == "makeCoarseTimeMeasure":
        return {
            "inProbe": [True, False]
        }
    elif specName == "makeTimePeriodsMeasure":
        return {
            **allConsideredTimePeriods
        }
    elif specName == "makePerPeriodMeasure":
        return {
            "mode": ["bout", "excursion"],
            "inProbe": [True, False]
        }
    elif specName == "makeTotalDurationMeasure":
        return {
            "inProbe": [True, False]
        }
    elif specName == "makeNumWellsVisitedTimeMeasure":
        return {
            "countReturns": [True, False],
            "offWallWells": [True, False],
            **allConsideredTimePeriods
        }
    elif specName == "makeGravityMeasure":
        return {
            "passRadius": np.linspace(0.5, 1.5, 3),
            "visitRadiusFactor": np.linspace(0.2, 0.4, 2),
            "passDenoiseFactor": [1.25],
            "bp": allConsideredBPs,
            "smoothDist": [0, 0.5]
        }
    elif specName == "makeDotProdMeasure":
        return {
            "distanceWeight": np.linspace(-1, 1, 5),
            "normalize": [True, False],
            "onlyPositive": [True, False],
            "bp": [
                BP(probe=False, inclusionFlags="awayTrial"),
                BP(probe=False, inclusionFlags=["awayTrial", "moving"]),
                BP(probe=False, inclusionFlags="moving"),
                BP(probe=False, inclusionFlags="awayTrial", erode=3),
                BP(probe=False, inclusionFlags=["awayTrial", "moving"], erode=3),
                BP(probe=False, inclusionFlags="moving", erode=3),
                BP(probe=True),
                BP(probe=True, timeInterval=BTSession.fillTimeInterval),
            ],
            "smoothDist": allConsideredSmoothDists,
        }

    else:
        raise ValueError(f"Unrecognized specName: {specName}")


def lookAtShuffles(specName, func, testData=False, filters: Optional[Callable[[Dict], bool]] = None,
                   plotShuffles=True, plotCorrelations=True, appendCorrelations=False, forceRecompute=False):
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

    if os.path.exists(outputFileName) and not appendCorrelations and not forceRecompute:
        print(f"Output file {outputFileName} already exists!")
    else:
        if not os.path.exists(outputFileName):
            print(f"Output file {outputFileName} does not exist, creating it now...")
        elif forceRecompute:
            print(f"Recomputing {outputFileName}")
            os.remove(outputFileName)
        else:
            print("appending to existing file")
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
                statsDir = findStatsDir(os.path.join(
                    "figures", "GreatBigDatamining_datamined", "B17", specName), name)

                # Can't do the below because sometimes it exists but was canceled so some files are missing
                # The above will find the directory with the most files in it
                # likelyStatsDir = noCorrStatsDir[:-6] + f"_X_{correlationName}/stats"
                # # check if it exists and there are 22 files in it
                # if not os.path.exists(likelyStatsDir) or len(glob(os.path.join(likelyStatsDir, "*.h5"))) != 22:
                #     statsDir = findStatsDir(os.path.join(
                #         "figures", "GreatBigDatamining_datamined", "B17", specName), name)
                #     # print(f"Found stats dir: {statsDir}")
                # else:
                #     statsDir = likelyStatsDir
                #     # print(f"Found expected stats dir: {likelyStatsDir}")
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
                                justGlobal=True, skipShuffles=appendCorrelations)

    if plotShuffles:
        significantShuffles = pd.read_hdf(outputFileName, key="significantShuffles")
        if "pvalIndex" in significantShuffles.columns:
            significantShuffles = significantShuffles[[
                "plot", "shuffle", "pval", "direction", "pvalIndex"]]
        else:
            significantShuffles = significantShuffles[["plot", "shuffle", "pval", "direction"]]
        significantShuffles = significantShuffles[significantShuffles["shuffle"].str.startswith(
            "[GLO") & ~ significantShuffles["plot"].str.contains("_X_")]
        # significantShuffles = significantShuffles[significantShuffles["plot"].str.endswith("diff.h5")]

        paramsByPlotData = []
        for combo in paramValueCombos:
            params = dict(zip(specParams.keys(), combo))
            name = getNameFromParams(specName, params, func, None)
            if specName in ["measureFromFunc", "makeGravityMeasure", "makeDotProdMeasure"]:
                if "bp" in params:
                    bpIsConvex = isConvex(params["bp"])
                    params["bp"] = params["bp"].conciseString()
                else:
                    bpIsConvex = False
                paramsByPlotData.append((name + "_measureByCondition.h5", *
                                        params.values(), "", "", bpIsConvex))
                # away control
                paramsByPlotData.append(
                    (name + "_" + name[3:] + "_ctrl_away.h5", *params.values(), "away", "", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_away_cond.h5", *
                                        params.values(), "away", "cond", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_away_diff.h5", *
                                        params.values(), "away", "diff", bpIsConvex))
                # later sessions control
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_latersessions.h5",
                                        *params.values(), "later", "", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_latersessions_cond.h5",
                                        *params.values(), "later", "cond", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_latersessions_diff.h5",
                                        *params.values(), "later", "diff", bpIsConvex))
                # next session control
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_nextsession.h5", *
                                        params.values(), "next", "", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_nextsession_cond.h5",
                                        *params.values(), "next", "cond", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_nextsession_diff.h5",
                                        *params.values(), "next", "diff", bpIsConvex))
                # next session control aways excluded
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_nextsessionnoaway.h5", *
                                        params.values(), "next", "", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_nextsessionnoaway_cond.h5",
                                        *params.values(), "next", "cond", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_nextsessionnoaway_diff.h5",
                                        *params.values(), "next", "diff", bpIsConvex))
                # other sessions control
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_othersessions.h5",
                                        *params.values(), "other", "", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_othersessions_cond.h5",
                                        *params.values(), "other", "cond", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_othersessions_diff.h5",
                                        *params.values(), "other", "diff", bpIsConvex))
                # symmetric control
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_symmetric.h5", *
                                        params.values(), "symmetric", "", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_symmetric_cond.h5", *
                                        params.values(), "symmetric", "cond", bpIsConvex))
                paramsByPlotData.append((name + "_" + name[3:] + "_ctrl_symmetric_diff.h5", *
                                        params.values(), "symmetric", "diff", bpIsConvex))

            elif specName in ["makeNumWellsVisitedMeasure", "makeTotalDurationMeasure", "makePerPeriodMeasure"]:
                paramsByPlotData.append(
                    (name + "_" + name[3:] + ".h5", *params.values(), "", "", False))
            elif specName in ["makeCoarseTimeMeasure", "makeTimePeriodsMeasure", "makeNumWellsVisitedTimeMeasure"]:
                paramsByPlotData.append(
                    (name + "_" + name[4:] + ".h5", *params.values(), "", "", False))
                paramsByPlotData.append(
                    (name + "_" + name[4:] + "_avgs_byCond.h5", *params.values(), "", "avgs_byCond", False))

        paramsByPlot = pd.DataFrame(paramsByPlotData, columns=[
            "plot", *specParams.keys(), "ctrlName", "suffix", "bpIsConvex"])
        # print(significantShuffles.head().to_string())
        # print(paramsByPlot.head().to_string())
        significantShuffles = significantShuffles.merge(paramsByPlot, on="plot")
        # print(significantShuffles.head().to_string())

        if specName in ["measureFromFunc", "makeGravityMeasure", "makeDotProdMeasure"]:
            # If ctrlName is empty, suffix should be empty too
            badFlags = (paramsByPlot["ctrlName"] == "") & (paramsByPlot["suffix"] != "")
            # print(f"Bad flags: {badFlags.sum()}")
            # print(paramsByPlot[badFlags])
            assert badFlags.sum() == 0
            # Now add these columns to the significant shuffles dataframe, merging on the plot name
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
            if "bp" in significantShuffles.columns:
                # Although let's ignore the ones where ctrlName is away or symmetric and isn't during the probe
                significantShuffles["flag"] = (significantShuffles["flag"]) & \
                    ((significantShuffles["suffix"] != "") |
                     ((significantShuffles["ctrlName"] != "away") &
                     (significantShuffles["ctrlName"] != "symmetric")) |
                     (significantShuffles["bp"].str.contains("probe")))
            # # _ctrl_cond plot shows effect of cond  -  Don't need ot look at interaction, already taken care of in diff plot
            # significantShuffles["flag"] = (significantShuffles["flag"]) | \
            #     (significantShuffles["suffix"] == "cond") & \
            #     (significantShuffles["ctrlName"] != "") & \
            #     (significantShuffles["shuffleCategory"] == "condition") & \
            #     (significantShuffles["shuffleCategoryValue"] == "SWR")

            if "bp" in significantShuffles.columns:
                # If a plot name has erode in it, that's the one we care about, can get rid of similar plot name without erode
                hasErode = significantShuffles["bp"].str.contains("erode")
                # Get list of plot names with erode
                erodePlots = significantShuffles[hasErode]["bp"].unique()
                # Now map each of those plot names to the equialent plot name without erode
                noErodeNames = [plot.replace(" erode 3", "").replace("_erode_3", "")
                                for plot in erodePlots]
                # Now get rid of rows in significantShuffles that have a plot name that's in noErodeNames
                significantShuffles["flag"] = significantShuffles["flag"] & ~(significantShuffles["bp"].isin(
                    noErodeNames))

            if func == BTSession.avgDwellTimeAtPosition or func == BTSession.numVisitsToPosition:
                significantShuffles["flag"] = significantShuffles["flag"] & significantShuffles["bpIsConvex"]

            significantShuffles = significantShuffles[significantShuffles["flag"]]
            # can drop the flag column now
            significantShuffles = significantShuffles.drop(
                columns=["flag", "bpIsConvex"])
            # columns=["flag", "shuffleCategoryValue", "shuffleCategory", "bpIsConvex"])
            # print(significantShuffles.head(20))
            # print(significantShuffles.shape)

            # lets also drop columns with only one value
            significantShuffles = significantShuffles.drop(
                columns=significantShuffles.columns[significantShuffles.nunique() == 1])

        # All column names except for plot, shuffle, pval, and direction are parameters
        allParams = significantShuffles.drop(
            columns=["plot", "shuffle", "pval", "direction", "shuffleCategory", "shuffleCategoryValue"])
        # print("All params: ")
        # print(allParams.head().to_string())
        # Separate into numerical columns and categorical columns
        numericalParams = allParams.select_dtypes(include=np.number)
        if len(numericalParams.columns) == 0:
            # add a dummy column of all 1's
            numericalParams = pd.DataFrame(np.ones(
                (allParams.shape[0], 1)), columns=["dummy"], index=allParams.index)
            significantShuffles = significantShuffles.join(numericalParams)
        categoricalParams = allParams.select_dtypes(exclude=np.number)
        # print these out to confirm
        # print(numericalParams.columns)
        # print(categoricalParams.columns)

        if specName in ["measureFromFunc", "makeGravityMeasure", "makeDotProdMeasure"]:
            def acceptComboDf(comboDf):
                if comboDf.shape[0] <= 3:
                    return False
                return True
        else:
            def acceptComboDf(comboDf):
                if comboDf.shape[0] == 0:
                    return False
                return True

        # For each combination of values for the categorical params, plot the effect of changing the numerical params on the pvalues
        categoricalCombinations = list(itertools.product(
            *[categoricalParams[col].unique() for col in categoricalParams.columns]))

        numWithData = 0
        for combo in categoricalCombinations:
            # get a df that's just the rows with this combination of values
            comboDf = significantShuffles.copy()
            # If this is empty, skip it
            for i, col in enumerate(categoricalParams.columns):
                comboDf = comboDf[comboDf[col] == combo[i]]
            # print("Combo: ", combo)
            # print("Combo df shape: ", comboDf.shape)
            # print(comboDf)
            if not acceptComboDf(comboDf):
                continue
            numWithData += 1
        print("Number of combinations with data: ", numWithData)
        print("Number of combinations total: ", len(categoricalCombinations))

        waitForCategory = None
        # waitForCategory = "Probe"
        waitForNextCategory = False
        # lastCategory = None
        numSkipped = 0
        for combo in categoricalCombinations:
            if waitForCategory is not None:
                if combo[0] == waitForCategory:
                    lastCategory = waitForCategory
                    waitForCategory = None
                    print("Number of combinations skipped: ", numSkipped)
                else:
                    numSkipped += 1
                    continue
            if waitForNextCategory:
                if combo[0] != lastCategory:
                    waitForNextCategory = False
                    print("Number of combinations skipped: ", numSkipped)
                else:
                    numSkipped += 1
                    continue
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
            if not acceptComboDf(comboDf):
                continue

            # "shuffleCategoryValue", "shuffleCategory",
            # Value in shuffleCategoryValue and shuffleCategory should be the same for all rows of comboDF, print the first one
            print("Plotting for combination: ", combo,
                  "shuffleCategory: ", comboDf["shuffleCategory"].unique(),
                  "shuffleCategoryValue: ", comboDf["shuffleCategoryValue"].unique())
            # print("shuffle column: ", comboDf["shuffle"].unique())

            fig, axs = plt.subplots(1, len(numericalParams.columns), figsize=(20, 6))
            for i, col in enumerate(numericalParams.columns):
                if len(numericalParams.columns) == 1:
                    ax = axs
                else:
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
        print("Number of significant correlations: ", len(significantCorrelations))
        print(significantCorrelations.head().to_string())
        print(significantCorrelations.columns)
        try:
            info = pd.read_hdf(outputFileName, key="corrInfo")
            corrFuncVersion = info["corrFuncVersion"].iloc[0]
        except KeyError:
            info = None
            corrFuncVersion = 0.1

        #    columns=["plot", "categories", "correlation", "pval", "direction", "measure", "plotSuffix"]
        # Possibly interesting things to see:
        # - Correlation between measue and any SM: plot ends in "measure"
        # - Difference in correlation between condition groups: plot ends in "measureByCondition", and one is in table and not other, or both in table but direciton is different
        # - Same but for measureVsCtrl: plot ends in "ctrl_{ctrlName}"
        # - Correlation of diff measure in one conditoin and not another: plot ends in "ctrl_{ctrlName}_diff_byCond"
        # - Correlation of diff measure and any SM: plot ends in "ctrl_{ctrlName}_diff"
        significantCorrelations = significantCorrelations[[
            "plot", "categories", "correlation", "pval", "direction"]]
        # print(significantCorrelations.head().to_string())
        # print(significantCorrelations.columns)

        if specName in ["measureFromFunc", "makeGravityMeasure", "makeDotProdMeasure"]:
            ctrlNames = ["away", "symmetric", "latersessions",
                         "nextsession", "nextsessionnoaway", "othersessions"]
        else:
            ctrlNames = []
        paramsByPlotData = []
        for combo in paramValueCombos:
            for sm in correlationNames:
                params = dict(zip(specParams.keys(), combo))
                name = getNameFromParams(specName, params, func, sm)
                if "bp" in params:
                    bpIsConvex = isConvex(params["bp"])
                    params["bp"] = params["bp"].conciseString()
                else:
                    bpIsConvex = False

                paramsByPlotData.append(
                    (name + "_measure.h5", * params.values(), "", "measure", sm, name, bpIsConvex))
                paramsByPlotData.append((name + "_measureByCondition.h5", *
                                        params.values(), "", "measureByCondition", sm, name, bpIsConvex))
                for ctrlName in ctrlNames:
                    paramsByPlotData.append(
                        (name + f"_ctrl_{ctrlName}.h5", * params.values(), ctrlName, "ctrl", sm, name, bpIsConvex))
                    paramsByPlotData.append(
                        (name + f"_ctrl_{ctrlName}_byCond.h5", * params.values(), ctrlName, "ctrlByCond", sm, name, bpIsConvex))
                    paramsByPlotData.append(
                        (name + f"_ctrl_{ctrlName}_diff.h5", * params.values(), ctrlName, "ctrlDiff", sm, name, bpIsConvex))
                    paramsByPlotData.append(
                        (name + f"_ctrl_{ctrlName}_diff_byCond.h5", * params.values(), ctrlName, "ctrlDiffByCond", sm, name, bpIsConvex))

        paramsByPlot = pd.DataFrame(paramsByPlotData, columns=[
            "plot", *specParams.keys(), "ctrlName", "plotType", "correlationName", "measureName", "bpIsConvex"])
        significantCorrelations = significantCorrelations.merge(
            paramsByPlot, on="plot")
        # print("Significant correlations:")
        # print(significantCorrelations.head().to_string())
        # print(significantCorrelations.columns)
        # print(significantCorrelations[significantCorrelations["plotType"]
        #       == "measureByCondition"].head(100).to_string())
        # print(significantCorrelations.head(3))

        if corrFuncVersion == 0.1:
            # In first version, direction and pval were calculated incorrectly, so if direction is True the pval
            # is fine, but if direction is False, the pval is actually 1-pval, and can be discarded
            print("Fixing direction and pval for old version of correlation function")
            significantCorrelations = significantCorrelations[significantCorrelations["direction"]]
            significantCorrelations["direction"] = significantCorrelations["correlation"] > 0

        if specName in ["measureFromFunc", "makeGravityMeasure", "makeDotProdMeasure"]:
            # for measurebycondition, only want if there's one category but not the other, or if both categories but direction is different
            # to do this, group by plot column, and count the number of unique values in categories and direction
            # if there's only one unique value in categories, or there's two unique values in direction, then keep
            counts = significantCorrelations.groupby("measureName").agg(
                {"categories": lambda x: len(x.unique()), "direction": lambda x: len(x.unique())})
            # print(counts.head().to_string())
            counts["flag"] = (counts["categories"] == 1) | (counts["direction"] == 2)
            significantCorrelations["measureByConditionFlag"] = significantCorrelations["measureName"].map(
                counts["flag"])
            # print(significantCorrelations.head().to_string())

            significantCorrelations["flag"] = False
            # These will already by covered by the control plots
            # significantCorrelations["flag"] = significantCorrelations["plotType"] == "measure"
            significantCorrelations["flag"] = significantCorrelations["flag"] | (
                significantCorrelations["plotType"] == "ctrlDiff")
            significantCorrelations["flag"] = significantCorrelations["flag"] | (
                (significantCorrelations["plotType"] == "measureByCondition") & significantCorrelations["measureByConditionFlag"]) | (
                (significantCorrelations["plotType"] == "ctrl") & significantCorrelations["measureByConditionFlag"]) | (
                (significantCorrelations["plotType"] == "ctrlDiffByCond") & significantCorrelations["measureByConditionFlag"])

            if func == BTSession.totalTimeAtPosition:
                # Don't care about correlations with numRipplesProbeStats, numStims, or numRipplesPreStats
                significantCorrelations["flag"] = significantCorrelations["flag"] & (
                    significantCorrelations["correlationName"] != "numRipplesProbeStats")
                significantCorrelations["flag"] = significantCorrelations["flag"] & (
                    significantCorrelations["correlationName"] != "numStims")
                # significantCorrelations["flag"] = significantCorrelations["flag"] & (
                #     significantCorrelations["correlationName"] != "numRipplesPreStats")

            # Just gonna look at probe stats
            significantCorrelations["flag"] = significantCorrelations["flag"] & (
                significantCorrelations["correlationName"] != "rippleRatePreStats")
            significantCorrelations["flag"] = significantCorrelations["flag"] & (
                significantCorrelations["correlationName"] != "numRipplesPreStats")

            # If a plot name has erode in it, that's the one we care about, can get rid of similar plot name without erode
            hasErode = significantCorrelations["bp"].str.contains("erode")
            # Get list of plot names with erode
            erodePlots = significantCorrelations[hasErode]["bp"].unique()
            # Now map each of those plot names to the equialent plot name without erode
            noErodeNames = [plot.replace(" erode 3", "").replace("_erode_3", "")
                            for plot in erodePlots]
            # Now get rid of rows in significantCorrelations that have a plot name that's in noErodeNames
            significantCorrelations["flag"] = significantCorrelations["flag"] & ~(significantCorrelations["bp"].isin(
                noErodeNames))

            if func == BTSession.avgDwellTimeAtPosition or func == BTSession.numVisitsToPosition:
                significantCorrelations["flag"] = significantCorrelations["flag"] & significantCorrelations["bpIsConvex"]

            significantCorrelations = significantCorrelations[significantCorrelations["flag"]]
            # can drop flag column now
            significantCorrelations = significantCorrelations.drop(
                columns=["flag", "measureByConditionFlag", "bpIsConvex"])

        print("Significant correlations:")
        print(significantCorrelations.head().to_string())
        print(significantCorrelations.head())

        # columns not included in the following are parameters: plot, categories, correlation, pval, direction
        allParams = significantCorrelations.drop(columns=[
            "plot", "categories", "correlation", "pval", "direction", "measureName", "ctrlName"])
        print(allParams.columns)
        numericalParams = allParams.select_dtypes(include=np.number)
        categoricalParams = allParams.select_dtypes(exclude=np.number)
        # Reorder the columns in categoricalParams so the last two are at the start
        categoricalParams = categoricalParams[[
            categoricalParams.columns[-2], categoricalParams.columns[-1], *categoricalParams.columns[:-2]]]

        if specName in ["measureFromFunc", "makeGravityMeasure", "makeDotProdMeasure"]:
            def acceptComboDf(comboDf):
                if comboDf.shape[0] <= 3:
                    return False
                # If all values in correlation are small, skip it
                if np.all(np.abs(comboDf["correlation"]) < 0.1):
                    return False
                return True
        else:
            def acceptComboDf(comboDf):
                if comboDf.shape[0] == 0:
                    return False
                return True

        # For each combination of values for the categorical params, plot the effect of changing the numerical params on the pvalues
        categoricalCombinations = list(itertools.product(
            *[categoricalParams[col].unique() for col in categoricalParams.columns]))
        numWithData = 0
        for combo in categoricalCombinations:
            # get a df that's just the rows with this combination of values
            comboDf = significantCorrelations.copy()
            # If this is empty, skip it
            for i, col in enumerate(categoricalParams.columns):
                comboDf = comboDf[comboDf[col] == combo[i]]
            if not acceptComboDf(comboDf):
                continue
            numWithData += 1
        print("Number of combinations with data: ", numWithData)
        print("Number of combinations total: ", len(categoricalCombinations))

        # waitForCategory = None
        waitForCategory = "ctrlDiffByCond"
        waitForNextCategory = False
        # lastCategory = None
        numSkipped = 0
        lastCombo = None
        for combo in categoricalCombinations:
            if waitForCategory is not None:
                if combo[0] == waitForCategory:
                    lastCategory = waitForCategory
                    waitForCategory = None
                    print("Number of combinations skipped: ", numSkipped)
                else:
                    numSkipped += 1
                    continue
            if waitForNextCategory:
                if combo[0] != lastCategory:
                    waitForNextCategory = False
                    print("Number of combinations skipped: ", numSkipped)
                else:
                    numSkipped += 1
                    continue
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
            comboDf = significantCorrelations.copy()
            # If this is empty, skip it
            for i, col in enumerate(categoricalParams.columns):
                comboDf = comboDf[comboDf[col] == combo[i]]
            if not acceptComboDf(comboDf):
                continue

            if lastCombo is not None:
                if any([lastCombo[i] != combo[i] for i in range(len(lastCombo)-1)]):
                    print("New category")
                    # On the plt, put some text and show the plot
                    plt.text(0.5, 0.5, "New category", horizontalalignment='center',
                             verticalalignment='center', transform=plt.gca().transAxes)
                    plt.show()
            lastCombo = combo

            print("Plotting for combination: ", combo)
            # print(comboDf)

            # For each numerical parameter, plot the effect of changing it on the pvalue
            # in one subplot, and the effect on the correlation in another subplot
            fig, axes = plt.subplots(2, len(numericalParams.columns), figsize=(
                5 * len(numericalParams.columns), 10))
            for i, col in enumerate(numericalParams.columns):
                pvalAx = axes[0, i]
                assert isinstance(pvalAx, plt.Axes)
                pvalAx.set_title(col)
                jitteri = 0

                # Set the min and max of the y axis according to the min and max of the pvalues
                maxPval = significantCorrelations["pval"].max()
                pvalAx.set_ylim(-0.01, maxPval * 1.1)
                # Set the min and max of the x axis according to the min and max of the numerical param
                minX = significantCorrelations[col].min()
                if minX > 0:
                    minX = minX * 0.9
                elif minX < 0:
                    minX = minX * 1.1
                else:
                    minX = -0.1
                maxX = significantCorrelations[col].max()
                if maxX > 0:
                    maxX = maxX * 1.2
                elif maxX < 0:
                    maxX = maxX * 0.8
                else:
                    maxX = 0.2
                pvalAx.set_xlim(minX, maxX)
                # Set the x ticks to be the unique values of the numerical param
                pvalAx.set_xticks(significantCorrelations[col].unique())

                # correlation values
                corrAx = axes[1, i]
                assert isinstance(corrAx, plt.Axes)
                corrAx.set_ylim(-1, 1)
                corrAx.set_xlim(minX, maxX)
                corrAx.set_xticks(significantCorrelations[col].unique())

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
                    yvals = (otherNumComboDf["pval"]).to_numpy()
                    categoryMarkers = {
                        "home": "H",
                        "same": " ",
                        "away": "3",
                        "symmetric": "2",
                        "next": ">",
                        "nextsession": ">",
                        "nextsessionnoaway": ">",
                        "later": "<",
                        "latersessions": "<",
                        "other": "^",
                        "othersessions": "^",
                        # "": "o"
                        "SWR": "o",
                        "Ctrl": "o",
                    }
                    conditionColors = {
                        "SWR": "orange",
                        "Ctrl": "cyan",
                    }

                    if combo[0] == "ctrl":
                        cats = otherNumComboDf["categories"]
                        colors = None
                    elif combo[0] == "ctrlDiff":
                        cats = otherNumComboDf["ctrlName"]
                        colors = None
                    elif combo[0] == "ctrlDiffByCond":
                        cats = otherNumComboDf["ctrlName"]
                        colors = otherNumComboDf["categories"]
                        colors = np.array([conditionColors[c] for c in colors])
                    elif combo[0] == "measureByCondition":
                        cats = otherNumComboDf["categories"]
                        colors = otherNumComboDf["categories"]
                        colors = np.array([conditionColors[c] for c in colors])
                    elif combo[0] == "measure":
                        cats = np.full_like(otherNumComboDf["categories"], "home")
                        colors = None

                    uniqueCategories = set(cats)
                    # Choose a color here so both plots are the same color
                    if colors is None:
                        colors = np.full_like(cats, next(pvalAx._get_lines.prop_cycler)["color"])
                    for category in uniqueCategories:
                        marker = categoryMarkers[category]
                        catBool = cats == category
                        jitter = jitteri * (maxX - minX) * 0.005
                        jitteri += 1
                        xvals = (otherNumComboDf[col] + jitter).to_numpy()
                        if any(catBool):
                            lab = " ".join(str(v) for v in otherNumCombo) + " " + category
                            pvalAx.scatter(xvals[catBool], yvals[catBool], marker=marker,
                                           label=lab, color=colors[catBool])
                            corrAx.scatter(xvals[catBool], otherNumComboDf["correlation"][catBool], marker=marker,
                                           label=lab, color=colors[catBool])

                            # pvalAx.plot(xvals[catBool], yvals[catBool], marker,
                            #                label=lab,
                            #                color=colors)
                            # corrAx.plot(xvals[catBool], otherNumComboDf["correlation"][catBool], marker,
                            #             label=lab,
                            #             color=colors)

                # in pvalAx, add a legend in top right
                if i == 0:
                    pvalAx.legend(loc="upper right")
                # corrAx.legend()

            # Add a title to the whole figure
            fig.suptitle(
                "    ".join([f"{col}={val}" for col, val in zip(categoricalParams.columns, combo)]))

            plt.tight_layout()
            plt.show()


def getChosenParams(specName, func) -> List[Dict[str, Any]]:
    if specName == "measureFromFunc" and func == BTSession.totalTimeAtPosition:
        chosenParams = [
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                "radius": 0.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags="offWall"),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, inclusionFlags="explore"),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 3)),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(3, 4)),
                "radius": 0.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 7), inclusionFlags="awayTrial"),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 7), inclusionFlags="homeTrial"),
                "radius": 0.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(10, 15), inclusionFlags="awayTrial"),
                "radius": 1.0,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(10, 15), inclusionFlags="homeTrial"),
                "radius": 1.0,
                "smoothDist": 0.5,
            },
        ]
        # chosenParams = [
        #     {
        #         "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(2, 7)),
        #         "radius": 1.5,
        #         "smoothDist": 0.5,
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(2, 7)),
        #         "radius": 1.5,
        #         "smoothDist": 0.5,
        #     },
        #     {
        #         "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags=["offWall", "moving"]),
        #         "radius": 0.25,
        #         "smoothDist": 0,
        #     },
        #     {
        #         "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags=["offWall", "moving"]),
        #         "radius": 0.5,
        #         "smoothDist": 0,
        #     },
        #     {
        #         "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
        #         "radius": 0.5,
        #         "smoothDist": 0.5,
        #     },
        #     {
        #         "bp": BP(probe=False),
        #         "radius": 1,
        #         "smoothDist": 0
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3),
        #         "radius": 1,
        #         "smoothDist": 0
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3),
        #         "radius": 1,
        #         "smoothDist": 0.5
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3, trialInterval=(2, 7)),
        #         "radius": 1,
        #         "smoothDist": 0.5
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3, trialInterval=(2, 7)),
        #         "radius": 1.5,
        #         "smoothDist": 1.0
        #     },
        #     {
        #         "bp": BP(probe=False, erode=3, trialInterval=(0, 1)),
        #         "radius": 0.5,
        #         "smoothDist": 0.5
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(10, 15)),
        #         "radius": 0.5,
        #         "smoothDist": 0.5
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(10, 15)),
        #         "radius": 0.5,
        #         "smoothDist": 0.5
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="homeTrial", erode=3, trialInterval=(10, 15)),
        #         "radius": 0.5,
        #         "smoothDist": 0.5
        #     },
        #     {
        #         "bp": BP(probe=False, inclusionFlags="explore"),
        #         "radius": 0.5,
        #         "smoothDist": 0.5
        #     },
        # ]
    elif specName == "measureFromFunc" and func == BTSession.avgDwellTimeAtPosition:
        chosenParams = [
            {
                "bp": BP(probe=False),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 7)),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(10, 15)),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=True, timeInterval=(0, 120)),
                "radius": 1.0,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 3)),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(3, 4)),
                "radius": 1,
                "smoothDist": 0.5,
            },
        ]
    elif specName == "measureFromFunc" and func == BTSession.avgCurvatureAtPosition:
        chosenParams = [
            # {
            #     "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3),
            #     "radius": 1.0,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=False, inclusionFlags="explore"),
            #     "radius": 1.0,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=False, inclusionFlags="explore"),
            #     "radius": 0.5,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=False, erode=3, trialInterval=(0, 1)),
            #     "radius": .25,
            #     "smoothDist": 0,
            # },
            # {
            #     "bp": BP(probe=False, erode=3, trialInterval=(2, 3)),
            #     "radius": 1,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=False, erode=3, trialInterval=(3, 4)),
            #     "radius": .5,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=False, erode=3, trialInterval=(2, 7), inclusionFlags="awayTrial"),
            #     "radius": .5,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=False, erode=3, trialInterval=(2, 7)),
            #     "radius": .5,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=False, erode=3, trialInterval=(2, 7)),
            #     "radius": 1,
            #     "smoothDist": 0.5,
            # },
            # {
            #     "bp": BP(probe=True, inclusionFlags="offWall"),
            #     "radius": 1,
            #     "smoothDist": 0.5,
            # },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval, inclusionFlags="offWall"),
                "radius": 1,
                "smoothDist": 0.5,
            },
        ]
    elif specName == "measureFromFunc" and func == BTSession.numVisitsToPosition:
        chosenParams = [
            {
                "bp": BP(probe=False, trialInterval=(0, 1)),
                "radius": 0.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(0, 1)),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 3)),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 3)),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(3, 4)),
                "radius": 1,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(3, 4)),
                "radius": .5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False, trialInterval=(10, 15)),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=False),
                "radius": 1.0,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                "radius": 1.0,
                "smoothDist": 0.5,
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                "radius": 1.5,
                "smoothDist": 0.5,
            },
        ]
    elif specName == "makeNumWellsVisitedMeasure":
        chosenParams = [
            {
                "countRepeats": False,
                "offWallOnly": True,
                "inProbe": True,
                "mode": "excursion",
            },
            {
                "countRepeats": True,
                "offWallOnly": False,
                "inProbe": False,
                "mode": "bout",
            },
            {
                "countRepeats": True,
                "offWallOnly": True,
                "inProbe": True,
                "mode": "excursion",
            },
        ]
    elif specName == "makeCoarseTimeMeasure":
        chosenParams = [
            {
                "inProbe": True
            },
            {
                "inProbe": False
            }
        ]
    elif specName == "makeTimePeriodsMeasure":
        chosenParams = [
            {
                "timePeriodsGenerator": (TimeMeasure.trialTimePeriodsFunction(), "trial")
            },
            {
                "timePeriodsGenerator": (TimeMeasure.trialTimePeriodsFunction((2, 6)), "trial26"),
            },
            {
                "timePeriodsGenerator": (TimeMeasure.trialTimePeriodsFunction((2, 4)), "trial24"),
            },
            {
                "timePeriodsGenerator": (TimeMeasure.earlyVsLateTrials(), "evl"),
            },
            {
                "timePeriodsGenerator": (TimeMeasure.earlyVsLateTrials((0, 1), (2, 3)), "evl0123")
            }
        ]
    elif specName == "makeNumWellsVisitedTimeMeasure":
        chosenParams = [
            {
                "countReturns": True,
                "offWallWells": False,
                "timePeriodsGenerator": (TimeMeasure.earlyVsLateTrials((0, 1), (2, 3)), "evl0123")
            },
            {
                "countReturns": False,
                "offWallWells": True,
                "timePeriodsGenerator": (TimeMeasure.earlyVsLateTrials((0, 1), (2, 3)), "evl0123")
            },
            {
                "countReturns": False,
                "offWallWells": False,
                "timePeriodsGenerator": (TimeMeasure.earlyVsLateTrials((0, 1), (2, 3)), "evl0123")
            },
            {
                "countReturns": True,
                "offWallWells": True,
                "timePeriodsGenerator": (TimeMeasure.earlyVsLateTrials((0, 1), (2, 3)), "evl0123")
            },
            {
                "countReturns": True,
                "offWallWells": False,
                "timePeriodsGenerator": (TimeMeasure.trialTimePeriodsFunction(), "trial")
            },
            {
                "countReturns": False,
                "offWallWells": True,
                "timePeriodsGenerator": (TimeMeasure.trialTimePeriodsFunction(), "trial")
            },
            {
                "countReturns": False,
                "offWallWells": False,
                "timePeriodsGenerator": (TimeMeasure.trialTimePeriodsFunction(), "trial")
            },
            {
                "countReturns": True,
                "offWallWells": True,
                "timePeriodsGenerator": (TimeMeasure.trialTimePeriodsFunction(), "trial")
            },
        ]
    elif specName == "makeGravityMeasure":
        chosenParams = [
            # "bp": allConsideredBPs,
            # "passRadius": np.linspace(0.5, 1.5, 3),
            # "visitRadiusFactor": np.linspace(0.2, 0.4, 2),
            # "passDenoiseFactor": [1.25],
            # "smoothDist": [0, 0.5]
            {
                "bp": BP(probe=False, trialInterval=(3, 4)),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, trialInterval=(3, 4)),
                "passRadius": 1.0,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 3)),
                "passRadius": 1.0,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 7), inclusionFlags="awayTrial"),
                "passRadius": 1.0,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="awayTrial"),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="awayTrial"),
                "passRadius": 1.0,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, trialInterval=(2, 7)),
                "passRadius": 1.0,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial"),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags=["explore", "offWall"]),
                "passRadius": 1.0,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial", trialInterval=(2, 7)),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial", trialInterval=(2, 7)),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="homeTrial", trialInterval=(10, 15)),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="explore"),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="explore"),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="explore"),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, inclusionFlags="offWall"),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=False, trialInterval=(0, 1)),
                "passRadius": 1.0,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.5
            },
            {
                "bp": BP(probe=True),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True, inclusionFlags="moving", timeInterval=BTSession.fillTimeInterval),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True, inclusionFlags="moving", timeInterval=BTSession.fillTimeInterval),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True, inclusionFlags="moving"),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True, timeInterval=(0, 120)),
                "passRadius": 0.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
            {
                "bp": BP(probe=True, timeInterval=(0, 120)),
                "passRadius": 1.5,
                "visitRadiusFactor": 0.2,
                "passDenoiseFactor": 1.25,
                "smoothDist": 0.0
            },
        ]

    elif specName == "makeDotProdMeasure":
        # "distanceWeight": np.linspace(-1, 1, 5),
        # "normalize": [True, False],
        # "onlyPositive": [True, False],
        # "bp": [
        #     BP(probe=False, inclusionFlags="awayTrial"),
        #     BP(probe=False, inclusionFlags=["awayTrial", "moving"]),
        #     BP(probe=False, inclusionFlags="moving"),
        #     BP(probe=False, inclusionFlags="awayTrial", erode=3),
        #     BP(probe=False, inclusionFlags=["awayTrial", "moving"], erode=3),
        #     BP(probe=False, inclusionFlags="moving", erode=3),
        #     BP(probe=True),
        #     BP(probe=True, timeInterval=BTSession.fillTimeInterval),
        # ],
        # "smoothDist": allConsideredSmoothDists,
        chosenParams = [
            {
                "distanceWeight": 0.0,
                "normalize": False,
                "onlyPositive": False,
                "bp": BP(probe=False, inclusionFlags="moving"),
                "smoothDist": 0.0
            },
            {
                "distanceWeight": -1.0,
                "normalize": False,
                "onlyPositive": False,
                "bp": BP(probe=False, inclusionFlags="moving"),
                "smoothDist": 0.0
            },
            {
                "distanceWeight": 0.0,
                "normalize": False,
                "onlyPositive": False,
                "bp": BP(probe=True),
                "smoothDist": 0.0
            },
            {
                "distanceWeight": -1.0,
                "normalize": False,
                "onlyPositive": False,
                "bp": BP(probe=True),
                "smoothDist": 0.0
            },
        ]

    else:
        raise Exception("No chosen params for this specName and func")

    return chosenParams


def generateChosenPlots(specName, func, chosenParams, withCorrelations=True):
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

    if withCorrelations:
        correlationSMs = [
            # SessionMeasure("numStims", lambda sesh: len(
            #     sesh.btLFPBumps_posIdx), sessions, parallelize=False),
            # SessionMeasure("numRipplesPreStats", lambda sesh: len(
            #     sesh.btRipsPreStats), sessions, parallelize=False),
            # SessionMeasure("numRipplesProbeStats", lambda sesh: len(
            #     sesh.btRipsProbeStats), sessions, parallelize=False),
            SessionMeasure("stimRate", lambda sesh: len(sesh.btLFPBumps_posIdx) /
                           sesh.taskDuration, sessions, parallelize=False),
            # SessionMeasure("rippleRatePreStats", lambda sesh: len(
            #     sesh.btRipsPreStats) / sesh.taskDuration, sessions, parallelize=False),
            SessionMeasure("rippleRateProbeStats", lambda sesh: len(
                sesh.btRipsProbeStats) / sesh.taskDuration, sessions, parallelize=False),
        ]
    else:
        correlationSMs = []

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
            for sm in correlationSMs:
                lm.makeCorrelationFigures(
                    pp, sm, excludeFromCombo=True)
        elif specName == "makeDotProdMeasure":
            bp = params["bp"]
            distanceWeight = params["distanceWeight"]
            normalize = params["normalize"]
            smoothDist = params["smoothDist"]
            onlyPositive = params["onlyPositive"]
            lm = LocationMeasure(f"dotprod {bp.filenameString()} "
                                 f"dw{distanceWeight:.2f} "
                                 f"n{normalize} "
                                 f"sd={smoothDist:.2f} "
                                 f"op{onlyPositive}",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.getDotProductScore(bp, pos,
                                                                         distanceWeight=distanceWeight,
                                                                         normalize=normalize, onlyPositive=onlyPositive)),
                                 sessions,
                                 smoothDist=smoothDist)
            lm.makeFigures(pp, everySessionBehaviorPeriod=bp, excludeFromCombo=True)
        elif specName == "makeGravityMeasure":
            bp = params["bp"]
            passRadius = params["passRadius"]
            visitRadiusFactor = params["visitRadiusFactor"]
            passDenoiseFactor = params["passDenoiseFactor"]
            smoothDist = params["smoothDist"]
            lm = LocationMeasure(f"gravity {bp.filenameString()} "
                                 f"pr{passRadius:.2f} "
                                 f"vrf{visitRadiusFactor:.2f} "
                                 f"df{passDenoiseFactor:.2f} "
                                 f"sd{smoothDist:.2f}",
                                 lambda sesh: sesh.getValueMap(
                                     lambda pos: sesh.getGravity(bp, pos,
                                                                 passRadius=passRadius,
                                                                 visitRadius=passRadius * visitRadiusFactor,
                                                                 passDenoiseFactor=passDenoiseFactor)),
                                 sessions,
                                 smoothDist=smoothDist)
            lm.makeFigures(pp, everySessionBehaviorPeriod=bp, excludeFromCombo=True)
            # for sm in correlationSMs:
            #     lm.makeCorrelationFigures(
            #         pp, sm, excludeFromCombo=True)
        elif specName == "makeNumWellsVisitedMeasure":
            mode = params["mode"]
            inProbe = params["inProbe"]
            countRepeats = params["countRepeats"]
            offWallOnly = params["offWallOnly"]
            probeStr = "probe" if inProbe else "bt"
            smeas = SessionMeasure(f"numWellsVisitedPer{mode.capitalize()}_{probeStr}_cr{countRepeats}_ow{offWallOnly}",
                                   partial(avgNumWellsVisitedPerPeriod, countRepeats,
                                           offWallOnly, mode, inProbe),
                                   sessions)
            smeas.makeFigures(pp, excludeFromCombo=True)
            for sm in correlationSMs:
                smeas.makeCorrelationFigures(
                    pp, sm, excludeFromCombo=True)
        elif specName == "makeCoarseTimeMeasure":
            inProbe = params["inProbe"]
            probeStr = "probe" if inProbe else "bt"
            windowSize = 30 if inProbe else 90
            tm = TimeMeasure(f"{func.__name__}_{probeStr}", lambda sesh, t0, t1, ip, _: func(sesh, t0, t1, ip),
                             sessions, timePeriodsGenerator=(windowSize, 15, inProbe))
            tm.makeFigures(pp, excludeFromCombo=True)
        elif specName == "makeTimePeriodsMeasure":
            timePeriodsGenerator = params["timePeriodsGenerator"]
            tm = TimeMeasure(f"{func.__name__}_{timePeriodsGenerator[1]}", lambda sesh, t0, t1, inProbe, _: func(sesh, t0, t1, inProbe),
                             sessions, timePeriodsGenerator=timePeriodsGenerator[0])
            tm.makeFigures(pp, excludeFromCombo=True)
        elif specName == "makePerPeriodMeasure":
            probeStr = "probe" if inProbe else "bt"
            mode = params["mode"]
            smeas = SessionMeasure(f"{func.__name__}_{probeStr}_{mode}",
                                   partial(func, mode, inProbe), sessions)
            smeas.makeFigures(pp, excludeFromCombo=True)
            for sm in correlationSMs:
                smeas.makeCorrelationFigures(
                    pp, sm, excludeFromCombo=True)
        elif specName == "makeTotalDurationMeasure":
            probeStr = "probe" if inProbe else "bt"
            smeas = SessionMeasure(f"totalDuration_{probeStr}",
                                   lambda sesh: sesh.probeDuration if inProbe else sesh.btDuration,
                                   sessions)
            smeas.makeFigures(pp, excludeFromCombo=True)
            for sm in correlationSMs:
                smeas.makeCorrelationFigures(
                    pp, sm, excludeFromCombo=True)
        elif specName == "makeNumWellsVisitedTimeMeasure":
            offWallWells = params["offWallWells"]
            timePeriodsGenerator = params["timePeriodsGenerator"]
            countReturns = params["countReturns"]
            wells = offWallWellNames if offWallWells else allWellNames
            tm = TimeMeasure(f"numWellsVisited_cr{countReturns}_ow{offWallWells}_{timePeriodsGenerator[1]}",
                             partial(numWellsVisitedFunc, countReturns, wells), sessions,
                             timePeriodsGenerator=timePeriodsGenerator[0])
            tm.makeFigures(pp, excludeFromCombo=True)
        else:
            raise ValueError(f"unknown specName {specName}")


if __name__ == "__main__":
    func = None
    # specName = "measureFromFunc"
    # func = BTSession.totalTimeAtPosition
    # func = BTSession.numVisitsToPosition
    # func = BTSession.avgDwellTimeAtPosition
    # func = BTSession.avgCurvatureAtPosition
    # specName = "makeNumWellsVisitedMeasure"
    # specName = "makeCoarseTimeMeasure"
    # func = speedFunc
    # func = fracExploreFunc
    # func = fracOffWallFunc
    # specName = "makeTimePeriodsMeasure"
    # func = durationFunc
    # func = pathOptimalityFunc
    # func = pathLengthFunc
    # specName = "makePerPeriodMeasure" these all have warnings which turn into errors
    # func = avgDurationPerPeriod
    # func = avgPathLengthPerPeriod
    # specName = "makeTotalDurationMeasure" error here cause not enough params
    # specName = "makeNumWellsVisitedTimeMeasure"
    # specName = "makeGravityMeasure"
    specName = "makeDotProdMeasure"

    filters = None
    # filters = [
    #     {
    #         "bp": BP(probe=False, inclusionFlags="awayTrial", erode=3, trialInterval=(2, 7)),
    #         "ctrlName": "symmetric",
    #         "suffix": "diff"
    #     },
    # ]
    # lookAtShuffles(specName, func, filters=filters, appendCorrelations=False,
    #                plotCorrelations=True, plotShuffles=True)

    chosenParams = getChosenParams(specName, func)
    generateChosenPlots(specName, func, chosenParams)
