from glob import glob
import os
from itertools import product
from pprint import pprint
import random
import pandas as pd

from Shuffler import Shuffler
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP
from UtilFunctions import findDataDir
from tqdm import tqdm


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
    print(f"Could not find stats dir for {name}")
    print(f"Checked {len(possibleLocations)} possible locations")
    print(f"Possible locations:")
    pprint(possibleLocations)
    return None


def main():
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

    specName = "measureFromFunc"
    func = BTSession.totalTimeAtPosition
    specParams = {
        # "func": [BTSession.totalTimeAtPosition,
        #          BTSession.numVisitsToPosition,
        #          BTSession.avgDwellTimeAtPosition,
        #          BTSession.avgCurvatureAtPosition],
        "radius": allConsideredRadii,
        **basicParams
    }

    testData = False
    dataDir = findDataDir()
    outputFileName = os.path.join(
        dataDir, f"{specName}{'' if func is None else '_' + func.__name__}{'_test' if testData else ''}.h5")

    if os.path.exists(outputFileName):
        print(f"Output file {outputFileName} already exists!")
    else:
        # get all combinations of parameters in specParams
        paramValues = list(specParams.values())
        paramValueCombos = list(product(*paramValues))
        print(f"Gathering files for {len(paramValueCombos)} param combos")

        correlationNames = [
            "numStims",
            "numRipplesPreStats",
            "numRipplesProbeStats",
            "stimRate",
            "rippleRatePreStats",
            "rippleRateProbeStats",
        ]

        savedStatsNames = []

        epvc = enumerate(paramValueCombos)
        for ci, combo in tqdm(epvc, total=len(paramValueCombos)):
            if testData and ci > 10:
                break
            params = dict(zip(specParams.keys(), combo))
            # pprint(params)
            name = getNameFromParams(specName, params, func, None)
            noCorrStatsDir = findStatsDir(os.path.join(
                "figures", "GreatBigDatamining_datamined", "B17", specName), name)
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
                                savedStatsFiles=savedStatsNames, outputFileName=outputFileName)

    print("Here's the output")
    # sdf.to_hdf(outputFileName, key="significantShuffles")
    significantShuffles = pd.read_hdf(outputFileName, key="significantShuffles")
    print(significantShuffles.head().to_string())
    print(significantShuffles.shape)
    # sdf.to_hdf(outputFileName, key="significantCorrelations")
    significantCorrelations = pd.read_hdf(outputFileName, key="significantCorrelations")
    print(significantCorrelations.head().to_string())
    print(significantCorrelations.shape)

    # Next steps:
    #
    # Choose a set of parameters to make figures for
    # chosen set should be "robust" in that modifying any parameter somewhat should not change the conclusions
    # For probe stats BP, can prefer BPs that include more behavior over less, and ones that don't have inclusion flags
    # For numerical parameters, plot the effect of changing them on the pvalues over a sampling of the other parameters.
    # For categorical parameters, plot the effect of changing them on the pvalues over a sampling of the other parameters.
    # For task measures, I think just have to choose each set of BPs and go through it choosing all the other parameters, and then
    # select from the BPs at the end


if __name__ == "__main__":
    main()
