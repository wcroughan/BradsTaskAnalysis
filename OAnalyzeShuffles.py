from glob import glob
import os
from itertools import product

from Shuffler import Shuffler
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP


def getNameFromParams(name, params, func):
    if name == "measureFromFunc":
        bp = params["bp"]
        radius = params["radius"]
        smoothDist = params["smoothDist"]
        return f"{func.__name__}_{bp.filenameString()}_r{radius:.2f}_sd{smoothDist:.2f}"
    else:
        raise ValueError(f"Unrecognized name: {name}")


def main():
    basicParams = {
        "bp": allConsideredBPs,
        "smoothDist": allConsideredSmoothDists,
    }
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

    allFigNames = [
        "",
    ]

    # get all combinations of parameters in specParams
    paramValues = list(specParams.values())
    paramValueCombos = list(product(*paramValues))
    print(f"Running {len(paramValueCombos)} shuffles")
    # print the first 5 combos
    print(paramValueCombos[:5])

    savedStatsNames = []

    for combo in paramValueCombos:
        params = dict(zip(specParams.keys(), combo))
        name = getNameFromParams(specName, params, func)
        print(f"Running {name}")
        for figName in allFigNames:
            savedStatsNames.append((name, figName))
            # TODO actually save the right thing here...
            # Look in an existing info file

    shuffler = Shuffler()
    outputFileName = f"{specName}{'' if func is None else '_' + func.__name__}.h5"
    shuffler.runAllShuffles(None, numShuffles=100,
                            savedStatsNames=savedStatsNames, outputFileName=outputFileName)


if __name__ == "__main__":
    main()
