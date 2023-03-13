from typing import Callable, Iterable, Dict, List, Optional, Tuple
from MeasureTypes import LocationMeasure, TrialMeasure, TimeMeasure, SessionMeasure
from PlotUtil import PlotManager
from DataminingFactory import runEveryCombination
from UtilFunctions import parseCmdLineAnimalNames, findDataDir, getLoadInfo, numWellsVisited
import os
from datetime import datetime
import time
from BTData import BTData
from BTSession import BTSession
import pandas as pd
import numpy as np
from BTSession import BehaviorPeriod as BP
from functools import partial
from consts import allWellNames, offWallWellNames, TRODES_SAMPLING_RATE
import warnings


def maxDuration(sesh: BTSession, bp: BP):
    return sesh.getBehaviorPeriodDuration(bp)


def doubleDuration(sesh: BTSession, bp: BP):
    return sesh.getBehaviorPeriodDuration(bp) * 2


def speedFunc(sesh, t0, t1, inProbe) -> float:
    if inProbe:
        return np.nanmean(sesh.probeVelCmPerS[t0:t1])
    else:
        return np.nanmean(sesh.btVelCmPerS[t0:t1])


def fracExploreFunc(sesh, t0, t1, inProbe) -> float:
    return np.count_nonzero(
        sesh.evalBehaviorPeriod(BP(probe=inProbe, timeInterval=(t0, t1, "posIdx"), inclusionFlags="explore"))) / (t1 - t0)


def fracOffWallFunc(sesh, t0, t1, inProbe) -> float:
    return np.count_nonzero(
        sesh.evalBehaviorPeriod(BP(probe=inProbe, timeInterval=(t0, t1, "posIdx"), inclusionFlags="offWall"))) / (t1 - t0)


def avgNumWellsVisitedPerPeriod(countRepeats: bool, offWallOnly: bool, mode: str, inProbe: bool, sesh: BTSession):
    nw = sesh.probeNearestWells if inProbe else sesh.btNearestWells
    if mode == "bout":
        starts = sesh.probeExploreBoutStart_posIdx if inProbe else sesh.btExploreBoutStart_posIdx
        ends = sesh.probeExploreBoutEnd_posIdx if inProbe else sesh.btExploreBoutEnd_posIdx
    elif mode == "excursion":
        starts = sesh.probeExcursionStart_posIdx if inProbe else sesh.btExcursionStart_posIdx
        ends = sesh.probeExcursionEnd_posIdx if inProbe else sesh.btExcursionEnd_posIdx
    return np.nanmean([numWellsVisited(nw[s:b], countRepeats, offWallWellNames if offWallOnly else allWellNames) for s, b in zip(starts, ends)])


def avgDurationPerPeriod(mode: str, inProbe: bool, sesh: BTSession):
    if mode == "bout":
        starts = sesh.probeExploreBoutStart_posIdx if inProbe else sesh.btExploreBoutStart_posIdx
        ends = sesh.probeExploreBoutEnd_posIdx if inProbe else sesh.btExploreBoutEnd_posIdx
    elif mode == "excursion":
        starts = sesh.probeExcursionStart_posIdx if inProbe else sesh.btExcursionStart_posIdx
        ends = sesh.probeExcursionEnd_posIdx if inProbe else sesh.btExcursionEnd_posIdx
    return np.nanmean(ends - starts) / TRODES_SAMPLING_RATE


def avgPathLengthPerPeriod(mode: str, inProbe: bool, sesh: BTSession):
    xs = sesh.probePosXs if inProbe else sesh.btPosXs
    ys = sesh.probePosYs if inProbe else sesh.btPosYs
    if mode == "bout":
        starts = sesh.probeExploreBoutStart_posIdx if inProbe else sesh.btExploreBoutStart_posIdx
        ends = sesh.probeExploreBoutEnd_posIdx if inProbe else sesh.btExploreBoutEnd_posIdx
    elif mode == "excursion":
        starts = sesh.probeExcursionStart_posIdx if inProbe else sesh.btExcursionStart_posIdx
        ends = sesh.probeExcursionEnd_posIdx if inProbe else sesh.btExcursionEnd_posIdx
    pathLengths = []
    for s, b in zip(starts, ends):
        x = xs[s:b]
        y = ys[s:b]
        pathLengths.append(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))
    return np.nanmean(pathLengths)


def durationFunc(sesh, t0, t1, inProbe) -> float:
    ts = sesh.probePos_secs if inProbe else sesh.btPos_secs
    return ts[t1] - ts[t0]


def pathOptimalityFunc(sesh, t0, t1, inProbe) -> float:
    return sesh.pathOptimality(inProbe, timeInterval=(t0, t1, "posIdx"))


def pathLengthFunc(sesh, t0, t1, inProbe) -> float:
    xs = sesh.probePosXs if inProbe else sesh.btPosXs
    ys = sesh.probePosYs if inProbe else sesh.btPosYs
    x = xs[t0:t1]
    y = ys[t0:t1]
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))


def numWellsVisitedFunc(countReturns, wells, sesh, t0, t1, inProbe, _) -> float:
    nw = sesh.probeNearestWells if inProbe else sesh.btNearestWells
    return numWellsVisited(nw[t0:t1], countReturns, wells)


def runDMFOnM(measureFunc: Callable[..., LocationMeasure | SessionMeasure | TimeMeasure],
              allParams: Dict[str, Iterable],
              pp: PlotManager,
              correlationSMs: Optional[List[SessionMeasure]] = None,
              minParamsSequential=2, verbose=False):
    def measureFuncWrapper(params: Dict[str, Iterable]):
        if verbose:
            print(f"running {measureFunc.__name__} with params {params}")
        try:
            meas = measureFunc(**params)
            if "bp" in params and not isinstance(meas, TimeMeasure):
                meas.makeFigures(pp, excludeFromCombo=True, everySessionBehaviorPeriod=params["bp"])
            else:
                meas.makeFigures(pp, excludeFromCombo=True)
            if correlationSMs is not None and not isinstance(meas, TimeMeasure):
                for sm in correlationSMs:
                    try:
                        meas.makeCorrelationFigures(pp, sm, excludeFromCombo=True)
                    except Exception as e:
                        print(
                            f"\n\nerror in {measureFunc.__name__}, correlations with {sm.name} with params {params}\n\n")
                        print(e)
                        exit()
        except Exception as e:
            print(f"\n\nerror in {measureFunc.__name__} with params {params}\n\n")
            print(e)
            exit()

    if minParamsSequential == -1:
        minParamsSequential = len(allParams) - 1
    runEveryCombination(measureFuncWrapper, allParams,
                        numParametersSequential=min(minParamsSequential, len(allParams)-1))


def main(plotFlags: List[str] | str = "tests",
         dataMine=False, testData=False):
    if isinstance(plotFlags, str):
        plotFlags = [plotFlags]

    dataDir = findDataDir()
    outputDir = "GreatBigDatamining"
    if testData:
        outputDir += "_test"
    if dataMine:
        outputDir += "_datamined"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)

    animalNames = parseCmdLineAnimalNames(default=["B17"])
    if dataMine and len(animalNames) > 1:
        raise ValueError("can't datamine multiple animals at once")

    infoFileName = datetime.now().strftime("_".join(animalNames) + "_%Y%m%d_%H%M%S" + ".txt")
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=2 if dataMine else 3)

    allSessionsByRat = {}
    for animalName in animalNames:
        loadDebug = False
        if animalName[-1] == "d":
            loadDebug = True
            animalName = animalName[:-1]
        animalInfo = getLoadInfo(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        if loadDebug:
            dataFilename += ".debug.dat"
        print("loading from " + dataFilename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[animalName] = ratData.getSessions()

    for ratName in animalNames:
        print("======================\n", ratName)
        if ratName[-1] == "d":
            ratName = ratName[:-1]
        sessions: List[BTSession] = allSessionsByRat[ratName]
        if testData:
            sessions = sessions[:6]
        # nSessions = len(sessions)
        sessionsWithProbe = [sesh for sesh in sessions if sesh.probePerformed]
        # numSessionsWithProbe = len(sessionsWithProbe)
        # sessionsWithLog = [sesh for sesh in sessions if sesh.hasActivelinkLog]

        ctrlSessionsWithProbe = [sesh for sesh in sessions if (
            not sesh.isRippleInterruption) and sesh.probePerformed]
        swrSessionsWithProbe = [
            sesh for sesh in sessions if sesh.isRippleInterruption and sesh.probePerformed]
        nCtrlWithProbe = len(ctrlSessionsWithProbe)
        nSWRWithProbe = len(swrSessionsWithProbe)

        # ctrlSessions = [sesh for sesh in sessions if not sesh.isRippleInterruption]
        # swrSessions = [sesh for sesh in sessions if sesh.isRippleInterruption]
        # nCtrlSessions = len(ctrlSessions)
        # nSWRSessions = len(swrSessions)

        print(f"{len(sessions)} sessions ({len(sessionsWithProbe)} "
              f"with probe: {nCtrlWithProbe} Ctrl, {nSWRWithProbe} SWR)")

        data = [[
            sesh.name, sesh.conditionString(), sesh.homeWell, len(
                sesh.visitedAwayWells), sesh.fillTimeCutoff()
        ] for si, sesh in enumerate(sessions)]
        df = pd.DataFrame(data, columns=[
            "name", "condition", "home well", "num aways found", "fill time"
        ])
        s = df.to_string()
        pp.writeToInfoFile(s)
        print(s)

        # if hasattr(sessions[0], "probeFillTime"):
        #     sessionsWithProbeFillPast90 = [s for s in sessionsWithProbe if s.probeFillTime > 90]
        # else:
        #     sessionsWithProbeFillPast90 = sessionsWithProbe

        pp.pushOutputSubDir(ratName)
        if len(animalNames) > 1:
            pp.setStatCategory("rat", ratName)

        if "tests" in plotFlags:
            plotFlags.remove("tests")

            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # tm = TimeMeasure(f"test", lambda sesh, t0, t1, inProbe, _: func(sesh, t0, t1, inProbe),
                #                  sessions, timePeriodsGenerator=timePeriodsGenerator, parallelize=False)
                # tm.makeFigures(pp, excludeFromCombo=True)

                lm = LocationMeasure(f"test",
                                     lambda sesh: sesh.getValueMap(
                                         lambda pos: sesh.fracExcursionsVisited(
                                             BP(probe=True, timeInterval=(0, 120)), pos, radius=0.25, normalization="bp"),
                                     ), sessions, smoothDist=0,  parallelize=False)

                # lm.makeFigures(pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(
                #     probe=True, timeInterval=BTSession.fillTimeInterval))

                correlationSMs = [
                    SessionMeasure("numStims", lambda sesh: len(
                        sesh.btLFPBumps_posIdx), sessions, parallelize=False),
                    SessionMeasure("rippleRatePreStats", lambda sesh: len(
                        sesh.btRipsPreStats) / (sesh.btPos_secs[-1] - sesh.btPos_secs[0]), sessions, parallelize=False),
                ]

                for sm in correlationSMs:
                    sm.makeFigures(pp, excludeFromCombo=True)
                    lm.makeCorrelationFigures(pp, sm, excludeFromCombo=True)

            # lm = LocationMeasure("LMtest", lambda sesh: sesh.getValueMap(
            #     lambda pos: sesh.fracExcursionsVisited(BP(probe=True), pos, 0.25)),
            #     sessions, smoothDist=0)
            # # lm.makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True))
            # sm = SessionMeasure("SMtest", lambda sesh: np.remainder(
            #     np.sqrt(float(sesh.name[-1])), 1), sessions)
            # # sm.makeFigures(pp)
            # sm2 = SessionMeasure("SMtest2", lambda sesh: np.remainder(
            #     np.sqrt(float(sesh.name[-3])), 1), sessions)
            # # sm2.makeFigures(pp)

            # lm.makeCorrelationFigures(pp, sm, excludeFromCombo=True)
            # sm2.makeCorrelationFigures(pp, sm, excludeFromCombo=True)

        if "earlyVsLate" in plotFlags:
            plotFlags.remove("earlyVsLate")
            TimeMeasure("evl_latency", lambda sesh, t0, t1, _, _2: sesh.btPos_secs[t1] - sesh.btPos_secs[t0],
                        sessions, timePeriodsGenerator=TimeMeasure.earlyVsLateTrials()).makeFigures(
                pp, excludeFromCombo=True)
            TimeMeasure("1v2_latency", lambda sesh, t0, t1,  _, _2: sesh.btPos_secs[t1] - sesh.btPos_secs[t0],
                        sessions, timePeriodsGenerator=TimeMeasure.earlyVsLateTrials(earlyRange=(0, 1), lateRange=(1, 2))).makeFigures(
                pp, excludeFromCombo=True)
            TimeMeasure("1v23_latency", lambda sesh, t0, t1, _, _2: sesh.btPos_secs[t1] - sesh.btPos_secs[t0],
                        sessions, timePeriodsGenerator=TimeMeasure.earlyVsLateTrials(earlyRange=(0, 1), lateRange=(1, 3))).makeFigures(
                pp, excludeFromCombo=True)

        if "T2" in plotFlags:
            plotFlags.remove("T2")

            def T2Latency(sesh: BTSession):
                trialPosIdxs = sesh.getAllTrialPosIdxs()
                h2Start = trialPosIdxs[2][0]
                h2End = trialPosIdxs[2][1]
                return sesh.btPos_secs[h2End] - sesh.btPos_secs[h2Start]
            SessionMeasure("TH2 Latency", T2Latency, sessions).makeFigures(
                pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=False, trialInterval=(2, 3)))

        if "generalBehavior" in plotFlags:
            plotFlags.remove("generalBehavior")
            for inProbe in [True, False]:
                probeStr = "probe" if inProbe else "bt"
                TimeMeasure(f"speed_{probeStr}", lambda sesh, t0, t1, _, _2: np.nanmean(sesh.probeVelCmPerS[t0:t1]),
                            sessions, timePeriodsGenerator=(30, 15, inProbe)).makeFigures(
                    pp, excludeFromCombo=True)
                TimeMeasure(f"fracExplore_{probeStr}", lambda sesh, t0, t1, _, _2: np.count_nonzero(
                    sesh.evalBehaviorPeriod(BP(probe=inProbe, timeInterval=(t0, t1, "posIdx"), inclusionFlags="explore"))) / (t1 - t0),
                    sessions, timePeriodsGenerator=(30, 15, inProbe)).makeFigures(
                    pp, excludeFromCombo=True)
                TimeMeasure(f"fracOffWall_{probeStr}", lambda sesh, t0, t1, _, _2: np.count_nonzero(
                    sesh.evalBehaviorPeriod(BP(probe=inProbe, timeInterval=(t0, t1, "posIdx"), inclusionFlags="offWall"))) / (t1 - t0),
                    sessions, timePeriodsGenerator=(30, 15, inProbe)).makeFigures(
                    pp, excludeFromCombo=True)

                def avgNumWellsVisitedPerBout(countRepeats: bool, offWallOnly: bool, sesh: BTSession):
                    nw = sesh.probeNearestWells if inProbe else sesh.btNearestWells
                    boutStarts = sesh.probeExploreBoutStart_posIdx if inProbe else sesh.btExploreBoutStart_posIdx
                    boutEnds = sesh.probeExploreBoutEnd_posIdx if inProbe else sesh.btExploreBoutEnd_posIdx
                    return np.nanmean([numWellsVisited(nw[s:b], countRepeats, offWallWellNames if offWallOnly else allWellNames) for s, b in zip(boutStarts, boutEnds)])

                SessionMeasure(f"numWellsVisitedPerExplorationBout_{probeStr}", partial(avgNumWellsVisitedPerBout, False, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))
                SessionMeasure(f"numWellsVisitedPerExplorationBout_withRepeats_{probeStr}", partial(avgNumWellsVisitedPerBout, True, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))
                SessionMeasure(f"numWellsVisitedPerExplorationBout_offWall_{probeStr}", partial(avgNumWellsVisitedPerBout, False, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))
                SessionMeasure(f"numWellsVisitedPerExplorationBout_withRepeats_offWall_{probeStr}", partial(avgNumWellsVisitedPerBout, True, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="explore"))

                def avgNumWellsVisitedPerExcursion(countRepeats: bool, offWallOnly: bool, sesh: BTSession):
                    nw = sesh.probeNearestWells if inProbe else sesh.btNearestWells
                    excursionStarts = sesh.probeExcursionStart_posIdx if inProbe else sesh.btExcursionStart_posIdx
                    excursionEnds = sesh.probeExcursionEnd_posIdx if inProbe else sesh.btExcursionEnd_posIdx
                    return np.nanmean([numWellsVisited(nw[s:b], countRepeats, offWallWellNames if offWallOnly else allWellNames)
                                       for s, b in zip(excursionStarts, excursionEnds)])

                SessionMeasure(f"numWellsVisitedPerExcursion_{probeStr}", partial(avgNumWellsVisitedPerExcursion, False, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))
                SessionMeasure(f"numWellsVisitedPerExcursion_withRepeats_{probeStr}", partial(avgNumWellsVisitedPerExcursion, True, False), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))
                SessionMeasure(f"numWellsVisitedPerExcursion_offWall_{probeStr}", partial(avgNumWellsVisitedPerExcursion, False, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))
                SessionMeasure(f"numWellsVisitedPerExcursion_withRepeats_offWall_{probeStr}", partial(avgNumWellsVisitedPerExcursion, True, True), sessions).makeFigures(
                    pp, excludeFromCombo=True, everySessionBehaviorPeriod=BP(probe=inProbe, inclusionFlags="offWall"))

        if "fullDatamine" in plotFlags:
            plotFlags.remove("fullDatamine")

            pp.defaultSavePlot = False
            allSpecs = []

            if testData:
                allConsideredBPs = [
                    BP(probe=False),
                    BP(probe=True),
                    BP(probe=False, inclusionFlags="moving"),
                    BP(probe=False, inclusionFlags=["explore", "offWall"]),
                    BP(probe=False, inclusionFlags="awayTrial", erode=3),
                    BP(probe=False, trialInterval=(2, 7), erode=3),
                    BP(probe=False, trialInterval=(0, 1)),
                    BP(probe=True, timeInterval=(0, 120)),
                    BP(probe=True, timeInterval=BTSession.fillTimeInterval),
                    BP(probe=True, inclusionFlags="moving"),
                ]
                allConsideredSmoothDists = [0]
                allConsideredRadii = [0.25]
            else:
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

            def measureFromFunc(func: Callable, bp: BP, radius: float, smoothDist: float):
                return LocationMeasure(f"{func.__name__}_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: func(sesh, bp, pos, radius),
                                       ), sessions, smoothDist=smoothDist, parallelize=False)
            allSpecs.append((measureFromFunc, {
                "func": [BTSession.totalTimeAtPosition,
                            BTSession.numVisitsToPosition,
                            BTSession.avgDwellTimeAtPosition,
                            BTSession.avgCurvatureAtPosition],
                "radius": allConsideredRadii,
                **basicParams
            }))

            def makePathOptimalityMeasure(func: Callable, bp: BP, radius: float, smoothDist: float, fillNanMode: float | str):
                return LocationMeasure(f"{func.__name__}_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}__fn{fillNanMode}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: func(sesh, bp, pos, radius,
                                                            startPoint="beginning"),
                                           fillNanMode=fillNanMode
                                       ), sessions, smoothDist=smoothDist,  parallelize=False)
            # Actually not gonna use excursion start point because it's the same as pathLengthInExcursion with first flag
            allSpecs.append((makePathOptimalityMeasure, {
                "func": [BTSession.pathOptimalityToPosition,
                         BTSession.pathLengthToPosition],
                "fillNanMode": [np.nan, 6 * np.sqrt(2), "max", "mean"],
                "radius": allConsideredRadii,
                **basicParams
            }))

            def makePathLengthInExcursionMeasure(bp: BP, radius: float, smoothDist: float, noVisitVal: float, mode: str, normalizeByDisplacement: bool, fillNanMode: float | str):
                return LocationMeasure(f"pathLengthInExcursion_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}__nvv{noVisitVal}_m{mode}_nd{normalizeByDisplacement}__fn{fillNanMode}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.pathLengthInExcursionToPosition(bp, pos, radius,
                                                                                            noVisitVal=noVisitVal, mode=mode, normalizeByDisplacement=normalizeByDisplacement),
                                           fillNanMode=fillNanMode
                                       ), sessions, smoothDist=smoothDist,  parallelize=False)
            allSpecs.append((makePathLengthInExcursionMeasure, {
                "noVisitVal": [np.nan, 6 * np.sqrt(2), 1000],
                "mode": ["first", "last", "firstvisit", "lastvisit", "mean", "meanIgnoreNoVisit"],
                "normalizeByDisplacement": [True, False],
                "fillNanMode": [np.nan, 6 * np.sqrt(2), "max", "mean"],
                "radius": allConsideredRadii,
                **basicParams
            }))

            def makeLatencyMeasure(bp: BP, radius: float, smoothDist: float, emptyVal: str | float | Callable[[BTSession, BP], float]):
                if callable(emptyVal):
                    emptyValStr = emptyVal.__name__
                else:
                    emptyValStr = str(emptyVal)
                return LocationMeasure(f"latency_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}_e{emptyValStr}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.latencyToPosition(bp, pos, radius),
                                           fillNanMode=emptyVal(sesh, bp) if callable(
                                               emptyVal) else emptyVal
                                       ), sessions, smoothDist=smoothDist,  parallelize=False)

            allSpecs.append((makeLatencyMeasure, {
                "emptyVal": [np.nan, maxDuration, doubleDuration, "max", "mean"],
                "radius": allConsideredRadii,
                **basicParams
            }))

            def makeFracExcursionsMeasure(bp: BP, radius: float, smoothDist: float, normalization: str):
                return LocationMeasure(f"fracExcursions_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}_n{normalization}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.fracExcursionsVisited(
                                               bp, pos, radius, normalization),
                                       ), sessions, smoothDist=smoothDist,  parallelize=False)
            allSpecs.append((makeFracExcursionsMeasure, {
                "normalization": ["session", "bp", "none"],
                "radius": allConsideredRadii,
                **basicParams
            }))

            def makeDotProdMeasure(bp: BP, distanceWeight, normalize, smoothDist, onlyPositive):
                return LocationMeasure(f"dotprod {bp.filenameString()} "
                                       f"dw{distanceWeight:.2f} "
                                       f"n{normalize} "
                                       f"sd={smoothDist:.2f} "
                                       f"op{onlyPositive}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.getDotProductScore(bp, pos,
                                                                               distanceWeight=distanceWeight,
                                                                               normalize=normalize, onlyPositive=onlyPositive)),
                                       sessions,
                                       smoothDist=smoothDist, parallelize=False)
            allSpecs.append((makeDotProdMeasure, {
                "distanceWeight": np.linspace(-1, 1, 5),
                "normalize": [True, False],
                "onlyPositive": [True, False],
                **basicParams
            }))

            def makeGravityMeasure(bp: BP, passRadius, visitRadiusFactor, passDenoiseFactor, smoothDist):
                return LocationMeasure(f"gravity {bp.filenameString()} "
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
                                       smoothDist=smoothDist, parallelize=False)
            allSpecs.append((makeGravityMeasure, {
                "passRadius": np.linspace(0.5, 1.5, 3),
                "visitRadiusFactor": np.linspace(0.2, 0.4, 2),
                "passDenoiseFactor": np.linspace(1.25, 2, 3),
                **basicParams
            }))

            def makeCoarseTimeMeasure(func: Callable, inProbe: bool):
                probeStr = "probe" if inProbe else "bt"
                return TimeMeasure(f"{func.__name__}_{probeStr}", lambda sesh, t0, t1, ip, _: func(sesh, t0, t1, ip),
                                   sessions, timePeriodsGenerator=(30, 15, inProbe))

            allSpecs.append((makeCoarseTimeMeasure, {
                "func": [speedFunc, fracExploreFunc, fracOffWallFunc],
                "inProbe": [True, False]
            }))

            def makeNumWellsVisitedMeasure(countRepeats: bool, offWallOnly: bool, inProbe: bool, mode: str):
                probeStr = "probe" if inProbe else "bt"
                return SessionMeasure(f"numWellsVisitedPer{mode.capitalize()}_{probeStr}_cr{countRepeats}_ow{offWallOnly}",
                                      partial(avgNumWellsVisitedPerPeriod, countRepeats,
                                              offWallOnly, mode, inProbe), sessions,
                                      parallelize=False)

            allSpecs.append((makeNumWellsVisitedMeasure, {
                "countRepeats": [True, False],
                "offWallOnly": [True, False],
                "inProbe": [True, False],
                "mode": ["bout", "excursion"]
            }))

            def makePerPeriodMeasure(func: Callable, mode: str, inProbe: bool):
                probeStr = "probe" if inProbe else "bt"
                return SessionMeasure(f"{func.__name__}_{probeStr}_{mode}", partial(func, mode, inProbe), sessions,
                                      parallelize=False)

            allSpecs.append((makePerPeriodMeasure, {
                "func": [avgDurationPerPeriod, avgPathLengthPerPeriod],
                "mode": ["bout", "excursion"],
                "inProbe": [True, False]
            }))

            def makeTimePeriodsMeasure(func: Callable, timePeriodsGenerator):
                return TimeMeasure(f"{func.__name__}_{timePeriodsGenerator[1]}", lambda sesh, t0, t1, inProbe, _: func(sesh, t0, t1, inProbe),
                                   sessions, timePeriodsGenerator=timePeriodsGenerator[0], parallelize=False)

            allConsideredTimePeriods = {"timePeriodsGenerator":
                                        [(TimeMeasure.trialTimePeriodsFunction(), "trial"),
                                         (TimeMeasure.trialTimePeriodsFunction((2, 6)), "trial26"),
                                         (TimeMeasure.trialTimePeriodsFunction((2, 4)), "trial24"),
                                         (TimeMeasure.earlyVsLateTrials(), "evl"),
                                         (TimeMeasure.earlyVsLateTrials((0, 1), (2, 3)), "evl0123")]}

            allSpecs.append((makeTimePeriodsMeasure, {
                "func": [durationFunc, pathOptimalityFunc, pathLengthFunc],
                **allConsideredTimePeriods
            }))

            def makeNumWellsVisitedTimeMeasure(countReturns: bool, offWallWells: bool, timePeriodsGenerator):
                wells = offWallWellNames if offWallWells else allWellNames
                return TimeMeasure(f"numWellsVisited_cr{countReturns}_ow{offWallWells}_{timePeriodsGenerator[1]}",
                                   partial(numWellsVisitedFunc, countReturns, wells), sessions,
                                   timePeriodsGenerator=timePeriodsGenerator[0])

            allSpecs.append((makeNumWellsVisitedTimeMeasure, {
                "countReturns": [True, False],
                "offWallWells": [True, False],
                **allConsideredTimePeriods
            }))

            if testData:
                correlationSMs = [
                    SessionMeasure("numStims", lambda sesh: len(
                        sesh.btLFPBumps_posIdx), sessions, parallelize=False),
                    SessionMeasure("rippleRatePreStats", lambda sesh: len(
                        sesh.btRipsPreStats) / (sesh.btPos_secs[-1] - sesh.btPos_secs[0]), sessions, parallelize=False),
                ]
            else:
                correlationSMs = [
                    SessionMeasure("numStims", lambda sesh: len(
                        sesh.btLFPBumps_posIdx), sessions, parallelize=False),
                    SessionMeasure("numRipplesPreStats", lambda sesh: len(
                        sesh.btRipsPreStats), sessions, parallelize=False),
                    SessionMeasure("numRipplesProbeStats", lambda sesh: len(
                        sesh.btRipsProbeStats), sessions, parallelize=False),
                    SessionMeasure("stimRate", lambda sesh: len(sesh.btLFPBumps_posIdx) /
                                   (sesh.btPos_secs[-1] - sesh.btPos_secs[0]), sessions, parallelize=False),
                    SessionMeasure("rippleRatePreStats", lambda sesh: len(
                        sesh.btRipsPreStats) / (sesh.btPos_secs[-1] - sesh.btPos_secs[0]), sessions, parallelize=False),
                    SessionMeasure("rippleRateProbeStats", lambda sesh: len(
                        sesh.btRipsProbeStats) / (sesh.btPos_secs[-1] - sesh.btPos_secs[0]), sessions, parallelize=False),
                ]

            for sm in correlationSMs:
                sm.makeFigures(pp, excludeFromCombo=True)

            for mi, (makeMeasure, paramDict) in enumerate(allSpecs):
                # For debugging, removing stuff that's already been run
                if testData:
                    if makeMeasure.__name__ in ["measureFromFunc",
                                                "makePathOptimalityMeasure",
                                                "makePathLengthInExcursionMeasure",
                                                "makeLatencyMeasure",
                                                "makeFracExcursionsMeasure",
                                                "makeDotProdMeasure",
                                                "makeGravityMeasure",
                                                "makeCoarseTimeMeasure",
                                                "makeNumWellsVisitedMeasure",
                                                "makePerPeriodMeasure",
                                                "makeTimePeriodsMeasure"
                                                ]:
                        continue

                    for k, v in paramDict.items():
                        # if all items in v are numeric, shorten it to just the first two elements
                        if k == "fillNanMode":
                            paramDict[k] = [np.nan]
                        elif all(isinstance(x, (int, float)) for x in v):
                            paramDict[k] = v[:2]

                print("Running", makeMeasure.__name__, f"({mi+1}/{len(allSpecs)})")

                pp.pushOutputSubDir(makeMeasure.__name__)
                if testData:
                    # runDMFOnM(makeMeasure, paramDict, pp, correlationSMs,
                    #           verbose=True, minParamsSequential=-1)
                    runDMFOnM(makeMeasure, paramDict, pp, correlationSMs)
                else:
                    runDMFOnM(makeMeasure, paramDict, pp, correlationSMs)
                pp.popOutputSubDir()

            if testData:
                pp.runShuffles(numShuffles=6)
            else:
                pp.runShuffles()

        if len(plotFlags) != 0:
            print("Warning, unused plot flags:", plotFlags)


if __name__ == "__main__":
    main("fullDatamine", dataMine=True, testData=True)
    # main(testData=True)


# Note from lab meeting 2023-3-7
# In paper, possible effect in first few trials
# BP flag for early trials/trial nums
# frac excursion, add over excursions instead of session so don't weight low exploration sessions higher
# General behavior differences like speed, amt exploration, wells visited per time/per bout
# Instead of pseudo p-val, just call it p-val, take lower number, and indicate separately which direction
# Shuffle histogram below violin plots with p vals
# Should I be more neutral in choosing which control location, or bias for away fine?
# Dot product but only positive side?
# Carefully look for opposite direction of effect with different parameters
# Starting excursion near home well - diff b/w conditions?
# Average of aways with nans biases to lower values. Esp latency, worst case is b-line to home but pass through a couple aways first
#   maybe set nans to max possible value in BP? Maybe add penalty?
# Overal narrative may support strat of direct paths from far away vs roundabout searching with familiarity
# Path opt but from start of excursion
# Run stats on cumulative amt of stims

# To finish datamining B17 and apply those measures to all animals, should complete the following steps:
# - Instead of ctrl vs stim, correlate with ripple count. Stim count too? might as well
# - test out new measures and new flag arguments
# - remember to add in all new measures and parameters to the datamining factory. Can also remove results that get filtered out later based on BP
# And while that's running, figure these out:
# - look at outlier sessions according to cum stim rates. Compare to behavior measures
# - Why are ripple counts so low?
# - Should prefer away control location? Or neutral? Dependent on measure?
# - Add in check when summarizing results for difference direction. Separate significant results this way
# - Difference in stim rates b/w conditions? Probs not, but should run stats

# COMPLETED:
# - Import or move important functions so can easily run lots of measures from any file
# - Change how psudo-pval is presented. Histogram of shuffle values, line for real data, p value, and written i.e. "ctrl > swr"
# - implement the following new measures
#   - path length from start of excursion
#       - sym ctrl locations here. Unless want to subdivide by well location type or something fancy
#   - General behavior differences: speed, num wells visited per excursion/bout, frac in exploration, others from previous lab meetings/thesis meetings
# - the following mods to existing measures:
#   - dot prod just positive
#   - frac excursion, avg over excursion not session
#   - Latency nans => max value + penalty. Possibly also path opt, others?
#   - path opt from start of excursion
# - look also at specific task behavior periods like early learning
# - Try different measures of learning rate, specifically targetted at possible effect from paper
# - Think about each measure during task, what possible caveats are there?
#   - Is erode 3 enough?
# - in testing found that avg home minus avg away trial duration difference b/w conditions is significant. Should run stats on this
