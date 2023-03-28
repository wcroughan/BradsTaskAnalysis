from typing import Callable, Iterable, Dict, List, Optional, Tuple
from MeasureTypes import LocationMeasure, TrialMeasure, TimeMeasure, SessionMeasure
from PlotUtil import PlotManager
from DataminingFactory import runEveryCombination
from UtilFunctions import parseCmdLineAnimalNames, findDataDir, getLoadInfo, numWellsVisited, posOnWall
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
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing
from glob import glob


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


def checkIfMeasureAlreadyDone(pp: PlotManager,
                              measure: LocationMeasure | SessionMeasure | TimeMeasure,
                              corrMeasure: Optional[SessionMeasure] = None,
                              verbose=False):
    classPrefixes = {
        TimeMeasure: "TiM",
        SessionMeasure: "SM",
        LocationMeasure: "LM",
        TrialMeasure: "TrM",
        # WellMeasure: "WM"
    }
    pfx = classPrefixes[type(measure)]
    if corrMeasure is None:
        doneFile = os.path.join(f"{pfx}_{measure.name.replace(' ', '_')}", "processed.txt")
        errFile = os.path.join(f"{pfx}_{measure.name.replace(' ', '_')}", "error.txt")
    else:
        doneFile = os.path.join(
            f"{pfx}_{measure.name.replace(' ', '_')}_X_{corrMeasure.name.replace(' ', '_')}", "corr_processed.txt")
        errFile = os.path.join(
            f"{pfx}_{measure.name.replace(' ', '_')}_X_{corrMeasure.name.replace(' ', '_')}", "corr_error.txt")

    possibleDrives = [
        "/media/WDC11/",
        "/media/WDC10/",
        "/media/WDC9/",
        "/media/WDC4/",
    ]

    drivesToAppend = []
    for pd in possibleDrives:
        if os.path.exists(os.path.join(pd, "BY_PID")):
            gl = glob(os.path.join(pd, "BY_PID", "*"))
            drivesToAppend.extend(gl)
    possibleDrives.extend(drivesToAppend)
    if verbose:
        print(f"possibleDrives: {possibleDrives}")

    ppOutputDirInDrive = os.path.sep.join([*(pp.outputDir.split(os.path.sep)[3:]), pp.outputSubDir])
    possibleLocations = [f"{pd}{os.path.sep}{ppOutputDirInDrive}" for pd in possibleDrives]

    for pl in possibleLocations:
        checkFileName = os.path.join(pl, doneFile)
        if os.path.exists(checkFileName):
            with open(checkFileName, "r") as f:
                makeFigsInfoFileName = f.read()
            return makeFigsInfoFileName, checkFileName
        errFileName = os.path.join(pl, errFile)
        if os.path.exists(errFileName):
            return "Error", errFileName
    if verbose:
        print(f"could not find {doneFile} in any of {possibleLocations}")
    return "None", None


def runDMFOnM(measureFunc: Callable[..., LocationMeasure | SessionMeasure | TimeMeasure],
              allParams: Dict[str, Iterable],
              pp: PlotManager,
              correlationSMs: Optional[List[SessionMeasure]] = None,
              minParamsSequential=2, verbose=False, overwriteOldResults=False):
    def measureFuncWrapper(params: Dict[str, Iterable]):

        if verbose:
            print(
                f"running {measureFunc.__name__} with params {params} in process {multiprocessing.current_process()} (pid {multiprocessing.current_process().pid})")

        try:
            pid = multiprocessing.current_process().pid
            pp.setDriveOutputDir(f"BY_PID{os.path.sep}{str(pid)[-2:]}")
            if verbose:
                print(f"pp output dir is {pp.fullOutputDir} (pid {pid})")

            meas = measureFunc(**params)
            classPrefixes = {
                TimeMeasure: "TiM",
                SessionMeasure: "SM",
                LocationMeasure: "LM",
                TrialMeasure: "TrM",
                # WellMeasure: "WM"
            }
            pfx = classPrefixes[type(meas)]
            savePoint = pp.getOutputSubDirSavepoint()
        except Exception as e:
            print(f"Error just making the measure ... this shouldn't happen")
            return

        try:
            makeFigsInfoFileName = "None"
            corrInfoFileNames = []
            if not overwriteOldResults:
                makeFigsInfoFileName, checkFileName = checkIfMeasureAlreadyDone(
                    pp, meas, corrMeasure=None, verbose=False)

                if makeFigsInfoFileName == "Error":
                    # Ran into an error last time, skip
                    if verbose:
                        print(
                            f"skipping {measureFunc.__name__} with params {params} because it errored last time")
                    return

                if isinstance(meas, (LocationMeasure, SessionMeasure)) and makeFigsInfoFileName != "None":
                    for sm in correlationSMs:
                        infoFileName, _ = checkIfMeasureAlreadyDone(
                            pp, meas, corrMeasure=sm, verbose=False)
                        if infoFileName == "Error":
                            # Ran into an error last time, skip
                            if verbose:
                                print(
                                    f"skipping {measureFunc.__name__} with params {params} because it errored last time")
                            return

                        corrInfoFileNames.append(infoFileName)
                        # corrInfoFileNames.append(
                        #     checkIfMeasureAlreadyDone(pp, meas, corrMeasure=sm, verbose=verbose)[0])

                if makeFigsInfoFileName != "None" and (len(corrInfoFileNames) == 0 or all([c != "None" for c in corrInfoFileNames])):
                    # We have done this one entirely already, skip running the measure func
                    if verbose:
                        print(
                            f"skipping {measureFunc.__name__} with params {params} because it has already been run")

                    pp.writeToInfoFile(
                        f"already run:{checkFileName}__!__{makeFigsInfoFileName}__!__{'__!__'.join(corrInfoFileNames)}\n")
                    return

            os.makedirs(os.path.join(pp.fullOutputDir,
                        f"{pfx}_{meas.name.replace(' ', '_')}"), exist_ok=True)
            stdOutFileName = os.path.join(
                pp.fullOutputDir, f"{pfx}_{meas.name.replace(' ', '_')}", "stdout.txt")
            stdErrFileName = os.path.join(
                pp.fullOutputDir, f"{pfx}_{meas.name.replace(' ', '_')}", "stderr.txt")
            with open(stdOutFileName, "w") as stdout, open(stdErrFileName, "w") as stderr:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    meas.runMeasureFunc()

                    # For now, just remake all the figs if any need to be remade
                    if "bp" in params and not isinstance(meas, TimeMeasure):
                        meas.makeFigures(pp, excludeFromCombo=True,
                                         everySessionBehaviorPeriod=params["bp"])
                    else:
                        meas.makeFigures(pp, excludeFromCombo=True)
                    if correlationSMs is not None and not isinstance(meas, TimeMeasure):
                        for sm in correlationSMs:
                            meas.makeCorrelationFigures(pp, sm, excludeFromCombo=True)

            # if stderr and stdout files are empty, delete them
            if os.stat(stdOutFileName).st_size == 0:
                os.remove(stdOutFileName)
            if os.stat(stdErrFileName).st_size == 0:
                os.remove(stdErrFileName)
        except Exception as e:
            # print(f"\n\nerror in {measureFunc.__name__} with params {params}\n\n")
            # print(e)
            # make a blank error file
            pp.restoreOutputSubDirSavepoint(savePoint)
            pp.setDriveOutputDir("")
            errFileName = os.path.join(
                pp.fullOutputDir, f"{pfx}_{meas.name.replace(' ', '_')}", "error.txt")
            with open(errFileName, "w") as f:
                f.write("error")

        # Now wait a few milliseconds to make sure the file is written, and hopefully drive won't get corrupted
        time.sleep(0.1)

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

                # "noVisitVal": [np.nan, 6 * np.sqrt(2), 1000],
                # "mode": ["first", "last", "firstvisit", "lastvisit", "mean", "meanIgnoreNoVisit"],
                # "normalizeByDisplacement": [True, False],
                # "fillNanMode": [np.nan, 6 * np.sqrt(2), "max", "mean"],
                # # "radius": allConsideredRadii,
                # "radius": [0.25, 0.5, 1],
                # "reciprocal": [True, False],
                # # **basicParams
                # "bp": allConsideredBPs,
                # "smoothDist": [0.5, 1]

                totalReplaceTime = 0
                totalBpEvalTime = 0
                totalInRadiusTime = 0
                totalFilterTime = 0
                totalRemakeTime = 0
                totalTotalDistTime = 0
                totalTotalDivTime = 0
                totalEndOfDistTime = 0
                totalReciprocalTime = 0
                totalRetTime = 0
                totalEvalTime = 0
                totalCopyTime = 0

                for bp in [
                    # BP(probe=True, timeInterval=(0, 60)),
                    BP(probe=False, trialInterval=(2, 7)),
                ]:
                    for radius in [0.25, 1]:
                        print("bp =", bp, "radius =", radius)
                        noVisitVal = np.nan
                        mode = "mean"
                        normalizeByDisplacement = True
                        # fillNanMode = np.nan
                        reciprocal = True
                        # smoothDist = 0.5

                        resolution = 36
                        pxls = np.linspace(-0.5, 6.5, resolution, endpoint=False)
                        pxls += (pxls[1] - pxls[0]) / 2
                        for x in pxls:
                            for y in pxls:
                                for sesh in sessions:
                                    pos = (x, y)
                                    ret = sesh.pathLengthInExcursionToPosition(bp, pos, radius,
                                                                               noVisitVal=noVisitVal, mode=mode,
                                                                               normalizeByDisplacement=normalizeByDisplacement,
                                                                               reciprocal=reciprocal,
                                                                               timeFunction=True,
                                                                               showPlot=False)

                                    totalReplaceTime += ret[2] - ret[1]
                                    totalBpEvalTime += ret[3] - ret[2]
                                    totalInRadiusTime += ret[4] - ret[3]
                                    if ret[5] != 0:
                                        totalFilterTime += ret[5] - ret[4]
                                    if ret[6] != 0:
                                        totalRemakeTime += ret[6] - ret[5]
                                    if ret[7] != 0:
                                        totalTotalDistTime += ret[7]
                                    if ret[8] != 0:
                                        totalTotalDivTime += ret[8]
                                    if ret[9] != 0:
                                        totalEndOfDistTime += ret[9] - ret[6]
                                    if ret[10] != 0:
                                        totalReciprocalTime += ret[10] - ret[9]
                                    if ret[11] != 0:
                                        totalRetTime += ret[11] - ret[10]
                                    totalEvalTime += ret[12]
                                    totalCopyTime += ret[13]

                allTimesTotal = totalReplaceTime + totalBpEvalTime + totalInRadiusTime + totalFilterTime + totalRemakeTime + \
                    totalTotalDistTime + totalTotalDivTime + totalEndOfDistTime + totalReciprocalTime + totalRetTime

                print(f"allTimesTotal: {allTimesTotal / 1e9:.2f}s")
                print(
                    f"totalReplaceTime:\t{totalReplaceTime / 1e9:.2f}s\t{totalReplaceTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalBpEvalTime:\t{totalBpEvalTime / 1e9:.2f}s\t{totalBpEvalTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalInRadiusTime:\t{totalInRadiusTime / 1e9:.2f}s\t{totalInRadiusTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalFilterTime:\t{totalFilterTime / 1e9:.2f}s\t{totalFilterTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalRemakeTime:\t{totalRemakeTime / 1e9:.2f}s\t{totalRemakeTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalTotalDistTime:\t{totalTotalDistTime / 1e9:.2f}s\t{totalTotalDistTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalTotalDivTime:\t{totalTotalDivTime / 1e9:.2f}s\t{totalTotalDivTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalEndOfDistTime:\t{totalEndOfDistTime / 1e9:.2f}s\t{totalEndOfDistTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalReciprocalTime:\t{totalReciprocalTime / 1e9:.2f}s\t{totalReciprocalTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalRetTime:\t{totalRetTime / 1e9:.2f}s\t{totalRetTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalEvalTime:\t{totalEvalTime / 1e9:.2f}s\t{totalEvalTime / allTimesTotal * 100:.2f}%")
                print(
                    f"totalCopyTime:\t{totalCopyTime / 1e9:.2f}s\t{totalCopyTime / allTimesTotal * 100:.2f}%")

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

            def isConvex(bp: BP):
                return bp.inclusionArray is None and bp.inclusionFlags is None and bp.moveThreshold is None
            allConvexBPs = [bp for bp in allConsideredBPs if isConvex(bp)]
            # nonConvexBPs = [bp for bp in allConsideredBPs if not isConvex(bp)]

            basicParams = {
                "bp": allConsideredBPs,
                "smoothDist": allConsideredSmoothDists,
            }

            def measureFromFunc(func: Callable, bp: BP, radius: float, smoothDist: float):
                return LocationMeasure(f"{func.__name__}_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: func(sesh, bp, pos, radius),
                                       ), sessions, smoothDist=smoothDist, parallelize=False, runImmediately=False)
            allSpecs.append((measureFromFunc, {
                "func": [BTSession.totalTimeAtPosition,
                            BTSession.numVisitsToPosition,
                            BTSession.avgDwellTimeAtPosition,
                            BTSession.avgCurvatureAtPosition],
                "radius": allConsideredRadii,
                **basicParams
            }))

            def makePathOptimalityMeasure(func: Callable, bp: BP, radius: float, smoothDist: float, fillNanMode: float | str, reciprocal: bool):
                return LocationMeasure(f"{func.__name__}_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}_fn{fillNanMode}_rec{reciprocal}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: func(sesh, bp, pos, radius,
                                                            startPoint="beginning", reciprocal=reciprocal),
                                           fillNanMode=fillNanMode
                                       ), sessions, smoothDist=smoothDist,  parallelize=False, runImmediately=False)
            # Actually not gonna use excursion start point because it's the same as pathLengthInExcursion with first flag
            allSpecs.append((makePathOptimalityMeasure, {
                "func": [BTSession.pathOptimalityToPosition,
                         BTSession.pathLengthToPosition],
                "fillNanMode": [np.nan, 6 * np.sqrt(2), "max", "mean"],
                # "radius": allConsideredRadii,
                "radius": [0.25, 0.5, 1],
                "reciprocal": [True, False],
                # **basicParams
                "bp": allConsideredBPs,
                "smoothDist": [0.5, 1]
            }))

            def makePathLengthInExcursionMeasure(bp: BP, radius: float, smoothDist: float,
                                                 noVisitVal: float, mode: str, normalizeByDisplacement: bool,
                                                 fillNanMode: float | str, reciprocal: bool):
                return LocationMeasure(f"pathLengthInExcursion_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}__nvv{noVisitVal}_m{mode}_nd{normalizeByDisplacement}_fn{fillNanMode}_r{reciprocal}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.pathLengthInExcursionToPosition(bp, pos, radius,
                                                                                            noVisitVal=noVisitVal, mode=mode, normalizeByDisplacement=normalizeByDisplacement, reciprocal=reciprocal),
                                           fillNanMode=fillNanMode
                                       ), sessions, smoothDist=smoothDist,  parallelize=False, runImmediately=False)
            allSpecs.append((makePathLengthInExcursionMeasure, {
                "noVisitVal": [np.nan, 6 * np.sqrt(2), 1000],
                "mode": ["first", "last", "firstvisit", "lastvisit", "mean", "meanIgnoreNoVisit"],
                "normalizeByDisplacement": [True, False],
                "fillNanMode": [np.nan, 6 * np.sqrt(2), "max", "mean"],
                # "radius": allConsideredRadii,
                "radius": [0.25, 0.5, 1],
                "reciprocal": [True, False],
                # **basicParams
                "bp": allConvexBPs,
                "smoothDist": [0.5, 1]
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
                                       ), sessions, smoothDist=smoothDist,  parallelize=False, runImmediately=False)

            allSpecs.append((makeLatencyMeasure, {
                "emptyVal": [np.nan, maxDuration, doubleDuration, "max", "mean"],
                "radius": allConsideredRadii,
                # **basicParams
                "bp": allConvexBPs,
                "smoothDist": [0, 0.5, 1]
            }))

            def makeFracExcursionsMeasure(bp: BP, radius: float, smoothDist: float, normalization: str):
                return LocationMeasure(f"fracExcursions_{bp.filenameString()}_r{radius:.2f}_s{smoothDist:.2f}_n{normalization}",
                                       lambda sesh: sesh.getValueMap(
                                           lambda pos: sesh.fracExcursionsVisited(
                                               bp, pos, radius, normalization),
                                       ), sessions, smoothDist=smoothDist,  parallelize=False, runImmediately=False)
            allSpecs.append((makeFracExcursionsMeasure, {
                "normalization": ["session", "bp", "none"],
                "radius": allConsideredRadii,
                # **basicParams
                "bp": allConvexBPs,
                "smoothDist": [0, 0.5, 1]
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
                                       smoothDist=smoothDist, parallelize=False, runImmediately=False)
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
                                       smoothDist=smoothDist, parallelize=False, runImmediately=False)
            allSpecs.append((makeGravityMeasure, {
                "passRadius": np.linspace(0.5, 1.5, 3),
                "visitRadiusFactor": np.linspace(0.2, 0.4, 2),
                "passDenoiseFactor": [1.25],
                # "passDenoiseFactor": np.linspace(1.25, 2, 3),
                # **basicParams
                "bp": allConsideredBPs,
                "smoothDist": [0, 0.5]
            }))

            def makeCoarseTimeMeasure(func: Callable, inProbe: bool):
                probeStr = "probe" if inProbe else "bt"
                return TimeMeasure(f"{func.__name__}_{probeStr}", lambda sesh, t0, t1, ip, _: func(sesh, t0, t1, ip),
                                   sessions, timePeriodsGenerator=(30, 15, inProbe), runImmediately=False)

            allSpecs.append((makeCoarseTimeMeasure, {
                "func": [speedFunc, fracExploreFunc, fracOffWallFunc],
                "inProbe": [True, False]
            }))

            def makeNumWellsVisitedMeasure(countRepeats: bool, offWallOnly: bool, inProbe: bool, mode: str):
                probeStr = "probe" if inProbe else "bt"
                return SessionMeasure(f"numWellsVisitedPer{mode.capitalize()}_{probeStr}_cr{countRepeats}_ow{offWallOnly}",
                                      partial(avgNumWellsVisitedPerPeriod, countRepeats,
                                              offWallOnly, mode, inProbe), sessions,
                                      parallelize=False, runImmediately=False)

            allSpecs.append((makeNumWellsVisitedMeasure, {
                "countRepeats": [True, False],
                "offWallOnly": [True, False],
                "inProbe": [True, False],
                "mode": ["bout", "excursion"]
            }))

            def makeTotalDurationMeasure(inProbe: bool):
                probeStr = "probe" if inProbe else "bt"
                return SessionMeasure(f"totalDuration_{probeStr}",
                                      lambda sesh: sesh.probeDuration if inProbe else sesh.btDuration,
                                      sessions,
                                      parallelize=False, runImmediately=False)

            allSpecs.append((makeTotalDurationMeasure, {
                "inProbe": [True, False]
            }))

            def makePerPeriodMeasure(func: Callable, mode: str, inProbe: bool):
                probeStr = "probe" if inProbe else "bt"
                return SessionMeasure(f"{func.__name__}_{probeStr}_{mode}", partial(func, mode, inProbe), sessions,
                                      parallelize=False, runImmediately=False)

            allSpecs.append((makePerPeriodMeasure, {
                "func": [avgDurationPerPeriod, avgPathLengthPerPeriod],
                "mode": ["bout", "excursion"],
                "inProbe": [True, False]
            }))

            def makeTimePeriodsMeasure(func: Callable, timePeriodsGenerator):
                return TimeMeasure(f"{func.__name__}_{timePeriodsGenerator[1]}", lambda sesh, t0, t1, inProbe, _: func(sesh, t0, t1, inProbe),
                                   sessions, timePeriodsGenerator=timePeriodsGenerator[0], parallelize=False, runImmediately=False)

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
                                   timePeriodsGenerator=timePeriodsGenerator[0], runImmediately=False)

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
                        sesh.btRipsPreStats) / sesh.taskDuration, sessions, parallelize=False),
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
                                   sesh.taskDuration, sessions, parallelize=False),
                    SessionMeasure("rippleRatePreStats", lambda sesh: len(
                        sesh.btRipsPreStats) / sesh.taskDuration, sessions, parallelize=False),
                    SessionMeasure("rippleRateProbeStats", lambda sesh: len(
                        sesh.btRipsProbeStats) / sesh.taskDuration, sessions, parallelize=False),
                ]

            for sm in correlationSMs:
                sm.makeFigures(pp, excludeFromCombo=True)

            reversedSpecs = reversed(list(enumerate(allSpecs)))
            for mi, (makeMeasure, paramDict) in reversedSpecs:
                # For debugging, removing stuff that's already been run
                if testData:
                    if makeMeasure.__name__ in [
                        "measureFromFunc",
                        "makeGravityMeasure",
                        "makePathLengthInExcursionMeasure"
                        "makePathOptimalityMeasure",
                        "makeFracExcursionsMeasure",
                        "makeNumWellsVisitedMeasure",
                    ]:
                        continue
                    # if makeMeasure.__name__ not in [
                    #     "makePathLengthInExcursionMeasure"
                    # ]:
                    #     continue

                    hadABool = False
                    for k, v in paramDict.items():
                        if k == "fillNanMode":
                            paramDict[k] = [np.nan]
                        elif all(isinstance(x, (int, float)) for x in v) and not all(isinstance(x, bool) for x in v):
                            paramDict[k] = v[:1]
                        elif k == "bp":
                            if makeMeasure.__name__ == "makeFracExcursionsMeasure":
                                paramDict[k] = [v[2], v[7]]
                            elif makeMeasure.__name__ == "makeDotProdMeasure":
                                paramDict[k] = [BP(probe=True, timeInterval=(30, 31))]
                            else:
                                paramDict[k] = v[:2]
                        elif k == "onlyPositive":
                            paramDict[k] = [True]
                        elif k == "inProbe":
                            paramDict[k] = [True]
                        elif all(isinstance(x, bool) for x in v):
                            hadABool = True

                    if not hadABool and makeMeasure.__name__ != "makeFracExcursionsMeasure":
                        continue

                print("Running", makeMeasure.__name__, f"({mi+1}/{len(allSpecs)})")

                pp.pushOutputSubDir(makeMeasure.__name__)
                if testData:
                    runDMFOnM(makeMeasure, paramDict, pp, correlationSMs,
                              verbose=True, minParamsSequential=-1)
                    # runDMFOnM(makeMeasure, paramDict, pp, correlationSMs)
                else:
                    # runDMFOnM(makeMeasure, paramDict, pp, correlationSMs, verbose=True)
                    runDMFOnM(makeMeasure, paramDict, pp, correlationSMs)
                pp.popOutputSubDir()

            if testData:
                pp.runShuffles(numShuffles=6, significantThreshold=0.5)
            else:
                pp.runShuffles()

        if len(plotFlags) != 0:
            print("Warning, unused plot flags:", plotFlags)


if __name__ == "__main__":
    main("fullDatamine", dataMine=True, testData=False)
    # main()

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
# - Instead of ctrl vs stim, correlate with ripple count. Stim count too? might as well
# - test out new measures and new flag arguments
# - remember to add in all new measures and parameters to the datamining factory. Can also remove results that get filtered out later based on BP
