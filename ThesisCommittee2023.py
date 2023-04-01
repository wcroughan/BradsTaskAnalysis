import os
from datetime import datetime
import time
from functools import partial

from BTData import BTData
from PlotUtil import PlotManager
from UtilFunctions import getLoadInfo, findDataDir
from MeasureTypes import LocationMeasure, SessionMeasure
from BTSession import BTSession
from BTSession import BehaviorPeriod as BP

# how often are aways checked on way to home. Specifically t2-7
# How often are previously rewarded wells checked during away trials? Specifically t2-7
# : does ripple count correlate with first trial duration? same with stim count
# : more stims = more probe OW/E/M time?
# :just scatter stim rate vs rip rate
# :remember with avg time about missing values too!
# :ripple rate seems more highly correlated with task things, whereas stim rate correlates with probe things ... interpretation?
# :I think it's really important to see where the ripples/stims are happening. For each correlation explanation, can p easily think of where I think they're happening, need to test that
# :and all these things as a function of session index


def makeFigures(plotFlags="all"):
    if isinstance(plotFlags, str):
        plotFlags = [plotFlags]

    ratName = "B17"
    ratInfo = getLoadInfo(ratName)
    dataFilename = os.path.join(ratInfo.output_dir, ratInfo.out_filename)
    print("loading from " + dataFilename)
    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    sessions = ratData.getSessions()

    infoFileName = datetime.now().strftime("Thesis2023_%Y%m%d_%H%M%S.txt")
    dataDir = findDataDir()
    outputDir = "Thesis2023"
    globalOutputDir = os.path.join(dataDir, "figures", outputDir)
    rseed = int(time.perf_counter())
    print("random seed =", rseed)
    pp = PlotManager(outputDir=globalOutputDir, randomSeed=rseed,
                     infoFileName=infoFileName, verbosity=3)

    if "all" in plotFlags or "rippleLocations" in plotFlags:
        try:
            plotFlags.remove("rippleLocations")
        except ValueError:
            pass

        LocationMeasure("taskRippleCount", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.numRipplesAtLocation, sesh, BP(probe=False))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)
        LocationMeasure("probeRippleCount", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.numRipplesAtLocation, sesh, BP(probe=True))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)
        LocationMeasure("taskRippleRate", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.rippleRateAtLocation, sesh, BP(probe=False))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)
        LocationMeasure("probeRippleRate", lambda sesh:
                        sesh.getValueMap(
                            partial(BTSession.rippleRateAtLocation, sesh, BP(probe=True))
                        ), sessions, smoothDist=0.5).makeFigures(pp, everySessionBehaviorPeriod=BP(probe=True), excludeFromCombo=True)

    if "all" in plotFlags or "ripVsStimRate" in plotFlags:
        try:
            plotFlags.remove("ripVsStimRate")
        except ValueError:
            pass

        stimCount = SessionMeasure("stimCount", lambda sesh: len(sesh.btLFPBumps_posIdx), sessions)
        rippleCount = SessionMeasure(
            "rippleCount", lambda sesh: len(sesh.btRipsProbeStats), sessions)
        stimRate = SessionMeasure("stimRate", lambda sesh: len(sesh.btLFPBumps_posIdx) /
                                  sesh.taskDuration, sessions)
        rippleRate = SessionMeasure("rippleRate", lambda sesh: len(
            sesh.btRipsProbeStats) / sesh.taskDuration, sessions)

        stimCount.makeFigures(pp, excludeFromCombo=True)
        rippleCount.makeFigures(pp, excludeFromCombo=True)
        stimCount.makeCorrelationFigures(rippleCount)

        stimRate.makeFigures(pp, excludeFromCombo=True)
        rippleRate.makeFigures(pp, excludeFromCombo=True)
        stimRate.makeCorrelationFigures(rippleRate)


def main():
    makeFigures()


if __name__ == "__main__":
    main()
