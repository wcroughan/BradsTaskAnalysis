import os

from BTData import BTData
from BTSession import BTSession
from UtilFunctions import getLoadInfo
from MeasureTypes import TimeMeasure, TrialMeasure, SessionMeasure, LocationMeasure
from PlotUtil import PlotManager


def main():
    animalName = "B17"
    animalInfo = getLoadInfo(animalName)
    # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
    dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
    print("loading from " + dataFilename)
    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    sessions = ratData.getSessions()

    pm = PlotManager.getGlobalPlotManager(
        outputDir=os.path.join(os.curdir, "examplePlots"), randomSeed=0)

    # :param measureFunc: a function that takes a session, a start posIdx, an end posIdx, a bool indicating whether posIdxs are in probe,
    #     and a value of type T and returns a float
    m1 = TimeMeasure("time", lambda sesh, start, end, inProbe, val: end - start, sessions)
    m1.makeFigures(pm)

    # measureFunc(session, trialStart_posIdx, trialEnd_posIdx, trial type ("home" | "away")) -> measure value
    m2 = TrialMeasure("trial", lambda sesh, start, end, type: end - start, sessions)
    m2.makeFigures(pm)

    m3 = SessionMeasure("session", lambda sesh: sesh.btPos_secs[-1], sessions)
    m3.makeFigures(pm)

    m4 = LocationMeasure("location", lambda sesh: sesh.getTestMap(), sessions, smoothDist=0.5)
    m4.makeFigures(pm)


if __name__ == "__main__":
    main()
