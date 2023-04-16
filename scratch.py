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
    sessions = ratData.getSessions()[:6]

    pm = PlotManager.getGlobalPlotManager(
        outputDir=os.path.join(os.curdir, "examplePlots"), randomSeed=0)

    m4 = LocationMeasure("location", lambda sesh: sesh.getTestMap(), sessions, smoothDist=0.5)
    m4.makeFigures(pm)

    m4 = LocationMeasure("location2", lambda sesh: sesh.getTestMap() -
                         1000, sessions, smoothDist=0.5)
    m4.makeFigures(pm)

    pm.makeCombinedFigs()
    pm.runImmediateShufflesAcrossPersistentCategories()


if __name__ == "__main__":
    main()
