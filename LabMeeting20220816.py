import numpy as np
from PlotUtil import PlotCtx, plotIndividualAndAverage, setupBehaviorTracePlot, boxPlot
import MountainViewIO
from BTData import BTData
from BTSession import BTSession
import os
from UtilFunctions import getInfoForAnimal, findDataDir, parseCmdLineAnimalNames
from consts import TRODES_SAMPLING_RATE, offWallWellNames
import math


def makeFigures(
    MAKE_LAST_MEETING_FIGS=True,
    MAKE_SPIKE_ANALYSIS_FIGS=True,
    MAKE_CLOSE_PASS_FIGS=True,
    MAKE_INITIAL_HEADING_FIGS=True
):
    dataDir = findDataDir()
    globalOutputDir = os.path.join(dataDir, "figures", "20220607_labmeeting")
    rseed = int(time.perf_counter())
    print("random seed =", rseed)
    pp = PlotCtx(outputDir=globalOutputDir, randomSeed=rseed, priorityLevel=1)

    animalNames = parseCmdLineAnimalNames(default=["B13", "B14", "Martin"])
    allSessionsByRat = {}
    for animalName in animalNames:
        animalInfo = getInfoForAnimal(animalName)
        # dataFilename = os.path.join(dataDir, animalName, "processed_data", animalInfo.out_filename)
        dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        allSessionsByRat[animalName] = ratData.getSessions()

    for ratName in animalNames:
        print("======================\n", ratName)
        sessions = allSessionsByRat[ratName]

        if MAKE_LAST_MEETING_FIGS:
            # All probe measures, within-session differences:
            # avg dwell time
            # avg dwell time, 90sec
            # curvature
            # curvature, 90sec
            # num entries
            # latency
            # optimality
            # gravity
            # gravity from offwall

            # task measures, within-session:
            # number of repeat well visits
            # gravity
            # gravity off-wall

            # follow-ups
            # correlation b/w trial duration and probe behavior?
            # task measures but only after learning (maybe T3-5 and onward?)
            # Evening vs morning session

            pass
        else:
            print("warning: skipping last meeting figs")

        if MAKE_SPIKE_ANALYSIS_FIGS:
            # Any cells for B13? If so, make spike calib plot
            # Latency of detection. Compare to Jadhav, or others that have shown this
            pass
        else:
            print("warning: skipping spike analysis figs")


if __name__ == "__main__":
    makeFigures()


# TODO: new analyses
# in addition to gravity modifications, should see how often goes near and doesn't get it,
# or how close goes and doesn't get it. Maybe histogram of passes where he does or doesn't find reward
#
# another possible measure, when they find away and are facing away from the home well, how often do they continue
# the direction they're facing (most of the time they do this) vs turn around immediately.
# More generally, some bias level between facing dir and home dir
#
# color swarm plots by trial index or date
#
# Pseudo-probe
# Any chance they return to same-condition home well during this? I.e. during pseudoprobe, will check out previous interruption home well
# on an interruption session even though there's been delay sessions in between? (and vice versa)
