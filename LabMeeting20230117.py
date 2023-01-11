import os
import math
import time

from MeasureTypes import WellMeasure
from UtilFunctions import findDataDir, parseCmdLineAnimalNames, getInfoForAnimal
from PlotUtil import PlotCtx, setupBehaviorTracePlot
from BTData import BTData
from datetime import datetime
from consts import offWallWellNames
from BTSession import BTSession

# TODOs
# Also use other sessions same well as comparison not just same session other wells
#   b/c tracking is different around some specific wells
#   Of course if DLC works well don't have to worry