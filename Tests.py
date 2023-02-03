# from UtilFunctions import getRipplePower
import unittest
from ImportData import extractAndSave
from UtilFunctions import ImportOptions, getLoadInfo, LoadInfo
from BTData import BTData
import os


class SaveLoadTest(unittest.TestCase):
    def testSaveAndLoad(self):
        configName = "B17"
        importOptions = ImportOptions()
        importOptions.debugMode = True
        importOptions.debug_dontSave = False
        importOptions.debug_maxNumSessions = 1
        extractAndSave(configName, importOptions)

        loadInfo = getLoadInfo(configName)
        dataFilename = os.path.join(loadInfo.output_dir, loadInfo.out_filename)
        ratData = BTData()
        ratData.loadFromFile(dataFilename)
        print(ratData.allSessions)
