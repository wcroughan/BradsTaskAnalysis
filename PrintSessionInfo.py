import sys
import pandas as pd
import os
import numpy as np

from UtilFunctions import findDataDir, AnimalInfo, getInfoForAnimal
from BTData import BTData

# dataDir = findDataDir()

if len(sys.argv) >= 2:
    animalNames = sys.argv[1:]
else:
    animalNames = ['B13', 'B14', 'Martin']
print("Printing info for animals ", animalNames)

for animalName in animalNames:
    animalInfo = getInfoForAnimal(animalName)
    dataFilename = os.path.join(animalInfo.output_dir, animalInfo.out_filename)

    ratData = BTData()
    ratData.loadFromFile(dataFilename)
    sessions = ratData.getSessions()

    nCtrl = np.count_nonzero([not sesh.isRippleInterruption for sesh in sessions])
    nSWR = np.count_nonzero([sesh.isRippleInterruption for sesh in sessions])
    nCtrlWithProbe = np.count_nonzero(
        [(not sesh.isRippleInterruption) and sesh.probe_performed for sesh in sessions])
    nSWRWithProbe = np.count_nonzero(
        [sesh.isRippleInterruption and sesh.probe_performed for sesh in sessions])

    d = {"# sessions": [nCtrl, nSWR],
         "# sessions w/ probe": [nCtrlWithProbe, nSWRWithProbe]}
    df1 = pd.DataFrame(data=d, index=["Ctrl", "SWR"])

    print("==========================")
    print(animalName)
    print(df1)

    for si, sesh in enumerate(sessions):
        print("{}\t{}\t{}".format(si, sesh.name, "SWR" if sesh.isRippleInterruption else "Ctrl"))
