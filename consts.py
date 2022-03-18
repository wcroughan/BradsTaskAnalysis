import numpy as np

all_well_names = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])
allWellNames = all_well_names
TRODES_SAMPLING_RATE = 30000
LFP_SAMPLING_RATE = 1500.0
