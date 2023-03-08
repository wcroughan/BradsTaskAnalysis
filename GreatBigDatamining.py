from typing import Callable, Iterable, Dict
from MeasureTypes import LocationMeasure
from PlotUtil import PlotManager
from DataminingFactory import runEveryCombination


def runDMFOnLM(measureFunc: Callable[..., LocationMeasure], allParams: Dict[str, Iterable], pp: PlotManager, minParamsSequential=2):
    def measureFuncWrapper(params: Dict[str, Iterable]):
        lm = measureFunc(**params)
        if "bp" in params:
            lm.makeFigures(pp, excludeFromCombo=True, everySessionBehaviorPeriod=params["bp"])
        else:
            lm.makeFigures(pp, excludeFromCombo=True)

    runEveryCombination(measureFuncWrapper, allParams,
                        numParametersSequential=min(minParamsSequential, len(allParams)))

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
# - Import or move important functions so can easily run lots of measures from any file
# - Why are ripple counts so low?
# - Think about each measure during task, what possible caveats are there?
#   - Is erode 3 enough?
# - Should prefer away control location? Or neutral? Dependent on measure?
# - Add in check when summarizing results for difference direction. Separate significant results this way
# - look also at specific task behavior periods like early learning
# - Try different measures of learning rate, specifically targetted at possible effect from paper
# - the following mods to existing measures:
#   - dot prod just positive
#   - frac excursion, avg over excursion not session
#   - Latency nans => max value + penalty. Possibly also path opt, others?
#   - path opt from start of excursion
# - implement the following new measures
#   - path length from start of excursion
#       - sym ctrl locations here. Unless want to subdivide by well location type or something fancy
#   - General behavior differences: speed, num wells visited per excursion/bout, frac in exploration, others from previous lab meetings/thesis meetings
# - Instead of ctrl vs stim, correlate with ripple count. Stim count too? might as well
# - Change how psudo-pval is presented. Histogram of shuffle values, line for real data, p value, and written i.e. "ctrl > swr"
# - Difference in stim rates b/w conditions? Probs not, but should run stats
