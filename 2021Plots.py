import os

from MyPlottingFunctions import MyPlottingFunctions
from BTData import BTData

animals = ["B13", "B14"]

TRODES_SAMPLING_RATE = 30000

for animal_name in animals:
    if animal_name == "B13":
        data_filename = "/media/WDC7/B13/processed_data/B13_bradtask.dat"
        output_dir = "/media/WDC7/B13/processed_data/behavior_figures/"
    elif animal_name == "B14":
        data_filename = "/media/WDC7/B14/processed_data/B14_bradtask.dat"
        output_dir = "/media/WDC7/B14/processed_data/behavior_figures/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alldata = BTData()
    alldata.loadFromFile(data_filename)
    all_sessions = alldata.getSessions()
    P = MyPlottingFunctions(all_sessions, output_dir)
    tlbls = [P.trial_label(sesh) for sesh in all_sessions]

    # probe_avg_dwell_time_60sec
    P.makeAPersevMeasurePlot("probe_total_dwell_time_60sec",
                             lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_avg_dwell_time_60sec",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))

    # probe_avg_dwell_time
    P.makeAPersevMeasurePlot("probe_total_dwell_time",
                             lambda s, w: s.total_dwell_time(True, w))
    P.makeAPersevMeasurePlot("probe_avg_dwell_time",
                             lambda s, w: s.avg_dwell_time(True, w))

    # meandisttohomewell
    P.makeABoxPlot([sesh.avg_dist_to_home_well(True) for sesh in all_sessions],
                   tlbls, ['Condition', 'MeanDistToHomeWell'], title="Probe Mean Dist to Home")
    P.makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well) for sesh in all_sessions],
                   tlbls, ['Condition', 'MeanDistToCtrlHomeWell'], title="Probe Mean Dist to Ctrl Home")

    P.makeABoxPlot([sesh.avg_dist_to_home_well(True, timeInterval=[0, 60]) for sesh in all_sessions],
                   tlbls, ['Condition', 'MeanDistToHomeWell'], title="Probe Mean Dist to Home")
    P.makeABoxPlot([sesh.avg_dist_to_well(True, sesh.ctrl_home_well, timeInterval=[0, 60]) for sesh in all_sessions],
                   tlbls, ['Condition', 'MeanDistToCtrlHomeWell'], title="Probe Mean Dist to Ctrl Home")
