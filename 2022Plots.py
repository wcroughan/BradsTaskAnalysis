import os

from MyPlottingFunctions import MyPlottingFunctions
from BTData import BTData

animals = ["B13", "B14"]

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
    P = MyPlottingFunctions(alldata, output_dir)

    # probe_avg_dwell_time_60sec
    P.makeAPersevMeasurePlot("probe_total_dwell_time_60sec",
                             lambda s, w: s.total_dwell_time(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_avg_dwell_time_60sec",
                             lambda s, w: s.avg_dwell_time(True, w, timeInterval=[0, 60]))
    P.makeAPersevMeasurePlot("probe_total_dwell_time",
                             lambda s, w: s.total_dwell_time(True, w))
    P.makeAPersevMeasurePlot("probe_avg_dwell_time",
                             lambda s, w: s.avg_dwell_time(True, w))

    # meandisttohomewell
    P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_home_well(True),
                         "Probe Mean Dist to Home", yAxisName="Mean Dist to Home")
    P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_well(True, s.ctrl_home_well),
                         "Probe Mean Dist to Ctrl Home", yAxisName="Mean Dist to Ctrl Home")

    P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_home_well(True, timeInterval=[0, 60]),
                         "Probe Mean Dist to Home 60sec", yAxisName="Mean Dist to Home")
    P.makeASimpleBoxPlot(lambda s: s.avg_dist_to_well(True, s.ctrl_home_well, timeInterval=[0, 60]),
                         "Probe Mean Dist to Ctrl Home 60sec", yAxisName="Mean Dist to Ctrl Home")
