import os
import re
# import cv2
import sys
import csv
import time
import pprint   # Pretty printing of lists, dictionaries etc.
import numpy as np
import itertools
import datetime
from scipy.stats import linregress

# Qt5 imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QDialog, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QPushButton, QSlider, QRadioButton, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout

# Matplotlib in Qt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, writers

# Local imports
import MountainViewIO
import QtHelperUtils

MODULE_IDENTIFIER = "[PositionProcessor] "
SPEED_THRESHOLD = 40.0
MIN_CAMERA_TSTAMP_JUMP = 100
ANIMATION_FRAME_INTERVAL = 20
N_MOVIE_FRAMES = 1000
SMOOTHING_WINDOW_LENGTH = 3
DEFAULT_SMOOTHING_CHOICE = True
DEFAULT_CLIPPING_CHOICE = True
DEFAULT_INTERPOLATION_CHOICE = True
DEFAULT_LINEARIZATION_CHOICE = False
POSITION_INTERPOLATION_TSTAMP_JUMP = 1500.0
MAX_JUMP_DISTANCE = 20.0
SLIDER_TIME_RANGE = 60.0
PROCESSED_POSITION_DTYPE = [(MountainViewIO.TIMESTAMP_LABEL, 'float'), 
        (MountainViewIO.X_LOC1_LABEL, 'float'), (MountainViewIO.Y_LOC1_LABEL, 'float')]
CLIP_FILE_EXTENSION = '.clips2'
MAX_X_LOC = 1075
MIN_X_LOC = -80
MAX_Y_LOC = 1225
MIN_Y_LOC = 100
TIME_CHUNK_SIZE = 300 # 5 mins of data will be displayed at once.
MIRROR_X_LOCATION = True

def readClips(clips_file=None, data_dir=None):
    """
    Read clips written for a particular position-tracking file.

    :clips_file: Location of the clips file.
    :returns: Continuous time periods for which recording data is available.
    """
    if clips_file is None:
        if data_dir is None:
            data_dir = os.getcwd()
        clips_file = QtHelperUtils.get_open_file_name(data_dir, message="Choose position clips file",
                file_format='PositionClips (*' + CLIP_FILE_EXTENSION + ')')

    # TODO: Add a check to test if the file already exists, in which case we need to ask the user if he wants the files to be overwritten
    clip_data = list()
    try:
        with open(clips_file, 'r') as f_csv:
            csv_reader = csv.reader(f_csv)
            for data_row in csv_reader:
                if data_row:
                    clip_data.append(data_row)
    except (FileNotFoundError, IOError) as err:
        print(MODULE_IDENTIFIER + 'Unable to read clips file. Aborting')
        print(err)
    finally:
        return clip_data

class PositionPlot(QDialog):

    """
    Widget for showing position data and working with it to extract useful
    information in an experiment.
    """
    def clearAxes(self):
        self.position_axes.clear()
        self.position_axes.set_xlabel('X (cm)')
        self.position_axes.set_ylabel('Y (cm)')
        self.position_axes.grid(True)

        self.other_axes.clear()
        self.other_axes.set_xlabel('Time (s)')
        self.other_axes.set_ylabel(self.other_axes_label)
        self.other_axes.grid(True)

        self.canvas.draw()

    def pauseAnimation(self):
        self.play_button.setText('Play')
        self.animation_paused = True
        if self.anim_obj is not None:
            self.anim_obj.event_source.stop()

    def getAnimationFrameIdx(self):
        if self.clips:
            last_tpt = self.clips[self.clip_selection.currentIndex()][-1]
        else:
            last_tpt = int(self.position_data[MountainViewIO.TIMESTAMP_LABEL][-1])

        while self.last_animation_step < last_tpt:
            if not self.animation_paused:
                self.last_animation_step += self.animation_increment
            else:
                time.sleep(0.005)
            yield self.last_animation_step 

    def initAnimationFrame(self):
        self.clearAxes()
        self.refresh()
        self.other_frame, = self.other_axes.plot(np.full(10, self.start_time_slider.value()), np.linspace(0, SPEED_THRESHOLD, 10), animated=True)
        self.position_frame, = self.position_axes.plot(self.position_data[MountainViewIO.X_LOC1_LABEL][0], self.position_data[MountainViewIO.Y_LOC1_LABEL][0], linewidth=2.0, color='black', animated=True)
        return self.position_frame, self.other_frame

    def nextAnimationFrame(self, step=0):
        # print('Animation frame: ' + str(step))
        if not self.animation_paused:
            self.start_time_value.setText(str(datetime.timedelta(seconds=int(self.first_animation_step/MountainViewIO.SPIKE_SAMPLING_RATE))))
            self.last_animation_step = step
            self.first_animation_step = step - self.time_length_slider.value()
            position_start_idx = np.searchsorted(self.position_data[MountainViewIO.TIMESTAMP_LABEL], self.first_animation_step)
            position_finish_idx = np.searchsorted(self.position_data[MountainViewIO.TIMESTAMP_LABEL], self.last_animation_step)

            self.position_frame.set_data(self.position_data[MountainViewIO.X_LOC1_LABEL][position_start_idx:position_finish_idx], \
                    self.position_data[MountainViewIO.Y_LOC1_LABEL][position_start_idx:position_finish_idx])
            self.other_frame.set_data(np.full(10, self.first_animation_step), \
                    np.linspace(0, SPEED_THRESHOLD, 10))
        return self.position_frame, self.other_frame

    def Load(self):
        self.clearAxes()
        self.position_axes.plot(self.position_data[MountainViewIO.X_LOC1_LABEL], self.position_data[MountainViewIO.Y_LOC1_LABEL], color='blue')
        self.canvas.draw()

    def initTimeChunks(self):
        raise NotImplementedError()

    def updateTimeSliderRange(self):
        if self.time_chunks is None:
            return

        # Set the application sliders
        self.start_time_slider.setMinimum(self.time_chunks[self.current_chunk_idx,0])
        self.start_time_slider.setMaximum(self.time_chunks[self.current_chunk_idx,1])
        self.plotTimeValueChanged()

    def NextChunk(self):
        if self.time_chunks is None:
            return

        # Move to the next time chunk if there is room to move
        if self.current_chunk_idx < len(self.time_chunks):
            self.current_chunk_idx += 1
            self.start_time_slider.setValue(self.time_chunks[self.current_chunk_idx,0])
            self.updateTimeSliderRange()

    def PrevChunk(self):
        if self.time_chunks is None:
            return

        # Move to the prev time chunk if there 
        if self.current_chunk_idx > 0:
            self.current_chunk_idx -= 1
            self.start_time_slider.setValue(self.time_chunks[self.current_chunk_idx,1])
            self.updateTimeSliderRange()

    def PlayPause(self, _, save_file_name=None):
        if self.animation_paused:
            self.animation_paused = False
            if self.anim_obj is None:
                if save_file_name is None:
                    self.anim_obj = FuncAnimation(self.canvas.figure, self.nextAnimationFrame, 
                            self.getAnimationFrameIdx, init_func=self.initAnimationFrame,
                            interval=ANIMATION_FRAME_INTERVAL, blit=True, repeat=False)
                else:
                    self.anim_obj = FuncAnimation(self.canvas.figure, self.nextAnimationFrame,\
                            self.getAnimationFrameIdx, init_func=self.initAnimationFrame,\
                            interval=ANIMATION_FRAME_INTERVAL, blit=True, save_count=N_MOVIE_FRAMES)
                    # Set up formatting for the movie files
                    animation_writer = writers['ffmpeg']
                    animation_writer = animation_writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                    self.anim_obj.save(save_file_name, writer=animation_writer)
                    QtHelperUtils.display_information('Movie saved as ' + save_file_name)
                    self.anim_obj.event_source.stop()
            else:
                self.initAnimationFrame()
                self.anim_obj.event_source.start()
            self.play_button.setText('Pause')
        else:
            self.pauseAnimation()
            self.start_time_slider.setValue(self.first_animation_step)
            self.position_frame = None
            self.other_frame = None
            self.posterior_frame = None
            self.plotTimeValueChanged()

    def getPositionProcessingArgs(self):
        processing_args = list()
        processing_args.append("Smooth")
        processing_args.append("Interpolate")
        processing_args.append("Clip")
        processing_args.append("Linearize")
        user_choices = QtHelperUtils.CheckBoxWidget(processing_args, message="Select position processing options.", \
                default_choices=[DEFAULT_SMOOTHING_CHOICE, DEFAULT_INTERPOLATION_CHOICE, DEFAULT_CLIPPING_CHOICE, \
                DEFAULT_LINEARIZATION_CHOICE]).exec_()
        user_processing_choices = dict()
        user_processing_choices['smooth'] = DEFAULT_SMOOTHING_CHOICE
        user_processing_choices['interp'] = DEFAULT_INTERPOLATION_CHOICE
        user_processing_choices['clip'] = DEFAULT_CLIPPING_CHOICE
        user_processing_choices['linear'] = DEFAULT_LINEARIZATION_CHOICE
        if user_choices[0] == QMessageBox.Ok:
            if 0 in user_choices[1]:
                user_processing_choices['smooth'] = True
            else:
                user_processing_choices['smooth'] = False

            if 1 in user_choices[1]:
                user_processing_choices['interp'] = True
            else:
                user_processing_choices['interp'] = False

            if 2 in user_choices[1]:
                user_processing_choices['clip'] = True
            else:
                user_processing_choices['clip'] = False

            if 3 in user_choices[1]:
                user_processing_choices['linear'] = True
            else:
                user_processing_choices['linear'] = False
        return user_processing_choices

    def updateSpeedInfo(self):
        # Get user choices for what all needs to be done
        user_processing_choices = self.getPositionProcessingArgs()

        # Look at position data before any cleanup
        if self.show_debug_plots:
            self.other_axes.plot(self.position_data[MountainViewIO.TIMESTAMP_LABEL], \
                    self.position_data[MountainViewIO.X_LOC1_LABEL])
            self.other_axes.set_xlabel('timestamp')
            self.other_axes.set_ylabel('position (cm)')
            self.canvas.draw()
            QtHelperUtils.display_information('Click OK to see processed position data.')

        # Calculate timestamp jumps to get any discrepancies in the data
        last_tstamp_jump = self.position_data[MountainViewIO.TIMESTAMP_LABEL][-1] - \
                self.position_data[MountainViewIO.TIMESTAMP_LABEL][-2]
        tstamp_jumps  = np.append(np.diff(self.position_data[MountainViewIO.TIMESTAMP_LABEL]), \
                last_tstamp_jump)

        if (tstamp_jumps < 0.0).any():
            QtHelperUtils.display_warning('Position timestamps have NEGATIVE jumps!')

        increasing_timestamps = (tstamp_jumps > 0.0)
        if not increasing_timestamps.all():
            if self.show_debug_plots:
                QtHelperUtils.display_warning('Position timestamps have repeats!')
            self.position_data = self.position_data[increasing_timestamps]

        # Clean-up any large jumps in the position data
        for clean_round in range(5):
            position_jumps = np.sqrt(np.square(np.diff(\
                    self.position_data[MountainViewIO.X_LOC1_LABEL], prepend=self.position_data[MountainViewIO.X_LOC1_LABEL][0])) + \
                    np.square(np.diff(\
                    self.position_data[MountainViewIO.Y_LOC1_LABEL], prepend=self.position_data[MountainViewIO.Y_LOC1_LABEL][0])))
            accepted_position_jumps = position_jumps < MAX_JUMP_DISTANCE 
            print(max(position_jumps))
            self.position_data = self.position_data[accepted_position_jumps]

        if user_processing_choices['interp']:
            # Interpolate both the position and timestamp data.
            n_new_timestamps = int((self.position_data[MountainViewIO.TIMESTAMP_LABEL][-1] -\
                    self.position_data[MountainViewIO.TIMESTAMP_LABEL][0])/POSITION_INTERPOLATION_TSTAMP_JUMP)
            new_timestamps = np.linspace(self.position_data[MountainViewIO.TIMESTAMP_LABEL][0], \
                    self.position_data[MountainViewIO.TIMESTAMP_LABEL][-1], n_new_timestamps)
            interp_x_locs = np.interp(new_timestamps, self.position_data[MountainViewIO.TIMESTAMP_LABEL], \
                    self.position_data[MountainViewIO.X_LOC1_LABEL])
            interp_y_locs = np.interp(new_timestamps, self.position_data[MountainViewIO.TIMESTAMP_LABEL], \
                    self.position_data[MountainViewIO.Y_LOC1_LABEL])
            self.position_data = np.ndarray(n_new_timestamps, dtype=PROCESSED_POSITION_DTYPE)
            self.position_data[MountainViewIO.TIMESTAMP_LABEL] = new_timestamps
            self.position_data[MountainViewIO.X_LOC1_LABEL] = interp_x_locs
            self.position_data[MountainViewIO.Y_LOC1_LABEL] = interp_y_locs

        if user_processing_choices['smooth']:
            # Smooth the position data at the given timestamps
            smoothing_weights = np.ones(SMOOTHING_WINDOW_LENGTH)/SMOOTHING_WINDOW_LENGTH
            self.position_data[MountainViewIO.X_LOC1_LABEL]= np.convolve(\
                    self.position_data[MountainViewIO.X_LOC1_LABEL], smoothing_weights, mode='same')
            self.position_data[MountainViewIO.Y_LOC1_LABEL]= np.convolve(\
                    self.position_data[MountainViewIO.Y_LOC1_LABEL], smoothing_weights, mode='same')

        if user_processing_choices['linear']:
            # Linearize the data. The result could still be 2D data (as in
            # along a straight line), or we could just set one of the variables
            # to a constant value.

            # First, fit a line to the data
            # TODO: Can probably use the p-value to warn the user of a bad line fit!
            l_slope, l_intercept, _, _, _ = linregress(self.position_data[MountainViewIO.X_LOC1_LABEL],\
                    self.position_data[MountainViewIO.Y_LOC1_LABEL])

            # Point on the line which is closest to a given point x0, y0 is
            # x = ((x0 + my0) - mc)/(1 + m^2), y = mx + c
            x0 = np.array(self.position_data[MountainViewIO.X_LOC1_LABEL])
            y0 = np.array(self.position_data[MountainViewIO.Y_LOC1_LABEL])

            # Calculate the new position coordinates
            conv_denominator = 1.0/(1.0 + l_slope*l_slope)
            self.position_data[MountainViewIO.X_LOC1_LABEL] = conv_denominator * \
                    ((x0 + l_slope*y0) - l_slope*l_intercept)

            self.position_data[MountainViewIO.Y_LOC1_LABEL] = np.median(l_slope * \
                    self.position_data[MountainViewIO.X_LOC1_LABEL] + \
                    l_intercept)

        x_pixel_jumps = np.abs(np.diff(self.position_data[MountainViewIO.X_LOC1_LABEL]))
        y_pixel_jumps = np.abs(np.diff(self.position_data[MountainViewIO.Y_LOC1_LABEL]))
        time_jumps  = np.diff(self.position_data[MountainViewIO.TIMESTAMP_LABEL])/MountainViewIO.SPIKE_SAMPLING_RATE

        # Just look at taxi-cab distance to get the speed measure for now.
        self.speed_data = np.empty(len(self.position_data), dtype='float')
        np.divide(np.sqrt(np.square(x_pixel_jumps) + np.square(y_pixel_jumps)), \
                self.pixel_to_cm_conversion * time_jumps, out=self.speed_data[:-1])
        self.speed_data[-1] = self.speed_data[-2]

        # See if position data has been cleaned up
        if self.show_debug_plots:
            self.other_axes.plot(self.position_data[MountainViewIO.TIMESTAMP_LABEL], \
                    self.position_data[MountainViewIO.X_LOC1_LABEL])
            self.other_axes.set_xlabel('timestamp')
            self.other_axes.set_ylabel('position (cm)')
            self.canvas.draw()

        if self.show_speed_data or (QtHelperUtils.display_information('Click OK to see speed data.') == QMessageBox.Ok):
            self.clearAxes()
            self.other_axes.plot(self.position_data[MountainViewIO.TIMESTAMP_LABEL], \
                    self.speed_data)
            self.other_axes.set_xlabel('timestamp')
            self.other_axes.set_ylabel('speed (cm/s)')
            self.other_axes.set_ylim((0.0, SPEED_THRESHOLD))
            self.canvas.draw()

            # QtHelperUtils.display_information('Click OK to continue!')

    def setTimeChunks(self, time_chunk_data):
        self.time_chunks = np.copy(time_chunk_data)

    def setPositionData(self, data, update_speed=True):
        n_data_entries = len(data)
        self.position_data = np.ndarray(n_data_entries, dtype=PROCESSED_POSITION_DTYPE)
        self.position_data[MountainViewIO.TIMESTAMP_LABEL] = data[MountainViewIO.TIMESTAMP_LABEL]
        self.position_data[MountainViewIO.X_LOC1_LABEL] = data[MountainViewIO.X_LOC1_LABEL]
        self.position_data[MountainViewIO.Y_LOC1_LABEL] = data[MountainViewIO.Y_LOC1_LABEL]

        inv_pixel_to_cm_conversion = 1.0/self.pixel_to_cm_conversion
        self.position_data[MountainViewIO.X_LOC1_LABEL] *= inv_pixel_to_cm_conversion
        self.position_data[MountainViewIO.Y_LOC1_LABEL] *= inv_pixel_to_cm_conversion

        if MIRROR_X_LOCATION:
            self.position_data[MountainViewIO.X_LOC1_LABEL] *= -1

        if update_speed:
            self.updateSpeedInfo()

    def setLayout(self):
        # Set the layout
        # Put toolbar and canvas in a vertical box
        parent_layout_box = QVBoxLayout()
        parent_layout_box.addWidget(self.toolbar)
        parent_layout_box.addWidget(self.canvas)
        parent_layout_box.addStretch(1)
        hbox_controls = QHBoxLayout()

        vbox_time_labels = QVBoxLayout()
        vbox_time_labels.addWidget(self.start_time_value)
        vbox_time_labels.addWidget(self.time_length_value)

        vbox_sliders = QVBoxLayout()
        vbox_sliders.addWidget(self.start_time_slider)
        vbox_sliders.addWidget(self.time_length_slider)
        vbox_sliders.addStretch(1)
        hbox_frame_buttons = QHBoxLayout()
        hbox_frame_buttons.addWidget(self.next_chunk_button)
        hbox_frame_buttons.addWidget(self.prev_chunk_button)
        vbox_sliders.addLayout(hbox_frame_buttons)

        vbox_buttons = QVBoxLayout()
        vbox_buttons.addWidget(self.play_button)
        vbox_buttons.addWidget(self.clear_button)
        vbox_buttons.addStretch(1)

        hbox_controls.addLayout(vbox_time_labels)
        hbox_controls.addLayout(vbox_sliders)
        hbox_controls.addLayout(vbox_buttons)
        parent_layout_box.addLayout(hbox_controls)
        QDialog.setLayout(self, parent_layout_box)

    def refresh(self):
        self.position_axes.set_xlim((MIN_X_LOC/self.pixel_to_cm_conversion, \
                MAX_X_LOC/self.pixel_to_cm_conversion))
        self.position_axes.set_ylim((MIN_Y_LOC/self.pixel_to_cm_conversion, \
                MAX_Y_LOC/self.pixel_to_cm_conversion))
        if self.position_data is not None:
            self.trajectory_frame, = self.position_axes.plot(self.position_data[MountainViewIO.X_LOC1_LABEL],
                    self.position_data[MountainViewIO.Y_LOC1_LABEL], color='gray', alpha=0.5)
            """
            self.position_axes.set_xlim((min(self.position_data[MountainViewIO.X_LOC1_LABEL]), \
                    max(self.position_data[MountainViewIO.X_LOC1_LABEL])))
            self.position_axes.set_ylim((min(self.position_data[MountainViewIO.Y_LOC1_LABEL]), \
                    max(self.position_data[MountainViewIO.Y_LOC1_LABEL])))
            """
        if self.speed_data is not None:
            self.other_axes.plot(self.position_data[MountainViewIO.TIMESTAMP_LABEL], \
                    self.speed_data, c='r')
            self.other_axes.set_xlabel('timestamp')
            self.other_axes.set_ylabel('speed (cm/s)')
            self.other_axes.set_ylim((0.0, SPEED_THRESHOLD))

        self.canvas.draw()

    def plotTimeValueChanged(self, update_plots=True):
        """
        Action to be taken when the value of the start time slider has changed.
        """
        if self.position_data is None:
            return

        current_start_time = self.start_time_slider.value()
        current_time_length = self.time_length_slider.value()

        # Update the time labels
        self.start_time_value.setText(str(datetime.timedelta(seconds=int(current_start_time/MountainViewIO.SPIKE_SAMPLING_RATE))))
        self.time_length_value.setText(str(datetime.timedelta(seconds=int(current_time_length/MountainViewIO.SPIKE_SAMPLING_RATE))))

        # Show where we are in the OTHER frame. This could be LFP, population
        # activity, anything basically.
        other_axis_lims = self.other_axes.get_ylim()
        current_time_end_in_s = (current_start_time+current_time_length)/MountainViewIO.SPIKE_SAMPLING_RATE
        self.first_animation_step = current_start_time
        self.last_animation_step = current_start_time + current_time_length

        # Get start and stop indices for data corresponding to start and stop times.
        start_time_idx = np.searchsorted(self.position_data[MountainViewIO.TIMESTAMP_LABEL], current_start_time)
        end_time_idx = np.searchsorted(self.position_data[MountainViewIO.TIMESTAMP_LABEL], 
                current_start_time+current_time_length)

        # Use these two attributes to plot the trajectory that starts at the
        # current start time and goes on until the current time length.
        if self.anim_obj is not None:
            # Remove all the animation elements from here.
            self.pauseAnimation()
            self.anim_obj = None
            self.refresh()

        if self.position_frame is not None:
            self.position_frame.set_data(self.position_data[MountainViewIO.X_LOC1_LABEL][start_time_idx:end_time_idx], 
                    self.position_data[MountainViewIO.Y_LOC1_LABEL][start_time_idx:end_time_idx])
        else:
            self.position_frame, = self.position_axes.plot(self.position_data[MountainViewIO.X_LOC1_LABEL][
                start_time_idx:end_time_idx], self.position_data[MountainViewIO.Y_LOC1_LABEL][
                start_time_idx:end_time_idx], linewidth=2.0, color='black', animated=False)

        if self.other_frame is not None:
            self.other_frame.set_data(np.full(10, current_start_time), np.linspace(0, SPEED_THRESHOLD, 10))
        else:
            self.other_frame, = self.other_axes.plot(current_start_time * np.ones(10), np.linspace(0, SPEED_THRESHOLD, 10), animated=False, c='r')

        if update_plots:
            self.canvas.draw()

    def getStartTime(self):
        return self.start_time_slider.value()

    def getFinishTime(self):
        return self.time_length_slider.value() + self.start_time_slider.value()

    def __init__(self, init_display=True):
        """TODO: to be defined1. """
        QDialog.__init__(self)
        plt.ion()

        # Flags
        # TODO: Allow these to be set externally
        self.show_debug_plots = False
        self.show_speed_data = True

        # Set up the plot area
        self.figure = Figure(figsize=(12,16))
        self.canvas = FigureCanvas(self.figure)
        plot_grid   = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        self.position_axes = self.figure.add_subplot(plot_grid[0])
        self.other_axes = self.figure.add_subplot(plot_grid[1])
        self.anim_obj = None
        self.last_animation_step = 0
        self.first_animation_step = 0
        self.animation_paused = True
        self.trajectory_frame = None
        self.position_frame = None
        self.other_frame = None
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.other_axes_label = 'speed (cm/s)'

        # Animation parameters
        self.animation_increment = 0.05 * MountainViewIO.SPIKE_SAMPLING_RATE

        # Set slider for start and finish time
        self.start_time_slider = QSlider(Qt.Horizontal)
        self.start_time_slider.setTickInterval(SLIDER_TIME_RANGE * MountainViewIO.SPIKE_SAMPLING_RATE)
        self.start_time_slider.setTickPosition(QSlider.TicksBelow)
        self.start_time_slider.valueChanged.connect(self.plotTimeValueChanged)
        self.start_time_value = QLabel('00:00:00')

        self.time_length_slider = QSlider(Qt.Horizontal)
        self.time_length_slider.setMinimum(0.05 * MountainViewIO.SPIKE_SAMPLING_RATE)
        self.time_length_slider.setMaximum(SLIDER_TIME_RANGE * MountainViewIO.SPIKE_SAMPLING_RATE)
        self.time_length_slider.setTickInterval(1.0 * MountainViewIO.SPIKE_SAMPLING_RATE)
        self.time_length_slider.setTickPosition(QSlider.TicksBelow)
        self.time_length_slider.valueChanged.connect(self.plotTimeValueChanged)
        self.time_length_value = QLabel('00:00:00')

        # Button setup
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.PlayPause)
        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.clearAxes)

        # Update 2019/10/25: We are now working with very large time periods
        # and need to be able to scroll through time in smaller chunks. Adding
        # Next and Prev Buttons to navigate these chunks.
        self.time_chunks = None
        self.current_chunk_idx = -1
        self.next_chunk_button = QPushButton('Next')
        self.next_chunk_button.clicked.connect(self.NextChunk)
        self.prev_chunk_button = QPushButton('Prev')
        self.prev_chunk_button.clicked.connect(self.PrevChunk)

        self.clips = list()
        self.pixel_to_cm_conversion = 5.0
        self.position_data = None
        self.speed_data = None
        if init_display:
            self.setLayout()
            self.clearAxes()
        self.time_length_slider.setValue(0.5 * MountainViewIO.SPIKE_SAMPLING_RATE)
        
class PositionProcessorWindow(QMainWindow):

    """
    Main Window for processing position data.
    """

    def extractPosition(self):
        """
        Extract position data from video. Video must have been loaded first.
        TODO: If video hasn't been loaded, first load video from disk.

        See Also: loadPosition
        """
        raise NotImplementedError()

    def loadVideo(self):
        """
        Load Video (.mp4 or .h264) video.
        """
        self.raw_video_file = QtHelperUtils.get_open_file_name(\
                file_format="Video (*.mp4)", data_dir=self.data_dir)

        # Load the video metadata, the Camera Sync timestamps are needed to
        # look at the tracking data.
        # TODO

        # Get video parameters and set-up the sliders to mimic it
        # TODO
        return

    def loadTrajectory(self):
        self.trajectory_data_file = QtHelperUtils.get_open_file_name(\
                file_format="PositionTracking (*.handPositionTracking)",\
                data_dir=self.data_dir)
        self.loadTrajectoryFromFile(self.trajectory_data_file)

    def loadNextTrajectory(self):
        pattern = "(.*).1T(.*).handPositionTracking"
        ss = re.search(pattern, self.trajectory_data_file)
        if ss is None:
            print("Weird error parsing file name, cant get next file")
            return
            
        fn = ss.group(1) + ".1T" + str(int(ss.group(2))+1) + ".handPositionTracking"
        if not os.path.exists(fn):
            print("Couldnt find file, you need to load manually")
            return

        self.trajectory_data_file = fn
        self.loadTrajectoryFromFile(self.trajectory_data_file)

    def loadPrevTrajectory(self):
        pattern = "(.*).1T(.*).handPositionTracking"
        ss = re.search(pattern, self.trajectory_data_file)
        if ss is None:
            print("Weird error parsing file name, cant get next file")
            return
            
        fn = ss.group(1) + ".1T" + str(int(ss.group(2))-1) + ".handPositionTracking"
        if not os.path.exists(fn):
            print("Couldnt find file, you need to load manually")
            return

        self.trajectory_data_file = fn
        self.loadTrajectoryFromFile(self.trajectory_data_file)

    def loadTrajectoryFromFile(self, filename):
        """
        Load trajectory extracted from video and use it to clip trial ends.
        Useful for Water-Maze like experiments where the starting and
        end-points of a given trial might not be clear.
        """
        self.clips = list()

        try:
            with open(filename, 'r') as tf:
                print("loading from " + filename)

                timestamp_data = []
                x1_data = []
                y1_data = []
                x2_data = []
                y2_data = []
                csv_reader = csv.reader(tf)
                for data_row in csv_reader:
                    if data_row:
                        timestamp_data.append(int(data_row[0]))
                        x1_data.append(int(data_row[1]))
                        y1_data.append(int(data_row[2]))
                        x2_data.append(int(data_row[3]))
                        y2_data.append(int(data_row[4]))

                # Initialize an empty array
                n_data_entries = len(timestamp_data)
                position_data_type = [(MountainViewIO.TIMESTAMP_LABEL, \
                        MountainViewIO.TIMESTAMP_DTYPE), \
                        (MountainViewIO.X_LOC1_LABEL, MountainViewIO.POSITION_DTYPE), \
                        (MountainViewIO.Y_LOC1_LABEL, MountainViewIO.POSITION_DTYPE), \
                        (MountainViewIO.X_LOC2_LABEL, MountainViewIO.POSITION_DTYPE), \
                        (MountainViewIO.Y_LOC2_LABEL, MountainViewIO.POSITION_DTYPE)]
                trajectory_array = np.empty(n_data_entries, dtype=position_data_type)
                # print("Initialized empty array for position data tracking.")
    

                # Fill in the array with data extracted above
                # print(np.array(timestamp_data))
                np.copyto(trajectory_array[MountainViewIO.TIMESTAMP_LABEL], \
                        np.array(timestamp_data, dtype=MountainViewIO.TIMESTAMP_DTYPE))
                np.copyto(trajectory_array[MountainViewIO.X_LOC1_LABEL], \
                        np.array(x1_data, dtype=MountainViewIO.POSITION_DTYPE))
                np.copyto(trajectory_array[MountainViewIO.Y_LOC1_LABEL], \
                        np.array(y1_data, dtype=MountainViewIO.POSITION_DTYPE))
                np.copyto(trajectory_array[MountainViewIO.X_LOC2_LABEL], \
                        np.array(x2_data, dtype=MountainViewIO.POSITION_DTYPE))
                np.copyto(trajectory_array[MountainViewIO.Y_LOC2_LABEL], \
                        np.array(y2_data, dtype=MountainViewIO.POSITION_DTYPE))

                # Keep the data in a single chunk
                session_time_start = trajectory_array[MountainViewIO.TIMESTAMP_LABEL][0]
                session_time_end = trajectory_array[MountainViewIO.TIMESTAMP_LABEL][-1]
                time_chunks = np.empty((1, 2), dtype='float')
                time_chunks[0,0] = session_time_start
                time_chunks[0,1] = session_time_end

                self.statusBar().showMessage('Position data loaded!')
                self.plot.setTimeChunks(time_chunks)
                self.plot.setPositionData(trajectory_array)
                self.plot.updateTimeSliderRange()
                self.plot.refresh()
        except Exception as err:
            QtHelperUtils.display_warning("Unable to open/read file.")
            print(err)
            return

        

    def playVideo(self):
        """
        Play corresponding video data (matched up to the decoded trajectory).
        """
        if self.video_playing:
            return

        # if self.raw_video_file is None:
        #     QtHelperUtils.display_information("Video not loaded for playback.")

        # if self.video_capture is None:
        #     try:
        #         self.video_capture = cv2.VideoCapture(self.raw_video_file)
        #         cv2.startWindowThread()
        #         cv2.namedWindow(self.video_window_identifier, cv2.WINDOW_NORMAL)
        #         cv2.resizeWindow(self.video_window_identifier, \
        #                 int(self.FRAME_SIZE[1] * self.inv_pixel_to_cm_conversion), \
        #                 int(self.FRAME_SIZE[0] * self.inv_pixel_to_cm_conversion))
        #     except Exception as err:
        #         QtHelperUtils.display_warning("Unable to load video file.")

        # # TODO: Implement the actual video playback in a separate window. This
        # # playback should be interruptible from the parent window's controls.
        # self.video_playing = True
        # video_frame_delay = 1.0/self.VIDEO_FRAME_RATE
        # while self.video_playing and self.video_capture.isOpened():
        #     ret, frame = self.video_capture.read()
        #     if not ret:
        #         continue

        #     cv2.imshow(self.video_window_identifier, frame)
        #     cv2.waitKey(video_frame_delay)
            
        raise NotImplementedError()

    def pauseVideo(self):
        """
        If a video is currently playing out, pause it.
        """
        if self.video_capture is None:
            QtHelperUtils.display_warning("No video playback to Pause!")
        self.video_playing = False

    def rewindVideo(self):
        pass

    def forwardVideo(self):
        pass

    def loadPosition(self):
        """
        Load Position data from rec file, or data sorted.

        :returns: Position data from a run session.
        """
        data_file = QFileDialog.getOpenFileName(self, "Choose position tracking file",
                self.data_dir, 'PositionTracking (*.videoPositionTracking)')
        self.video_tracking_file = data_file[0]

        try:
            position_data_file = open(self.video_tracking_file, 'rb')
        except (ValueError, FileNotFoundError, IOError) as err:
            QtHelperUtils.display_warning('File not found, or not specified!')
            return

        self.setWindowTitle('Position Processor - ' + self.video_tracking_file)

        for _ in range(8):
            # TODO: Could use the file header to process the information. Right now we will just believe that 
            position_data_file.readline()

        position_data_type = [(MountainViewIO.TIMESTAMP_LABEL, MountainViewIO.TIMESTAMP_DTYPE), 
                (MountainViewIO.X_LOC1_LABEL, MountainViewIO.POSITION_DTYPE), (MountainViewIO.Y_LOC1_LABEL, MountainViewIO.POSITION_DTYPE),
                (MountainViewIO.X_LOC2_LABEL, MountainViewIO.POSITION_DTYPE), (MountainViewIO.Y_LOC2_LABEL, MountainViewIO.POSITION_DTYPE)]
        try:
            position_data = np.fromfile(position_data_file, dtype=position_data_type)
        except Exception as err:
            QtHelperUtils.display_warning(err)
            return

        # Break up the data into time chunks
        session_time_start = position_data[MountainViewIO.TIMESTAMP_LABEL][0]
        session_time_end = position_data[MountainViewIO.TIMESTAMP_LABEL][-1]
        n_time_chunks = int((session_time_end - session_time_start) / (TIME_CHUNK_SIZE * MountainViewIO.SPIKE_SAMPLING_RATE))
        time_chunk_boundaries = np.linspace(session_time_start, session_time_end, n_time_chunks + 1)
        time_chunks = np.empty((n_time_chunks, 2), dtype='float')
        time_chunks[:,1] = time_chunk_boundaries[1:]
        time_chunks[:,0] = time_chunk_boundaries[:-1]

        self.statusBar().showMessage('Position data loaded!')
        self.plot.setTimeChunks(time_chunks)
        self.plot.setPositionData(position_data)
        self.plot.updateTimeSliderRange()
        self.plot.refresh()


    def startClip(self):
        """
        Start clip at the current slider time. Slider is reset to start at the
        current slider value. End point does not change.
        """

        clip_start = self.plot.getStartTime()
        self.clips.append([0, clip_start, clip_start])
        print(MODULE_IDENTIFIER + "Clip started %.1f"%clip_start)

    def finishClip(self):
        """
        Finish the current clip at the current slider value. Slider is reset to
        end at the current value. Start point does not change.
        """

        clip_finish = self.plot.getFinishTime()
        if self.clips:
            self.clips[-1][2] = clip_finish
        print(MODULE_IDENTIFIER + "Clip finished at %.1f"%clip_finish)

    def writeClips(self):
        """
        Save the store clips to a log file for later use.
        """
        if not self.clips:
            QtHelperUtils.display_warning('No clips to write to file!')
            return

        clip_str = ["S%d: (%d, %d)"%(x[0], x[1], x[2]) for x in self.clips]
        user_clip_selection = QtHelperUtils.CheckBoxWidget(clip_str, "Select clips to be written to File.")
        ok, accepted_clips = user_clip_selection.exec_()
        if ok == int(QMessageBox.Cancel):
            return

        if self.video_tracking_file is not None:
            # Check if the data file can be found
            save_file = self.video_tracking_file.split(MountainViewIO.VIDEO_TRACKING_EXTENSION)[0] + CLIP_FILE_EXTENSION
        elif self.trajectory_data_file is not None:
            save_file = self.trajectory_data_file.split(MountainViewIO.VIDEO_TRACKING_EXTENSION)[0] + CLIP_FILE_EXTENSION
        else:
            QtHelperUtils.display_warning('Could not find a filename to write to!')


        # TODO: Add a check to test if the file already exists, in which case we need to ask the user if he wants the files to be overwritten
        with open(save_file, 'w') as f_csv:
            csv_writer = csv.writer(f_csv)
            for user_accepted_clip in accepted_clips:
                csv_writer.writerow(self.clips[user_accepted_clip])
        print(MODULE_IDENTIFIER + 'Clips written to file ' + save_file)

    def setupMenus(self):
        # Set up the menu bar
        menu_bar = self.menuBar()

        # File menu - Load (Processed Data), Quit
        file_menu = menu_bar.addMenu('&File')

        open_menu = file_menu.addMenu('&Open')

        position_action = open_menu.addAction('&Position')
        position_action.setShortcut('Ctrl+P')
        position_action.triggered.connect(self.loadPosition)
        position_action.setStatusTip('Load videoPositionTracking file from SpikeGadgets.')

        trajectory_action = open_menu.addAction('&Trajectory')
        trajectory_action.setShortcut('Ctrl+T')
        trajectory_action.triggered.connect(self.loadTrajectory)
        trajectory_action.setStatusTip('Load hand curated trajectory.')

        next_traj_action = open_menu.addAction('&Next Trajectory')
        next_traj_action.setShortcut('Ctrl+N')
        next_traj_action.triggered.connect(self.loadNextTrajectory)
        next_traj_action.setStatusTip('Load next hand curated trajectory.')

        prev_traj_action = open_menu.addAction('&Prev Trajectory')
        prev_traj_action.setShortcut('Ctrl+Shift+N')
        prev_traj_action.triggered.connect(self.loadPrevTrajectory)
        prev_traj_action.setStatusTip('Load prev hand curated trajectory.')

        extract_action = QAction('&Extract')
        extract_action.setStatusTip('Extract position from Video')
        extract_action.triggered.connect(self.extractPosition)

        quit_action = QAction('&Exit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.setStatusTip('Exit Program')
        quit_action.triggered.connect(qApp.quit)

        # Add actions to the file menu
        file_menu.addAction(quit_action)

        # Create a menu for video controls
        video_menu = menu_bar.addMenu('&Video')
        load_video_action = video_menu.addAction('&Video')
        load_video_action.triggered.connect(self.loadVideo)
        load_video_action.setStatusTip('Load video file.')

        play_video_action = QAction('P&lay', self)
        play_video_action.triggered.connect(self.playVideo)
        play_video_action.setShortcut('Ctrl+y')

        pause_video_action = QAction('Pa&use', self)
        pause_video_action.triggered.connect(self.pauseVideo)
        pause_video_action.setShortcut('Ctrl+u')

        rewind_video_action = QAction('&Rewind')
        rewind_video_action.triggered.connect(self.rewindVideo)
        rewind_video_action.setShortcut('Ctrl+h')

        forward_video_action = QAction('&Forward')
        forward_video_action.triggered.connect(self.forwardVideo)
        forward_video_action.setShortcut('Ctrl+l')

        video_menu.addAction(play_video_action)
        video_menu.addAction(pause_video_action)

        # Preferences menu
        preferences_menu = menu_bar.addMenu('&Preferences')

        # Clips menu
        clips_menu = menu_bar.addMenu('&Clip')

        clip_start_action = QAction('&Start', self)
        clip_start_action.setShortcut('Ctrl+x')
        clip_start_action.triggered.connect(self.startClip)

        clip_finish_action = QAction('&Finish', self)
        clip_finish_action.setShortcut('Ctrl+e')
        clip_finish_action.triggered.connect(self.finishClip)

        write_clips_action = QAction('&Write', self)
        write_clips_action.setShortcut('Ctrl+w')
        write_clips_action.triggered.connect(self.writeClips)
        
        # Add actions to the clips menu
        clips_menu.addAction(clip_start_action)
        clips_menu.addAction(clip_finish_action)
        clips_menu.addAction(write_clips_action)

    def __init__(self, args):
        """Class constructor """
        QMainWindow.__init__(self)

        # Keep track of time clips
        self.clips = list()

        # Add new class members
        if args.data_dir:
            self.data_dir = args.data_dir
        else:
            self.data_dir = os.getcwd()

        self.FRAME_SIZE = [1280, 984]
        self.VIDEO_FRAME_RATE = 17.0
        self.video_tracking_file = None
        self.raw_video_file = None
        self.video_capture = None
        self.trajectory_data_file = None
        self.video_playing = False
        self.video_window_identifier = 'Video'

        self.plot = PositionPlot()

        # Call parent class functions to setup window
        self.setWindowTitle('Position Processor')
        self.statusBar().showMessage('Open a Position Timestamps file to process it.')
        self.setupMenus()
        self.setCentralWidget(self.plot)
        self.setGeometry(100, 100, 750, 1000)

def launchPositionProcessorApplication(args):
    """
    Launch the main Position Analysis application using the SAMainWindow class

    :args: Arguments to be passed into the class
    """

    qt_args = list()
    qt_args.append(args[0])
    qt_args.append('-style')
    qt_args.append('Windows')
    app = QApplication(qt_args)
    print(MODULE_IDENTIFIER + "Parsing Input Arguments: " + str(sys.argv))

    parsed_arguments = QtHelperUtils.parseQtCommandlineArgs(args)
    sa_window = PositionProcessorWindow(parsed_arguments)
    sa_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    launchPositionProcessorApplication(sys.argv)

