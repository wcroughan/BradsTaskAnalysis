import os
import glob
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QLabel, QPushButton, QStyle, QHBoxLayout, QListWidget, QAction, QMainWindow, QSizePolicy, QFileDialog, QInputDialog, QListWidgetItem, QGraphicsView, QGraphicsScene, qApp
from PyQt5.QtCore import QDir, QUrl, QPointF, QSizeF, Qt, QRectF, QTimer
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
import matplotlib.pyplot as plt
import MountainViewIO
from functools import partial
import csv
import ffmpeg
import ffms2
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from UtilFunctions import getInfoForAnimal, AnimalInfo, readRawPositionData
from TrodesCameraExtrator import processUSBVideoData
from consts import TRODES_SAMPLING_RATE, all_well_names


class AnnotatorRegionListItem(QListWidgetItem):
    def __init__(self, clipIdx, clip):
        if clipIdx == -2:
            self.clipTypeLbl = "Task"
        elif clipIdx == -1:
            self.clipTypeLbl = "Probe"
        else:
            self.clipTypeLbl = str(clipIdx)

        cs1 = self.clipTimeString(clip[1])
        cs2 = self.clipTimeString(clip[2])
        QListWidgetItem.__init__(self, "{}: {} - {}".format(self.clipTypeLbl, cs1, cs2))
        self.clipIdx = clipIdx

    def __lt__(self, other):
        return self.clipIdx < other.clipIdx

    def clipTimeString(self, clipTime):
        seconds = clipTime / TRODES_SAMPLING_RATE
        mins = int(seconds / 60)
        seconds -= mins * 60
        seconds = float(int(seconds * 10)) / 10
        return "{}:{}".format(mins, seconds)


class PositionPlot(QWidget):
    """
    Widget for showing position data and working with it to extract useful
    information in an experiment.

    heavily borrowed and edited from Archit's PositionProcessor.py code
    """

    def __init__(self, xs, ys, ts):
        QWidget.__init__(self)
        print("Hello from Position Plot!")
        plt.ion()

        self.xs = - np.array(xs)
        self.ys = np.array(ys)
        self.ts = np.array(ts)

        self.initUI()
        self.clearAxes()
        self.initTrajectoryPlot()

    def initTrajectoryPlot(self):
        self.fullTrajectoryPlot, = self.position_axes.plot(self.xs,
                                                           self.ys, color='gray', alpha=0.5, zorder=0)
        self.clipTrajectoryPlot = None
        self.wellCoordPlot = None

    def initUI(self):
        # Set up the plot area
        self.figure = Figure(figsize=(12, 16))
        self.canvas = FigureCanvas(self.figure)
        self.position_axes = self.figure.subplots()

        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clearAxes(self):
        self.position_axes.clear()
        self.position_axes.set_xlabel('X (px)')
        self.position_axes.set_ylabel('Y (px)')
        self.position_axes.grid(True)

        self.canvas.draw()

    def updateClipEdges(self, startTimestamp, stopTimestamp):
        # print("position plot updating with {} - {}".format(startTimestamp, stopTimestamp))
        if startTimestamp is not None:
            start_time_idx = np.searchsorted(self.ts, startTimestamp)
        if stopTimestamp is not None:
            end_time_idx = np.searchsorted(self.ts, stopTimestamp)

        if startTimestamp is None:
            start_time_idx = max(0, end_time_idx - 30)
        if stopTimestamp is None:
            end_time_idx = min(len(self.xs), start_time_idx + 30)

        if self.clipTrajectoryPlot is None:
            self.clipTrajectoryPlot, = self.position_axes.plot(
                self.xs[start_time_idx:end_time_idx], self.ys[start_time_idx:end_time_idx], linewidth=2.0, color='black', animated=False, zorder=1)
        else:
            self.clipTrajectoryPlot.set_data(self.xs[start_time_idx:end_time_idx],
                                             self.ys[start_time_idx:end_time_idx])
        self.canvas.draw()

    def setWellCoord(self, wellCoord):
        if self.wellCoordPlot is not None:
            self.wellCoordPlot.remove()
        self.wellCoordPlot = self.position_axes.scatter(-wellCoord[0], wellCoord[1], zorder=2)

    def clearWellCoord(self):
        if self.wellCoordPlot is not None:
            self.wellCoordPlot.remove()
            self.wellCoordPlot = None


class USBVideoWidget(QWidget):
    def __init__(self, usbVideoFileName, tsToFrameFunc):
        QWidget.__init__(self)
        print("hello from usbvideowidghet!")

        self.usbVideoFileName = usbVideoFileName
        self.tsToFrameFunc = tsToFrameFunc

        self.isAnimatingClip = False
        self.clips = {}
        self.taskClip = None
        self.probeClip = None
        self.animationSpeed = 4.0

        probe = ffmpeg.probe(usbVideoFileName)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        self.videoWidth = int(video_stream['width'])
        self.videoHeight = int(video_stream['height'])

        print(self.videoWidth, self.videoHeight)
        if self.videoWidth == 252 and self.videoHeight == 288:
            # tall video
            self.videoType = 1
        elif self.videoWidth == 640 and self.videoHeight == 720:
            # B16-18 video
            self.videoType = 3
        else:
            # wide video
            self.videoType = 2

        self.animPauseTimer = QTimer()
        self.animPauseTimer.setSingleShot(True)
        self.animPauseTimer.timeout.connect(self.endAnimateClip)
        self.initUI()
        self.initMedia()

    def initUI(self):
        if self.videoType == 1:
            videoOffset = QPointF(0, 0)
            videoSize = QSizeF(700, 500)
            viewSceneRect = QRectF(120, -5, 1, 1)
            sceneSceneRect = QRectF(0, 0, 1, 1)

            smallVideoOffset = QPointF(0, 0)
            smallVideoSize = QSizeF(350, 160)
            smallViewSceneRect = QRectF(100, -2, 1, 1)
            smallSceneSceneRect = QRectF(0, 0, 1, 1)

        elif self.videoType == 2:
            videoOffset = QPointF(0, 0)
            videoSize = QSizeF(750, 500)
            viewSceneRect = QRectF(-10, 20, 1, 1)
            sceneSceneRect = QRectF(0, 0, 1, 1)

            smallVideoOffset = QPointF(0, 0)
            smallVideoSize = QSizeF(375, 250)
            smallViewSceneRect = QRectF(-10, 20, 1, 1)
            smallSceneSceneRect = QRectF(0, 0, 1, 1)
        else:
            videoOffset = QPointF(0, 0)
            videoSize = QSizeF(700, 500)
            viewSceneRect = QRectF(120, -5, 1, 1)
            sceneSceneRect = QRectF(0, 0, 1, 1)

            smallVideoOffset = QPointF(0, 0)
            smallVideoSize = QSizeF(350, 160)
            smallViewSceneRect = QRectF(100, -2, 1, 1)
            smallSceneSceneRect = QRectF(0, 0, 1, 1)

        scene = QGraphicsScene()
        self.videoWidget = QGraphicsVideoItem()
        self.videoWidget.setSize(videoSize)
        self.videoWidget.setOffset(videoOffset)
        scene.addItem(self.videoWidget)
        scene.setSceneRect(sceneSceneRect)
        mainVideoParent = QGraphicsView(scene)
        mainVideoParent.setSceneRect(viewSceneRect)
        mainVideoParent.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        scene = QGraphicsScene()
        self.startFrameVideoWidget = QGraphicsVideoItem()
        self.startFrameVideoWidget.setSize(smallVideoSize)
        self.startFrameVideoWidget.setOffset(smallVideoOffset)
        scene.addItem(self.startFrameVideoWidget)
        scene.setSceneRect(smallSceneSceneRect)
        startFrameVideoParent = QGraphicsView(scene)
        startFrameVideoParent.setSceneRect(smallViewSceneRect)
        startFrameVideoParent.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        scene = QGraphicsScene()
        self.endFrameVideoWidget = QGraphicsVideoItem()
        self.endFrameVideoWidget.setSize(smallVideoSize)
        self.endFrameVideoWidget.setOffset(smallVideoOffset)
        scene.addItem(self.endFrameVideoWidget)
        scene.setSceneRect(smallSceneSceneRect)
        endFrameVideoParent = QGraphicsView(scene)
        endFrameVideoParent.setSceneRect(smallViewSceneRect)
        endFrameVideoParent.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.regionListWidget = QListWidget()
        self.regionListWidget.setSortingEnabled(True)

        if self.videoType == 1 or self.videoType == 3:
            # tall video
            l2 = QHBoxLayout()
            l2.addWidget(startFrameVideoParent)
            l2.addWidget(endFrameVideoParent)

            l1 = QVBoxLayout()
            l1.addLayout(l2, 1)
            l1.addWidget(self.regionListWidget, 2)

            layout = QHBoxLayout()
            layout.addLayout(l1, 2)
            layout.addWidget(mainVideoParent, 3)
            self.setLayout(layout)
        else:
            l2 = QHBoxLayout()
            l2.addWidget(startFrameVideoParent, 1)
            l2.addWidget(endFrameVideoParent, 1)

            # l1 = QVBoxLayout()
            # l1.addLayout(l2, 1)
            # l1.addWidget(self.regionListWidget, 1)

            layout = QVBoxLayout()
            layout.addWidget(mainVideoParent, 2)
            layout.addLayout(l2, 1)
            layout.addWidget(self.regionListWidget, 2)
            self.setLayout(layout)

    def initMedia(self):
        self.mainMediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mainMediaPlayer.setVideoOutput(self.videoWidget)
        self.mainMediaPlayer.error.connect(self.handleMediaError)
        print("\n\n")
        print("1")
        t1 = QUrl.fromLocalFile(self.usbVideoFileName)
        print("2")
        t2 = QMediaContent(t1)
        print("3")
        self.mainMediaPlayer.setMedia(t2)
        print("4")
        print("\n\n")
        self.mainMediaPlayer.positionChanged.connect(self.mediaPositionChanged)
        self.mainMediaPlayer.setVolume(0)
        self.mainMediaPlayer.setPlaybackRate(self.animationSpeed)

        self.startFrameMediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.startFrameMediaPlayer.setVideoOutput(self.startFrameVideoWidget)
        self.startFrameMediaPlayer.error.connect(self.handleMediaError)
        self.startFrameMediaPlayer.setMedia(QMediaContent(
            QUrl.fromLocalFile(self.usbVideoFileName)))
        self.startFrameMediaPlayer.setVolume(0)

        self.endFrameMediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.endFrameMediaPlayer.setVideoOutput(self.endFrameVideoWidget)
        self.endFrameMediaPlayer.error.connect(self.handleMediaError)
        self.endFrameMediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.usbVideoFileName)))
        self.endFrameMediaPlayer.setVolume(0)

        ffmsvid = ffms2.VideoSource(self.usbVideoFileName)
        self.frameTimes = np.array(ffmsvid.track.timecodes)

        self.startPosition = None
        self.stopPosition = None

    def handleMediaError(self, err):
        print("handleMediaError {}".format(err))

    def updateClipEdges(self, startTimestamp, stopTimestamp):
        startPositionUpdated = False
        if startTimestamp is not None:
            startFrame = self.tsToFrameFunc(startTimestamp)
            newStartPosition = int(self.frameTimes[min(startFrame, len(self.frameTimes) - 1)])
            if self.startPosition != newStartPosition:
                startPositionUpdated = True
            self.startPosition = newStartPosition
            self.startFrameMediaPlayer.setPosition(self.startPosition)
            self.startFrameMediaPlayer.pause()
        else:
            self.startPosition = None

        stopPositionUpdated = False
        if stopTimestamp is not None:
            stopFrame = self.tsToFrameFunc(stopTimestamp)
            newStopPosition = int(self.frameTimes[stopFrame])
            if self.stopPosition != newStopPosition:
                stopPositionUpdated = True
            self.stopPosition = newStopPosition
            self.endFrameMediaPlayer.setPosition(self.stopPosition)
            self.endFrameMediaPlayer.pause()
        else:
            self.stopPosition = None

        if (self.stopPosition is None or self.startPosition is None) and self.isAnimatingClip:
            self.mainMediaPlayer.pause()
            self.isAnimatingClip = False
            self.animPauseTimer.stop()

        if self.isAnimatingClip:
            self.animateClip()
        elif (stopPositionUpdated and not startPositionUpdated) or (self.startPosition is None):
            self.mainMediaPlayer.setPosition(self.stopPosition)
        else:
            self.mainMediaPlayer.setPosition(self.startPosition)

    def animateClip(self):
        if self.startPosition is not None and self.stopPosition is not None:
            self.isAnimatingClip = True
            self.mainMediaPlayer.setPosition(self.startPosition)
            self.mainMediaPlayer.play()
            # QTimer.singleShot((self.stopPosition - self.startPosition) // self.animationSpeed,
            #                   Qt.TimerType.PreciseTimer, self.endAnimateClip)
            self.animPauseTimer.stop()
            self.animPauseTimer.start(int(
                (self.stopPosition - self.startPosition) // self.animationSpeed))

    def endAnimateClip(self):
        self.mainMediaPlayer.pause()
        self.mainMediaPlayer.setPosition(self.stopPosition)
        self.isAnimatingClip = False

    def mediaPositionChanged(self, position):
        # if self.isAnimatingClip:
        #     if self.stopPosition is None or position > self.stopPosition:
        #         self.isAnimatingClip = False
        #         if self.stopPosition is not None:
        #             self.mainMediaPlayer.setPosition(self.stopPosition)
        #         self.mainMediaPlayer.pause()
        pass

    def updateClipList(self, clips):
        self.clips = clips
        self.remakeClipList()

    def updateTaskClip(self, taskClip):
        self.taskClip = taskClip
        self.remakeClipList()

    def updateProbeClip(self, probeClip):
        self.probeClip = probeClip
        self.remakeClipList()

    def remakeClipList(self):
        self.regionListWidget.clear()
        for c in self.clips:
            self.regionListWidget.addItem(AnnotatorRegionListItem(c, self.clips[c]))
        if self.taskClip is not None:
            self.regionListWidget.addItem(AnnotatorRegionListItem(-2, self.taskClip))
        if self.probeClip is not None:
            self.regionListWidget.addItem(AnnotatorRegionListItem(-1, self.probeClip))


class AnnotatorWindow(QMainWindow):
    def __init__(self, trodesXs, trodesYs, trodesTs, trodesLightOffTime, trodesLightOnTime,
                 usbVideoFileName, wellEntryTimes, wellExitTimes, foundWells, skipProbe,
                 outputFileNameStart, wellCoordMap):
        QMainWindow.__init__(self)
        print("Hello from annotator!")

        self.smallMovementAmt = int(float(TRODES_SAMPLING_RATE) / 15.0)
        self.largeMovementAmt = TRODES_SAMPLING_RATE
        self.extraLargeMovementAmt = TRODES_SAMPLING_RATE * 20

        self.clips = {}
        self.skipProbe = skipProbe
        self.outputFileNameStart = outputFileNameStart

        self.STATE_FOUND_WELLS = 0
        self.STATE_TASK_START = 1
        self.STATE_TASK_END = 2
        self.STATE_PROBE_START = 3
        self.STATE_PROBE_END = 4
        self.STATE_WAIT_CONFIRM = 5
        self.currentState = self.STATE_FOUND_WELLS

        self.trodesXs = trodesXs
        self.trodesYs = trodesYs
        self.trodesTs = trodesTs
        self.trodesLightOffTime = trodesLightOffTime
        self.trodesLightOnTime = trodesLightOnTime
        print(self.trodesLightOffTime, self.trodesLightOnTime)
        self.wellEntryTimes = wellEntryTimes
        self.wellExitTimes = wellExitTimes
        self.foundWells = foundWells
        self.wellCoordMap = wellCoordMap
        # print(wellCoordMap)

        self.usbVideoFileName = usbVideoFileName
        self.hasUsbVideo = self.usbVideoFileName is not None
        if self.hasUsbVideo:
            print("Processing USB Video for light times")
            self.usbLightOffFrame, self.usbLightOnFrame = processUSBVideoData(
                usbVideoFileName, overwriteMode="loadOld", showVideo=False)
            print("Done processing USB Video for light times")

            self.ttf_m = (float(self.usbLightOnFrame) - float(self.usbLightOffFrame)) / \
                (float(trodesLightOnTime) - float(trodesLightOffTime))
            self.ttf_b = self.ttf_m * - float(trodesLightOffTime) + float(self.usbLightOffFrame)

            print("USB frames: {}, {}".format(self.usbLightOffFrame, self.usbLightOnFrame))
            print("timestamps: {}, {}".format(self.trodesLightOffTime, self.trodesLightOnTime))
            print("func test : {}, {}".format(self.timestampToUSBFrame(
                self.trodesLightOffTime), self.timestampToUSBFrame(self.trodesLightOnTime)))
        else:
            print("No USB video")
        print("initializing UI")
        self.initUI()
        print("initializing menu items")
        self.setupMenu()

        self.clipStart = -1
        self.clipEnd = -1
        self.currentFoundWellIdx = 0
        print("displaying first well")
        self.setupForWell(0)
        print("done setting up annotator window")

    def timestampToUSBFrame(self, timestamp):
        return int(self.ttf_m * float(timestamp) + self.ttf_b)

    def initUI(self):
        self.positionPlot = PositionPlot(self.trodesXs, self.trodesYs, self.trodesTs)
        # self.positionPlot = QLabel("hi!")

        if self.hasUsbVideo:
            self.videoWidget = USBVideoWidget(self.usbVideoFileName, self.timestampToUSBFrame)
        else:
            self.videoWidget = QLabel("no usb video!")

        centralLayout = QHBoxLayout()
        centralLayout.addWidget(self.positionPlot, 1)
        centralLayout.addWidget(self.videoWidget, 1)

        self.statusLabel = QLabel("hello!")
        self.statusLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.addLayout(centralLayout)
        layout.addWidget(self.statusLabel)

        # self.setLayout(layout)
        wid = QWidget(self)
        wid.setLayout(layout)
        self.setCentralWidget(wid)

    def setupMenu(self):
        menu_bar = self.menuBar()
        clip_menu = menu_bar.addMenu('&Clip')

        startClipSmallIncrement = QAction('start inc sm', self)
        startClipSmallIncrement.triggered.connect(
            partial(self.moveClipStart, self.smallMovementAmt))
        startClipSmallIncrement.setShortcut('s')
        clip_menu.addAction(startClipSmallIncrement)
        startClipLargeIncrement = QAction('start inc lg', self)
        startClipLargeIncrement.triggered.connect(
            partial(self.moveClipStart, self.largeMovementAmt))
        startClipLargeIncrement.setShortcut('w')
        clip_menu.addAction(startClipLargeIncrement)
        startClipExtraLargeIncrement = QAction('start inc lg', self)
        startClipExtraLargeIncrement.triggered.connect(
            partial(self.moveClipStart, self.extraLargeMovementAmt))
        startClipExtraLargeIncrement.setShortcut('2')
        clip_menu.addAction(startClipExtraLargeIncrement)
        startClipSmallDecrement = QAction('start dec sm', self)
        startClipSmallDecrement.triggered.connect(
            partial(self.moveClipStart, -self.smallMovementAmt))
        startClipSmallDecrement.setShortcut('a')
        clip_menu.addAction(startClipSmallDecrement)
        startClipLargeDecrement = QAction('start dec lg', self)
        startClipLargeDecrement.triggered.connect(
            partial(self.moveClipStart, -self.largeMovementAmt))
        startClipLargeDecrement.setShortcut('q')
        clip_menu.addAction(startClipLargeDecrement)
        startClipExtraLargeDecrement = QAction('start dec lg', self)
        startClipExtraLargeDecrement.triggered.connect(
            partial(self.moveClipStart, -self.extraLargeMovementAmt))
        startClipExtraLargeDecrement.setShortcut('1')
        clip_menu.addAction(startClipExtraLargeDecrement)

        endClipSmallIncrement = QAction('end inc sm', self)
        endClipSmallIncrement.triggered.connect(
            partial(self.moveClipEnd, self.smallMovementAmt))
        endClipSmallIncrement.setShortcut('f')
        clip_menu.addAction(endClipSmallIncrement)
        endClipLargeIncrement = QAction('end inc lg', self)
        endClipLargeIncrement.triggered.connect(
            partial(self.moveClipEnd, self.largeMovementAmt))
        endClipLargeIncrement.setShortcut('r')
        clip_menu.addAction(endClipLargeIncrement)
        endClipExtraLargeIncrement = QAction('end inc lg', self)
        endClipExtraLargeIncrement.triggered.connect(
            partial(self.moveClipEnd, self.extraLargeMovementAmt))
        endClipExtraLargeIncrement.setShortcut('4')
        clip_menu.addAction(endClipExtraLargeIncrement)
        endClipSmallDecrement = QAction('end dec sm', self)
        endClipSmallDecrement.triggered.connect(
            partial(self.moveClipEnd, -self.smallMovementAmt))
        endClipSmallDecrement.setShortcut('d')
        clip_menu.addAction(endClipSmallDecrement)
        endClipLargeDecrement = QAction('end dec lg', self)
        endClipLargeDecrement.triggered.connect(
            partial(self.moveClipEnd, -self.largeMovementAmt))
        endClipLargeDecrement.setShortcut('e')
        clip_menu.addAction(endClipLargeDecrement)
        endClipExtraLargeDecrement = QAction('end dec lg', self)
        endClipExtraLargeDecrement.triggered.connect(
            partial(self.moveClipEnd, -self.extraLargeMovementAmt))
        endClipExtraLargeDecrement.setShortcut('3')
        clip_menu.addAction(endClipExtraLargeDecrement)

        saveClipAction = QAction('save clip', self)
        saveClipAction.triggered.connect(self.saveAndNextWell)
        saveClipAction.setShortcut('g')
        clip_menu.addAction(saveClipAction)
        undoSaveClipAction = QAction('undo save clip', self)
        undoSaveClipAction.triggered.connect(self.undoSaveAndNextWell)
        undoSaveClipAction.setShortcut('j')
        clip_menu.addAction(undoSaveClipAction)
        resetClipsAction = QAction('reset clips', self)
        resetClipsAction.triggered.connect(self.resetAllClips)
        resetClipsAction.setShortcut('k')
        clip_menu.addAction(resetClipsAction)
        nextEntryAction = QAction('next entry', self)
        nextEntryAction.triggered.connect(self.showNextEntrySameWell)
        nextEntryAction.setShortcut('x')
        clip_menu.addAction(nextEntryAction)
        prevEntryAction = QAction('prev entry', self)
        prevEntryAction.triggered.connect(self.showPrevEntrySameWell)
        prevEntryAction.setShortcut('z')
        clip_menu.addAction(prevEntryAction)

        playClipAction = QAction('play clip', self)
        playClipAction.triggered.connect(self.playClip)
        playClipAction.setShortcut('t')
        clip_menu.addAction(playClipAction)

    def setupForWell(self, foundWellIdx, currentTime=None, reverseTime=False):
        wellName = self.foundWells[foundWellIdx]
        wellIdx = np.argmax(all_well_names == wellName)
        entryTimes = self.wellEntryTimes[wellIdx]
        exitTimes = self.wellExitTimes[wellIdx]
        if currentTime is None:
            if reverseTime:
                currentTime = self.clipStart
            else:
                currentTime = self.clipEnd

        entryIdx = None
        if reverseTime:
            for ei, et in reversed(list(enumerate(exitTimes))):
                if et <= currentTime:
                    entryIdx = ei
                    break
        else:
            for ei, et in enumerate(entryTimes):
                if et >= currentTime:
                    entryIdx = ei
                    break

        if entryIdx is None:
            print("ERROR: No more well entries to well {} {} time {}".format(
                wellName, "before" if reverseTime else "after", currentTime))
            return

        self.clipStart = entryTimes[entryIdx]
        self.clipEnd = exitTimes[entryIdx]

        print("Setting up for visit {} to well {}, ({} - {})".format(foundWellIdx,
              wellName, self.clipStart, self.clipEnd))
        self.statusLabel.setText("visit {}, well {}".format(foundWellIdx, wellName))

        self.positionPlot.setWellCoord(self.wellCoordMap[str(wellName)])
        self.updateClipInWidgets(animateAnew=True)

    def setupForTaskStart(self):
        self.statusLabel.setText("Task start")
        self.clipStart = (self.trodesLightOffTime + self.clips[0][1]) / 2
        # print(self.clipStart)
        # print(int(self.clipStart))
        self.clipEnd = None
        self.positionPlot.clearWellCoord()
        self.updateClipInWidgets()

    def setupForTaskEnd(self):
        self.statusLabel.setText("Task end")
        self.clipStart = None
        self.clipEnd = self.clips[len(self.foundWells) - 1][2]
        self.positionPlot.clearWellCoord()
        self.updateClipInWidgets()

    def setupForProbeStart(self):
        self.statusLabel.setText("probe start")
        self.clipStart = self.taskClip[2] + TRODES_SAMPLING_RATE * 60
        self.clipEnd = None
        self.positionPlot.clearWellCoord()
        self.updateClipInWidgets()

    def setupForProbeEnd(self):
        self.statusLabel.setText("probe end")
        self.clipEnd = self.clipStart + TRODES_SAMPLING_RATE * 60 * 5
        self.clipStart = None
        self.positionPlot.clearWellCoord()
        self.updateClipInWidgets()

    def updateClipInWidgets(self, animateAnew=False):
        self.positionPlot.updateClipEdges(self.clipStart, self.clipEnd)
        if self.hasUsbVideo:
            self.videoWidget.updateClipEdges(self.clipStart, self.clipEnd)
        if animateAnew:
            self.playClip()

    def playClip(self):
        if self.hasUsbVideo:
            self.videoWidget.animateClip()

    def moveClipStart(self, moveAmt):
        if self.clipStart is not None:
            self.clipStart += moveAmt
            self.updateClipInWidgets()

    def moveClipEnd(self, moveAmt):
        if self.clipEnd is not None:
            self.clipEnd += moveAmt
            self.updateClipInWidgets()

    def showNextEntrySameWell(self):
        self.setupForWell(self.currentFoundWellIdx)

    def showPrevEntrySameWell(self):
        self.setupForWell(self.currentFoundWellIdx, reverseTime=True)

    def saveClip(self):
        if self.currentState == self.STATE_FOUND_WELLS:
            clip = (0, int(self.clipStart), int(self.clipEnd))
            self.clips[self.currentFoundWellIdx] = clip
            if self.hasUsbVideo:
                self.videoWidget.updateClipList(self.clips)
        elif self.currentState == self.STATE_TASK_START:
            self.taskClipStart = int(self.clipStart)
            print(self.taskClipStart)
        elif self.currentState == self.STATE_TASK_END:
            self.taskClip = (0, self.taskClipStart, int(self.clipEnd))
            print(self.taskClip)
            if self.hasUsbVideo:
                self.videoWidget.updateTaskClip(self.taskClip)
        elif self.currentState == self.STATE_PROBE_START:
            self.probeClipStart = int(self.clipStart)
        elif self.currentState == self.STATE_PROBE_END:
            self.probeClip = (0, self.probeClipStart, int(self.clipEnd))
            if self.hasUsbVideo:
                self.videoWidget.updateProbeClip(self.probeClip)

    def saveAndNextWell(self):
        self.saveClip()
        if self.currentState == self.STATE_FOUND_WELLS:
            self.currentFoundWellIdx += 1
            if self.currentFoundWellIdx == len(self.foundWells):
                self.doneWithFoundWells()
            else:
                self.setupForWell(self.currentFoundWellIdx)
        elif self.currentState == self.STATE_TASK_START:
            self.currentState = self.STATE_TASK_END
            self.setupForTaskEnd()
        elif self.currentState == self.STATE_TASK_END:
            if self.skipProbe:
                self.doneWithAllClips()
            else:
                self.currentState = self.STATE_PROBE_START
                self.setupForProbeStart()
        elif self.currentState == self.STATE_PROBE_START:
            self.currentState = self.STATE_PROBE_END
            self.setupForProbeEnd()
        elif self.currentState == self.STATE_PROBE_END:
            if self.validateClips():
                self.statusLabel.setText("Press G to save clips, K to reset them")
            else:
                self.statusLabel.setText("Warning! Clips aren't good!")
            self.currentState = self.STATE_WAIT_CONFIRM
        elif self.currentState == self.STATE_WAIT_CONFIRM:
            self.doneWithAllClips()

    def undoSaveAndNextWell(self):
        if self.currentState == self.STATE_WAIT_CONFIRM:
            self.currentState = self.STATE_PROBE_END
            self.probeClip = None
            self.setupForProbeEnd()
        elif self.currentState == self.STATE_PROBE_END:
            self.currentState = self.STATE_PROBE_START
            self.probeClipStart = None
            self.setupForProbeStart()
        elif self.currentState == self.STATE_PROBE_START:
            self.currentState = self.STATE_TASK_END
            self.taskClip = None
            self.setupForTaskEnd()
        elif self.currentState == self.STATE_TASK_END:
            self.currentState = self.STATE_TASK_START
            self.taskClipStart = None
            self.setupForTaskStart()
        elif self.currentState == self.STATE_TASK_START:
            self.currentState = self.STATE_FOUND_WELLS

        if self.currentState == self.STATE_FOUND_WELLS:
            if self.currentFoundWellIdx == 0:
                return
            self.currentFoundWellIdx -= 1
            del self.clips[self.currentFoundWellIdx]
            self.setupForWell(self.currentFoundWellIdx)

    def validateClips(self):
        for ci in range(len(self.clips)):
            c = self.clips[ci]
            if c[1] >= c[2]:
                return False
            if c[1] <= self.taskClip[1] or c[2] > self.taskClip[2]:
                return False

        if self.taskClip[1] >= self.taskClip[2]:
            return False

        if not self.skipProbe:
            if self.taskClip[2] >= self.probeClip[1] or self.probeClip[1] >= self.probeClip[2]:
                return False

        return True

    def resetAllClips(self):
        self.currentState = self.STATE_FOUND_WELLS
        self.clipStart = -1
        self.clipEnd = -1
        self.clips = {}
        self.taskClip = None
        self.probeClip = None
        self.currentFoundWellIdx = 0
        if self.hasUsbVideo:
            self.videoWidget.updateClipList(self.clips)
            self.videoWidget.updateTaskClip(self.taskClip)
            self.videoWidget.updateProbeClip(self.probeClip)
        print("displaying first well")
        self.setupForWell(0)

    def doneWithFoundWells(self):
        self.currentState = self.STATE_TASK_START
        self.setupForTaskStart()

    def doneWithAllClips(self):
        clipsFileName = self.outputFileNameStart + ".clips"
        with open(clipsFileName, 'w') as f_csv:
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(self.taskClip)
            if not self.skipProbe:
                csv_writer.writerow(self.probeClip)
        print("saved to {}".format(clipsFileName))

        rewardClipsFileName = self.outputFileNameStart + ".rewardclips"
        with open(rewardClipsFileName, 'w') as f_csv:
            csv_writer = csv.writer(f_csv)
            for i in range(len(self.foundWells)):
                csv_writer.writerow(self.clips[i])
        print("saved to {}".format(rewardClipsFileName))
        self.close()


def runPositionAnnotator(trodesXs, trodesYx, trodesTs, trodesLightOffTime, trodesLightOnTime,
                         usbVideoFileName, wellEntryTimes, wellExitTimes, foundWells, skipProbe, outputFileNameStart):
    parent_app = QApplication(sys.argv)
    ann = AnnotatorWindow(trodesXs, trodesYx, trodesTs, trodesLightOffTime, trodesLightOnTime,
                          usbVideoFileName, wellEntryTimes, wellExitTimes, foundWells, skipProbe, outputFileNameStart)
    ann.resize(600, 600)
    ann.show()
    parent_app.exec_()
