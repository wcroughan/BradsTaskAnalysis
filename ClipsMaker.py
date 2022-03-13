import os
import glob
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QLabel, QPushButton, QStyle, QHBoxLayout, QListWidget, QAction, QMainWindow, QSizePolicy, QFileDialog, QInputDialog, QListWidgetItem, QGraphicsView, QGraphicsScene, qApp
from PyQt5.QtCore import QDir, QUrl, QPointF, QSizeF
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
import matplotlib.pyplot as plt
import MountainViewIO
from functools import partial
import csv
import ffmpeg
import ffms2

from UtilFunctions import getInfoForAnimal, AnimalInfo, readRawPositionData
from TrodesCameraExtrator import processUSBVideoData
from consts import TRODES_SAMPLING_RATE


class PositionPlot(QWidget):
    """
    Widget for showing position data and working with it to extract useful
    information in an experiment.

    heavily borrowed and edited from Archit's PositionProcessor.py code
    """

    def __init__(self, xs, ys, ts):
        QWidget.__init__(self)
        plt.ion()

        self.xs = xs
        self.ys = ys
        self.ts = ts

        self.initUI()
        self.clearAxes()
        self.initTrajectoryPlot()

    def initTrajectoryPlot(self):
        self.fullTrajectoryPlot, = self.position_axes.plot(self.xs,
                                                           self.ys, color='gray', alpha=0.5)
        self.clipTrajectoryPlot = None

    def initUI(self):
        # Set up the plot area
        self.figure = Figure(figsize=(12, 16))
        self.canvas = FigureCanvas(self.figure)
        self.position_axes = self.figure.subplot(111)

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
        if startTimestamp is not None:
            start_time_idx = np.searchsorted(self.ts, startTimestamp)
        if endTimestamp is not None:
            end_time_idx = np.searchsorted(self.ts, stopTimestamp)

        if startTimestamp is None:
            start_time_idx = end_time_idx - 1
        if endTimestamp is None:
            end_time_idx = start_time_idx + 1

        if self.clipTrajectoryPlot is None:
            self.clipTrajectoryPlot, = self.position_axes.plot(
                self.xs[start_time_idx:end_time_idx], self.ys[start_time_idx:end_time_idx], linewidth=2.0, color='black', animated=False)
        else:
            self.clipTrajectoryPlot.set_data(self.xs[start_time_idx:end_time_idx],
                                             self.ys[start_time_idx:end_time_idx])
        self.canvas.draw()


class USBVideoWidget(QWidget):
    def __init__(self, usbVideoFileName, tsToFrameFunc):
        QWidget.__init__(self)

        self.usbVideoFileName = usbVideoFileName
        self.tsToFrameFunc = tsToFrameFunc

        self.isAnimatingClip = False

        probe = ffmpeg.probe(usbVideoFileName)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        self.videoWidth = int(video_stream['width'])
        self.videoHeight = int(video_stream['height'])

        print(self.videoWidth, self.videoHeight)
        if self.videoWidth / self.videoHeight < 0.6:
            self.videoType = 1
        else:
            self.videoType = 2

        self.initUI()
        self.initMedia()

    def initUI(self):
        if self.videoType == 2:
            videoOffset = QPointF(-1200, -800)
            videoSize = QSizeF(4000, 2000)
        else:
            videoOffset = QPointF(-1200, -800)
            videoSize = QSizeF(4000, 2000)

        scene = QGraphicsScene()
        self.videoWidget = QGraphicsVideoItem()
        self.videoWidget.setOffset(videoOffset)
        self.videoWidget.setSize(videoSize)
        scene.addItem(self.videoWidget)
        mainVideoParent = QGraphicsView(scene)

        scene = QGraphicsScene()
        self.startFrameVideoWidget = QGraphicsVideoItem()
        self.startFrameVideoWidget.setOffset(videoOffset)
        self.startFrameVideoWidget.setSize(videoSize)
        scene.addItem(self.startFrameVideoWidget)
        startFrameVideoParent = QGraphicsView(scene)

        scene = QGraphicsScene()
        self.endFrameVideoWidget = QGraphicsVideoItem()
        self.endFrameVideoWidget.setOffset(videoOffset)
        self.endFrameVideoWidget.setSize(videoSize)
        scene.addItem(self.endFrameVideoWidget)
        endFrameVideoParent = QGraphicsView(scene)

        self.regionListWidget = QListWidget()
        self.regionListWidget.setSortingEnabled(True)

        if self.videoType == 1:
            # tall video
            l2 = QHBoxLayout()
            l2.addWidget(startFrameVideoParent)
            l2.addWidget(endFrameVideoParent)

            l1 = QVBoxLayout()
            l1.addLayout(l2)
            l1.addWidget(self.regionListWidget)

            layout = QHBoxLayout()
            layout.addLayout(l1)
            layout.addWidget(mainViewParent)
            self.setLayout(layout)
        else:
            l2 = QHBoxLayout()
            l2.addWidget(startFrameVideoParent)
            l2.addWidget(endFrameVideoParent)

            l1 = QVBoxLayout()
            l1.addLayout(l2)
            l1.addWidget(self.regionListWidget)

            layout = QVBoxLayout()
            layout.addWidget(mainVideoParent)
            layout.addLayout(l1)
            self.setLayout(layout)

    def initMedia(self):
        self.mainMediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mainMediaPlayer.setVideoOutput(self.videoWidget)
        self.mainMediaPlayer.error.connect(self.handleMediaError)
        self.mainMediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.usbVideoFileName)))
        self.mainMediaPlayer.positionChanged.connect(self.mediaPositionChanged)

        self.startFrameMediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.startFrameMediaPlayer.setVideoOutput(self.startFrameVideoWidget)
        self.startFrameMediaPlayer.error.connect(self.handleMediaError)
        self.startFrameMediaPlayer.setMedia(QMediaContent(
            QUrl.fromLocalFile(self.usbVideoFileName)))

        self.endFrameMediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.endFrameMediaPlayer.setVideoOutput(self.endFrameVideoWidget)
        self.endFrameMediaPlayer.error.connect(self.handleMediaError)
        self.endFrameMediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.usbVideoFileName)))

        ffmsvid = ffms2.VideoSource(fileName)
        self.frameTimes = np.array(ffmsvid.track.timecodes)

    def handleMediaError(self, err):
        print("handleMediaError {}".format(err))

    def updateClipEdges(self, startTimestamp, stopTimestamp):
        if startTimestamp is not None:
            startFrame = self.tsToFrameFunc(startTimestamp)
            self.startPosition = int(self.frameTimes[startFrame])
            self.startFrameMediaPlayer.setPosition(self.startPosition)
            self.mainMediaPlayer.setPosition(self.startPosition)
        else:
            self.startPosition = None

        if stopTimestamp is not None:
            stopFrame = self.tsToFrameFunc(stopTimestamp)
            self.stopPosition = int(self.frameTimes[stopFrame])
            self.endFrameMediaPlayer.setPosition(self.stopPosition)
        else:
            self.stopPosition = None

        if self.startPosition is None:
            self.mainMediaPlayer.setPosition(self.stopPosition)

    def animateClip(self):
        self.isAnimatingClip = True
        self.mainMediaPlayer.play()

    def mediaPositionChanged(self, position):
        if self.isAnimatingClip:
            if self.stopPosition is None or position > self.stopPosition:
                self.mainMediaPlayer.pause()


class AnnotatorWindow(QMainWindow):
    def __init__(self, trodesXs, trodesYx, trodesTs, trodesLightOffTime, trodesLightOnTime,
                 usbVideoFileName, wellEntryTimes, wellExitTimes, foundWells, skipProbe, outputFileNameStart):
        QMainWindow.__init__(self)

        self.smallMovementAmt = int(float(TRODES_SAMPLING_RATE) / 15.0)
        self.largeMovementAmt = TRODES_SAMPLING_RATE

        self.clips = {}
        self.skipProbe = skipProbe
        self.outputFileNameStart = outputFileNameStart

        self.STATE_FOUND_WELLS = 0
        self.STATE_TASK_START = 1
        self.STATE_TASK_END = 2
        self.STATE_PROBE_START = 3
        self.STATE_PROBE_END = 4
        self.currentState = self.STATE_FOUND_WELLS

        self.trodesXs = trodesXs
        self.trodesYs = trodesYs
        self.trodesTs = trodesTs
        self.trodesLightOffTime = trodesLightOffTime
        self.trodesLightOnTime = trodesLightOnTime
        self.wellEntryTimes = wellEntryTimes
        self.wellExitTimes = wellExitTimes
        self.foundWells = foundWells

        self.usbVideoFileName = usbVideoFileName
        self.usbLightOffFrame, self.usbLightOnFrame = processUSBVideoData(
            usbVideoFileName, overwriteMode="loadOld", showVideo=False)

        self.ttf_m = (float(self.usbLightOnFrame) - float(self.usbLightOffFrame)) / \
            (float(trodesLightOnTime) - float(trodesLightOffTime))
        self.ttf_b = self.ttf_m * - float(trodesLightOffTime) + float(self.usbLightOffFrame)

        self.initUI()
        self.setupMenu()

        self.clipStart = -1
        self.clipEnd = -1
        self.currentFoundWellIdx = 0
        self.setupForWell(0)

    def timestampToUSBFrame(self, timestamp):
        return int(self.ttf_m * float(timestamp) + self.ttf_b)

    def initUI(self):
        self.positionPlot = PositionPlot(self.trodesXs, self.trodesYs, self.trodesTs)

        self.videoWidget = USBVideoWidget(self.usbVideoFileName, self.timestampToUSBFrame)

        centralLayout = QHBoxLayout()
        centralLayout.addWidget(self.positionPlot)
        centralLayout.addWidget(self.videoWidget)

        self.statusLabel = QLabel("hello!")
        self.statusLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.addWidget(centralLayout)
        layout.addWidget(self.statusLabel)

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

        saveClipAction = QAction('save clip', self)
        saveClipAction.triggered.connect(self.saveAndNextWell)
        saveClipAction.setShortcut('g')
        clip_menu.addAction(saveClipAction)
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

    def setupForWell(self, wellIdx, currentTime=None, reverseTime=False):
        wellName = self.foundWells[wellIdx]
        entryTimes = self.wellEntryTimes[wellName]
        exitTimes = self.wellExitTimes[wellName]
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
        self.clipStart = entryTimes[entryIdx]
        self.clipEnd = exitTimes[entryIdx]

        self.updateClipInWidgets(animateAnew=True)

    def setupForTaskStart(self):
        self.clipStart = self.wellEntryTimes[7][0]
        self.clipEnd = None
        self.updateClipInWidgets()

    def setupForTaskEnd(self):
        self.clipStart = None
        self.clipEnd = self.clips[len(self.foundWells)-1][2]
        self.updateClipInWidgets()

    def setupForProbeStart(self):
        self.clipStart = self.taskClip[2] + TRODES_SAMPLING_RATE * 60
        self.clipEnd = None
        self.updateClipInWidgets()

    def setupForProbeEnd(self):
        self.clipEnd = self.clipStart + TRODES_SAMPLING_RATE * 60 * 5
        self.clipStart = None
        self.updateClipInWidgets()

    def updateClipInWidgets(self, animateAnew=False):
        self.positionPlot.updateClipEdges(self.clipStart, self.clipEnd)
        self.videoWidget.updateClipEdges(self.clipStart, self.clipEnd)
        if animateAnew:
            self.playClip()

    def playClip(self):
        self.videoWidget.animateClip()

    def moveClipStart(self, moveAmt):
        self.clipStart += moveAmt
        self.updateClipInWidgets()

    def moveClipEnd(self, moveAmt):
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
        elif self.currentState == self.STATE_TASK_START:
            self.taskClip = int(self.clipStart)
        elif self.currentState == self.STATE_TASK_END:
            self.taskClip = (0, self.taskClip, int(self.clipEnd))
        elif self.currentState == self.STATE_PROBE_START:
            self.probeClip = int(self.clipStart)
        elif self.currentState == self.STATE_PROBE_END:
            self.probeClip = (0, self.probeClip, int(self.clipEnd))

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
            self.doneWithAllClips()

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

        rewardClipsFileName = self.outputFileNameStart + ".rewardclips"
        with open(rewardClipsFileName, 'w') as f_csv:
            csv_writer = csv.writer(f_csv)
            for i in range(len(self.foundWells)):
                csv_writer.writerow(self.clips[i])


def runPositionAnnotator(trodesXs, trodesYx, trodesTs, trodesLightOffTime, trodesLightOnTime,
                         usbVideoFileName, wellEntryTimes, wellExitTimes, skipProbe, outputFileNameStart):
    parent_app = QApplication()
    ann = AnnotatorWindow(trodesXs, trodesYx, trodesTs, trodesLightOffTime, trodesLightOnTime,
                          usbVideoFileName, wellEntryTimes, wellExitTimes, skipProbe, outputFileNameStart)
    ann.resize(600, 600)
    ann.show()
    parent_app.exec_()
