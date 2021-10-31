from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QLabel, QPushButton, QStyle, QHBoxLayout, QListWidget, QAction, QMainWindow, QSizePolicy, QFileDialog, QInputDialog, QListWidgetItem, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import QDir, QUrl, QPointF, QSizeF
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
import sys
import ffms2
import numpy as np
import json
import csv


class CMRegionListItem(QListWidgetItem):
    def __init__(self, start, stop, well):
        QListWidgetItem.__init__(self, "{} - {}  ({})".format(start/1000.0, stop/1000.0, well))
        self.start = start
        self.stop = stop
        self.well = well

    def __lt__(self, other):
        return self.start > other.start


class CMWidget(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.SPEED_CHANGE_FACTOR = 1.25
        self.JUMP_DIST = 1000  # ms
        self.deletedItem = None
        self.videoFileName = None

        self.preTrialLightOffMark = None
        self.trialStartMark = None
        self.trialStopMark = None
        self.probeStartMark = None
        self.probeStopMark = None
        self.postProbeLightOnMark = None

        self.initUI()
        self.initMediaPlayer()

        # self.loadVideoFromFilename("/home/wcroughan/glasses_data/facial_recog/outvid2.mp4")
        # self.loadVideoFromFilename()
        self.openVideo()

    def initUI(self):
        self.setWindowTitle("Camera Marker")

        self.scene = QGraphicsScene()
        # self.videoWidget = QVideoWidget()
        self.videoWidget = QGraphicsVideoItem()
        self.videoWidget.setOffset(QPointF(-1200, -800))
        self.videoWidget.setSize(QSizeF(4000, 2000))
        self.scene.addItem(self.videoWidget)
        # self.scene.setSceneRect(0, 0, 1400, 1400)

        self.videoParent = QGraphicsView(self.scene)

        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu("File")
        ac = QAction('&Export Regions', self)
        ac.setShortcut('Shift+E')
        ac.triggered.connect(self.exportRegions)
        fileMenu.addAction(ac)
        ac = QAction('&Import Regions', self)
        ac.triggered.connect(self.importRegions)
        fileMenu.addAction(ac)
        ac = QAction('&Open Video', self)
        ac.setShortcut('Ctrl+O')
        ac.triggered.connect(self.openVideo)
        fileMenu.addAction(ac)
        ac = QAction('Exit', self)
        ac.triggered.connect(self.exitCall)
        fileMenu.addAction(ac)

        mediaMenu = menuBar.addMenu("Media")

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        ac = QAction('&Play', self)
        ac.setShortcut('5')
        ac.triggered.connect(self.play)
        mediaMenu.addAction(ac)

        self.speedUpButton = QPushButton("speed up")
        self.speedUpButton.setEnabled(False)
        self.speedUpButton.clicked.connect(self.speedUp)
        ac = QAction('Speed &Up', self)
        ac.setShortcut('+')
        ac.triggered.connect(self.speedUp)
        mediaMenu.addAction(ac)

        self.slowDownButton = QPushButton("slow down")
        self.slowDownButton.setEnabled(False)
        self.slowDownButton.clicked.connect(self.slowDown)
        ac = QAction('Slow &Down', self)
        ac.setShortcut('-')
        ac.triggered.connect(self.slowDown)
        mediaMenu.addAction(ac)

        self.jumpFwdButton = QPushButton("jump fwd")
        self.jumpFwdButton.setEnabled(False)
        self.jumpFwdButton.clicked.connect(self.jumpFwd)
        ac = QAction('Jump Fwd', self)
        ac.setShortcut('3')
        ac.triggered.connect(self.jumpFwd)
        mediaMenu.addAction(ac)

        self.jumpBackButton = QPushButton("jump back")
        self.jumpBackButton.setEnabled(False)
        self.jumpBackButton.clicked.connect(self.jumpBack)
        ac = QAction('Jump Back', self)
        ac.setShortcut('1')
        ac.triggered.connect(self.jumpBack)
        mediaMenu.addAction(ac)

        self.fwdFrameButton = QPushButton("fwd frame")
        self.fwdFrameButton.setEnabled(False)
        self.fwdFrameButton.clicked.connect(self.fwdFrame)
        ac = QAction('fwd Frame', self)
        ac.setShortcut('6')
        ac.triggered.connect(self.fwdFrame)
        mediaMenu.addAction(ac)

        self.backFrameButton = QPushButton("back frame")
        self.backFrameButton.setEnabled(False)
        self.backFrameButton.clicked.connect(self.backFrame)
        ac = QAction('back Frame', self)
        ac.setShortcut('4')
        ac.triggered.connect(self.backFrame)
        mediaMenu.addAction(ac)

        regionsMenu = menuBar.addMenu("Regions")
        self.setStartMarkButton = QPushButton("set start")
        self.setStartMarkButton.setEnabled(False)
        self.setStartMarkButton.clicked.connect(self.setStartMark)
        ac = QAction('Set Start Mark', self)
        ac.setShortcut('7')
        ac.triggered.connect(self.setStartMark)
        regionsMenu.addAction(ac)

        self.setStopMarkButton = QPushButton("set stop")
        self.setStopMarkButton.setEnabled(False)
        self.setStopMarkButton.clicked.connect(self.setStopMark)
        ac = QAction('Set Stop Mark', self)
        ac.setShortcut('9')
        ac.triggered.connect(self.setStopMark)
        regionsMenu.addAction(ac)

        self.jumpToStartMarkButton = QPushButton("->start")
        self.jumpToStartMarkButton.setEnabled(False)
        self.jumpToStartMarkButton.clicked.connect(self.jumpToStartMark)
        ac = QAction('Jump to Start Mark', self)
        ac.setShortcut('/')
        ac.triggered.connect(self.jumpToStartMark)
        regionsMenu.addAction(ac)

        self.jumpToStopMarkButton = QPushButton("->stop")
        self.jumpToStopMarkButton.setEnabled(False)
        self.jumpToStopMarkButton.clicked.connect(self.jumpToStopMark)
        ac = QAction('Jump to Stop Mark', self)
        ac.setShortcut('*')
        ac.triggered.connect(self.jumpToStopMark)
        regionsMenu.addAction(ac)

        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.speedUpButton)
        controlLayout.addWidget(self.slowDownButton)
        controlLayout.addWidget(self.jumpFwdButton)
        controlLayout.addWidget(self.jumpBackButton)
        controlLayout.addWidget(self.setStartMarkButton)
        controlLayout.addWidget(self.setStopMarkButton)
        controlLayout.addWidget(self.jumpToStartMarkButton)
        controlLayout.addWidget(self.jumpToStopMarkButton)
        controlLayout.addWidget(self.fwdFrameButton)
        controlLayout.addWidget(self.backFrameButton)

        self.saveRegionButton = QPushButton("save")
        # self.saveRegionButton.setEnabled(False)
        self.saveRegionButton.clicked.connect(self.saveRegion)
        ac = QAction('Save Region', self)
        ac.setShortcut('8')
        ac.triggered.connect(self.saveRegion)
        regionsMenu.addAction(ac)

        self.loadRegionButton = QPushButton("load")
        # self.loadRegionButton.setEnabled(False)
        self.loadRegionButton.clicked.connect(self.loadRegion)
        ac = QAction('Load Region', self)
        ac.triggered.connect(self.loadRegion)
        regionsMenu.addAction(ac)

        self.deleteRegionButton = QPushButton("delete")
        # self.deleteRegionButton.setEnabled(False)
        self.deleteRegionButton.clicked.connect(self.deleteRegion)
        ac = QAction('Delete Region', self)
        ac.triggered.connect(self.deleteRegion)
        regionsMenu.addAction(ac)

        self.restoreRegionButton = QPushButton("restore")
        # self.restoreRegionButton.setEnabled(False)
        self.restoreRegionButton.clicked.connect(self.restoreRegion)
        ac = QAction('Restore Region', self)
        ac.triggered.connect(self.restoreRegion)
        regionsMenu.addAction(ac)

        self.regionListWidget = QListWidget()
        self.regionListWidget.setSortingEnabled(True)

        regionListButtonsLayout = QVBoxLayout()
        regionListButtonsLayout.addWidget(self.saveRegionButton)
        regionListButtonsLayout.addWidget(self.loadRegionButton)
        regionListButtonsLayout.addWidget(self.deleteRegionButton)
        regionListButtonsLayout.addWidget(self.restoreRegionButton)

        regionListLayout = QHBoxLayout()
        regionListLayout.addWidget(self.regionListWidget)
        regionListLayout.addLayout(regionListButtonsLayout)

        self.setMetaLabelMarkButton = QPushButton("Set Meta Label")
        self.setMetaLabelMarkButton.clicked.connect(self.setMetaLabelMark)
        ac = QAction('Set Meta Label Mark', self)
        ac.setShortcut('2')
        ac.triggered.connect(self.setMetaLabelMark)

        regionsMenu.addAction(ac)
        ac = QAction('Jump to Meta Label Mark', self)
        ac.setShortcut('j')
        ac.triggered.connect(self.jumpToMetaLabelMark)
        regionsMenu.addAction(ac)

        self.preTrialLightOffMarkLabel = QLabel(str(self.preTrialLightOffMark))
        self.trialStartMarkLabel = QLabel(str(self.trialStartMark))
        self.trialStopMarkLabel = QLabel(str(self.trialStopMark))
        self.probeStartMarkLabel = QLabel(str(self.probeStartMark))
        self.probeStopMarkLabel = QLabel(str(self.probeStopMark))
        self.postProbeLightOnMarkLabel = QLabel(str(self.postProbeLightOnMark))

        metaLabelsLayout = QHBoxLayout()
        metaLabelsLayout.addWidget(QLabel("Pretrial light off:"))
        metaLabelsLayout.addWidget(self.preTrialLightOffMarkLabel)
        metaLabelsLayout.addWidget(QLabel("Trial start"))
        metaLabelsLayout.addWidget(self.trialStartMarkLabel)
        metaLabelsLayout.addWidget(QLabel("Trial stop"))
        metaLabelsLayout.addWidget(self.trialStopMarkLabel)
        metaLabelsLayout.addWidget(QLabel("Probe start"))
        metaLabelsLayout.addWidget(self.probeStartMarkLabel)
        metaLabelsLayout.addWidget(QLabel("Probe stop"))
        metaLabelsLayout.addWidget(self.probeStopMarkLabel)
        metaLabelsLayout.addWidget(QLabel("Postprobe light on:"))
        metaLabelsLayout.addWidget(self.postProbeLightOnMarkLabel)
        metaLabelsLayout.addWidget(self.setMetaLabelMarkButton)

        self.statusLabel = QLabel("yo")
        self.statusLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout = QVBoxLayout()
        layout.addWidget(self.videoParent)
        layout.addLayout(controlLayout)
        layout.addLayout(regionListLayout)
        layout.addLayout(metaLabelsLayout)
        layout.addWidget(self.statusLabel)

        wid = QWidget(self)
        wid.setLayout(layout)
        self.setCentralWidget(wid)

    def initMediaPlayer(self):
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.mediaPositionChanged)
        self.mediaPlayer.durationChanged.connect(self.mediaDurationChanged)
        self.mediaPlayer.error.connect(self.handleMediaError)

    def openVideo(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video", QDir.homePath())

        if fileName != '':
            self.loadVideoFromFilename(fileName)

    def loadVideoFromFilename(self, fileName):
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
        self.playButton.setEnabled(True)
        self.speedUpButton.setEnabled(True)
        self.slowDownButton.setEnabled(True)
        self.jumpFwdButton.setEnabled(True)
        self.jumpBackButton.setEnabled(True)
        self.setStartMarkButton.setEnabled(True)
        self.setStopMarkButton.setEnabled(True)
        self.fwdFrameButton.setEnabled(True)
        self.backFrameButton.setEnabled(True)

        self.startMark = None
        self.stopMark = None

        ffmsvid = ffms2.VideoSource(fileName)
        # print("Frames:{}".format(ffmsvid.properties.NumFrames))
        # print("Times:{}".format(ffmsvid.track.timecodes))
        self.frameTimes = np.array(ffmsvid.track.timecodes)

        self.videoFileName = fileName

        self.setWindowTitle("Camera Marker - {}".format(fileName))

    def exitCall(self):
        sys.exit(0)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.statusLabel.setText("Pausing")
        else:
            self.mediaPlayer.play()
            self.statusLabel.setText("Playing")

    def speedUp(self):
        self.mediaPlayer.setPlaybackRate(self.mediaPlayer.playbackRate() * 1.25)
        self.statusLabel.setText("New playback rate: {}".format(self.mediaPlayer.playbackRate()))

    def slowDown(self):
        self.mediaPlayer.setPlaybackRate(self.mediaPlayer.playbackRate() / self.SPEED_CHANGE_FACTOR)
        self.statusLabel.setText("New playback rate: {}".format(self.mediaPlayer.playbackRate()))

    def jumpFwd(self):
        self.mediaPlayer.setPosition(self.mediaPlayer.position() + self.JUMP_DIST)
        self.statusLabel.setText("Jumped by {} ms. New position: {}".format(
            self.JUMP_DIST, self.mediaPlayer.position()))

    def jumpBack(self):
        self.mediaPlayer.setPosition(self.mediaPlayer.position() - self.JUMP_DIST)
        self.statusLabel.setText("Jumped by {} ms. New position: {}".format(
            -self.JUMP_DIST, self.mediaPlayer.position()))

    def setStartMark(self):
        self.startMark = self.mediaPlayer.position()
        self.statusLabel.setText("Start Mark set to {}".format(self.startMark))

    def setStopMark(self):
        self.stopMark = self.mediaPlayer.position()
        self.statusLabel.setText("Stop Mark set to {}".format(self.stopMark))

    def setMetaLabelMark(self):
        option, ok = QInputDialog.getInt(
            self, "Which mark are you making", "1 - pretrial light off\n2 - trial start\n3 - trial end\n4 - probe start\n5 - probe end\n6 - post probe light on")
        if ok:
            if option == 1:
                self.preTrialLightOffMark = self.mediaPlayer.position()
                self.statusLabel.setText(
                    "pre trial light off mark set to {}".format(self.preTrialLightOffMark))
                self.preTrialLightOffMarkLabel.setText(str(self.preTrialLightOffMark))
            elif option == 2:
                self.trialStartMark = self.mediaPlayer.position()
                self.statusLabel.setText(
                    "trial start mark set to {}".format(self.trialStartMark))
                self.trialStartMarkLabel.setText(str(self.trialStartMark))
            elif option == 3:
                self.trialStopMark = self.mediaPlayer.position()
                self.statusLabel.setText(
                    "trial Stop mark set to {}".format(self.trialStopMark))
                self.trialStopMarkLabel.setText(str(self.trialStopMark))
            elif option == 4:
                self.probeStartMark = self.mediaPlayer.position()
                self.statusLabel.setText(
                    "probe start mark set to {}".format(self.probeStartMark))
                self.probeStartMarkLabel.setText(str(self.probeStartMark))
            elif option == 5:
                self.probeStopMark = self.mediaPlayer.position()
                self.statusLabel.setText(
                    "probe Stop mark set to {}".format(self.probeStopMark))
                self.probeStopMarkLabel.setText(str(self.probeStopMark))
            elif option == 6:
                self.postProbeLightOnMark = self.mediaPlayer.position()
                self.statusLabel.setText(
                    "probe Stop mark set to {}".format(self.postProbeLightOnMark))
                self.postProbeLightOnMarkLabel.setText(str(self.postProbeLightOnMark))

    def jumpToMetaLabelMark(self):
        option, ok = QInputDialog.getInt(
            self, "Which mark are you jumping to", "1 - pretrial light off\n2 - trial start\n3 - trial end\n4 - probe start\n5 - probe end\n6 - post probe light on")
        if ok:
            if option == 1:
                if self.preTrialLightOffMark is not None:
                    self.statusLabel.setText(
                        "Jumping to mark at {}".format(self.preTrialLightOffMark))
                    self.mediaPlayer.setPosition(self.preTrialLightOffMark)
                else:
                    self.statusLabel.setText("no mark to jump to")
            elif option == 2:
                if self.trialStartMark is not None:
                    self.statusLabel.setText("Jumping to mark at {}".format(self.trialStartMark))
                    self.mediaPlayer.setPosition(self.trialStartMark)
                else:
                    self.statusLabel.setText("no mark to jump to")
            elif option == 3:
                if self.trialStopMark is not None:
                    self.statusLabel.setText("Jumping to mark at {}".format(self.trialStopMark))
                    self.mediaPlayer.setPosition(self.trialStopMark)
                else:
                    self.statusLabel.setText("no mark to jump to")
            elif option == 4:
                if self.probeStartMark is not None:
                    self.statusLabel.setText("Jumping to mark at {}".format(self.probeStartMark))
                    self.mediaPlayer.setPosition(self.probeStartMark)
                else:
                    self.statusLabel.setText("no mark to jump to")
            elif option == 5:
                if self.probeStopMark is not None:
                    self.statusLabel.setText("Jumping to mark at {}".format(self.probeStopMark))
                    self.mediaPlayer.setPosition(self.probeStopMark)
                else:
                    self.statusLabel.setText("no mark to jump to")
            elif option == 6:
                if self.postProbeLightOnMark is not None:
                    self.statusLabel.setText(
                        "Jumping to mark at {}".format(self.postProbeLightOnMark))
                    self.mediaPlayer.setPosition(self.postProbeLightOnMark)
                else:
                    self.statusLabel.setText("no mark to jump to")

    def jumpToStartMark(self):
        if self.startMark is not None:
            self.statusLabel.setText("Jumping to start mark at {}".format(self.startMark))
            self.mediaPlayer.setPosition(self.startMark)
        else:
            self.statusLabel.setText("no start mark to jump to")

    def jumpToStopMark(self):
        if self.stopMark is not None:
            self.statusLabel.setText("Jumping to stop mark at {}".format(self.stopMark))
            self.mediaPlayer.setPosition(self.stopMark)
        else:
            self.statusLabel.setText("no stop mark to jump to")

    def fwdFrame(self):
        currentFrame = np.searchsorted(self.frameTimes, self.mediaPlayer.position())
        if currentFrame < len(self.frameTimes):
            self.mediaPlayer.setPosition(int(self.frameTimes[currentFrame+1]))
            self.statusLabel.setText("Going fwd one frame")
        else:
            self.statusLabel.setText("Can't go fwd a frame, already at end")

    def backFrame(self):
        currentFrame = np.searchsorted(self.frameTimes, self.mediaPlayer.position())
        if currentFrame > 0:
            self.mediaPlayer.setPosition(int(self.frameTimes[currentFrame-1]))
            self.statusLabel.setText("Going back one frame")
        else:
            self.statusLabel.setText("Can't go back a frame, already at start")

    def saveRegion(self):
        well, ok = QInputDialog.getInt(self, "Well", "well")
        if ok:
            reg = dict()
            reg['start'] = self.startMark
            reg['stop'] = self.stopMark
            reg['well'] = well
            # self.regionListWidget.addItem(json.dumps(reg))
            self.regionListWidget.addItem(CMRegionListItem(self.startMark, self.stopMark, well))

    def loadRegion(self):
        listItem = self.regionListWidget.currentItem()
        self.startMark = listItem.start
        self.stopMark = listItem.stop
        # d = json.loads(listItem.text())
        # self.startMark = d['start']
        # self.stopMark = d['stop']
        self.statusLabel.setText("Loaded item {}".format(listItem))

    def deleteRegion(self):
        self.deletedItem = self.regionListWidget.takeItem(self.regionListWidget.currentRow())
        self.statusLabel.setText("Deleted {}".format(self.deletedItem))

    def restoreRegion(self):
        if self.deletedItem is None:
            self.statusLabel.setText("No item to restore")
        else:
            self.regionListWidget.addItem(self.deletedItem)
            self.statusLabel.setText("Restored item {}".format(self.deletedItem))

    def mediaStateChanged(self, state):
        # print("mediaStateChanged {}".format(state))
        pass

    def mediaPositionChanged(self, position):
        # print("mediaPositionChanged {}".format(position))
        pass

    def mediaDurationChanged(self, dur):
        # print("mediaDurationChanged {}".format(dur))
        pass

    def handleMediaError(self, er):
        print("handleMediaError {}".format(er))

    def exportRegions(self):
        it = reversed([self.regionListWidget.item(r) for r in range(self.regionListWidget.count())])
        lastStop = -1
        for i in it:
            if i.stop <= i.start or i.start <= lastStop:
                print("error in interval: {} - {}. Cancelling save".format(i.start, i.stop))
                return
            lastStop = i.stop

        fnfilt = "rgs(*.rgs)"
        defaultFileName = self.videoFileName + ".rgs"
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Regions", defaultFileName, fnfilt)

        if fileName != '':
            if not fileName.endswith(".rgs"):
                fileName += ".rgs"
            self.saveRegionsFromFilename(fileName)

    def saveRegionsFromFilename(self, fileName):
        it = reversed([self.regionListWidget.item(r) for r in range(self.regionListWidget.count())])
        with open(fileName, 'w') as csvfile:
            w = csv.writer(csvfile)
            w.writerow([self.preTrialLightOffMark, self.trialStartMark, self.trialStopMark])
            w.writerow([self.probeStartMark, self.probeStopMark, self.postProbeLightOnMark])
            for i in it:
                w.writerow([i.start, i.stop, i.well])

    def importRegions(self):
        fnfilt = "rgs(*.rgs)"
        if self.videoFileName is not None:
            defaultFileName = self.videoFileName + ".rgs"
        else:
            defaultFileName = QDir.homePath()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Regions", defaultFileName, fnfilt)

        if fileName != '':
            self.loadRegionsFromFilename(fileName)

    def loadRegionsFromFilename(self, fileName):
        with open(fileName, 'r') as csvfile:
            reader = csv.reader(csvfile)
            r = [[int(v) if len(v) > 0 else None for v in l] for l in list(reader)]

            self.regionListWidget.clear()

            self.preTrialLightOffMark = r[0][0]
            self.trialStartMark = r[0][1]
            self.trialStopMark = r[0][2]
            self.probeStartMark = r[1][0]
            self.probeStopMark = r[1][1]
            self.postProbeLightOnMark = r[1][2]
            self.preTrialLightOffMarkLabel.setText(str(self.preTrialLightOffMark))
            self.trialStartMarkLabel.setText(str(self.trialStartMark))
            self.trialStopMarkLabel.setText(str(self.trialStopMark))
            self.probeStartMarkLabel.setText(str(self.probeStartMark))
            self.probeStopMarkLabel.setText(str(self.probeStopMark))
            self.postProbeLightOnMarkLabel.setText(str(self.postProbeLightOnMark))

            for i in r[2:]:
                self.regionListWidget.addItem(CMRegionListItem(int(i[0]), int(i[1]), int(i[2])))


def main():
    parent_app = QApplication(sys.argv)
    cmw = CMWidget()
    cmw.resize(600, 600)
    cmw.show()
    sys.exit(parent_app.exec_())


if __name__ == "__main__":
    main()
