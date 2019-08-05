"""
realtime_audio_chromagram.py:

Qt application which records chunks of audio and extracts chroma features
from recorded audio. Recording of audio and chroma extraction are separated
into two different threads, which interact with the same threadsafe queue.

"""
###########import statements#################
##standard PyQt imports (thanks christos!)###
from PyQt5 import QtGui, QtCore, QtSvg
from PyQt5.QtWidgets import (QMainWindow, QApplication, QCheckBox, QComboBox,
        QDateTimeEdit,QMessageBox,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget,QLCDNumber, QDoubleSpinBox,QGraphicsItem,
        QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsObject,
        QGraphicsLineItem,
        QGraphicsScene, QGraphicsView, QStyle, QWidget, QLabel, QHBoxLayout,
        QMenuBar, QTextEdit, QGridLayout, QAction, QActionGroup, QToolBar,
        QToolBox, QToolButton, QInputDialog, QFileDialog)
from PyQt5.QtCore import (QObject, pyqtSignal, QTimer, Qt, pyqtSlot, QThread,
                            QPointF, QRectF, QLineF, QRect)
from PyQt5.QtGui import (QPen, QTransform)
from PyQt5.QtSvg import QGraphicsSvgItem
import pyqtgraph as pg
##############################################
###############################################
import queue #threadsafe queue
import sys
################################################
from AudioRecorder import AudioRecorder
from Chromatizer import Chromatizer
from OnlineDTW import OnlineDTW
from MusicXMLprocessor import MusicXMLprocessor
from OSCClient import OSCclient

#####################################################
## Qt app instantiation -> thread setup
#####################################################
class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = "Realtime Chroma Extraction"
        self.width = 640
        self.height = 400
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.win = pg.GraphicsWindow(title="OnlineDTW")
        self.p = self.win.addPlot(title = "Minimum Cost Path",
                                  labels = {
                                  'bottom':"Audio Frame (U(t))",
                                  'left':"Score Frame (V(j))"},
                                   backround = "white")

        self.curve = self.p.plot(pen="r", background="w")
        self.setupThreads()
        self.signalsandSlots()

    def setupThreads(self):
        file = "twinklescore.musicxml"
        self.scorechroma = MusicXMLprocessor(file)
        self.scorechroma.musicXMLtoChroma()
        cues = self.scorechroma.eventTriggerOnset
        self.readQueue = queue.Queue()
        self.chromaQueue = queue.Queue()
        ## threads
        self.audioThread = QThread()
        self.dtwThread = QThread()
        self.chromaThread = QThread()
        self.oscthread = QThread()
        self.audioRecorder = AudioRecorder(self.readQueue, wavfile = "twinkleaudio.wav")
        self.audioRecorder.moveToThread(self.audioThread)
        self.chromatizer = Chromatizer(inputqueue = self.readQueue,
                                    outputqueue = self.chromaQueue)
        self.chromatizer.moveToThread(self.chromaThread)
        self.onlineDTW = OnlineDTW(self.scorechroma.chroma, self.chromaQueue, cues)
        self.onlineDTW.moveToThread(self.dtwThread)
        self.oscclient = OSCclient(ip = "10.5.30.72")
        self.oscclient.moveToThread(self.oscthread)
        self.audioThread.start()
        self.chromaThread.start()
        self.dtwThread.start()
        self.oscthread.start()

    def closeEvent2(self):
        self.audioRecorder.stopStream()

    def signalsandSlots(self):
        self.audioRecorder.signalToChromatizer.connect(self.chromatizer.calculate)
        self.chromatizer.signalToOnlineDTW.connect(self.onlineDTW.align)
        self.onlineDTW.signalToGUIThread.connect(self.plotter)
        self.onlineDTW.signalToOSCclient.connect(self.oscclient.emit)
        #self.onlineDTW.signalToPlotter.connect(self.plotter.animate)

    @pyqtSlot(object, object)
    def plotter(self, line, matrix):
        line.sort(axis = 0)
        self.curve.setData(line)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = App()
    exit_code = app.exec_()
    sys.exit(exit_code)
