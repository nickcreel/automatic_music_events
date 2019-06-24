"""
realtime_audio_chromagram.py:

Qt application which records chunks of audio and extracts chroma features
from recorded audio. Recording of audio and chroma extraction are separated
into two different threads, which interact with the same threadsafe queue.

"""
###########import statements#################
##standard PyQt imports (thanks christos!)###
from PyQt5 import QtGui, QtCore, QtSvg
from PyQt5.QtWidgets import (QMainWindow, QApplication, QCheckBox, QComboBox, QDateTimeEdit,QMessageBox,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget,QLCDNumber, QDoubleSpinBox,QGraphicsItem, QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsObject, QGraphicsLineItem,
                         QGraphicsScene, QGraphicsView, QStyle, QWidget, QLabel, QHBoxLayout, QMenuBar, QTextEdit, QGridLayout, QAction, QActionGroup, QToolBar, QToolBox, QToolButton)
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, Qt, pyqtSlot, QThread, QPointF, QRectF, QLineF, QRect
from PyQt5.QtGui import (QPen, QTransform)
from PyQt5.QtSvg import QGraphicsSvgItem
##############################################
import pyaudio #audio streams
import numpy as np 
from scipy import ndimage
from librosa import feature, display, decompose 
from music21 import *
#audio -> chroma information and display plots of chromagrams
###############################################
import matplotlib
matplotlib.use("Qt5Agg")# displaying matplotlib plots in Qt
import queue #threadsafe queue
################################################
import sys 
################################################
from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
################################################

## threads #####################################

class AudioRecorder(QObject):
    '''
    AudioRecorder(QObject): thread which accepts input from specified
    audio input device (default is 0) in chunks, then pushes audio to 
    queue for processing by Chromatizer thread. 
    '''
    
    def __init__(self, queue,  rate = 22050, chunk = 2048, input_device_index=0): #rate = librosa default
        QObject.__init__(self) #getting all the qthread stuff
        self.rate = rate
        self.chunk = chunk
        self.queue = queue
        self.p = pyaudio.PyAudio()
        self.input_device_index = input_device_index
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels = 1,
                                  rate = self.rate,
                                  input = True,
                                  input_device_index = self.input_device_index,
                                  frames_per_buffer = self.chunk,
                                  stream_callback = self._callback)
        self.stop = False
        
    #def startStream(self):
    def stopStream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
    def  _callback(self, in_data, frame_count, time_info, status):
        """
        grab data from buffer,
        put data and rate into queue
        continue
        """
        data = np.frombuffer(in_data, "float32")
        self.queue.put(data)
        return (data, pyaudio.paContinue)
    
class Chromatizer(QObject):
    '''
    Chromatizer(QObject): accepts chunks of audio information as input
    from audio buffer, calculates chroma matrix of audio chunk, 
    pushes chroma information to chroma queue for comparison to 
    reference chroma. Currently prints value of fundamental frequency
    of audio chunk. 
    '''
    def __init__(self, inputqueue, outputqueue):
        QObject.__init__(self)
        self.outputqueue = outputqueue
        self.inputqueue = inputqueue
        self.rate = 22050 
    def _display(self):
        chroma = self.chroma_frames.get_nowait()
        display.specshow(chroma, y_axis = "chroma", x_axis = "time")

    @pyqtSlot(object)    
    def calculate(self, frame):
        y = frame
        sr = self.rate
        chroma = feature.chroma_cqt(y, sr,
                                        bins_per_octave = 12*3)
        chroma_filtered = np.minimum(chroma, 
                                        decompose.nn_filter(chroma,
                                        aggregate = np.median,
                                        metric = 'cosine'))
        chroma_smooth = ndimage.median_filter(chroma_filtered,
                                                size = (1,9))
        self.outputqueue.put(chroma_smooth)
        print(np.argmax(np.mean(chroma_smooth,axis=1)))

class Reader(QObject):
    signalToChromatizer = pyqtSignal(object)
    def __init__(self, queue):
        QObject.__init__(self)
        self.queue = queue
        self.timer = QTimer()
        self.timer.timeout.connect(self.getter)
        self.timer.start(50)

    def getter(self):
        try:
            frame = self.queue.get_nowait()
        except:
            frame = None
        if frame is not None:
            self.signalToChromatizer.emit(frame)
         
class MusicXMLprocessor:
    '''
    MusicXMLprocessor: accepts musicxml file as input, calculates chroma
    vector representation of piece of music. intended as pre-processor 
    before actual comparison happens...
    '''
    def __init__(self, filename):
        self.file = filename
        # self.chroma[0] = C, ..., self.chroma[11] = B
        self.chroma = [[],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       [],
                       []]
        #key = note name
        #value = chroma index for chroma matrix 
        self._chromaToIndex = {'C':0,
                              'C#':1,
                              'D':2,
                              'D#':3,
                              'E':4,
                              'F':5,
                              'F#':6,
                              'G':7,
                              'G#':8,
                              'A':9,
                              'A#':10,
                              'B':11}

        #key = chroma index for a particular note (see chroma to index)
        #value = for each chroma index, amplitude of fund. freq. and partials
        #see https://en.wikipedia.org/wiki/Harmonic_series_(music)
        self._harmonics = {0:[1.5,0,0,0,.8,0,0,.66,0,0,.57,0],
                          1:[0,1.5,0,0,0,.8,0,0,.66,0,0,.57],                    
                          2:[.57,0,1.5,0,0,0,.8,0,0,.66,0,0],
                          3:[0,.57,0,1.5,0,0,0,.8,0,0,.66,0],
                          4:[0,0,.57,0,1.5,0,0,0,.8,0,0,.66],
                          5:[.66,0,0,.57,0,1.5,0,0,0,.8,0,0],
                          6:[0,.66,0,0,.57,0,1.5,0,0,0,.8,0],
                          7:[0,0,.66,0,0,.57,0,1.5,0,0,0,.8],
                          8:[.8,0,0,.66,0,0,.57,0,1.5,0,0,0],
                          9:[0,.8,0,0,.66,0,0,.57,0,1.5,0,0],
                          10:[0,0,.8,0,0,.66,0,0,.57,0,1.5,0],
                          11:[0,0,0,.8,0,0,.66,0,0,.57,0,1.5]}
    def musicXMLtoChroma(self):
        score = converter.parse(self.file)
        duration_of_score_in_seconds = score.seconds()
        notes = []

        for element in score.flat:
            if type(element) == music21.chord.Chord:
                for pitch in element.pitches:
                    notes.append((element.offset, element.seconds, 
                                        self._chromaToIndex.get(pitch.name)))
            elif type(element) == music21.note.Note:
                notes.append((element.offset, element.seconds,
                                    self._chromaToIndex.get(note.name)))


class App(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = "Realtime Chroma Extraction"
        self.width = 640
        self.height = 400
        self.setupThreads()
        self.signalsandSlots()
        self.timer = QTimer()
        self.timer.timeout.connect(self.closeEvent2)
        self.timer.setSingleShot(True)
        self.timer.start(5000)

    def setupThreads(self):
        self.readQueue = queue.Queue()
        self.chromaQueue = queue.Queue()
        self.audioThread = QThread()
        self.audioRecorder = AudioRecorder(self.readQueue)
        self.audioRecorder.moveToThread(self.audioThread)
        self.audioThread.start()
        self.readerThread = QThread()
        self.reader = Reader(self.readQueue)
        self.reader.moveToThread(self.readerThread)
        self.chromaThread = QThread()
        self.chromatizer = Chromatizer(inputqueue = self.readQueue,
                                    outputqueue = self.chromaQueue)
        self.chromatizer.moveToThread(self.chromaThread)
        self.readerThread.start()
        self.chromaThread.start()

    def closeEvent2(self):
        self.audioRecorder.stopStream()
        self.reader.timer.stop()
         
    def signalsandSlots(self):
        self.reader.signalToChromatizer.connect(self.chromatizer.calculate)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = App()
    mainwindow.show()
    exit_code = app.exec_()
    sys.exit(exit_code)
