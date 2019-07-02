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
import xml.etree.ElementTree as ET
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
## globals #####################################
RATE = 22050
CHUNK = 2048

## threads #####################################




class AudioRecorder(QObject):
    '''
    AudioRecorder(QObject): thread which accepts input from specified
    audio input device (default is 0) in chunks, then pushes audio to 
    queue for processing by Chromatizer thread. 
    '''
    
    def __init__(self, queue,  rate = RATE, chunk = CHUNK, input_device_index=0): 
#rate = librosa default
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
        self.chroma = np.array([[],
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
                                []])
#key = note name
#value = chroma index for chroma matrix 
        self._chromaToIndex = {'C':0,
                                'C#':1,
                                'D-':1,
                                'D':2,
                                'D#':3,
                                'E-':3,
                                'E':4,
                                'F':5,
                                'F#':6,
                                'G-':6,
                                'G':7,
                                'G#':8,
                                'A-':8,
                                'A':9,
                                'A#':10,
                                'B-':10,
                                'B':11}

#key = chroma index for a particular note (see chroma to index)
#value = for each chroma index, amplitude of fund. freq. and partials
#see https://en.wikipedia.org/wiki/Harmonic_series_(music)
        self._cOvertones = np.array([1.825,0,0,0,.2,0,0,.5,0,0,.143,0])
        self._harmonics = {}
        for i in range(0,12):
            self._harmonics[i] = np.roll(self._cOvertones, i)

    def musicXMLtoChroma(self):
        importer = musicxml.xmlToM21.MusicXMLImporter()
        
        root = ET.parse(self.file).getroot()
        beatUnit = ""
        perMinute = ""
        for attribute in root.iter('beat-unit'):
            beatUnit = attribute.text
        for attribute in root.iter('per-minute'):
            perMinute = attribute.text
        score = importer.scoreFromFile(self.file)
        #score.show()
        scoreTempo = tempo.MetronomeMark(None, int(perMinute), beatUnit)        
       # print("Debug: scoreTempo", scoreTempo)
        score.insert(scoreTempo)
        #print("Debug: score.seconds",score.seconds)
        parts_extracted = score.parts
        parts_and_voices = []
        for part in parts_extracted:
            part.insert(scoreTempo)
            if part.hasPartLikeStreams():
             #   print(part.partName, "has part-like substream")
                for voice in part.voices:
                    parts_and_voices.append(voice)
            else:
               # print(part.partName, "doesn't have part-like substream")
               # print(part.hasVoices())

                parts_and_voices.append(part)
        length_score_seconds = score.seconds
######### extracting note names and indexes by part ######
        chroma_nodes= {}
        chroma_per_part = {}

        for i in range(len(parts_and_voices)):
           
# cover all parts
# print("Debug: parts[i]._getSeconds()", parts[i]._getSeconds())
            chroma_nodes[parts_and_voices[i].partName] = []
#each part is key in dict, list of notes in each part
            for note in parts_and_voices[i].flat.notes:
#cover all notes in specific part
                if not chroma_nodes[parts_and_voices[i].partName] and not note.isChord:
                    start = 0
                    end = note.seconds + start
                    chroma_node = (note.name, start, end)
                    chroma_nodes[parts_and_voices[i].partName].append(chroma_node)

                elif chroma_nodes[parts_and_voices[i].partName] and not note.isChord:
#print(chroma_tuples_per_part[parts[i].partName][j-1])
                    start = chroma_nodes[parts_and_voices[i].partName][-1][2]
                    end = note.seconds + start
                    chroma_node = (note.name, start, end)
                    chroma_nodes[parts_and_voices[i].partName].append(chroma_node)
                elif not chroma_nodes[parts_and_voices[i].partName] and note.isChord:
                    note_temp = []
                    for pitch in note.pitches:
                        note_temp.append(pitch.name)
                    start = 0
                    end = note.seconds + start
                    chroma_node = (note_temp, start, end)
                    chroma_nodes[parts_and_voices[i].partName].append(chroma_node)
                elif chroma_nodes[parts_and_voices[i].partName] and note.isChord:
                    note_temp = []
                    for pitch in note.pitches:
                        note_temp.append(pitch.name)
                    start = chroma_nodes[parts_and_voices[i].partName][-1][2]
                    end = note.seconds + start
                    chroma_node = (note_temp,start,end)
                    chroma_nodes[parts_and_voices[i].partName].append(chroma_node)
        #print("Debug: chroma_nodes", chroma_nodes)
## generating chroma vectors for each part #####################
        num_of_frames = 0
        for part in chroma_nodes:
            
           # print("part:", part)
            chroma_per_part[part] = [[],
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


            for i in np.arange(0, length_score_seconds, ((CHUNK/4)/RATE)):                
                # print(num_of_frames)
                notes_in_frame = []
                chroma_in_frame = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                for node in chroma_nodes[part]:
                    if node[1] < i and i < node[2] and type(node[0]) == str:
                        #if in current frame and only one note
                        #print(node)
                        notes_in_frame.append(self._chromaToIndex[node[0]])
                    elif node[1] < i and i<node[2] and type(node[0]) == list:
                        #if in current frame and multiple notes
                        #print(node[0])
                        note_index = []
                        for note in node[0]:
                            note_index = self._chromaToIndex[note]

                       
                            if note_index not in notes_in_frame:
                                notes_in_frame.append(note_index)
                    #    print(notes_in_frame)
                    elif node[2] > i:
                        break
                                   
                for note in notes_in_frame:
                    if type(note) == int:
                        chroma_in_frame +=  self._harmonics[note]
               # print("Chroma in frame:", chroma_in_frame)
               # print("length of Chroma in frame:", len(chroma_in_frame))
                for j in range(len(chroma_per_part[part])):
                    num_of_frames = len(chroma_per_part[part][j])+1
                    chroma_per_part[part][j].append(chroma_in_frame[j])
        #print(chroma_per_part) 
        full_chroma = np.zeros((12, num_of_frames))
        for part in chroma_per_part:
            chroma_per_part[part] = np.array(chroma_per_part[part])
            for i in range(len(chroma_per_part[part])):
                full_chroma[i] += chroma_per_part[part][i] 

        chroma_normed = full_chroma / full_chroma.max(axis=0)
        np.place(chroma_normed, np.isnan(chroma_normed), [0])
        #print("self.chroma:", self.chroma)
        self.chroma = chroma_normed   
        return self.chroma

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
        self.timer.start(10000)

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
    musicxmlparser = MusicXMLprocessor("/Users/hypatia/Qt_projects/wtq.xml")
    chroma = musicxmlparser.musicXMLtoChroma()
    print(chroma[-11:-1])
    display.specshow(chroma, x_axis = "time", y_axis = "chroma", cmap = "viridis")
    plt.show()
    app = QApplication(sys.argv)
    mainwindow = App()
    mainwindow.show()
    exit_code = app.exec_()
sys.exit(exit_code)
