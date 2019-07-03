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
################################################

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
    signalToOnlineDTW = pyqtSignal(object)
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
        
"""
class OnlineDTW(QObject):
    
    def __init__(self, score_chroma):
        QObject.__init__(self)
        self.scoreChroma = score_chroma
    #### parameters ###############################
        self.search_band_size = 200
        self.weight = 0.9
        self.maxRunCount = 3
    ###############################################
    #### distance matrices ########################
        self.globalCostMatrix = None

    @pyqtSlot(object)
    def align(self):
        
        OnlineDTW.align(): finds the best alignment between two chromagrams
        and returns the position of the best alignment in the global cost matrix.
        see dixon 2005 online dtw for algorithm details. 
        also received much help from Bochen Li and his reference version in Matlab. 
        

        previous = None
        inputindex = 0
        scoreindex = 0

        framenumscore = len(self.scoreChroma[0])
        framenumaudio = int(1.5 * framenumscore)

        self.globalCostMatrix = np.matrix(np.ones((framenumscore, framenumaudio)) * np.inf)
        localCostMatrix = np.matrix(np.ones((framenumscore, framenumaudio)) * np.inf)  

        pathcost = self._evaluatePathCost(inputindex, scoreindex)

        direction, x, y = self._getInc(inputindex, scoreindex)



    def _getInc(self, inputindex, scoreindex):
    _getInc: takes input index, score index as arguments and returns a char where:
    B = both
    C = column
    R = row
    which indicates the direction of the algorithm's calculation.
        if inputIndex < frameSize:
            return "B"
        if runCount > self.maxRunCount:
            if previous == "R":
                return "C"
            else:
                return "R"
        x, y = np.argmin

    def _evaluatePathCost(self, inputindex, scoreindex):
        ## some kind of cost calculation function here
        ## probably a good idea to borrow from dixon? but maybe better funcitons exist
""" 

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
        #because tempo is impossible to get with music21 for some reason
        #i'm extracting it directly from the xml using a parser
        #it's very dinky but works...
        root = ET.parse(self.file).getroot()
        beatUnit = ""
        perMinute = ""
        for attribute in root.iter('beat-unit'):
            beatUnit = attribute.text
        for attribute in root.iter('per-minute'):
            perMinute = attribute.text
        #load file
        score = importer.scoreFromFile(self.file)
        #must make tempo a metronome mark object to attach to music21 streams
        scoreTempo = tempo.MetronomeMark(None, int(perMinute), beatUnit)        
        score.insert(scoreTempo)
        #print("Debug: score.seconds",score.seconds)
        parts_extracted = score.parts
        parts_and_voices = []
        for part in parts_extracted:
            part.insert(scoreTempo)
            print(part.partName, part.seconds)
            if part.hasVoices():
                #voices could get lost if not extracted separately...
                for voice in part.voices:
                    parts_and_voices.append(voice)
            else:
                parts_and_voices.append(part)
        #duration for calculations later
        length_score_seconds = score.seconds
        ################################################
        #extracting note names and indexes by part 
        #a chroma node is a tuple which contains:
        #    the pitch(es) of a particular frame
        #        (if there's only one, type(pitch)  == str
        #         else, type(pitches) == list
        #    the start time of those pitches
        #        (which is 0 if the pitch is first
        #         or the end of the previous note if
        #         i > 0)
        #    the end time of those pitches
        #         (duration of pitch in seconds + start time)
        # these nodes are stored in a dictionary, chroma_nodes_per_part
        # where the key == a part in the piece
        # and value == list of chroma nodes in that part.
        ################################################
        chroma_nodes_per_part= {}
        chroma_vector_per_part = {}

        for i in range(len(parts_and_voices)):
           
        # cover all parts
            chroma_nodes_per_part[parts_and_voices[i].partName] = []
            #each part is key in dict, list of notes in each part
            for note in parts_and_voices[i].flat.notesAndRests:
            #cover all notes in specific part
                if not chroma_nodes_per_part[parts_and_voices[i].partName] and not note.isChord:
                    start = 0
                    end = note.seconds + start
                    chroma_node = (note.name, start, end)
                    chroma_nodes_per_part[parts_and_voices[i].partName].append(chroma_node)

                elif chroma_nodes_per_part[parts_and_voices[i].partName] and not note.isChord:
                    start = chroma_nodes_per_part[parts_and_voices[i].partName][-1][2]
                    end = note.seconds + start
                    chroma_node = (note.name, start, end)
                    chroma_nodes_per_part[parts_and_voices[i].partName].append(chroma_node)
                elif not chroma_nodes_per_part[parts_and_voices[i].partName] and note.isChord:
                    note_temp = []
                    for pitch in note.pitches:
                        note_temp.append(pitch.name)
                    start = 0
                    end = note.seconds + start
                    chroma_node = (note_temp, start, end)
                    chroma_nodes_per_part[parts_and_voices[i].partName].append(chroma_node)
                elif chroma_nodes_per_part[parts_and_voices[i].partName] and note.isChord:
                    note_temp = []
                    for pitch in note.pitches:
                        note_temp.append(pitch.name)
                    start = chroma_nodes_per_part[parts_and_voices[i].partName][-1][2]
                    end = note.seconds + start
                    chroma_node = (note_temp,start,end)
                    chroma_nodes_per_part[parts_and_voices[i].partName].append(chroma_node)
#############################################################                    
## generating chroma vectors for each part ##################
        num_of_frames = 0
        for part in chroma_nodes_per_part:
            print("part:", part)
            chroma_vector_per_part[part] = [[],
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
                notes_in_frame = []
                chroma_in_frame = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                for node in chroma_nodes_per_part[part]:
                    if node[1] < i and i <= node[2] and type(node[0]) == str:
                        #if in current frame and only one note
                        #print(node)
                        if node[0] in self._chromaToIndex:
                            notes_in_frame.append(self._chromaToIndex[node[0]])
                        else:
                            notes_in_frame.append("R")
                    elif node[1] < i and i<= node[2] and type(node[0]) == list:
                        #if in current frame and multiple notes
                        #print(node[0])
                        note_index = []
                        for note in node[0]:
                            if note in self._chromaToIndex:
                                note_index = self._chromaToIndex[note]
                            elif note == "R":
                                pass
                            if note_index not in notes_in_frame:
                                notes_in_frame.append(note_index)
                    elif node[2] > i:
                        break
                print(notes_in_frame)                   
                for note in notes_in_frame:
                    if type(note) == int:
                        chroma_in_frame +=  self._harmonics[note]
                    elif note == "R":
                        pass
                for j in range(len(chroma_vector_per_part[part])):
                    num_of_frames = len(chroma_vector_per_part[part][j])+1
                    chroma_vector_per_part[part][j].append(chroma_in_frame[j]) 

        full_chroma = np.zeros((12, num_of_frames))
        for part in chroma_vector_per_part:
            chroma_vector_per_part[part] = np.array(chroma_vector_per_part[part])
            for i in range(len(chroma_vector_per_part[part])):
                full_chroma[i] += chroma_vector_per_part[part][i] 

        chroma_normed = full_chroma / full_chroma.max(axis=0)
        #norming columnwise to preserve intensity of each note that has been 
        #played in a particular frame. if we didnt, polyphonic frames
        #would be artificially "louder" than monophonic frames.
        np.place(chroma_normed, np.isnan(chroma_normed), [0])
        #any values that are nan are 0/0, which means they should be 0
        #anyway. a row that sums to 0 means silence, which we should preserve. 
        self.chroma = chroma_normed   
        return self.chroma

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

## currently just testing but u know...will be other things soon. 
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
