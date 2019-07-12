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
##############################################
import pyaudio #audio streams
import numpy as np
from scipy import ndimage
from librosa import feature, display, decompose
from music21 import *
import xml.etree.ElementTree as ET
#audio -> chroma information and display plots of chromagrams
###############################################
import queue #threadsafe queue
import sys
import socket
################################################
import matplotlib
matplotlib.use("Qt5Agg")# displaying matplotlib plots in Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from math import log2
################################################
## globals #####################################
RATE = 22050
CHUNK = 2048
## threads #####################################
################################################

###
def freq2MIDI(frequency):
    '''
    freq2MIDI: takes freq in hz as input nd returns corresponding midi number
    to that frequency.
    '''
    midinum = 69 + (12 *(log2(frequency / 440)))
    return midinum

def MIDI2Freq(midinum):
    '''converts midi number to corresponding frequency in hz'''
    frequency = 440 * (2**((midinum - 69) / 12))
###

class AudioRecorder(QObject):
    '''
    AudioRecorder(QObject): thread which accepts input from specified
    audio input device (default is 0) in chunks, then pushes audio to
    queue for processing by Chromatizer thread.
    '''

    def __init__(self, queue,  rate = RATE, chunk = CHUNK,
                                            input_device_index=0):
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
        print("calculating chroma...")
        i = 0
        y = frame
        sr = self.rate
        chroma = feature.chroma_cqt(y, sr,
                                        bins_per_octave = 12*3)
        #filtering reduces volume of noise/partials
        chroma_filtered = np.minimum(chroma,
                                        decompose.nn_filter(chroma,
                                        aggregate = np.median,
                                        metric = 'cosine'))
        chroma_smooth = ndimage.median_filter(chroma_filtered,
                                                size = (1,9))
        self.outputqueue.put_nowait(chroma_smooth)
        self.signalToOnlineDTW.emit()

class OnlineDTW(QObject):

    def __init__(self, score_chroma, inputqueue):

        QObject.__init__(self)
    #### parameters ###############################
        self.search_band_size = 512
        self.diagonalWeight = 0.9
        self.maxRunCount = 3
    ##############################################
        self.scoreChroma = score_chroma
        self.framenumscore = len(self.scoreChroma[0])
        print(self.framenumscore)
        self.framenumaudio = self.framenumscore
        self.pathLenMax = self.framenumscore + self.framenumaudio
        self.audioChroma = np.empty_like(self.scoreChroma)
        self.inputQueue = inputqueue
    ###############################################
    #### distance matrices ########################
        self.globalCostMatrix = np.matrix(np.ones((self.framenumscore,
                                                    self.framenumaudio))
                                                        * np.inf)
        self.localCostMatrix = np.matrix(np.ones((self.framenumscore,
                                                    self.framenumaudio))
                                                        * np.inf)
    ###############################################
    #### least cost path ##########################
        self.pathOnlineIndex = 0
        self.pathFront = np.zeros((self.pathLenMax, 2))
        self.pathOnline = np.zeros((self.pathLenMax, 2))
        self.pathFront[0,:]= [1,1]
        self.frameQueue = queue.Queue()
        self.inputindex = 1
        self.scoreindex = 1
        self.fnum = 0
        self.previous = None

    @pyqtSlot(object)
    def align(self):
        '''
        OnlineDTW.align(): using a modified version of the dynamic time warping
        algorithm, finds a path of best alignment between two sequences, one
        known and one partially known. As frames of the partially known sequence
        are fed into the function, the "cost" or difference between both
        sequences is calculated, and the algorithm decides which point is
        the optimal next point in the least cost path by choosing the point with
        the least cost. Cost is cumulative and the cost of the current point
        depends on the cost of previous points. previous points also determine
        the direction that the algorithm predicts the next least cost point will
        be.

        TODO: needs to emit current alignment point to a OSC signal generator
        so that signals can be sent to QLab based on current alignment point.
        '''
        #!!!!!!!!!!!please read!!!!!!!!!!!!!!!!!!!!
        #note: dixon's description of the algorithm has the input index as the
        #row index of the cost matrix and the score index as the column index. i
        #prefer to think of the score index as the y axis, so i use the score
        #index as the row index of the cost matrix and the input index as the
        #columnwise index.
        #
        #or:
        #self.globalCostMatrix[scoreindex][inputindex] == proper way to index
        #cost matrix, as i've written it.
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # t:=1, j:=1
        frameStart = 1
        runCount = 0
        needNewFrame = 1
        pathcostindex = 0
        #according to the boundary rules of dtw, both sequences must start and
        #end in the same place...or, more specifically
        #first point in least cost path = (1,1)
        #last point in least cost path = (last point in original,
        #                                      last point in recorded)
        if self.scoreindex < self.framenumscore and self.fnum < self.framenumaudio:
            ## getting next frame ##
            if needNewFrame == 1:
                inputData = self.inputQueue.get_nowait()
                for i in range(len(inputData[0])):
                    self.frameQueue.put_nowait(inputData[:,i])# INPUT u(t) (step 3)
                 # but also step 7
                ## base case for recursion
                if self.fnum == 0:
                    self.fnum = self.fnum + 1
                    # EvaluatePathCost(t, j) (step 4)
                    self.audioChroma[:,0] = self.frameQueue.get_nowait()
                    self.localCostMatrix[0,0] = sum(self.scoreChroma[:,0]-
                                                      self.audioChroma[:,0])**2
                    self.globalCostMatrix[0,0] = self.localCostMatrix[0,0]
                else:
                    self.audioChroma[:,self.inputindex] = self.frameQueue.get_nowait()
            needNewFrame = 0
            direction = self._getInc(self.scoreindex, self.inputindex,
                                        self.previous, runCount)
            # LOOP
            print(f"fnum is {self.fnum}")
            print(f"Direction is {direction}")
            print(f"Previous is {previous}")
            print(f"scoreindex is {self.scoreindex}")
            print(f"inputindex is {self.inputindex}")
            if direction != "C": # step 5
                self.inputindex += 1  # step 6
                needNewFrame = 1 # where is step 7 happening?
                for k in range((self.scoreindex - (self.search_band_size + 1)),
                                    self.scoreindex): # step 8
                    if k > 0: # step 9
                        pathCost = self._evaluatePathCost(k, self.inputindex)
                        self.globalCostMatrix[k,self.inputindex] = pathCost #step 10
            if direction != "R": #step 11
                self.scoreindex += 1 #step 12
                for k in range((self.inputindex -(self.search_band_size + 1)),
                                    self.inputindex): #step 13
                    if k > 0: #step 14
                        pathCost = self._evaluatePathCost(self.scoreindex, k)
                        self.globalCostMatrix[self.scoreindex,k] = pathCost #step 15
            if direction == previous: #step 16
                runCount += 1 #step 17
            else: #step 18
                runCount = 1 #step 19
            if direction != "B": #step 20
                previous = self._getInc(self.scoreindex, self.inputindex,
                                        previous, runCount) #step 21
            # end loop
##get direction ##################################
##################################################
    def _getInc(self, scoreindex, inputindex, previous, runCount):
        '''
        _getInc: takes input index, score index as arguments and returns a
        char where:
        B = both
        C = column
        R = row
        which indicates the direction of the next alignment point
        '''
        print("in _getInc")
        if inputindex < self.search_band_size: #step 1
            return "B" #step 2
        if runCount > self.maxRunCount: #step 3
            if previous == "R": #step 4
                return "C" #step 5
            else: #step 6
                return "R" #step 7
        scoreindex_min = None
        inputindex_min = None
        #####################################
        ## need to find the minimum cost given our current scoreindex
        ## and input index. need to find argmin where either scoreindex
        ## and inputindex are static, and the other varies up to the current
        ## index.
        #####################################
        ## step 8 ####
        for i in range(0, scoreindex):
            inputindex_min = np.argmin(self.globalCostMatrix[i,inputindex])
        for i in range(0, inputindex):
            scoreindex_min = np.argmin(self.globalCostMatrix[scoreindex,i])
        if (self.globalCostMatrix[scoreindex_min[0],scoreindex_min[1]] <
            self.globalCostMatrix[inputindex_min[0],inputindex_min[1]]):
            x, y = scoreindex_min[0], scoreindex_min[1]
        else:
            x, y = inputindex_min[0], inputindex_min[1]
        ## end step 8 ####
        self.pathOnlineIndex +=1
        self.pathOnline[self.pathOnlineIndex,:] = [x, y]
        self.pathFront[self.pathOnlineIndex,:] = [scoreindex, inputindex]
        print(f"Current alignment point is {self.pathOnline[self.pathOnlineIndex,:]}")
        print(f"current path front is {self.pathFront[self.pathOnlineIndex,:]}")
        if x < scoreindex: #step 9
            return "R" #step 10
        elif y < inputindex: #step 11
            return "C" #step 12
        else: #step 13
            return "B" #step 14

    def _evaluatePathCost(self, scoreindex, inputindex):
        '''
        OnlineDTW._evaluatePathCost:
        calculates the cost difference between the current
        frames of the score and audio chromagrams, returns pathCost.
        cost is weighted so that there is no bias towards the diagonal
        (see dixon 2005)
        cost of cell is based on cost of previous cells in the vertical,
        horizonal, or diagonal direction backward, hence /dynamic/ time warping.
        '''
        print("in _evaluatePathCost")
        dist = np.linalg.norm(self.scoreChroma[:,scoreindex] -
                                self.audioChroma[:,inputindex])
        print(f"dist is {dist}")
        pathCost = min(((self.globalCostMatrix[scoreindex,inputindex-1] +
                        dist),

                       (self.globalCostMatrix[scoreindex - 1,inputindex]+
                        dist),
                        #diagonal
                       (self.globalCostMatrix[scoreindex-1,inputindex-1]+
                            (self.diagonalWeight*dist))))
        print(f"pathCost is {pathCost}")
        return pathCost

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
'''
class OSCHandler(QObject):
#see https://nerdclub-uk.blogspot.com/2016/09/
#osc-udp-tcpip-and-choosing-right.html
    def __init__(self, cues, serverPort = 53000, clientPort = 53000, serverIP = 'localhost'):

        self.cues = cues
        self.serverPort = serverPort
        self.clientPort = clientPort
        self.serverIP = serverIP

    def setupClient(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.serverIP, self.serverPort)
        sock.connect(server_address)

    @pyqtSlot(object)
    def sendOSC(self):
        try:
            for
'''

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
        self.eventTriggerOnset = []
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
                if not (chroma_nodes_per_part[parts_and_voices[i].partName] or
                                    note.isChord):
                    start = 0
                    end = note.seconds + start
                    if parts_and_voices[i].partName != "GO":
                        chroma_node = (note.name, start, end)
                        chroma_nodes_per_part[parts_and_voices[i].partName].append(chroma_node)
                    else:
                        if note.name is not "rest":
                            trigger_node = (note.lyric, start, end)
                            chroma_nodes_per_part[parts_and_voices[i].partName].append(trigger_node)
                        else:
                            chroma_node = (note.name, start, end)
                            chroma_nodes_per_part[parts_and_voices[i].partName].append(chroma_node)
                elif chroma_nodes_per_part[parts_and_voices[i].partName] and not note.isChord:
                    start = chroma_nodes_per_part[parts_and_voices[i].partName][-1][2]
                    end = note.seconds + start
                    if parts_and_voices[i].partName != "GO":
                        chroma_node = (note.name, start, end)
                        chroma_nodes_per_part[parts_and_voices[i].partName].append(chroma_node)
                    else:
                        if note.name is not "rest":
                            trigger_node = (note.lyric, start, end)
                            chroma_nodes_per_part[parts_and_voices[i].partName].append(trigger_node)
                        else:
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
        seen_triggers = []
        for part in chroma_nodes_per_part:
            print("part:", part)
            if part != "GO":
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
                        if part != "GO":
                            if node[0] in self._chromaToIndex:
                                notes_in_frame.append(self._chromaToIndex[node[0]])
                            else:
                                notes_in_frame.append("R")
                        else:
                            if node[0] != 'rest' and node[0] not in seen_triggers:
                                self.eventTriggerOnset((i, node[0]))
                                seen_triggers.append(node[0])

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
        self.initUI()
        ## non-UI stuff
        self.setupThreads()
        self.signalsandSlots()
        self.timer = QTimer()
        self.timer.timeout.connect(self.closeEvent2)
        self.timer.setSingleShot(True)
        self.timer.start(10000)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    def setupThreads(self):
        ## score chroma calculations
        # for testing ###
        file = "/Users/hypatia/Twinkle_Twinkle_Little_Star.musicxml"
        self.scorechroma = MusicXMLprocessor(file)
        self.scorechroma.musicXMLtoChroma()
        #################
        ## queues
        self.readQueue = queue.Queue()
        self.chromaQueue = queue.Queue()
        ## threads
        self.audioThread = QThread()
        self.dtwThread = QThread()
        self.readerThread = QThread()
        self.chromaThread = QThread()
        #creating instances of objects and moving to threads
        self.audioRecorder = AudioRecorder(self.readQueue,
                                            input_device_index = 2)
        self.audioRecorder.moveToThread(self.audioThread)
        self.reader = Reader(self.readQueue)
        self.reader.moveToThread(self.readerThread)
        self.chromatizer = Chromatizer(inputqueue = self.readQueue,
                                    outputqueue = self.chromaQueue)
        self.chromatizer.moveToThread(self.chromaThread)
        self.onlineDTW = OnlineDTW(self.scorechroma.chroma, self.chromaQueue)
        self.onlineDTW.moveToThread(self.dtwThread)
        ## starting threads
        self.audioThread.start()
        self.readerThread.start()
        self.chromaThread.start()
        self.dtwThread.start()

    def closeEvent2(self):
        self.audioRecorder.stopStream()
        self.reader.timer.stop()

    def signalsandSlots(self):
        self.reader.signalToChromatizer.connect(self.chromatizer.calculate)
        self.chromatizer.signalToOnlineDTW.connect(self.onlineDTW.align)

    def openFilesDialogue(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Load your MusicXML file"
                            "", "MusicXML Files (*.musicxml, *.mxl, *.xml)",
                            options = options)

## currently just testing but u know...will be other things soon.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = App()
    exit_code = app.exec_()
    sys.exit(exit_code)
