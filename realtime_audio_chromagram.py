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
import wave
from math import log2, sqrt
################################################
## globals #####################################
RATE = 44100
CHUNK = 4096
## threads #####################################
################################################

class AudioRecorder(QObject):
    '''
    AudioRecorder(QObject): thread which accepts input from specified
    audio input device (default is 0) in chunks, then pushes audio to
    queue for processing by Chromatizer thread.
    '''

    def __init__(self, queue, wavfile=None, rate = RATE, chunk = CHUNK,
                       input_device_index = 0):
        #rate = librosa default
        QObject.__init__(self) #getting all the qthread stuff
        self.rate = rate
        self.i=0
        if wavfile != None:
            self.file = wave.open(wavfile, 'r')
            self.filelen = self.file.getnframes()
            print(self.filelen)
        else:
            self.file = None
        self.chunk = chunk
        self.queue = queue
        self.p = pyaudio.PyAudio()
        self.input_device_index = input_device_index
        if self.file == None:
            self.stream = self.p.open(format= pyaudio.paFloat32,
                                    channels = 1,
                                    rate = self.rate,
                                    input = True,
                                    input_device_index = self.input_device_index,
                                    frames_per_buffer = self.chunk,
                                    stream_callback = self._callback)
        else:
            self.stream = self.p.open(format=self.p.get_format_from_width(self.file.getsampwidth()),
                                    channels = self.file.getnchannels(),
                                    rate = self.rate,
                                    input = True,
                                    stream_callback = self._callback)
        self.stop = False

    def stopStream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.file.close()

    def  _callback(self, in_data, frame_count, time_info, status):
        """
        grab data from buffer,
        put data and rate into queue
        continue
        """
        self.i+=1
        print(self.i)
        if self.file != None:
            data = self.file.readframes(frame_count)
            data = np.frombuffer(data, "int16")
            print(len(data))
        else:
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
    signalToOnlineDTW = pyqtSignal()
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
        #print("calculating chroma...")
        i = 0
        y = frame.astype('float32')
        sr = self.rate
        mag = np.linalg.norm(y)
        if mag > .008:
            chroma = feature.chroma_cqt(y, sr,
                                        bins_per_octave = 12*3)
        #filtering reduces volume of noise/partials
            chroma_filtered = np.minimum(chroma,
                                            decompose.nn_filter(chroma,
                                            aggregate = np.median,
                                            metric = 'cosine'))
            chroma_smooth = ndimage.median_filter(chroma_filtered,
                                                    size = (1,9))
            np.place(chroma_smooth, np.isnan(chroma_smooth), [0])
            chroma_smooth = np.mean(chroma_smooth, axis = 1)
            #print(f'chroma_smooth is {chroma_smooth}')
        else:
            chroma_smooth = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
        self.outputqueue.put_nowait(chroma_smooth)
        self.signalToOnlineDTW.emit()

class OnlineDTW(QObject):
    signalToGUIThread = pyqtSignal(object)
    def __init__(self, score_chroma, inputqueue, cuelist):

        QObject.__init__(self)
    #### parameters ###############################
        self.search_band_size = 10
        self.maxRunCount = 3
    ##############################################
        self.scoreChroma = score_chroma
        self.framenumscore = len(self.scoreChroma[0])
        #print(self.framenumscore)
        self.framenumaudio = self.framenumscore * 2
        self.pathLenMax = self.framenumscore + self.framenumaudio
        self.audioChroma = np.zeros_like(self.scoreChroma)
        self.inputQueue = inputqueue
    ###############################################
    #### distance matrices ########################
        self.globalPathCost = np.matrix(np.ones((self.framenumscore,
                                                    self.framenumaudio))
                                                        * np.inf)
        #this is a matrix of the cost of a path which terminates at point [x, y]
        #print(self.globalPathCost)
        self.localEuclideanDistance = np.matrix(np.ones((self.framenumscore,
                                                    self.framenumaudio))
                                                        * np.inf)
        #this is a matrix of the euclidean distance between frames of audio
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
        self.needNewFrame = 1
        self.runCount = 0
        self.cuelist = cuelist

    @pyqtSlot()
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
        if self.scoreindex < self.framenumscore:
            print(f'score index (J) is {self.scoreindex}')
            if self.needNewFrame == 1 and not self.inputQueue.empty():
                inputData = self.inputQueue.get_nowait()
                self.frameQueue.put_nowait(inputData)
                if self.fnum == 0:
                    self.fnum = self.fnum + 1
                    self.audioChroma[:,0] = self.frameQueue.get_nowait()
                    diff = 0
                    for i in range(len(self.scoreChroma[:,0])):
                        diff += np.linalg.norm(self.scoreChroma[:,0][i] - self.audioChroma[:,0][i])
                    self.localEuclideanDistance[0,0]= diff
                    self.globalPathCost[0,0] = self.localEuclideanDistance[0,0]
                else:
                    self.fnum +=1
                    self.audioChroma[:,self.inputindex] = self.frameQueue.get_nowait()
                    np.place(self.audioChroma[:,self.inputindex], np.isnan(self.audioChroma[:,self.inputindex]), 0)
        #    print(f"audio chroma is {self.audioChroma[:,self.inputindex]}")
        #    print(f"score chroma is {self.scoreChroma[:,self.scoreindex]}")

            self.needNewFrame = 0
            direction = self._getInc(self.scoreindex, self.inputindex,
                                        self.previous, self.runCount)
        #    print(f'direction is {direction}')
            # LOOP
        #    print(f'self.runCount is {self.runCount}')
            if direction != "C":
                for k in range((self.inputindex -(self.search_band_size + 1)),
                                        self.inputindex):
                    if k > 0:
                        pathCost = self._evaluatePathCost(self.scoreindex, k)
                        self.globalPathCost[self.scoreindex,k] = pathCost
                        #print(f'self.globalPathCost[{self.scoreindex}, {k}] is {self.globalPathCost[self.scoreindex, k]}')
                self.scoreindex += 1

            if direction != "R":
                self.needNewFrame = 1
                for k in range((self.scoreindex - (self.search_band_size + 1)),
                                    self.scoreindex):
                    if k > 0:
                        pathCost = self._evaluatePathCost(k, self.inputindex)
                        self.globalPathCost[k,self.inputindex] = pathCost
                        #print(f'self.globalPathCost[{k}, {self.inputindex}] is {self.globalPathCost[k,self.inputindex]}')
                self.inputindex += 1
            if direction == self.previous:
                self.runCount += 1
            else:
                self.runCount = 1
            if direction != "B":
                self.previous = direction
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
        if inputindex < self.search_band_size:
            return "B"
        if self.runCount > self.maxRunCount:
            if previous == "R":
                return "C"
            else:
                return "R"

        path1 = self.globalPathCost[scoreindex-1, 0:inputindex]
        path2 = self.globalPathCost[0:scoreindex, inputindex-1]

        for i in range(len(path1)):
            path1[i] = path1[i] / sqrt((scoreindex-1)**2+i**2)
            path1[i] = path1[i] / (scoreindex-1 + i)
        for i in range(len(path2)):
            path2[i] = path2[i] / sqrt((inputindex-1)**2 + i **2)
            path2[i] = path2[i] / (inputindex-1 + i)

        minOfPath1, y = np.min(path1), np.argmin(path1)
        minOfPath2, x = np.min(path2), np.argmin(path2)

        print(f'x before if is {x}')
        print(f'y before if is {y}')

        if minOfPath1 < minOfPath2:
            x = scoreindex-1
        elif minOfPath1 > minOfPath2:
            y = inputindex-1
        else:
            y = inputindex-1
            x = scoreindex-1
        print(f'x after if is {x}')
        print(f'y after if is {y}')

        self.pathOnlineIndex +=1
        self.pathOnline[self.pathOnlineIndex,:] = [x, y]
        self.pathFront[self.pathOnlineIndex,:] = [scoreindex-1, inputindex-1]
        print(f"current alignment point is ({x}, {y}")

        self.signalToGUIThread.emit(self.pathOnline)

        for cue in self.cuelist:
            if scoreindex-1 == cue[0]:
                print(f"CUE HIT: CUE NUMBER {cue[1]}" +
                f"SCORE INDEX {self.scoreindex-1}")

        if x < scoreindex-1:
            return "C"
        elif y < inputindex-1:
            return "R"
        else:
            return "B"

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

        diff = 0
        #print(self.scoreChroma[:,scoreindex])
        for i in range(len(self.scoreChroma[:,scoreindex])):
            diff += np.linalg.norm(self.scoreChroma[:,scoreindex][i]-self.audioChroma[:,inputindex][i])
        #print(f'diff is {diff}')
        self.localEuclideanDistance[scoreindex, inputindex] = diff

        pathCost = np.min(((self.globalPathCost[scoreindex,inputindex-1] +
                        diff),

                       (self.globalPathCost[scoreindex - 1,inputindex]+
                        diff),

                       (self.globalPathCost[scoreindex-1,inputindex-1]+
                            (2*diff))))
        #print(f'pathCost is {pathCost}')
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
            #print(part.partName, part.seconds)
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
        counter = 0
        seen_triggers = []
        for part in chroma_nodes_per_part:
            #print("part:", part)
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

            for i in np.arange(0, length_score_seconds, ((CHUNK)/RATE)):
                counter += 1
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
                                self.eventTriggerOnset.append((counter, node[0]))
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
                #print(notes_in_frame)
                for note in notes_in_frame:
                    if type(note) == int:
                        chroma_in_frame +=  self._harmonics[note]
                    elif note == "R":
                        pass
                if part != "GO":
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

        ## non-UI stuff
        self.setupThreads()
        self.signalsandSlots()
        self.timer = QTimer()
        self.timer.timeout.connect(self.closeEvent2)
        self.timer.setSingleShot(True)
        self.timer.start(2000000)

    def setupThreads(self):
        file = "/Users/hypatia/Twinkle_with_Cues.musicxml"
        self.scorechroma = MusicXMLprocessor(file)
        self.scorechroma.musicXMLtoChroma()
        cues = self.scorechroma.eventTriggerOnset

        self.readQueue = queue.Queue()
        self.chromaQueue = queue.Queue()
        ## threads
        self.audioThread = QThread()
        self.dtwThread = QThread()
        self.readerThread = QThread()
        self.chromaThread = QThread()

        self.audioRecorder = AudioRecorder(self.readQueue, wavfile = "/Users/hypatia/Twinkle_with_Cues.wav")
        self.audioRecorder.moveToThread(self.audioThread)
        self.reader = Reader(self.readQueue)
        self.reader.moveToThread(self.readerThread)
        self.chromatizer = Chromatizer(inputqueue = self.readQueue,
                                    outputqueue = self.chromaQueue)
        self.chromatizer.moveToThread(self.chromaThread)
        self.onlineDTW = OnlineDTW(self.scorechroma.chroma, self.chromaQueue, cues)
        self.onlineDTW.moveToThread(self.dtwThread)

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
        self.onlineDTW.signalToGUIThread.connect(self.plotter)
        #self.onlineDTW.signalToPlotter.connect(self.plotter.animate)

    @pyqtSlot(object)
    def plotter(self, line):
        init = 0
        line = line
        self.x = [line[i, 1] for i in range(len(line))]
        self.y = [line[i, 0] for i in range(len(line))]
        self.y_axis_size = len(self.scorechroma.chroma[0])
        self.x_axis_size = self.y_axis_size #we don't know the actual size
        if init == 0:
            self.plt = pg.plot(self.x, self.y, pen="b", symbol='o', title="Alignment Path")
            self.plt.showGrid(x=True, y=True)
            init = 1
        else:
            self.plt.PlotDataItem(self.x, self.y, clear = True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = App()
    exit_code = app.exec_()
    sys.exit(exit_code)
