from PyQt5.QtCore import (QObject, pyqtSignal, QTimer, Qt, pyqtSlot, QThread,
                            QPointF, QRectF, QLineF, QRect)
import pyaudio
import queue
import wave
import numpy as np

RATE = 44100
CHUNK = 4096

class AudioRecorder(QObject):
    '''
    AudioRecorder(QObject): thread which accepts input from specified
    audio input device (default is 0) in chunks, then pushes audio to
    queue for processing by Chromatizer thread.
    '''
    signalToChromatizer = pyqtSignal(object)
    def __init__(self, queue, wavfile=None, rate = RATE, chunk = CHUNK,
                       input_device_index = 0):
        QObject.__init__(self)
        self.rate = rate
        self.i=0

        if wavfile != None:
            self.file = wave.open(wavfile, 'r')
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
                                    output = True,
                                    #output = True,
                                    frames_per_buffer = self.chunk,
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

        data = self.file.readframes(frame_count)

        if self.file != None:
            if data != '':
                data = np.frombuffer(data, "int16")
                data_per_channel=[data[chan::self.file.getnchannels()] for chan in range(self.file.getnchannels())]
                mono = (data_per_channel[0] + data_per_channel[1])/2
                self.signalToChromatizer.emit(data)
                self.i += 1
            else:
                self.stopStream()
        else:
            data = np.frombuffer(in_data, "float32")
            self.signalToChromatizer.emit(data)
        return (data, pyaudio.paContinue)
