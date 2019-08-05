from PyQt5.QtCore import (QObject, pyqtSignal, QTimer, Qt, pyqtSlot, QThread,
                            QPointF, QRectF, QLineF, QRect)
from scipy import ndimage
from librosa import feature, display, decompose
import queue
import numpy as np

RATE = 44100
CHUNK = 4096

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
        self.rate = RATE
    def _display(self):
        chroma = self.chroma_frames.get_nowait()
        display.specshow(chroma, y_axis = "chroma", x_axis = "time")

    @pyqtSlot(object)
    def calculate(self, frame):

        #print("calculating chroma...")
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
        else:
            chroma_smooth = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
        self.outputqueue.put_nowait(chroma_smooth)
        self.signalToOnlineDTW.emit()
