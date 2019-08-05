from PyQt5.QtCore import (QObject, pyqtSignal, QTimer, Qt, pyqtSlot, QThread,
                            QPointF, QRectF, QLineF, QRect)
import numpy as np
import queue
from math import sqrt

class OnlineDTW(QObject):
    signalToGUIThread = pyqtSignal(object)
    signalToOSCclient = pyqtSignal(object)
    def __init__(self, score_chroma, inputqueue, cuelist):

        QObject.__init__(self)
    #### parameters ###############################
        self.search_band_size = 10
        self.maxRunCount = 3
    ##############################################
        self.scoreChroma = score_chroma
        self.framenumscore = len(self.scoreChroma[0])
        #print("framenumscore is ", self.framenumscore)
        #print(self.framenumscore)
        self.framenumaudio = self.framenumscore * 2
        self.pathLenMax = self.framenumscore + self.framenumaudio
        self.audioChroma = np.zeros((12, self.framenumaudio))
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
        #self.pathFront = np.zeros((self.pathLenMax, 2))
        self.pathOnline = np.zeros((self.pathLenMax, 2))
    #    self.pathFront[0,:]= [1,1]
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
            #print(f'path online index is {self.pathOnlineIndex}')
            #print(f'score index (J) is {self.scoreindex}')

            if self.needNewFrame == 1 and not self.inputQueue.empty():
                inputData = self.inputQueue.get_nowait()
                #print(f'fnum is {self.fnum)
                if self.fnum == 0:
                    self.fnum = self.fnum + 1
                    #print(f"after {self.audioChroma[:,0]}")
                    self.audioChroma[:,0] = inputData #
                    #print(f"after {self.audioChroma[:,0]}")
                    diff = np.linalg.norm(self.scoreChroma[:,0] - self.audioChroma[:,0])
                    #print(diff)
                    self.localEuclideanDistance[0,0]= diff
                    self.globalPathCost[0,0] = self.localEuclideanDistance[0,0]
                else:
                    self.fnum +=1
                    self.audioChroma[:,self.inputindex] = inputData
                    np.place(self.audioChroma[:,self.inputindex], np.isnan(self.audioChroma[:,self.inputindex]), 0)
            #print(f"audio chroma is {self.audioChroma[:,self.inputindex]}")
            #print(f"score chroma is {self.scoreChroma[:,self.scoreindex]}")

            self.needNewFrame = 0
            direction = self._getInc()
            if direction != "C":
                for k in range((self.inputindex -(self.search_band_size + 1)),
                                        self.inputindex):
                    if k > 0:
                        pathCost = self._evaluatePathCost(self.scoreindex, k)
                        self.globalPathCost[self.scoreindex,k] = pathCost
                self.scoreindex += 1

            if direction != "R":
                self.needNewFrame = 1
                for k in range((self.scoreindex - (self.search_band_size + 1)),
                                    self.scoreindex):
                    if k > 0:
                        pathCost = self._evaluatePathCost(k, self.inputindex)
                        self.globalPathCost[k,self.inputindex] = pathCost
                self.inputindex += 1

            test = direction==self.previous
            #print(f"is direction == self.previous? {test}")
            if test == True:
                self.runCount += 1
            else:
                self.runCount = 1
            #print(f'self.runCount is {self.runCount}')
            if direction != "B":
                self.previous = direction
            # end loop
##get direction ##################################
##################################################
    def _getInc(self):
        '''
        _getInc: takes input index, score index as arguments and returns a
        char where:
        B = both
        C = column
        R = row
        which indicates the direction of the next alignment point
        '''

        if self.inputindex == 0 and self.scoreindex == 0:
            pass
        elif self.inputindex == 0:
            path1 = np.copy(self.globalPathCost[self.scoreindex-1, 0])
            path2 = np.copy(self.globalPathCost[0:self.scoreindex, 0])
        elif self.scoreindex == 0:
            path1 = np.copy(self.globalPathCost[0, 0:self.inputindex])
            path2 = np.copy(self.globalPathCost[0, self.inputindex-1])
        else:
            path1 = np.copy(self.globalPathCost[self.scoreindex-1, 0:self.inputindex])
            path2 = np.copy(self.globalPathCost[0:self.scoreindex, self.inputindex-1])
            path1 = path1.flatten()
            path2 = path2.flatten()


        for sidx in range(len(path1)):
            path1[sidx] = path1[sidx] / sqrt(sidx**2 + (self.scoreindex-1)**2)
            #path1[sidx] = path1[sidx] / (sidx + (self.scoreindex-1))
        for iidx in range(len(path2)):
            path2[iidx] = path2[iidx] / sqrt(iidx**2 + self.inputindex-1**2)
            #path2[iidx] = path2[iidx] / (iidx + (self.inputindex-1))

        if len(path1) > 0:
            minOfPath1 = np.min(path1)
        else:
            minOfPath1 = 0
        if len(path2) > 0:
            minOfPath2= np.min(path2)
        else:
            minOfPath2 = 0

        y = np.where(path1 == minOfPath1)[0]
        x = np.where(path2 == minOfPath2)[0]

        if minOfPath1 < minOfPath2:
            y = self.scoreindex-1
        elif minOfPath1 > minOfPath2:
            x = self.inputindex-1
        else:
            x = self.inputindex-1
            y = self.scoreindex-1
        self.pathOnlineIndex +=1
        self.pathOnline[self.pathOnlineIndex,:] = [x, y]
        #print(f"current alignment point is ({x}, {y}")

        seencues = []
        for cue in self.cuelist:
                if self.scoreindex-1 == cue[0] and cue[1] not in seencues:
                    self.signalToOSCclient.emit(cue[1])
                    seencues.append(cue[1])

        #self.pathFront[self.pathOnlineIndex,:] = [self.scoreindex-1-1, self.inputindex-1]
        # (i don't know what we need this for? but it was in bochen's code)


        self.signalToGUIThread.emit(self.pathOnline)


        if self.runCount > self.maxRunCount:
            if self.previous == "R":
                return "C"
            else:
                return "R"
        if self.inputindex < self.search_band_size:
            return "B"
        if y < self.scoreindex-1:
            return "C"
        elif x < self.inputindex-1:
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


        #print(self.scoreChroma[:,scoreindex])
        diff = np.linalg.norm(self.scoreChroma[:,scoreindex]-self.audioChroma[:,inputindex])
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
