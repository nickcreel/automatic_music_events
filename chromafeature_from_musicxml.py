'''
Extracting Chroma Features from audio files and musicxml scores.

Extracting chroma features from scores introduces complications,
most importantly the preservation of harmonics. In order to compare
directly to audio, the chroma generated from the musicxml file must
simulate the harmonic tones that would be present were the score to be
played by an actual instrument. 

In the opposite direction, it is also possible to remove partials
and overtones from the chromagram of the audio recording. Doing so
might be simpler/less error prone than estimating the harmonics, as 
different instruments produce different fundamental and partial frequencies
based on their timbre. 

https://librosa.github.io/librosa_gallery/auto_examples/plot_chroma.html

'''
##import statements ###########################
from librosa import core, display, feature, decompose, effects
from music21 import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl

##class definitions ###########################
class Chromatizer:
    """
    class Chromatizer: accepts either a musicxml file or wav file as
    input, returns 2D array, or chroma matrix, containing pitch information
    of score/audio. Initialize with path to file you intend to convert.
    """
    
    def __init__(self, file_name):
        
        self.chroma_key = { 'C': 0,
                            'C#': 1,
                            'D': 2,
                            'D#':3,
                            'E':4,
                            'F':5,
                            'F#':6,
                            'G':7,
                            'G#':8,
                            'A':9,
                            'A#':10,
                            'B':11
                                }
        self.file = file_name
        self.chroma_nodes = []
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
        
        self.chroma_flat = []
        self.chroma_labels = ["C", "C#", "D", "D#", "E", "F","F#",
                                "G", "G#", "A", "A#", "B"]
        self.duration_in_seconds = 0
        self.fig = None
        self.ax = None
        
    def musicxmlToChroma(self):
        """
        method Chromatizer.musicxmlToChroma(self): takes musicxml 
        file as input (as specified in self.file, returns chroma matrix 
        (2D array with 12 rows,  one for each chroma). If a non musicxml
        file is sent, leaves fxn and asks user if they meant to convert an 
        audio file. You DO NOT have to provide any arguments,
        as file path is provided when a Chromatizer object is instantiated. 
        TODO: need to make sure this can handle polyphonic input... probably
        just involves recursing over each part (except for cue part)
        """
        if self.file.endswith((".musicxml", ".xml")):   
            self.score = converter.parse(self.file)
        else:
            print("The file you've loaded is not a file type accepted")
            print(" by music21.")
            print("Did you mean to call audioToChroma?\n")
            
        self.notes = self.score.recurse().notes
        length_of_score = len(self.notes)
        #########
        ### extract note and duration information from score
        #########
        for i in range(length_of_score):
                #################
                ###chroma_node:
                ###chroma_node[0] = note name as number 
                ###(see notename_to_ylabel for key)
                ###chroma_node[1] = note duration seconds 
                ###(useful for graphing within a range...)
                ###chroma_node[2] = index of node in time sequence
                ##################
            if i == 0:
                chroma_node = (self.chroma_key.get(self.notes[i].name),
                               self.notes[i].seconds,
                               0)
                self.chroma_nodes.append(chroma_node)
            else:
                chroma_node = (self.chroma_key.get(self.notes[i].name),
                               self.notes[i].seconds,
                               self.chroma_nodes[i-1][2]+
                               self.chroma_nodes[i-1][1])
                self.chroma_nodes.append(chroma_node)
        
        self.duration_in_seconds = self.chroma_nodes[-1][2]
        
        ######################
        ### convert note and duration information into chroma information
        ### same form as librosa chroma, but also flattened version
        ### for graphing
        #######################
        for j in np.arange(0, self.duration_in_seconds, ((2048/4)/22050)):
            #############################
            ### see https://labrosa.ee.columbia.edu/matlab/chroma-ansyn/ for 
            ### an explanation of this frame size. In short: 
            ### 22050: default sample rate for librosa is 22050 
            ### 2048: cfft length
            ### 2048/4: "the frame advance is always one quarter of 
            ###the fft length."
            ### we have to use this number to increment frames so that 
            ###the musicxml chroma is the same length, and has the same
            ###number of frames, as the librosa chromagram. 
            ###############################
            for k in range(len(self.chroma)):
                ###
                # k == note
                ###
                for l in range(0, len(self.chroma_nodes)+1):
                    ###
                    # l: we want to hit every note of course...so we 
                    # index over whole score
                    ###
                    if self.chroma_nodes[l][0] == k and (
                                        self.chroma_nodes[l][2]<=j
                                         and j < (self.chroma_nodes[l][1] 
                                             + self.chroma_nodes[l][2])):
                        ###
                        # if k == note and note is in current time frame
                        ###
                        self.chroma[k].append(1)
                    elif self.chroma_nodes[l][0] != k and (
                                self.chroma_nodes[l][2]<=j and 
                                j < (self.chroma_nodes[l][1] +
                                    self.chroma_nodes[l][2])):
                        ###
                        # if k is not current note and current note in frame
                        ###
                        self.chroma[k].append(0)
                    elif self.chroma_nodes[l][2] > j:
                        ###
                        # we need this so that we only iterate over the notes 
                        # we need  for the current frame...
                        # otherwise this would be suuuuuper slow
                        # and we'd get more values
                        # than we actually want in our chroma matrix. 
                        ###
                        break
        self.chroma = np.asarray(self.chroma)
        return self.chroma
                        
    def audioToChroma(self):
        """
        method Chromatizer.audioToChroma: accepts .wav file as input,
        returns chroma matrix for said file. basically just a wrapper
        for some librosa functions. 

        see https://librosa.github.io/librosa_gallery/auto_examples/plot_chroma.html
        for information about filtering/smoothing functions. 
        """
        audio = core.load(self.file)
        audio_harmonic = effects.harmonic(y = audio[0], 
                                             margin = 8)
        chroma = feature.chroma_cqt(audio_harmonic, 
                                        audio[1], 
                                        bins_per_octave=12*3)
        chroma_filtered = np.minimum(chroma,
                                        decompose.nn_filter(chroma,
                                        aggregate=np.median,
                                        metric = 'cosine'))

                                            
        chroma_smooth = scipy.ndimage.median_filter(chroma_filtered,
                                                        size=(1,9))
            
    
        self.chroma = chroma_smooth
        return self.chroma
    
    def showChroma(self):
        """
        method Chromatizer.showChroma: displays matplotlib.pyplot
        plot depending on type of file used to create chroma. For libROSA
        , this is primarily just a wrapper, but some code was written to
        handle musicxml files. 
        """
        chroma_ax = display.specshow(self.chroma, x_axis = "time",
                                         y_axis = "chroma", cmap ="viridis")
        plt.colorbar(label = "intensity")
##debugging ###############################
def test():
    audio = "/Users/hypatia/MMM/teto_twinkle.wav"
    audio_chromatizer = Chromatizer(audio)
    audio_chromatizer.audioToChroma()
    for i in range(len(audio_chromatizer.chroma)):
        print(audio_chromatizer.chroma[i][0:10])
    audio_chromatizer.showChroma()
    plt.show()
    musicxml_chromatizer = Chromatizer("/Users/hypatia/Twinkle_Twinkle_Little_Star.musicxml")
    musicxml_chromatizer.musicxmlToChroma()
    musicxml_chromatizer.showChroma()
    plt.show()
    polyphonic = "/Users/hypatia/polyphonic_twinkle.wav"
    polyphonic_chromatizer_test = Chromatizer(polyphonic)
    polyphonic_chromatizer_test.audioToChroma()
    polyphonic_chromatizer_test.showChroma()
    plt.show()
test()
