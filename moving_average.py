import numpy as np
from scipy.io import wavfile
from ssqueezepy import ssq_cwt
import matplotlib.pyplot as plt

# Get wavelet from .wav file
def wav2wavelet(path):       
    sample_rate, x = wavfile.read(path) 
    Twx, Wx, *_ = ssq_cwt(x) #get Wx 
    return  Wx

# Moving average function
def MovAverage(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

# Moving average for wavelet
def wavelet2MovAverage(matrix, window):
    ma = []
    for i in matrix:
        i = MovAverage(i, window)
        i = np.expand_dims(i, axis=0)
        if ma == []:
            ma = i
        else:
            ma = np.concatenate((ma, i), axis=0)
    return ma

# Plot a chosen row in the matrix
def plotMA(matrix, row):
    get = matrix[row]
    plt.plot(get)
    plt.title('After moving average')
    plt.show()

if __name__ == '__main__':
    a = wav2wavelet('/home/ubuntu-pc/Enrico_boss/voices/wav/the_different_word/good_afternoon_Kendra_Female.wav') 
    b = wavelet2MovAverage(a, 2)
    plotMA(b, 100)