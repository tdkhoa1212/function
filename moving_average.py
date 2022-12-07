import numpy as np
from scipy.io import wavfile
from ssqueezepy import ssq_cwt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Get wavelet from .wav file
def wav_to_wavelet(path):
    sample_rate, x = wavfile.read(path) 
    twx, wx, *_ = ssq_cwt(x)  # use wx
    print(f'shape: {wx.shape}')
    return np.abs(wx)

# Moving average function
def moving_average(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

# Moving average for wavelet
def wavelet_to_moving_average(matrix, window):
    ma = []
    for i in matrix:
        i = moving_average(i, window)
        i = np.expand_dims(i, axis=0)
        if ma == []:
            ma = i
        else:
            ma = np.concatenate((ma, i), axis=0)
    return ma

# Plot a chosen row in the matrix
def plot_row(matrix, row, ax, c):
    get = matrix[row]
    for i in range(5):
        ax.plot(np.array([i]*len(get)), np.arange(len(get)), get, c=c)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python moving_average.py <path to wav file> <row number>')
        sys.exit(-1)

    wav_file = sys.argv[1]
    row = int(sys.argv[2])
    window = 1000

    wavelet = wav_to_wavelet(wav_file)
    ma = wavelet_to_moving_average(wavelet, window)

    # Plot wave -------------------------------------------------------------
    fig = plt.figure()  
    ax = fig.gca(projection='3d')

    plot_row(wavelet, row, ax, 'orange')
    plot_row(ma, row, ax, 'b')
    ax.set_title(f'Row {row} - Orange=source, Blue=averaged({window})')
    plt.show()
