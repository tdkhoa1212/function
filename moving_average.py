import numpy as np
from scipy.io import wavfile
from ssqueezepy import ssq_cwt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys

def scaler(array, min_, max_):
    array = array.reshape((-1, 1))
    scaler = MinMaxScaler(feature_range=(min_, max_))
    s_data = scaler.fit_transform(array)
    s_data = np.squeeze(s_data)
    return s_data

# Get wavelet from .wav file
def wav_to_wavelet(path):
    '''
    path: Direction to .wav file
    '''
    sample_rate, x = wavfile.read(path) 
    twx, wx, *_ = ssq_cwt(x)  # use wx
    print(f'shape: {wx.shape}')
    return np.abs(wx)

# Moving average function
def moving_average(array, window):
    '''
    array: short array have window length
    window: predefine value
    '''
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

# Moving average for wavelet
def wavelet_to_moving_average(matrix, window):
    '''
    matrix: a set of short array
    window: predefine value
    '''
    ma = []
    for i in matrix:
        i = moving_average(i, window)
        i = np.expand_dims(i, axis=0)
        if ma == []:
            ma = i
        else:
            ma = np.concatenate((ma, i), axis=0)
    return ma

def stairway(value, max_excursion, steps, row):
    '''
    value: wavelet
    max_excursion: highest chosen value in stairway
    steps: number of bins
    '''
    window = len(value) + 1 - max_excursion
    ma = wavelet_to_moving_average(value, window)
    get = ma[row]
    bins = np.linspace(np.min(get), np.max(get), steps)

    # Compute the histogram of a dataset.
    hist, bins = np.histogram(get, bins = bins)
    # plt.stairs(hist, bins)
    return hist, bins
    

# Plot a chosen row in the matrix
def plot_row(row, ax, c, matrix=None, option=None, bins=None, hist=None):
    if np.max(matrix) != None:
        get = matrix[row]

    if option == 'hist':
        # scaling bins to horizon range of MA
        bins = np.linspace(0, matrix.shape[-1], row)

        # scaling hist by min and max value of MA
        min_, max_ = np.min(get), np.max(get)
        hist = scaler(hist, min_, max_ )

        # plot histogram
        ax.hist(bins[:-1], bins, weights=hist, color='g')
    else:
        ax.plot(get, c=c)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python moving_average.py <path to wav file> <row number>')
        sys.exit(-1)

    wav_file = sys.argv[1]
    row = int(sys.argv[2])
    window = 1000
    max_excursion = 25000
    steps = 10

    # computing part -------------------------------------------------------
    wavelet = wav_to_wavelet(wav_file)
    ma = wavelet_to_moving_average(wavelet, window)
    hist, bins = stairway(wavelet, max_excursion, steps, row)

    # Plot part -------------------------------------------------------------
    fig = plt.figure()  
    ax = fig.add_subplot()
    plot_row(row, ax, 'orange', matrix=wavelet)
    plot_row(row, ax, 'b', matrix=ma)
    plot_row(row, ax, 'g', matrix=ma, option='hist', bins=bins, hist=hist)

    ax.set_title(f'Row {row} - Orange=source, Blue=averaged({window}), Green=histogram')
    plt.show()
