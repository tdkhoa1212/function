import numpy as np
from scipy.io import wavfile
from ssqueezepy import ssq_cwt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys

def scaler(array, min_, max_):
    '''
    array: input
    min_, max_: range of scaling
    '''
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

def stairway(ma, bin_count, max_excursion=None):
    all_dig = []
    for ma_row in ma:
        max_excursion = ma_row.max()
        bins = np.linspace(0, max_excursion, bin_count)
        dig = (np.digitize(ma_row, bins) - 1) * (max_excursion / bin_count)
        all_dig.append(dig.tolist())
    return np.array(all_dig) 

# Plot all visualizations
def plot_row(row, ax, c, matrix=None, hist=None):
    if np.max(hist) != None:
        z = hist[row]
    else:
        z = matrix[row]
    x = np.array([row]*len(z))
    y = np.arange(len(z))
    ax.plot(x, y, z, c=c)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python moving_average.py <path to wav file> <row number>')
        sys.exit(-1)

    wav_file = sys.argv[1]
    row = int(sys.argv[2])
    window = 1000
    bins = 10

    # computing part -------------------------------------------------------
    wavelet = wav_to_wavelet(wav_file)
    ma = wavelet_to_moving_average(wavelet, window)
    ma_hist = stairway(ma, bins)

    # Plot part -------------------------------------------------------------
    fig = plt.figure()  
    ax = fig.add_subplot(projection='3d')
    for each_r in range(150, 170):
        # plot_row(each_r, ax, 'bisque', matrix=wavelet)
        plot_row(each_r, ax, 'lightsteelblue', matrix=ma)
        plot_row(each_r, ax, 'g', hist=ma_hist)

    ax.set_title(f'Row {row} - Orange=source, Blue=averaged({window}), Green=histogram')
    plt.show()