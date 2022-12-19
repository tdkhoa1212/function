import numpy as np
from scipy.io import wavfile
from ssqueezepy import ssq_cwt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestCentroid

# Plot all visualizations
def plot_row(row, ax, c, matrix=None, hist=None):
    if np.max(hist) != None:
        z = hist[row]
    else:
        z = matrix[row]
    x = np.array([row]*len(z))
    y = np.arange(len(z))
    ax.plot(x, y, z, c=c)

def plot_3D(matrix, c=None, title=None, low_row=None, up_row=None, saved_name=None):
        fig = plt.figure()  
        ax = fig.add_subplot(projection='3d')
        for each_r in range(low_row, up_row):
            plot_row(each_r, ax, c, hist=matrix)
            ax.set_title(title)
            ax.set_xlabel('Frequency index')
            ax.set_ylabel('Samples')
            ax.set_zlabel('Frequency value')
        plt.savefig(saved_name)
        plt.show()

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
    '''
    ma: Moving average matrix
    bin_count: number of bins
    max_excursion: the upper limit of all_dig output
    '''
    all_dig = []
    for ma_row in ma:
        max_excursion = ma_row.max()
        bins = np.linspace(0, max_excursion, bin_count)
        dig = (np.digitize(ma_row, bins) - 1) * (max_excursion / bin_count)
        all_dig.append(dig.tolist())
    return np.array(all_dig) 

def matrix_to_vectors(matrix):
    x = np.array(matrix)
    y = []
    for i in x:
        y.append(list(range(len(i))))
    y = np.array(y).reshape(-1, )
    return x, y

def stra(matrix, dis = 5):
    shape = matrix.shape
    x, y = matrix_to_vectors(matrix)
    x_list = np.array(np.arange(np.min(x), np.max(x), dis))
    y_list = np.array(np.arange(np.min(y), np.max(y), dis))
    x_1, y_1 = np.meshgrid(x_list, y_list)   # a grid is created by standard straighten

    x_train = np.array([x_1[:, i] for i in range(x_1.shape[1])]).reshape(-1, ).astype(np.int32)
    y_train = np.array([y_1[:, i] for i in range(y_1.shape[1])]).reshape(-1, )

    y_train = np.concatenate((y_train.reshape(-1, 1), x_train.reshape(-1, 1)), axis=-1)


    ################################### Train with ML ###################################
    # This standard data ((y_train, x_train), x_train) is trained with a Machine Learning (ML) model(NearestCentroid). 
    # After that, using the machine learning
    # model predict original data ((y, x), y)
    clf = NearestCentroid()
    clf.fit(y_train, x_train)
    x = clf.predict(np.concatenate((y.reshape(-1, 1), x.reshape(-1, 1)), axis=-1))
    return x.reshape(shape)
