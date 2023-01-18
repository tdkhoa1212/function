from utils.tools import wav_to_wavelet, wavelet_to_moving_average, stairway, stra
import pytest
import numpy as np


# Parameter --------------------------------------------------------
@pytest.fixture
def path():
    return 'wav/apple_and_lemmon.wav'

@pytest.fixture
def window():
    return 1000 # pooling window in MA function

@pytest.fixture 
def time_window():
    return 0.5

@pytest.fixture 
def bins():
    return 10 # number of bins in stairway function

@pytest.fixture 
def dis():
    return 200 # distance between columns in stra function

# Assertions --------------------------------------------------------
def t_wx(wx, window, bins, dis):
    assert len(wx.shape) == 2, "wx doesn't get 2D-shape, must have a 2D-shape"
    assert window <= int(wx.shape[1]), f"The value of window must be less than or equal number of colums in the wx matrix, but they are {window} and {int(wx.shape[1])} of window and wx's column respectively"
    assert dis <= int(wx.shape[1]/2), f"The value of dis should be less than of colums in the wx matrix twice, but they are {dis} and {int(wx.shape[1]/2)} of dis and wx's column respectively"
    assert bins <= int(wx.shape[1]/2), f"The value of bins should be less than of colums in the wx matrix twice, but they are {bins} and {int(wx.shape[1]/2)} of bins and wx's column respectively"

def t_ma_hist(wx, ma_hist):
    assert wx.shape[0] == ma_hist.shape[0], f"The number of rows in matrices must be equated, but they are {wx.shape[0]} and {ma_hist.shape[0]} of wx and ma_hist respectively"

def t_ma_hist_stra(wx, ma_hist_stra):
    assert wx.shape[0] == ma_hist_stra.shape[0], f"The number of rows in matrices must be equated, but they are {wx.shape[0]} and {ma_hist_stra.shape[0]} of wx and ma_hist_stra respectively"    

@pytest.mark.xfail(raises=IndexError)
def test_pieces(window, time_window, bins, dis, path):
    for idx, wx in enumerate(wav_to_wavelet(path, time_window)):
        t_wx(wx, window, bins, dis)

        ma_hist = stairway(wavelet_to_moving_average(wx, window), bins) 
        t_ma_hist(wx, ma_hist)
        print(f'Shape of stair segment {idx+1}: {ma_hist.shape}')

        ma_hist_stra = stra(ma_hist, dis = dis)
        t_ma_hist_stra(wx, ma_hist_stra)
        print(f'Shape of straight segment {idx+1}: {ma_hist_stra.shape}\n')

