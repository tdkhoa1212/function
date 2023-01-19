from utils.tools import wav_to_wavelet, wavelet_to_moving_average, stairway, stra
import pytest
import psutil

# Parameters --------------------------------------------------------
path = 'wav/apple_and_lemmon.wav'
window = 1000 # pooling window in MA function
time_window = 0.3
bins = 10 # number of bins in stairway function
dis = 200 # distance between columns in stra function

@pytest.mark.xfail(raises=IndexError)
def test_memory():
    past_me = psutil.virtual_memory().percent
    for idx, wx in enumerate(wav_to_wavelet(path, time_window)):
        # Test memory------------------------------------------------------
        current_me = psutil.virtual_memory().percent
        assert past_me-2 < current_me < past_me+2, f"the percentage of used RAM suddently changed, past memory: {past_me}, current memory: {current_me}"
        
        # Test parameters------------------------------------------------------
        assert len(wx.shape) == 2, "wx doesn't get 2D-shape, must have a 2D-shape"
        assert window <= int(wx.shape[1]), f"The value of window must be less than or equal number of colums in the wx matrix, but they are {window} and {int(wx.shape[1])} of window and wx's column respectively"
        assert dis <= int(wx.shape[1]/2), f"The value of dis should be less than of colums in the wx matrix twice, but they are {dis} and {int(wx.shape[1]/2)} of dis and wx's column respectively"
        assert bins <= int(wx.shape[1]/2), f"The value of bins should be less than of colums in the wx matrix twice, but they are {bins} and {int(wx.shape[1]/2)} of bins and wx's column respectively"

        # Test histogram ma------------------------------------------------------
        ma_hist = stairway(wavelet_to_moving_average(wx, window), bins) 
        assert wx.shape[0] == ma_hist.shape[0], f"The number of rows in matrices must be equated, but they are {wx.shape[0]} and {ma_hist.shape[0]} of wx and ma_hist respectively"
        print(f'Shape of stair segment {idx+1}: {ma_hist.shape}')

        # Test histogram straighted ma------------------------------------------------------
        ma_hist_stra = stra(ma_hist, dis = dis)
        assert wx.shape[0] == ma_hist_stra.shape[0], f"The number of rows in matrices must be equated, but they are {wx.shape[0]} and {ma_hist_stra.shape[0]} of wx and ma_hist_stra respectively"    
        print(f'Shape of straight segment {idx+1}: {ma_hist_stra.shape}\n')

        
