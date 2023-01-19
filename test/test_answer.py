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
    for wx in wav_to_wavelet(path, time_window):
        # Test memory------------------------------------------------------
        current_me = psutil.virtual_memory().percent
        assert past_me-2 < current_me < past_me+2, f"the percentage of used RAM suddently changed, past memory: {past_me}, current memory: {current_me}"

        
