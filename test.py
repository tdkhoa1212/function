from utils.tools import wav_to_wavelet, wavelet_to_moving_average, stairway, stra
    
path = 'wav/apple_and_lemmon.wav'
window = 1000 # pooling window in MA function
time_window=0.5
bins = 10 # number of bins in stairway function
dis = 200 # distance between columns in stra function

def test_pieces():
    for idx, wx in enumerate(wav_to_wavelet(path, time_window)):
        # if idx == 1:
        #     break
        ma_hist = stairway(wavelet_to_moving_average(wx, window), bins)
        print(f'Shape of stair segment {idx+1}: {ma_hist.shape}')
        ma_hist_stra = stra(ma_hist, dis = dis)
        print(f'Shape of straight segment {idx+1}: {ma_hist_stra.shape}\n')
