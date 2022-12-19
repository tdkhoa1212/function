from utils.tools import stairway, wavelet_to_moving_average, wav_to_wavelet, stra, plot_3D
import sys


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python moving_average.py <path to wav file>')
        sys.exit(-1)

    wav_file = sys.argv[1] # direction of .wav file
    window = 1000 # pooling window in MA function
    bins = 10 # number of bins in stairway function
    low_row = 150 # get lower limit row
    up_row = 155 # get upper limit row
    dis = 1000 # distance between columns in stra function

    # computing part -------------------------------------------------------
    ma_hist = stairway(wavelet_to_moving_average(wav_to_wavelet(wav_file), window), bins)
    ma_hist_stra = stra(ma_hist, dis = dis)

    # Plot part ---------------------
    # color: 'lightsteelblue', 'g'

    plot_3D(ma_hist, c='lightsteelblue', title='before being straightened', low_row=low_row, up_row=up_row, saved_name='results/before.png')
    plot_3D(ma_hist_stra, c='g', title=f'after being straightened - dis={dis}', low_row=low_row, up_row=up_row, saved_name='results/after.png')