from utils.tools import plot_row, wav_to_wavelet, wavelet_to_moving_average, stairway
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print('Usage: python moving_average.py <path to wav file> <row number>')
    #     sys.exit(-1)
    if len(sys.argv) != 2:
        print('Usage: python moving_average.py <path to wav file>')
        sys.exit(-1)

    wav_file = sys.argv[1]
    # row = int(sys.argv[2])
    window = 1000
    bins = 10
    low_row = 150
    up_row = 170

    # computing part -------------------------------------------------------
    wavelet = wav_to_wavelet(wav_file)
    ma = wavelet_to_moving_average(wavelet, window)
    ma_hist = stairway(ma, bins)
    print(wavelet.shape)

    # Plot part -------------------------------------------------------------
    fig = plt.figure()  
    ax = fig.add_subplot(projection='3d')
    for each_r in range(low_row, up_row):
        # plot_row(each_r, ax, 'bisque', matrix=wavelet)
        plot_row(each_r, ax, 'lightsteelblue', matrix=ma)
        plot_row(each_r, ax, 'g', hist=ma_hist)

    ax.set_title(f'Row {low_row}->{up_row} - Orange=source, Blue=averaged({window}), Green=histogram', fontsize=10)
    ax.set_xlabel('Frequency index')
    ax.set_ylabel('Samples')
    ax.set_zlabel('Frequency value')
    plt.show()