from straight_fn import str
from moving_average import stairway, wavelet_to_moving_average, wav_to_wavelet

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
    if len(sys.argv) != 2:
        print('Usage: python moving_average.py <path to wav file>')
        sys.exit(-1)

    wav_file = sys.argv[1]
    window = 1000
    bins = 10
    low_row = 150
    up_row = 170

    # computing part -------------------------------------------------------
    ma_hist = stairway(wavelet_to_moving_average(wav_to_wavelet(wav_file), window), bins)
    print(ma_hist.shape)

    # Plot part -------------------------------------------------------------
    fig = plt.figure()  
    ax = fig.add_subplot(projection='3d')
    for each_r in range(low_row, up_row):
        plot_row(each_r, ax, 'g', hist=ma_hist)

    ax.set_title(f'Row {low_row}->{up_row} - Orange=source, Blue=averaged({window}), Green=histogram', fontsize=10)
    ax.set_xlabel('Frequency index')
    ax.set_ylabel('Samples')
    ax.set_zlabel('Frequency value')
    plt.show()