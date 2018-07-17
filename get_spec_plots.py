import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from utilities.preprocessing import  get_xhdr_sample_rate, load_raw_data, get_basic_block_len, persist_object, trim_iq_basic_block, iq2fft, get_config
from utilities.plots import save_fig_pickle, load_fig_pickle, save_fig


def save_alot_spec(data_dir):
    plots_path = os.path.join(data_dir, 'samples')
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    data_name = os.path.basename(data_dir)

    sample_rate = get_xhdr_sample_rate(data_dir)
    glob_data_iq = load_raw_data(data_dir)
    freqs, _, fft_d = iq2fft(glob_data_iq, sample_rate, rbw)
    glob_means = np.mean(fft_d, axis=0)
    glob_stds = np.std(fft_d, axis=0)
    del fft_d
    gc.collect()

    basic_len = get_basic_block_len(sample_rate)
    starts = np.random.randint(0, glob_data_iq.shape[0]-basic_len, (5,))
    print('random starts as indices:')
    print(starts)
    for j in starts:
        data_iq = glob_data_iq[j:j+basic_len,:]
        _, time, fft_d = iq2fft(data_iq, sample_rate, rbw)
        means = np.mean(fft_d, axis=0)
        stds = np.std(fft_d, axis=0)
        f, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        plt.sca(axes[0])
        plt.plot(freqs, glob_means, color='black', linewidth=3)
        plt.plot(freqs, glob_means - glob_stds, color='red', linewidth=3)
        plt.plot(freqs, glob_means + glob_stds, color='red', linewidth=3)

        plt.plot(freqs, means, color='green', linewidth=1)
        plt.plot(freqs, means - stds, color='green', linewidth=1)
        plt.plot(freqs, means + stds, color='green', linewidth=1)

        plt.sca(axes[1])
        plt.imshow(fft_d, aspect='auto', origin='lower', extent=[freqs[0], freqs[-1], time[0], time[-1]])

        f.suptitle(data_name)
        f.set_size_inches(8, 6.5, forward=True)

        # plt.show()
        # input('continue? [y]')

        fig_path = os.path.join(plots_path, data_name + '_sample_' + str(j))
        save_fig(f, fig_path)
        plt.close(f)
        print('workin on file ' + data_name + ' - {}/{}'.format(i,5*num_records))

conf=get_config()
rbw = conf['preprocessing']['ae']['rbw']

normal_path = 'iq_data\\CELL\\normal'
anomal_path = 'iq_data\\CELL\\anomal'

normal_records = os.listdir(normal_path)# discarding train record
anomal_records = os.listdir(anomal_path)

num_records = len(normal_records)+len(anomal_records)

records = []
preds = []
trues = []
i = 0


for r in normal_records:
    data_dir = os.path.join(normal_path, r)
    i+=1
    save_alot_spec(data_dir)

for r in anomal_records:
    data_dir = os.path.join(anomal_path, r)
    i+=1
    save_alot_spec(data_dir)

