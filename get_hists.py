import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from utilities.preprocessing import  get_xhdr_sample_rate, load_raw_data, get_basic_block_len, persist_object, trim_iq_basic_block, iq2fft, get_config


def save_alot_spec(data_dir):
    plots_path = data_dir
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    data_name = os.path.basename(data_dir)
    sample_rate = get_xhdr_sample_rate(data_dir)
    glob_data_iq = load_raw_data(data_dir)

    basic_len = get_basic_block_len(sample_rate)
    starts = np.random.randint(0, glob_data_iq.shape[0]-basic_len, (sample_per_file,))
    print('random starts as indices:')
    print(starts)
    for j in starts:
        gc.collect()
        f, axes = plt.subplots(nrows=1, ncols=len(complex2scalar_mode)+1, sharey=True)

        ## ploting spectrogram
        data_iq = glob_data_iq[j:j+basic_len,:]
        freqs, time, fft_d = iq2fft(data_iq, sample_rate, rbw)
        plt.sca(axes[0])
        plt.imshow(fft_d.T, aspect='auto', origin='lower', extent=[time[0], time[-1], freqs[0], freqs[-1]])
        plt.title('spectrogram')

        ## getting index of intrest freqs
        intrest_freqs_index = []
        for f_0 in intrest_freqs:
            intrest_freqs_index.append(np.argmin(np.abs(freqs - f_0)))

        ## ploting histograms of power
        _, _, ffts = iq2fft(glob_data_iq, sample_rate, rbw, mode=complex2scalar_mode)
        for l,fft_d in enumerate(ffts):
            plt.sca(axes[l+1])
            for freq_ind in intrest_freqs_index:
                freq_col = fft_d[:, freq_ind]
                hist, bins_edge = np.histogram(freq_col, bins=200)
                hist = (1e6/np.max(hist)) * hist
                plt.plot(bins_edge[:-1], hist+freqs[freq_ind])
                print('mode: '+complex2scalar_mode[l]+' ---  hist for freq: '+str(freqs[freq_ind]))
            plt.title('hist of: '+complex2scalar_mode[l])

        f.suptitle(data_name)
        f.set_size_inches(12, 6.5, forward=True)
        # plt.show()
        # input('continue? [y]')
        #
        plt.savefig(os.path.join(plots_path, data_name + '_hists.png'))
        plt.close(f)
        print('workin on file ' + data_name + ' - {}/{}'.format(i,sample_per_file*num_records))

conf=get_config()
rbw = conf['preprocessing']['ae']['rbw']

normal_path = 'iq_data\\CELL\\normal'
anomal_path = 'iq_data\\CELL\\anomal'

normal_records = os.listdir(normal_path)# discarding train record
anomal_records = os.listdir(anomal_path)

num_records = len(normal_records)+len(anomal_records)
sample_per_file = 1

records = []
preds = []
trues = []
i = 0

complex2scalar_mode = ['power', 'real', 'imag', 'angle']
intrest_freqs = [-30e6, -25e6, -17e6, -10e6, -3e6, 4e6, 11e6, 12e6, 13e6, 14e6, 19e6, 25e6, 30e6]

for r in normal_records:
    data_dir = os.path.join(normal_path, r)
    i+=1
    save_alot_spec(data_dir)

for r in anomal_records:
    data_dir = os.path.join(anomal_path, r)
    i+=1
    save_alot_spec(data_dir)

