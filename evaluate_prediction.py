# coding: utf-8

# In[1]:


import sys
import os
import gc
import numpy as np
import matplotlib.pyplot as plt

from utilities.config_handler import get_config
from utilities.detection import predict_by_ae, predict_folder_by_ae
from utilities.preprocessing import persist_object, load_object, get_xhdr_sample_rate, load_raw_data, trim_iq_basic_block, get_config, get_basic_block_len, iq2fft
from base_model.ae_model import AeModel
from base_model.amir_model import AmirModel

# # Hyper parameters

def save_plots(model, data_dir, log):
    plots_path = os.path.join(model.model_path, 'eval')
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    data_name = os.path.basename(data_dir)
    data_path = os.path.join(plots_path, data_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    sample_rate = get_xhdr_sample_rate(data_dir)
    glob_data_iq = load_raw_data(data_dir)

    basic_len = get_basic_block_len(sample_rate)
    starts = np.random.randint(0, glob_data_iq.shape[0] - basic_len, (5,))
    print('random starts as indices:')
    print(starts)
    for ind,j in enumerate(starts):
        data_iq = glob_data_iq[j:j + basic_len, :]

        model.plot_prediction(data_iq, sample_rate, log)

        f = plt.gcf()
        f.suptitle('useing model "' + model.name + '" on file: ' + data_name + \
                   '\n start index = ' + str(j) +' ;   log = ' + str(log))
        if log:
            fig_path = os.path.join(data_path, data_name + '_sample_' + str(j) + '_log.png')
        else:
            fig_path = os.path.join(data_path, data_name + '_sample_' + str(ind) + '.png')


        f.savefig(fig_path)
        plt.close()
        global i
        i+=1
        print('working on file ' + data_name + ' - {}/{}'.format(i,5*num_records))


ModelClass_dic = {'ae': AeModel,
                  'amir': AmirModel}

ModelClass = ModelClass_dic['amir']
model = ModelClass()

conf = get_config()
gpus = conf['gpus']
rbw = conf['preprocessing']['ae']['rbw']
train_params = conf['learning']['ae']
batch_size = conf['learning']['ae']['batch_size']
use_noise = conf['preprocessing']['ae']['use_noise']
rbw_set = conf['preprocessing']['ae']['rbw_set']


normal_records = ['CELL_NORM_2', 'CELL_NORM_3', 'CELL_NORM_4']
anomal_records = ['CELL_CW_-20MHz_0dB', 'CELL_CW_-20MHz_10dB', 'CELL_SWP_18MHz_50us_0dB', \
          'CELL_SWP_18MHz_50us_10dB', 'CELL_SWP_18MHz_100us_0dB', 'CELL_SWP_18MHz_100us_10dB']


normal_path = 'iq_data\\CELL\\normal'
anomal_path = 'iq_data\\CELL\\anomal'

num_records = len(normal_records)+len(anomal_records)

for log in [False, True]:
    i = 0
    for r in normal_records:
        data_dir = os.path.join(normal_path, r)
        save_plots(model, data_dir, log)

    for r in anomal_records:
        data_dir = os.path.join(anomal_path, r)
        save_plots(model, data_dir, log)
