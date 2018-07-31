import numpy as np
import matplotlib.pyplot as plt

import os
from base_model.amir_model import AmirModel
from utilities.preprocessing import  iq2fft, scale_train_vectors, whiten_train_data, get_config, reshape_to_blocks, \
    add_noise, persist_val_stat, load_object, whiten_test_data, scale_test_vectors, \
    load_val_stat, persist_object, get_basic_block_len, compute_fft_train_data
from skimage.util import view_as_blocks, view_as_windows

conf=get_config()
basic_time = conf['preprocessing']['basic_time']
lr=conf['learning']['ae']['lr']
rbw = conf['preprocessing']['cepstrum']['rbw']
use_whitening=conf['preprocessing']['use_whitening']
use_scaling = conf['preprocessing']['use_scaling']
feature_range = conf['preprocessing']['feature_range']
sigma_ae = conf['detection']['ae']['sigma']
block_shape = conf['learning']['ae']['block_shape']
train_params = conf['learning']['ae']
validation_split = conf['learning']['ae']['validation_split']
batch_size = conf['learning']['ae']['batch_size']
gpus = conf['gpus']
use_noise=conf['preprocessing']['ae']['use_noise']
loss_fn = 'mse'
cepstrum_window_size = conf['preprocessing']['cepstrum']['window_size']
basic_block_interval = conf['preprocessing']['basic_time']


class CWDedicated(AmirModel):
    def __init__(self, *args,**kwargs):
        super(CWDedicated,self).__init__(name='cw_dedicated', **kwargs)

    def predict_basic_block_score(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)

        score = np.max(np.mean(pred_matrix, axis=0))
        return score

    def get_score_methods(self):
        return {'normal': self.predict_basic_block_score}