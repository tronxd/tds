import numpy as np
import matplotlib.pyplot as plt

import os
from base_model.cepstrum_model import CepstrumModel
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


class GaussianCepstrum(CepstrumModel):
    def __init__(self):
        super(GaussianCepstrum,self).__init__(name='gaussian_cepstrum')
        self.amir_model = AmirModel(model_path=os.path.join(self.model_path,'amir'))

    # Preprocess raw data and persist scalers
    # Returns the preprocessed data
    def preprocess_train_data(self, iq_data, sample_rate):
        (time, fft_train) = self.amir_model.preprocess_train_data(iq_data,sample_rate,rbw=self.rbw)
        self.amir_model.train_data(fft_train)

        return (time,fft_train)

    # Preprocess raw data from loaded scalers
    # Returns the preprocessed data
    def preprocess_test_data(self, iq_data,sample_rate):
        (time, fft_test) = self.amir_model.preprocess_test_data(iq_data,sample_rate,rbw=self.rbw)
        # scaling spectrogram
        if use_scaling:
            fft_test = scale_test_vectors(fft_test, self.scaler_path)
        # getting spectrogram

        return (time, fft_test)

