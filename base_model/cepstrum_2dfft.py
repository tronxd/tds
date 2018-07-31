import numpy as np
import matplotlib.pyplot as plt

import os
from base_model.cepstrum_model import CepstrumModel
from base_model.amir_model import AmirModel
from utilities.preprocessing import  iq2fft, scale_train_vectors, whiten_train_data, get_config, reshape_to_blocks, \
    add_noise, persist_val_stat, load_object, whiten_test_data, scale_test_vectors, \
    load_val_stat, persist_object, get_basic_block_len, compute_fft_train_data,trim_iq_basic_block
from skimage.util import view_as_blocks, view_as_windows
from scipy.fftpack import fft2 , fftshift

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


class Cepstrum2DFFT(CepstrumModel):
    def __init__(self, *args, **kwargs):
        super(Cepstrum2DFFT,self).__init__(*args, name='cepstrum_2dfft', **kwargs)
        self.amir_model = AmirModel(model_path=os.path.join(self.model_path,'amir'))


    def preprocess_train_data(self, iq_data,sample_rate):
        (time ,fft_train) = super(Cepstrum2DFFT,self).preprocess_train_data(iq_data,sample_rate)

        freqs_mean = np.mean(fft_train, axis=0)
        freqs_std = np.std(fft_train, axis=0)

        fft_train, freqs_mean = np.broadcast_arrays(fft_train, freqs_mean)
        fft_train, freqs_std = np.broadcast_arrays(fft_train, freqs_std)
        fft_train_white = (fft_train - freqs_mean) / freqs_std

        fft_train_white = fft_train_white - np.mean(fft_train_white)

        return (time,fft_train_white)

    def preprocess_test_data(self, iq_data,sample_rate):
        (time ,fft_test) = super(Cepstrum2DFFT,self).preprocess_test_data(iq_data,sample_rate)


        freqs_mean = np.mean(fft_test, axis=0)
        freqs_std = np.std(fft_test, axis=0)

        fft_test, freqs_mean = np.broadcast_arrays(fft_test, freqs_mean)
        fft_test, freqs_std = np.broadcast_arrays(fft_test, freqs_std)
        fft_test_white = (fft_test - freqs_mean) / freqs_std

        fft_test_white = fft_test_white - np.mean(fft_test_white)

        return (time,fft_test_white)


    def train_data(self,preprocessed_data):
        preprocessed_data = preprocessed_data[:497] #TODO fix it to avarage basic blocks
        fft_train_2d = np.abs(fftshift(fft2(preprocessed_data)))
        self.cepstrum_max = np.max(fft_train_2d)
        self.save_model()
        self.loaded = True


    def predict_basic_block(self, iq_data_basic_block, sample_rate):
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise ("iq_data too long...")
        if not self.loaded:
            self.load_model()

        (time, fft_test) = self.preprocess_test_data(iq_data_basic_block, sample_rate)
        fft_test_2d = np.abs(fftshift(fft2(fft_test)))

        return np.max(fft_test_2d)

    def predict_basic_block_score(self, iq_data_basic_block, sample_rate):
        # call predict_basic_block and does voting
        # get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise ("iq_data too long...")
        cepstrum_max = self.predict_basic_block(iq_data_basic_block, sample_rate) / self.cepstrum_max
        score = cepstrum_max

        return score




    def plot_prediction(self, iq_data_basic_block,sample_rate):
        # get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_max = self.predict_basic_block(iq_data_basic_block, sample_rate)

        fig, ax1 = plt.subplots(1, figsize=(20, 15))
        ax1.axhline(self.cepstrum_max)
        ax1.plot(pred_max)
        ax1.set_title('Test cepstrum max', fontsize=30)


    def save_model(self):
        max_path = os.path.join(self.model_path, "cepstrum_max.pkl")
        persist_object(self.cepstrum_max , max_path)

    # Load model parameters
    def load_model(self):
        max_path = os.path.join(self.model_path, "cepstrum_max.pkl")
        scaler_path = os.path.join(self.model_path, "train_scaler.pkl")
        if use_scaling:
            self.scaler = load_object(scaler_path)

        self.cepstrum_max = load_object(max_path)
        self.loaded = True
