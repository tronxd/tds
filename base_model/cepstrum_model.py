import numpy as np
import matplotlib.pyplot as plt

import os
from base_model.base_model_class import BaseModel
from utilities.preprocessing import  iq2fft, scale_train_vectors, whiten_train_data, get_config, reshape_to_blocks, add_noise, persist_val_stat, load_object, whiten_test_data, scale_test_vectors, load_val_stat, persist_object, get_basic_block_len, compute_fft_train_data
from skimage.util import view_as_blocks, view_as_windows
from scipy.signal import welch


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




class  CepstrumModel(BaseModel):
    def __init__(self,*args,**kwargs):
        self.rbw = rbw
        if 'name' in kwargs:
            self.name = kwargs.pop('name')
        else:
            self.name = 'cepstrum'

        if 'model_path' in kwargs:
            self.model_path = kwargs.pop('model_path')
        else:
            self.model_path = os.path.join('model',self.name + '_' + str(int(self.rbw)))

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.loaded = False
        self.cepstrum_max = None
        self.cepstrum_means = None
        self.scaler = None

    def preprocess_train(self, iq_data, sample_rate):
        max_path = os.path.join(self.model_path, "cepstrum_max.pkl")
        means_path = os.path.join(self.model_path, "cepstrum_train_means.pkl")
        scaler_path = os.path.join(self.model_path, "train_scaler.pkl")

        ## getting spectrogram
        freqs, time, fft_train = iq2fft(iq_data, sample_rate, self.rbw)
        ## scaling spectrogram
        if use_scaling:
            (fft_train, scaler) = scale_train_vectors(fft_train, scaler_path, rng=feature_range)
            self.scaler = scaler

        cepstrum_train = np.abs(np.apply_along_axis(compute_welch_spectrum, 0, fft_train))
        cepstrum_train = cepstrum_train[50:]  # removing the zero frequency
        cepstrum_train_means_over_time = np.mean(cepstrum_train, axis=1)
        self.cepstrum_means = cepstrum_train_means_over_time
        self.cepstrum_max = np.max(self.cepstrum_means)
        persist_object(self.cepstrum_max, max_path)
        persist_object(self.cepstrum_means, means_path)

        self.loaded = True



    def test_model(self, iq_data, sample_rate):
        # splits iq_data to basic block
        raise NotImplementedError()

    def predict_basic_block_score(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_means = self.predict_basic_block(iq_data_basic_block, sample_rate) / self.cepstrum_max

        score = np.percentile(pred_means, 95)
        return score

    def plot_prediction(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_means = self.predict_basic_block(iq_data_basic_block, sample_rate)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 15))
        ax1.axhline(self.cepstrum_max)
        ax1.plot(pred_means)
        ax1.set_title('Test cepstrum', fontsize=30)
        ax2.axhline(self.cepstrum_max)
        ax2.plot(self.cepstrum_means)
        ax2.set_title('Train cepstrum', fontsize=30)

    def predict_basic_block(self, iq_data_basic_block, sample_rate):
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        if not self.loaded:
            self.load_model()

        ## getting spectrogram
        _, time, fft_test = iq2fft(iq_data_basic_block, sample_rate, self.rbw)

        ## scaling spectrogram
        if use_scaling:
            fft_test = scale_test_vectors(fft_test, self.scaler)

        cepstrum_test = np.abs(np.apply_along_axis(compute_welch_spectrum, 0, fft_test))
        cepstrum_test = cepstrum_test[50:]  # removing the zero frequency
        cepstrum_test_means_over_time = np.mean(cepstrum_test, axis=1)

        return cepstrum_test_means_over_time

    def save_model(self):
        raise NotImplementedError()

    def load_model(self):
        max_path = os.path.join(self.model_path, "cepstrum_max.pkl")
        means_path = os.path.join(self.model_path, "cepstrum_train_means.pkl")
        scaler_path = os.path.join(self.model_path, "train_scaler.pkl")

        if use_scaling:
            self.scaler = load_object(scaler_path)

        self.cepstrum_max = load_object(max_path)
        self.cepstrum_means = load_object(means_path)

        self.loaded = True

@staticmethod
def compute_welch_spectrum(freq):
    freq = freq - np.mean(freq)
    return welch(freq,nperseg=cepstrum_window_size , \
                 noverlap=3*cepstrum_window_size//4 , scaling = 'spectrum',window='hann')[1]