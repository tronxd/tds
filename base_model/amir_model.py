import numpy as np
import matplotlib.pyplot as plt

import os
from utilities.preprocessing import  iq2fft, scale_train_vectors, whiten_train_data, get_config, reshape_to_blocks, add_noise, persist_val_stat, load_object, whiten_test_data, scale_test_vectors, load_val_stat, persist_object, get_basic_block_len
from utilities.learning import split_train_validation, train_model, predict_ae_error_vectors
from base_deep.ae_deep_model import AeDeepModel
from keras.optimizers import Adam
from scipy.stats import multivariate_normal

conf=get_config()
basic_time = conf['preprocessing']['basic_time']
lr=conf['learning']['ae']['lr']
rbw = conf['preprocessing']['ae']['rbw']
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
opt = Adam(lr=lr)
loss_fn = 'mse'




class AmirModel(object):
    def __init__(self, model_path=None):
        self.rbw = rbw
        self.name = 'amir'
        if not model_path:
            self.model_path = os.path.join('model',self.name + '_' + str(int(self.rbw)))
        else:
            self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.loaded = False
        self.scaler = None
        self.freqs = None
        self.means = None
        self.stds = None
        self.gaussians = None

    def preprocess_train(self, iq_data, sample_rate):
        scaler_path = os.path.join(self.model_path, "train_scaler.pkl")
        params_path = os.path.join(self.model_path, "model_params.pkl")
        ## getting spectrogram
        self.freqs, time, fft_d = iq2fft(iq_data, sample_rate, self.rbw)
        ## scaling spectrogram
        if use_scaling:
            (fft_d, scaler) = scale_train_vectors(fft_d, scaler_path, rng=feature_range)
            self.scaler = scaler

        self.means = np.mean(fft_d, axis=0)
        self.stds = np.std(fft_d, axis=0)
        self.gen_gaussians()

        params_dic = {'freqs': self.freqs,
                      'means': self.means,
                      'stds': self.stds,
                      'gaussians': self.gaussians}
        persist_object(params_dic, params_path)

        self.loaded = True



    def test_model(self, iq_data, sample_rate):
        # splits iq_data to basic block
        raise NotImplementedError()

    def predict_score(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise "iq_data too long..."
        pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)

        mean_score_per_freq = np.mean(pred_matrix, axis=0)
        max_freq_score = np.max(mean_score_per_freq)
        return max_freq_score

    def plot_prediction(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise "iq_data too long..."
        _, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)
        # pred_matrix[0,0] = -10
        # pred_matrix[-1,-1] = 0
        _, time, fft_d = iq2fft(iq_data_basic_block, sample_rate, self.rbw)

        imshow_limits = [self.freqs[0], self.freqs[-1], time[0], time[-1]]
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        plt.sca(axes[0])
        plt.imshow(pred_matrix, aspect='auto', origin='lower', extent=imshow_limits)
        plt.sca(axes[1])
        plt.imshow(fft_d, aspect='auto', origin='lower', extent=imshow_limits)

    def predict_basic_block(self, iq_data_basic_block, sample_rate):
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise "iq_data too long..."
        if not self.loaded:
            self.load_model()

        ## whitening
        if use_whitening:
            iq_data_basic_block = whiten_test_data(iq_data_basic_block, self.whiten_path)

        ## getting spectrogram
        _, time, fft_d = iq2fft(iq_data_basic_block, sample_rate, self.rbw)

        ## scaling spectrogram
        if use_scaling:
            fft_d = scale_test_vectors(fft_d, self.scaler)

        num_freqs = len(self.freqs)
        ret = np.zeros(fft_d.shape)
        for i in range(num_freqs):
            ret[:,i] = -self.gaussians[i].pdf(fft_d[:,i])

        return time, ret

    def save_weights(self):
        raise NotImplementedError()

    def load_model(self):
        scaler_path = os.path.join(self.model_path, "train_scaler.pkl")
        params_path = os.path.join(self.model_path, "model_params.pkl")

        if use_scaling:
            self.scaler = load_object(scaler_path)

        params_dic = load_object(params_path)
        self.freqs = params_dic['freqs']
        self.means = params_dic['means']
        self.stds = params_dic['stds']
        self.gaussians = params_dic['gaussians']

        self.loaded = True

    def gen_gaussians(self):
        num_freqs = len(self.freqs)
        self.gaussians = []
        for i in range(num_freqs):
            self.gaussians.append(multivariate_normal(self.means[i], self.stds[0]))
        self.rbw = rbw
