import numpy as np
import matplotlib.pyplot as plt

import os
from base_model.base_model_class import BaseModel
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




class AmirModel(BaseModel):
    def __init__(self, *args,**kwargs):
        if 'rbw' in kwargs:
            self.rbw = kwargs.pop('rbw')
        else:
            self.rbw = rbw

        if 'name' in kwargs:
            self.name = kwargs.pop('name')
        else:
            self.name = 'amir'

        if 'model_path' in kwargs:
            self.model_path = kwargs.pop('model_path')
        else:
            if 'model_root' in kwargs:
                model_root = kwargs.pop('model_root')
            else:
                model_root = 'model'
            self.model_path = os.path.join(model_root, self.name + '_' + str(int(self.rbw)))


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.loaded = False
        self.scaler = None
        self.freqs = None
        self.means = None
        self.stds = None
        self.gaussians = None

    def preprocess_train_data(self, iq_data,sample_rate,rbw=None):
        scaler_path = os.path.join(self.model_path, "train_scaler.pkl")
        ## getting spectrogram
        if rbw:
            self.freqs, time, fft_d = iq2fft(iq_data, sample_rate, rbw)
        else:
            self.freqs, time, fft_d = iq2fft(iq_data, sample_rate, self.rbw)
        ## scaling spectrogram
        if use_scaling:
            (fft_d, scaler) = scale_train_vectors(fft_d, scaler_path, rng=feature_range)
            self.scaler = scaler
        return (time, fft_d)

    def preprocess_test_data(self, iq_data, sample_rate,rbw=None):
        ## getting spectrogram
        if rbw:
            _, time, fft_d = iq2fft(iq_data, sample_rate, rbw)
        else:
            _, time, fft_d = iq2fft(iq_data, sample_rate, self.rbw)

        ## scaling spectrogram
        if use_scaling:
            fft_d = scale_test_vectors(fft_d, self.scaler)
        return (time, fft_d)



    def train_data(self, preprocessed_data):
        params_path = os.path.join(self.model_path, "model_params.pkl")
        fft_d = preprocessed_data

        self.means = np.mean(fft_d, axis=0)
        self.stds = np.std(fft_d, axis=0)
        self.gen_gaussians()

        params_dic = {'freqs': self.freqs,
                      'means': self.means,
                      'stds': self.stds,
                      'gaussians': self.gaussians}
        persist_object(params_dic, params_path)

        self.loaded = True

    def predict_basic_block_score(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        return self.predict_basic_block_score_max(iq_data_basic_block, sample_rate)


    def predict_basic_block_score_max(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)

        score_per_time = np.max(pred_matrix, axis=1)
        score = np.mean(score_per_time)
        return score

    def predict_basic_block_score_mean(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)

        score = np.mean(pred_matrix)
        return score

    def predict_basic_block_score_percent(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)

        score = np.sum(pred_matrix>=3.5) / pred_matrix.size
        return score

    def predict_basic_block_score_for_CW(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)

        score = np.max(np.mean(pred_matrix, axis=0))
        return score

    def get_score_methods(self):
        dic = {'mean': self.predict_basic_block_score_mean,
               'max_per_time': self.predict_basic_block_score_max,
               'percent': self.predict_basic_block_score_percent}
        return dic

    def plot_prediction(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data not 1 [msec]...")
        _, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)
        pred_matrix[0,0] = 0
        pred_matrix[-1,-1] = 7

        _, time, fft_d = iq2fft(iq_data_basic_block, sample_rate, self.rbw)

        imshow_limits = [self.freqs[0], self.freqs[-1], time[0], time[-1]]
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        plt.sca(axes[0])
        pred_im = plt.imshow(pred_matrix, aspect='auto', origin='lower', extent=imshow_limits)
        plt.colorbar(pred_im)

        plt.sca(axes[1])
        fft_im = plt.imshow(fft_d, aspect='auto', origin='lower', extent=imshow_limits)
        plt.colorbar(fft_im)


    def predict_basic_block(self, iq_data_basic_block, sample_rate):
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        if not self.loaded:
            self.load_model()

        time, fft_d = self.preprocess_test_data(iq_data_basic_block, sample_rate)
        spectogram = np.abs(fft_d - np.expand_dims(self.means, axis=0)) / np.expand_dims(self.stds, axis=0)
        spectogram = np.clip(spectogram, 0, 7)

        return time, spectogram

    def save_model(self):
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
