import numpy as np
import matplotlib.pyplot as plt

import os
from utilities.preprocessing import  iq2fft, scale_train_vectors, whiten_train_data, get_config, reshape_to_blocks, add_noise, persist_val_stat, load_object, whiten_test_data, scale_test_vectors, load_val_stat, get_basic_block_len
from utilities.learning import split_train_validation, train_model, predict_ae_error_vectors
from base_deep.ae_deep_model import AeDeepModel
from keras.optimizers import Adam


conf=get_config()
lr=conf['learning']['ae']['lr']
basic_time = conf['preprocessing']['basic_time']
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




class AeModel(object):
    def __init__(self, model_path=None):
        self.rbw = rbw
        self.name = 'ae'
        if not model_path:
            self.model_path = os.path.join('model',self.name+'_' + str(int(self.rbw)))
        else:
            self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.loaded = False
        self.whiten_path = None
        self.scaler = None
        self.conv_model = None

    def preprocess_train(self, iq_data, sample_rate):
        scaler_path = os.path.join(self.model_path, "train_scaler.pkl")
        self.whiten_path = os.path.join(self.model_path, "zca_scaler.pkl")

        ## whitening
        if use_whitening:
            iq_data = whiten_train_data(iq_data, self.whiten_path)

        ## getting spectrogram
        freqs, time, fft_d = iq2fft(iq_data, sample_rate, self.rbw)

        ## scaling spectrogram
        if use_scaling:
            (fft_d, scaler) = scale_train_vectors(fft_d, scaler_path, rng=feature_range)
            self.scaler = scaler

        ## traing auto encoder
        self.conv_model = AeDeepModel(train_params, self.model_path, gpus)
        block_indices, X_train = reshape_to_blocks(fft_d, block_shape)
        (X_train, _, X_val, _) = split_train_validation(X_train, X_train, validation_split)
        self.conv_model.build_model(X_train.shape[1:], opt, loss_fn)
        if use_noise:
            X_train_noisy = add_noise(X_train)
            train_model(self.conv_model, X_train_noisy, X_train, X_val, X_val, train_params)
        else:
            train_model(self.conv_model, X_train, X_train, X_val, X_val, train_params)

        train_errors = predict_ae_error_vectors(X_train, X_train, self.conv_model, batch_size)
        val_errors = predict_ae_error_vectors(X_val, X_val, self.conv_model, batch_size)
        persist_val_stat(val_errors, self.conv_model.weights_path)
        self.loaded = True


    def test_model(self, iq_data, sample_rate):
        # splits iq_data to basic block
        raise NotImplementedError()

    def predict_score(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_freqs, pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)

        mean_score_per_freq = np.mean(pred_matrix, axis=0)
        max_freq_score = np.max(mean_score_per_freq)
        return max_freq_score

    def plot_prediction(self, iq_data_basic_block, sample_rate):
        ## get only basic_block_len
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        pred_freqs, pred_time, pred_matrix = self.predict_basic_block(iq_data_basic_block, sample_rate)
        pred_matrix[-1,-1] = sigma_ae
        freqs, time, fft_d = iq2fft(iq_data_basic_block, sample_rate, self.rbw)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        plt.sca(axes[0])
        plt.imshow(pred_matrix, aspect='auto', origin='lower', extent=[pred_freqs[0], pred_freqs[-1], pred_time[0], pred_time[-1]])
        plt.sca(axes[1])
        plt.imshow(fft_d, aspect='auto', origin='lower', extent=[freqs[0], freqs[-1], time[0], time[-1]])


    def predict_basic_block(self, iq_data_basic_block, sample_rate):
        basic_len = get_basic_block_len(sample_rate, basic_time)
        if basic_len != iq_data_basic_block.shape[0]:
            raise("iq_data too long...")
        if not self.loaded:
            self.load_model()

        ## whitening
        if use_whitening:
            iq_data_basic_block = whiten_test_data(iq_data_basic_block, self.whiten_path)

        ## getting spectrogram
        freqs, time, fft_d = iq2fft(iq_data_basic_block, sample_rate, self.rbw)

        ## scaling spectrogram
        if use_scaling:
            fft_d = scale_test_vectors(fft_d, self.scaler)

        ## predicts of auto encoder
        block_indices, data_blocks = reshape_to_blocks(fft_d, block_shape)
        data_ae_errors = predict_ae_error_vectors(data_blocks, data_blocks, self.conv_model, batch_size)
        data_ae_errors = np.reshape(data_ae_errors, (block_indices.shape[0], block_indices.shape[1]))

        ## getting erros per block
        error_median, error_std = load_val_stat(self.model_path)

        ret = (data_ae_errors - error_median) / error_std
        blocks_time = time[block_indices[:,0,0]]
        blocks_freqs = freqs[block_indices[0,:,1]]
        return blocks_freqs, blocks_time, ret

    def save_weights(self):
        raise NotImplementedError()

    def load_model(self):
        if use_whitening:
            self.whiten_path = os.path.join(self.model_path, "zca_scaler.pkl")
        self.scaler = load_object(os.path.join(self.model_path, 'train_scaler.pkl'))
        self.conv_model = AeDeepModel(train_params, self.model_path, gpus)
        deep_model_input_shape = (block_shape[0], block_shape[1], 1)
        self.conv_model.build_model(deep_model_input_shape, opt, loss_fn)
        self.conv_model.load_weights()
        self.loaded = True