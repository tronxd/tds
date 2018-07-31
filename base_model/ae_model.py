import numpy as np
import matplotlib.pyplot as plt

from base_model.base_net_model import BaseNetModel

import os
from utilities.preprocessing import  iq2fft, scale_train_vectors, whiten_train_data, get_config, reshape_to_blocks, add_noise, persist_val_stat, load_object, whiten_test_data, scale_test_vectors, load_val_stat, get_basic_block_len
from utilities.learning import split_train_validation, train_model, predict_ae_error_vectors
from base_deep.ae_deep_model import AeDeepModel
from utilities.learning import get_layer_height_width, get_crop_layer
from keras.optimizers import Adam
from keras import Input, Model, backend as K, regularizers
from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, Flatten, Dense, Reshape
from keras.utils.training_utils import multi_gpu_model

import tensorflow as tf




conf=get_config()
lr=conf['learning']['ae']['lr']
basic_time = conf['preprocessing']['basic_time']
rbw = conf['preprocessing']['ae']['rbw']
use_whitening=conf['preprocessing']['use_whitening']
use_scaling = conf['preprocessing']['ae']['use_scaling']
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




class AeModel(BaseNetModel):
    def __init__(self, *args,**kwargs):
        super(AeModel, self).__init__(train_params, gpus)
        self.rbw = rbw
        self.name = 'ae'
        if 'model_path' in kwargs:
            self.model_path = kwargs.pop('model_path')
        else:
            if 'model_root' in kwargs:
                model_root = kwargs.pop('model_root')
            else:
                model_root = 'model'
            self.model_path = os.path.join(model_root,self.name+'_' + str(int(self.rbw))+'_block_{}X{}'.format(block_shape[0], block_shape[1]))

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.loaded = False
        self.whiten_path = None
        self.scaler = None

    def preprocess_train_data(self, iq_data, sample_rate, rbw=None):
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
        return ((freqs, time), fft_d)

    def preprocess_test_data(self, iq_data, sample_rate, rbw=None):
        self.whiten_path = os.path.join(self.model_path, "zca_scaler.pkl")

        ## whitening
        if use_whitening:
            iq_data = whiten_test_data(iq_data, self.whiten_path)

        ## getting spectrogram
        freqs, time, fft_d = iq2fft(iq_data, sample_rate, self.rbw)

        ## scaling spectrogram
        if use_scaling:
            fft_d = scale_test_vectors(fft_d, scaler=self.scaler)
        return ((freqs, time), fft_d)


    def train_data(self, preprocessed_data):
        ## traing auto encoder
        fft_d = preprocessed_data
        block_indices, X_train = reshape_to_blocks(fft_d, block_shape)
        (X_train, _, X_val, _) = split_train_validation(X_train, X_train, validation_split)

        deep_model_input_shape = (block_shape[0], block_shape[1], 1)
        self.build_model(deep_model_input_shape, opt, loss_fn)
        if use_noise:
            X_train_noisy = add_noise(X_train)
            train_model(self, X_train_noisy, X_train, X_val, X_val, train_params)
        else:
            train_model(self, X_train, X_train, X_val, X_val, train_params)

        train_errors = predict_ae_error_vectors(X_train, X_train, self, batch_size)
        val_errors = predict_ae_error_vectors(X_val, X_val, self, batch_size)
        persist_val_stat(val_errors, self.model_path)
        self.loaded = True


    def test_model(self, iq_data, sample_rate):
        # splits iq_data to basic block
        raise NotImplementedError()

    def predict_basic_block_score(self, iq_data_basic_block, sample_rate):
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

        f_t, fft_d = self.preprocess_test_data(iq_data_basic_block, sample_rate)
        freqs, time = f_t

        ## predicts of auto encoder
        block_indices, data_blocks = reshape_to_blocks(fft_d, block_shape)
        data_ae_errors = predict_ae_error_vectors(data_blocks, data_blocks, self, batch_size)
        data_ae_errors = np.reshape(data_ae_errors, (block_indices.shape[0], block_indices.shape[1]))

        ## getting erros per block
        error_median, error_std = load_val_stat(self.model_path)

        ret = (data_ae_errors - error_median) / error_std
        blocks_time = time[block_indices[:,0,0]]
        blocks_freqs = freqs[block_indices[0,:,1]]
        return blocks_freqs, blocks_time, ret

    def save_model(self):
        raise NotImplementedError()

    def load_model(self):
        if use_whitening:
            self.whiten_path = os.path.join(self.model_path, "zca_scaler.pkl")
        self.scaler = load_object(os.path.join(self.model_path, 'train_scaler.pkl'))

        deep_model_input_shape = (block_shape[0], block_shape[1], 1)
        self.build_model(deep_model_input_shape, opt, loss_fn)
        self.load_weights()

        self.loaded = True

    def build_model(self,input_shape, opt=None, loss_fn=None):
        self.input_shape = input_shape
        self.net_model = AeDeepModel.get_conv_autoencoder_model(input_shape)
        if opt:
            if self.gpus <= 1:
                self.net_model.summary()
                self.net_model.compile(optimizer=opt, loss=loss_fn)
                # from keras.utils import plot_model
                # plot_model(self.model, 'model/ae/model_arch.png',show_layer_names=False, show_shapes=True)

            else:
                with tf.device("/cpu:0"):
                    self.gpu_model = multi_gpu_model(self.net_model, gpus=self.gpus)
                    self.gpu_model.compile(optimizer=opt,loss=loss_fn)
                    # from keras.utils import plot_model
                    # plot_model(self.model, 'model/ae/model_arch.png', show_layer_names=False, show_shapes=True)

    @staticmethod
    def get_conv_autoencoder_model(input_shape):
        height = input_shape[0]
        width = input_shape[1]
        encoding_dimesions = []
        kernel_shape = (height // 10) , (width // 10)
        inputs = Input(shape=input_shape,name='input')
        encoding_dimesions.append(get_layer_height_width(inputs))

        x = Conv2D(4, kernel_shape,padding='same',strides=2,name='conv1')(inputs)
        x = LeakyReLU(alpha=1e-1)(x)
        encoding_dimesions.append(get_layer_height_width(x))

        encoded = Conv2D(2, kernel_shape,padding='same',strides=2,name='conv2')(x)
        encoded = LeakyReLU(alpha=1e-1)(encoded)

        x = Conv2DTranspose(2, kernel_shape , strides=2,padding='same',name='deconv1')(encoded)
        x = LeakyReLU(alpha=1e-1)(x)
        x = get_crop_layer(x , encoding_dimesions.pop())(x)

        x = Conv2DTranspose(4, kernel_shape,strides=2,padding='same',name='deconv2')(x)
        x = LeakyReLU(alpha=1e-1)(x)
        x = get_crop_layer(x , encoding_dimesions.pop())(x)
        decoded = Conv2D(1 , kernel_shape,activation='linear',padding='same',name='conv5')(x)

        model=Model(inputs,decoded)
        return model