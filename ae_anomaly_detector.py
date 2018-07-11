
# coding: utf-8

# In[1]:


import sys
import argparse
from keras.optimizers import Adam
import os
import numpy as np
from utilities.config_handler import get_config
from utilities.learning import split_train_validation, train_model, predict_ae_error_vectors
from utilities.detection import detect_reconstruction_anomalies_median,plot_spectogram_anomalies, predict_folder_by_ae
from utilities.preprocessing import  add_noise,load_fft_test_data ,load_fft_train_data,  reshape_to_blocks, persist_val_stat, load_val_stat, persist_object
from base.ae_model import AeModel


# # Argument parsing



parser = argparse.ArgumentParser()
parser.prog = 'Spectrum Anomaly Detection'
parser.description = 'Use this command parser for training or testing the anomaly detector'
parser.add_argument('-m', '--mode', help='train or test mode', choices=['train', 'test'])
parser.add_argument('-d', '--data-dir', help='I/Q recording directory')
parser.add_argument('-w', '--weights-path', help='path for trained weights')


namespace = parser.parse_args(sys.argv[1:])
if not namespace.data_dir and namespace.mode == 'train':
    parser.error('the -d arg must be present when mode is train')
if not namespace.weights_path and namespace.mode == 'train':
    parser.error('the -w arg must be present when mode is train')

if not namespace.data_dir and namespace.mode == 'test':
    parser.error('the -d arg must be present when mode is test')
if not namespace.weights_path and namespace.mode == 'test':
    parser.error('the -w arg must be present when mode is test')


# # Hyper parameters

# In[5]:


conf=get_config()
gpus = conf['gpus']
lr=conf['learning']['ae']['lr']
batch_size = conf['learning']['ae']['batch_size']
validation_split = conf['learning']['ae']['validation_split']
train_params = conf['learning']['ae']
use_noise=conf['preprocessing']['ae']['use_noise']
feature_names = conf['preprocessing']['ae']['feature_names']
rbw_set = conf['preprocessing']['ae']['rbw_set']
feature_names = conf['preprocessing']['ae']['feature_names']


data_dir = namespace.data_dir
train = namespace.mode == 'train'
opt = Adam(lr=lr)
loss_fn = 'mse'

# In[6]:


assert len(data_dir) != 0
dataset_name = str.split(data_dir, '/')[-2]
if train:
    for rbw in rbw_set:
        weights_dir="_".join((dataset_name,str(rbw)))

        conv_model = AeModel(train_params, weights_dir,gpus)

        # # Loading,whitening,scaling,fft
        fft_train = load_fft_train_data(data_dir, rbw, conv_model.weights_path)

        block_shape = int(np.sqrt(fft_train.shape[0])),\
                             int(np.sqrt(fft_train.shape[1]))

        persist_object(block_shape, os.path.join(conv_model.weights_path,'block_shape.pkl'))

        block_indices, X_train = reshape_to_blocks(fft_train, block_shape)

        (X_train, _, X_val, _) = split_train_validation(X_train, X_train,validation_split)

        conv_model.build_model(X_train.shape[1:],opt,loss_fn)

        if use_noise:
            X_train_noisy = add_noise(X_train)
            train_model(conv_model,X_train_noisy,X_train , X_val, X_val,train_params)
        else:
            train_model(conv_model,X_train,X_train,X_val, X_val,train_params)

        train_errors = predict_ae_error_vectors(X_train, X_train, conv_model,batch_size)
        val_errors = predict_ae_error_vectors(X_val, X_val, conv_model, batch_size)
        persist_val_stat(val_errors, conv_model.weights_path)

else:
    for rbw in rbw_set:
        weights_dir = "_".join((dataset_name,str(rbw)))
        weights_load_path = os.path.join(namespace.weights_path,weights_dir)

        pred = predict_folder_by_ae(data_dir, weights_load_path)

        if pred:
            print('\n##############\n\n F O U N D   A N O M A L Y \n\n##############')
        else:
            print('\n##############\n\n N O   A N O M A L Y \n\n##############')
        # plot_spectogram_anomalies(X_test, anomalies_indices, freqs, time, conv_model.weights_path)


def compute_batch_error(x,pred_x):
    x = np.squeeze(x)
    pred_x = np.squeeze(pred_x)
    return np.mean(np.mean(np.square(x - pred_x),-1),1)