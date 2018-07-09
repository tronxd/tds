
# coding: utf-8

# In[1]:


import sys
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import argparse
from keras.optimizers import Adam
import os
import numpy as np
from utilities.config_handler import get_config
from utilities.learning import split_train_validation, train_model, predict_ae_error_vectors
from utilities.detection import detect_reconstruction_anomalies_median,plot_spectogram_anomalies
from utilities.preprocessing import  add_noise,load_fft_test_data ,load_fft_train_data,  reshape_to_blocks, persist_val_stat, load_val_stat
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


data_dir = namespace.data_dir
train = namespace.mode == 'train'
opt = Adam(lr=lr)
loss_fn = 'mse'

assert len(data_dir) != 0
dataset_name = str.split(data_dir, '/')[-2]


rbw  = rbw_set[0]
weights_dir = "_".join((dataset_name, str(rbw)))
weights_load_path = os.path.join(namespace.weights_path, weights_dir)

error_median, error_std = load_val_stat(weights_load_path)
print((error_median, error_std))