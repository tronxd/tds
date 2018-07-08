# coding: utf-8
import argparse
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
import os
from utilities.config_handler import get_config
from utilities.learning import  get_clipped_loss, split_train_validation, \
    train_model, predict_rnn_error_vectors, reshape_errors, train_gmm
from utilities.preprocessing import series_to_supervised, trim_by_seq_length, reshape_to_seq, \
    get_X_and_Y_columns,persist_object,load_object, load_iq_test_data, load_iq_train_data, scale_train_vectors, \
    scale_test_vectors
from utilities.detection import compute_emd_split_samples,compute_emd_distributions, detect_emd_anomalies_median
from base.rnn_model import RnnModel

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


conf=get_config()
gpus = conf['gpus']
seq_input_length = conf['learning']['rnn']['seq_input_length']
seq_output_length = conf['learning']['rnn']['seq_output_length']
output_padding = conf['learning']['rnn']['output_padding']
input_padding = conf['learning']['rnn']['input_padding']

batch_size=conf['learning']['rnn']['batch_size']
num_clusters=conf['learning']['rnn']['num_clusters']
validation_split=conf['learning']['rnn']['validation_split']
lr=conf['learning']['rnn']['lr']
feature_names_rnn = conf['preprocessing']['feature_names']
train_params = conf['learning']['rnn']

data_dir = namespace.data_dir
seq_pad_length = seq_input_length + seq_output_length
use_padding = seq_input_length != seq_output_length
train = namespace.mode == 'train'
opt = Adam(lr=lr)

assert len(data_dir) != 0
dataset_name = str.split(data_dir, '/')[-2]
weights_dir = "_".join((dataset_name,str(seq_input_length),str(seq_output_length)))
scaler_path = os.path.join(weights_dir, "train_scaler.pkl")
whiten_path = os.path.join(weights_dir, "zca_scaler.pkl")
train_errors_path = os.path.join(weights_dir,"train_errors.pkl")
error_scaler_path = os.path.join(weights_dir,"error_train_scaler.pkl")
gmm_save_path = os.path.join(weights_dir,'gmm.pkl')
train_scores_path = os.path.join(weights_dir ,'train_scores.pkl')
val_scores_path = os.path.join(weights_dir ,'val_scores.pkl')
val_errors_path = os.path.join(weights_dir, "val_errors.pkl")
val_emds_path = os.path.join(weights_dir,'val_emds.pkl')

if use_padding:
    loss_fn = get_clipped_loss()
else:
    loss_fn = 'mse'

model_obj = RnnModel(train_params,weights_dir,gpus)



def scale_error_vectors(errors,weights_dir):
    error_scaler_path = os.path.join(weights_dir,"error_train_scaler.pkl")
    scaled_errors_path = os.path.join(weights_dir,"train_errors.pkl")
    (scaled_errors, error_scaler) = scale_train_vectors(errors, error_scaler_path)
    persist_object(scaled_errors, scaled_errors_path)
    return scaled_errors , error_scaler


#loading,whitening,scaling
if train:
    train_data = load_iq_train_data(data_dir,weights_dir)
    # # Create the output sequences
    train_data = series_to_supervised(train_data, n_in=seq_input_length, n_out=seq_output_length)
    # # Trim the data to fit the sequence length (SEQ_LENGTH)
    train_data = trim_by_seq_length(train_data, seq_input_length)
    (X_train, Y_train) = get_X_and_Y_columns(train_data)

    X_train = reshape_to_seq(X_train, seq_input_length)
    Y_train = reshape_to_seq(Y_train, seq_output_length)
    # Pad input/output sequences
    if use_padding:
        X_train = pad_sequences(X_train, maxlen=seq_pad_length, dtype='float32', padding=input_padding)
        Y_train = pad_sequences(Y_train, maxlen=seq_pad_length, dtype='float32', padding=output_padding)
    input_shape = X_train.shape

    # # Model and loss definition
    model_obj.build_model(input_shape,opt,loss_fn)


    # # Model training

    weights_save_path = namespace.weights_path
    (X_train, Y_train, X_val, Y_val) = split_train_validation(X_train, Y_train,validation_split)
    train_model(model_obj, X_train, Y_train, X_val, Y_val,validation_split)

    #Predict errors
    train_errors = predict_rnn_error_vectors(X_train, Y_train, model_obj, batch_size)
    val_errors = predict_rnn_error_vectors(X_val, Y_val, model_obj)

    # # Scale errors

    (scaled_train_errors, error_scaler) = scale_error_vectors(train_errors, weights_dir)
    scaled_val_errors = error_scaler.transform(val_errors)
    persist_object(scaled_val_errors, val_errors_path)

    #GMM training
    gmm = train_gmm(gmm_save_path,scaled_train_errors,num_clusters)

    train_scores = (gmm.score_samples(scaled_train_errors))
    persist_object(train_scores, train_scores_path)

    val_scores = (gmm.score_samples(scaled_val_errors))
    persist_object(val_scores, val_scores_path)

    val_emds = compute_emd_split_samples(val_scores, train_scores)
    persist_object(val_emds, val_emds_path)

else:
    test_data = load_iq_test_data(data_dir,weights_dir)
    # # Create the output sequences
    test_data = series_to_supervised(test_data, n_in=seq_input_length, n_out=seq_output_length)
    # # Trim the data to fit the sequence length (SEQ_LENGTH)
    test_data = trim_by_seq_length(test_data, seq_input_length)
    (X_test, Y_test) = get_X_and_Y_columns(test_data)

    X_test = reshape_to_seq(X_test, seq_input_length)
    Y_test = reshape_to_seq(Y_test, seq_output_length)
    # Pad input/output sequences
    if use_padding:
        X_test = pad_sequences(X_test, maxlen=seq_pad_length, dtype='float32', padding=input_padding)
    inp_shape = X_test.shape

    model_obj.load_weights()
    test_errors = predict_rnn_error_vectors(X_test, Y_test, model_obj,batch_size)

    scaled_train_errors = load_object(train_errors_path)
    scaled_test_errors = scale_test_vectors(test_errors, error_scaler_path)

    gmm = load_object(gmm_save_path)

    test_scores = (gmm.score_samples(scaled_test_errors))
    try:
        train_scores = load_object(train_scores_path)
    except:
        raise Exception('No train scores are found, please train to obtain them')

    # ## Anomaly detection phase - EMD

    # Dataset-wise
    # for now, just return the EMD between the train and test scores
    emd_dists=compute_emd_distributions(train_scores,test_scores)
    print("Overall distributions EMD:", emd_dists)

    val_emds = load_object(val_emds_path)
    test_emds = compute_emd_split_samples(test_scores, train_scores)
    detect_emd_anomalies_median(val_emds,test_emds)
