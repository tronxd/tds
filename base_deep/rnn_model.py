from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras import Input, Model, backend as K, regularizers, Sequential
from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, Flatten, Dense, Reshape, LSTM, TimeDistributed, Dropout, \
    SimpleRNN
import os
from base_deep.base_deep_model import BaseDeepModel
from utilities.config_handler import get_config

conf=get_config()
seq_input_length = conf['learning']['rnn']['seq_input_length']
seq_output_length = conf['learning']['rnn']['seq_output_length']
output_padding = conf['learning']['rnn']['output_padding']
input_padding = conf['learning']['rnn']['input_padding']
seq_pad_length = seq_input_length + seq_output_length
use_padding = seq_input_length != seq_output_length

models_dir = conf['learning']['rnn']['models_dir']
class RnnDeepModel(BaseDeepModel):
    def __init__(self, train_params, weights_dir , gpus):
        super(RnnDeepModel, self).__init__(train_params, gpus)
        self.weights_path = os.path.join(models_dir, weights_dir)
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)

    def build_model(self,input_shape,opt,loss_fn):
        self.model = get_vannila_rnn_model()
        if self.gpus <= 1:
            self.model.summary()
            self.model.compile(optimizer=opt, loss=loss_fn)
        else:
            with tf.device("/cpu:0"):
                self.gpu_model = multi_gpu_model(self.model, gpus=self.gpus)
                self.gpu_model.compile(optimizer=opt, loss=loss_fn)


def get_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(seq_pad_length,input_shape = (input_shape[1], input_shape[2]),return_sequences=True,name='lstm1'))
#     model.add(Dropout(0.5))
    model.add(LSTM(seq_pad_length,return_sequences=True,name='lstm2'))
#     model.add(Dropout(0.5))
    if use_padding:
        model.add(TimeDistributed(Dense(units=3*seq_pad_length,activation='relu'),name='dense1'))
#         model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=2*seq_pad_length,activation='linear',name='dense2'))
        model.add(Reshape((seq_pad_length,2,)))
    else:
        model.add(TimeDistributed(Dense(units=2*seq_output_length,activation='relu'),name='dense1'))
#         model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=2*seq_output_length,activation='linear',name='dense2'))
        model.add(Reshape((seq_output_length,2,)))
    return model


def get_big_model(input_shape):
    model = Sequential()
    model.add(LSTM(seq_pad_length, input_shape=(input_shape[1], input_shape[2]), return_sequences=True, name='lstm1'))
    #     model.add(Dropout(0.5))
    model.add(LSTM(4*seq_pad_length, return_sequences=True, name='lstm2'))
    model.add(LSTM(4*seq_pad_length, return_sequences=True, name='lstm3'))
    model.add(LSTM(4*seq_pad_length, return_sequences=True, name='lstm4'))
    model.add(LSTM(4*seq_pad_length, return_sequences=True, name='lstm5'))
    model.add(LSTM(4*seq_pad_length, return_sequences=True, name='lstm6'))

    #     model.add(Dropout(0.5))
    if use_padding:
        model.add(TimeDistributed(Dense(units= 4*seq_pad_length, activation='relu'), name='dense1'))
        #model.add(Dropout(0.5))
       # model.add(TimeDistributed(Dense(units= seq_pad_length, activation='relu'), name='dense2'))
       # model.add(Dropout(0.5))
       # model.add(TimeDistributed(Dense(units= seq_pad_length, activation='relu'), name='dense3'))
       # model.add(Dropout(0.5))
       # model.add(TimeDistributed(Dense(units= seq_pad_length, activation='relu'), name='dense4'))
       # model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=2 * seq_pad_length, activation='linear', name='dense5'))
        model.add(Reshape((seq_pad_length, 2,)))
    else:
        model.add(TimeDistributed(Dense(units= 4*seq_output_length, activation='relu'), name='dense1'))
        model.add(Dropout(0.5))
       # model.add(TimeDistributed(Dense(units= seq_output_length, activation='relu'), name='dense2'))
       # model.add(Dropout(0.5))
       # model.add(TimeDistributed(Dense(units= seq_output_length, activation='relu'), name='dense3'))
       # model.add(Dropout(0.5))
       # model.add(TimeDistributed(Dense(units= seq_output_length, activation='relu'), name='dense4'))
      #  model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=2 * seq_output_length, activation='linear', name='dense4'))
        model.add(Reshape((seq_output_length, 2,)))
    return model


def get_lstm_dense_model(loss_fn):
    model = Sequential()
    model.add(LSTM(seq_input_length, input_shape=(inp_shape[1], inp_shape[2]), return_sequences=True, name='lstm1'))
    model.add(Dropout(0.5))
    model.add(LSTM(seq_input_length, return_sequences=True, name='lstm2'))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(units=seq_pad_length, activation='relu'), name='dense1'))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(units=12, activation='tanh'), name='dense2'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=seq_output_length * 2, name='dense3', activation='sigmoid'))
    model.add(Reshape((seq_output_length, 2,)))
    return model


def get_vannila_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(seq_pad_length, input_shape=(input_shape[1], input_shape[2]), return_sequences=True, name='rnn1'))
    #     model.add(Dropout(0.5))
    model.add(SimpleRNN(seq_pad_length, return_sequences=True, name='rnn2'))
    #     model.add(Dropout(0.5))
    if use_padding:
        model.add(TimeDistributed(Dense(units=2 * seq_pad_length, activation='relu'), name='dense1'))
        #         model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=2 * seq_pad_length, activation='linear', name='dense2'))
        model.add(Reshape((seq_pad_length, 2,)))
    else:
        model.add(TimeDistributed(Dense(units=2 * seq_output_length, activation='relu'), name='dense1'))
        #         model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=2 * seq_output_length, activation='linear', name='dense2'))
        model.add(Reshape((seq_output_length, 2,)))
    return model