import math
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras import Input, Model, backend as K, regularizers
from keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, Flatten, Dense, Reshape
import os
from base.base_model import BaseModel
from utilities.learning import get_layer_height_width, get_crop_layer
from utilities.config_handler import get_config

conf=get_config()

models_dir = conf['learning']['ae']['models_dir']
class AeModel(BaseModel):
    def __init__(self, train_params, weights_dir,gpus, direct=False):
        super(AeModel ,self).__init__(train_params,gpus)
        if direct:
            self.weights_path = weights_dir
        else:
            self.weights_path = os.path.join(models_dir, weights_dir)
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)

    def build_model(self,input_shape,opt=None,loss_fn=None):
        self.input_shape = input_shape
        self.model = get_conv_autoencoder_model(input_shape)
        if not (opt is None):
            if self.gpus <= 1:
                self.model.summary()
                self.model.compile(optimizer=opt,loss=loss_fn)
            else:
                with tf.device("/cpu:0"):
                    self.gpu_model = multi_gpu_model(self.model, gpus=self.gpus)
                    self.gpu_model.compile(optimizer=opt,loss=loss_fn)


def get_conv_autoencoder_model(input_shape):
    height = input_shape[0]
    width = input_shape[1]
    encoding_dimesions = []
    kernel_shape = (math.ceil(height / 32) , math.ceil(width / 32))
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


def get_conv_autoencoder_model_paper(input_shape):
    num_features=input_shape[0]
    block_length = input_shape[1]
    inputs = Input(shape=input_shape,name='input')
    conv1 = Conv2D(1, (num_features , int(block_length / 2) - 4 ), activation='linear', padding='same')(inputs)

    conv1_flat = Flatten()(conv1)
#      h1 = Dense((int(block_length / 2)), activation=K.hard_sigmoid ,
#                 activity_regularizer=regularizers.l1(10e-5) ,
#                 kernel_regularizer=regularizers.l2(0.5))(conv1_flat)

    h1=Dense(int(block_length) ,activation=K.sigmoid , activity_regularizer=regularizers.l1(0) , name='hidden1')(conv1_flat)
#     h2 = Dense(num_features * block_length , activation=K.hard_sigmoid ,
#                 activity_regularizer=regularizers.l1(10e-5) ,
#                 kernel_regularizer=regularizers.l2(0.1))(h1)

    h2=Dense(block_length*num_features , activation=K.sigmoid, activity_regularizer=regularizers.l1(0) , name='hidden2')(h1)
    h2_reshape = Reshape(input_shape)(h2)
    outputs = Conv2D(1, ((int(block_length / 2) - 4), 1), activation='linear', padding='same')(h2_reshape)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model