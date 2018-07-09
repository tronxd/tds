__author__ = 's5806074'

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import mean_squared_error
import numpy as np
from sklearn import mixture
from keras.layers import Cropping2D
from utilities.preprocessing import persist_object
from utilities.config_handler import get_config
import os

conf = get_config()
mode = conf['mode']
seq_input_length = conf['learning']['rnn']['seq_input_length']
seq_output_length = conf['learning']['rnn']['seq_output_length']
output_padding = conf['learning']['rnn']['output_padding']
input_padding = conf['learning']['rnn']['input_padding']
seq_pad_length = seq_input_length + seq_output_length
use_padding = seq_input_length != seq_output_length

num_clusters = conf['learning']['rnn']['num_clusters']
cov_types = conf['learning']['rnn']['cov_types']
gpus=conf["gpus"]

assert mode in ['development','production']
if mode == 'production':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


assert output_padding in ['pre', 'post']
assert input_padding in ['pre', 'post']



def predict_one_sample(sample, model):
    return np.squeeze(model.predict(np.expand_dims(sample, axis=0)))


def get_clipped_loss(seq_len=seq_output_length):
    def compute_clipped_error(y_true, y_pred):
        y_true_clipped = y_true[:, -seq_len:, :]
        y_pred_clipped = y_pred[:, -seq_len:, :]
        return K.mean(mean_squared_error(y_true_clipped, y_pred_clipped))

    return compute_clipped_error


def split_train_validation(X, Y, validation_split):
    X_val = X[-int(validation_split * len(X)):]
    Y_val = Y[-int(validation_split * len(Y)):]

    X_train = X[:-int(validation_split * len(X))]
    Y_train = Y[:-int(validation_split * len(Y))]

    return (X_train, Y_train, X_val, Y_val)


def train_model(model_obj, X_train, Y_train, X_val, Y_val ,train_params):
    # fit network
    checkpoint_path = os.path.join(model_obj.weights_path,'model_checkpoint.hdf5')
    batch_size = train_params["batch_size"]
    num_epochs = train_params["num_epochs"]
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    if model_obj.gpus <= 1:
        history = model_obj.model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size * gpus,
                            validation_data=(X_val, Y_val), shuffle=False, callbacks=[early_stop,checkpointer], verbose=2)
    else:
        history = model_obj.gpu_model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size * gpus,
                            validation_data=(X_val, Y_val), shuffle=False, callbacks=[early_stop], verbose=2)


    model_obj.save_weights()
    #plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.savefig(os.path.join(model_obj.weights_path , 'train_plot.png'))


def get_batch(data, batch_size):
    l = len(data)
    for ndx in range(0, l, batch_size):
        yield data[ndx:min(ndx + batch_size, l)]


def compute_rnn_batch_error(y_true, y_pred):
    if use_padding:
        y_true = y_true[:, -seq_output_length:, :]
        y_pred = y_pred[:, -seq_output_length:, :]
    return np.square(y_true - y_pred)

def compute_ae_batch_error(x,pred_x):
    x = np.squeeze(x,axis=-1)
    pred_x = np.squeeze(pred_x,axis=-1)
    return np.mean(np.mean(np.square(x - pred_x),-1),1)


def predict_rnn_error_vectors(X, Y, model_obj, batch_size):
    model = model_obj.model
    if use_padding:
        errors = np.empty((Y.shape[0], seq_output_length, Y.shape[2]))
    else:
        errors = np.empty_like(Y)

    i = 0
    for (batch_X, batch_Y) in zip(get_batch(X, batch_size), get_batch(Y, batch_size)):
        Y_pred = model.predict_on_batch(batch_X)
        batch_error = compute_rnn_batch_error(batch_Y, Y_pred)
        errors[i * batch_size:(i + 1) * batch_size] = batch_error
        i = i + 1
        if i % 50 == 0:
            print('Prediction batch {:d} / {:d}'.format(i, int(len(X) / (batch_size))))

    errors = reshape_errors(errors)
    return errors


def predict_ae_error_vectors(X,Y,model_obj,batch_size):
    i=0
    model = model_obj.model
    errors = np.empty((X.shape[0]))
    for (batch_X,batch_Y) in zip(get_batch(X,batch_size),get_batch(Y,batch_size)):
        Y_pred = model.predict_on_batch(batch_X)
        batch_error = compute_ae_batch_error(batch_Y,Y_pred)
        errors[i*batch_size:(i+1)*batch_size] = batch_error
        i=i+1
        if i%50 == 0:
            print('Prediction batch {:d} / {:d}'.format(i,int(len(X)/(batch_size))))
    return errors


def reshape_errors(errors):
    num_samples, seq_len, num_features = errors.shape
    errors = errors.reshape(num_samples, seq_len * num_features, order='F')
    return errors


def train_gmm(gmm_save_path,train_vectors,num_clusters,max_iter=500):
    # fit a Gaussian Mixture Model
    lowest_bic = np.infty
    bic = []
    best_component=''
    best_cv=''
    best_gmm={}
    n_components_range = range(2, num_clusters) # specifying maximum number of clusters
    cv_types = cov_types
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type,max_iter=max_iter)
            gmm.fit(train_vectors)
            bic.append(gmm.bic(train_vectors))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_component = n_components
                best_cv = cv_type
                best_gmm = gmm

    print("best n_component {}".format(best_component))
    print("best gmm type {}".format(best_cv))
    persist_object(best_gmm,gmm_save_path)
    return best_gmm

#works only on tensorflow backend , fragile code
def get_layer_height_width(layer):
    return layer._keras_shape[1:3]


#compute the #pixels to crop from the 2nd and 3rd channels of a given decode layer with respect to it's symmetrical encode layer
def compute_layer_crop(decode_layer_shape , encode_layer_shape):
    (decode_height , decode_width) = decode_layer_shape
    (encode_height , encode_width) = encode_layer_shape

    return (decode_height - encode_height , decode_width - encode_width)


def get_crop_layer(decode_layer, encode_shape):
    decode_shape = get_layer_height_width(decode_layer)
    crop_height, crop_width = compute_layer_crop(decode_shape, encode_shape)
    crop_layer = Cropping2D(cropping=((0, crop_height), (0, crop_width)))
    return crop_layer