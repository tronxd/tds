from sklearn.preprocessing import MinMaxScaler

__author__ = 's5806074'
from utilities.config_handler import get_config
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import struct
import xml.etree.ElementTree
import os
from zca.zca import ZCA
from skimage.util import view_as_blocks, view_as_windows
from scipy.fftpack import fft,fftshift
from scipy.signal import get_window
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

conf=get_config()
gpus = conf['gpus']
basic_time = conf['preprocessing']['basic_time']
use_noise=conf['preprocessing']['ae']['use_noise']
use_whitening=conf['preprocessing']['use_whitening']
series_offset=conf['preprocessing']['rnn']['series_offset']
feature_range = conf['preprocessing']['feature_range']
fft_window_name = conf['preprocessing']['ae']['window']
use_scaling = conf['preprocessing']['use_scaling']
mode = conf['mode']
assert mode in ['development','production']
if mode == 'production':
    import matplotlib
    matplotlib.use('Agg')


def persist_object(obj, path):
    (dir_name,_) = os.path.split(path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    joblib.dump(obj, path)
    print("saving to:" + path)

def load_object(path):
    return joblib.load(path)

def xdat2array(xdat_path,little_endian):
    with open(xdat_path,'rb') as f:
        b = f.read()
        num_shorts = int(len(b) / 2)
        if little_endian:
            xdat = struct.unpack("<{}h".format(num_shorts),b)
        else:
            xdat = struct.unpack(">{}h".format(num_shorts),b)
        xdat_array = np.array(xdat,dtype='float32')
        xdat_array = xdat_array.reshape((int(len(xdat_array)/2),2))
        return xdat_array

#calculates the overall number of samples (shorts) of all xdat files in a given directory (for efficient allocation)
def calc_num_xdat_samples(data_dir):
    num_samples = 0
    xdat_data_files =[os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.xdat')]
    for file in xdat_data_files:
        num_samples = num_samples + int(os.path.getsize(file)/2) #divide by 2 because short = 2 bytes
    return num_samples

def get_xhdr_scale_factor(xhdr_path):
    e=xml.etree.ElementTree.parse(xhdr_path)
    return float(e.find('captures').find('capture').get('acq_scale_factor'))

def get_xhdr_center_freq(data_dir):
    xhdr_files = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.xhdr')]
    xhdr_path = xhdr_files[0]
    e=xml.etree.ElementTree.parse(xhdr_path)
    return int(e.find('captures').find('capture').get('center_frequency'))

#assuming sampling rate is the same in consecutive xdat recordings
def get_xhdr_sample_rate(data_dir):
    xhdr_files = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.xhdr')]
    xhdr_path = xhdr_files[0]
    e=xml.etree.ElementTree.parse(xhdr_path)
    return int(e.find('captures').find('capture').get('sample_rate'))


def get_xhdr_little_endian(xhdr_path):
    e=xml.etree.ElementTree.parse(xhdr_path)
    return bool(e.find('data_files').find('data').get('little_endian'))



# In[5]:

def load_csv_data(data_files,feature_names):
        df = pd.Series()
        for file in data_files:
            part_df = pd.read_csv(file,names=feature_names)
            df = df.append(part_df,ignore_index=True)
        df = df.dropna(axis=1)
        data = df.values
        return data


def load_xdat_data(data_files,num_samples):
    xdat = np.zeros((int(num_samples/2) , 2),dtype='float32')
    pos = 0
    for file in data_files:
        xhdr_path = "".join((os.path.splitext(file)[0],'.xhdr'))
        scale_factor = get_xhdr_scale_factor(xhdr_path)
        little_endian = get_xhdr_little_endian(xhdr_path)
        xdat_part = xdat2array(file,little_endian)
        xdat_part = scale_factor * 2**16 * xdat_part
        curr = pos + len(xdat_part)
        xdat[pos : curr,:] = xdat_part
        pos = curr
    return xdat

#Return data,data_type
def load_raw_data(data_dir):
    csv_data_files = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    xdat_data_files =[os.path.join(data_dir,file) for file in os.listdir(data_dir) if file.endswith('.xdat')]
    data = []
    if len(csv_data_files) > 0:
        data = load_csv_data(csv_data_files)
    elif len(xdat_data_files) > 0:
        num_samples = calc_num_xdat_samples(data_dir)
        data = load_xdat_data(xdat_data_files,num_samples)
    return data

def get_basic_block_len(sample_rate, delta_t=None):
    if not delta_t:
        delta_t = basic_time
    return int(delta_t*sample_rate)

def trim_iq_basic_block(iq_data, sample_rate, start=0):
    basic_len = get_basic_block_len(sample_rate, basic_time)
    if iq_data.shape[0] > basic_len:
        # print('iq_data too long... shortening to basic block')
        return iq_data[start:start+basic_len, :]
    elif iq_data.shape[0] < basic_len:
        raise('not enough data! iq_data is too short...')
    else:
        return iq_data


def trim_by_slice_length(data, slice_length):
    num_samples = data.shape[0]
    trim_num_samples = num_samples - num_samples % slice_length
    #trim to the data to fit sequence length
    trim_data = data[:trim_num_samples]
    return trim_data


def trim_by_block_shape(data, block_shape):
    data_height = data.shape[0]
    data_width = data.shape[1]

    (block_height, block_width) = block_shape
    trim_height = data_height - data_height % block_height
    trim_width = data_width - data_width % block_width
    # trim to the data to fit block shape
    trim_data = data[:trim_height, :trim_width]
    return trim_data



#assuming data is trimmed to block_shaoe (data.shape[0] % block_height == 0 & data.shape[1] % block_width == 0 )
def reshape_to_blocks(data,block_shape):
    overlap_window = 0.5
    step = (int(block_shape[0]*overlap_window), int(block_shape[1]*overlap_window))
    blocks = view_as_windows(data, block_shape, step)
    axis_0 = np.expand_dims(np.arange(0, step[0] * blocks.shape[0], step[0]), axis=-1)
    axis_1 = np.expand_dims(np.arange(0, step[1] * blocks.shape[1], step[1]), axis=0)
    indices = np.stack([np.tile(axis_0, (1, blocks.shape[1])), np.tile(axis_1, (blocks.shape[0], 1))], axis=-1)
    #flatten the two first dimension
    blocks = blocks.reshape(-1 , blocks.shape[2], blocks.shape[3], 1)
    return indices, blocks #expand dims to fit keras format


def samples2complex(samples):
    return samples[:,0]+1j*samples[:,1]

def complex2power(complex_data):
    return 20*np.log10(1000*np.absolute(complex_data))

def complex2real(complex_data):
    return np.real(complex_data)

def complex2imag(complex_data):
    return np.imag(complex_data)

def complex2angle(complex_data):
    return np.angle(complex_data)

complex2scalar_dic = {'power': complex2power,
                      'real': complex2real,
                      'imag': complex2imag,
                      'angle': complex2angle}


def remove_dc(data):
    return data - np.mean(data)

def fft_to_matrix(fft):
    return np.stack(fft,axis=0)


def add_noise(X_train):
    noise_factor = 0.2
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    return X_train_noisy


def series_to_supervised(data, n_in, n_out,feature_names, offset=series_offset, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (feature_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (feature_names[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (feature_names[j], i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg = agg.reset_index(drop=True)
    agg = agg[agg.index % offset == 0]
    return agg


def trim_by_seq_length(data, seq_length):
    num_samples = data.shape[0]
    trim_num_samples = num_samples - num_samples % seq_length
    # trim to the data to fit sequence length
    trim_data = data[:trim_num_samples]
    return trim_data


# assuming data is trimmed to seq_length (data.shape[0] % seq_length == 0)
def reshape_to_seq(data, seq_length,feature_names):
    num_features = len(feature_names)
    assert data.shape[0] % seq_length == 0, "seq_length does not match"
    data_length = data.shape[0]
    return data.reshape((data_length, seq_length, num_features))


def get_X_and_Y_columns(data):
    X_columns = list(filter(lambda x: '-' in x, data.columns))
    Y_columns = list(filter(lambda x: '-' not in x, data.columns))

    X_shifted = data[X_columns].values
    Y_shifted = data[Y_columns].values
    return (X_shifted, Y_shifted)


def whiten_train_data(data,zca_path):
    trf = ZCA().fit(data)
    whitened_data = trf.transform(data)
    persist_object(trf, zca_path)
    return whitened_data


def whiten_test_data(data,zca_path):
    trf=load_object(zca_path)
    whitened_data=trf.transform(data)
    return whitened_data


def iq2fft_manually(data,sample_rate,rbw):
    data = samples2complex(data)
    # data = remove_dc(data)

    num_samples = len(data)
    acq_time = 1/rbw
    slice_size = int(sample_rate * acq_time)
    num_slices = int(num_samples / slice_size)
    data = trim_by_slice_length(data, slice_size)
    data_split = np.array_split(data, num_slices)
    window = get_window(fft_window_name, slice_size)
    fft_data = [complex2power(fftshift(fft(window*part))) for part in data_split]
    fft_data = fft_to_matrix(fft_data)
    return fft_data

def iq2fft(data,sample_rate,rbw, mode='power'):
    data = samples2complex(data)
    data = remove_dc(data)

    acq_time = 1/rbw
    slice_size = int(sample_rate * acq_time)
    data = trim_by_slice_length(data, slice_size)
    window = get_window(fft_window_name, slice_size)

    freqs, time, fft_d = spectrogram(data, fs=sample_rate, window=window, return_onesided=False, nperseg=slice_size,
                                         noverlap=3*slice_size//4, mode='complex')
    if type(mode) == list:
        fft_scalar = []
        for m in mode:
            complex2scalar = complex2scalar_dic[m]
            fft_scalar.append(complex2scalar(fftshift(fft_d.T)))
    else:
        complex2scalar = complex2scalar_dic[mode]
        fft_scalar = complex2scalar(fftshift(fft_d.T))

    mid_freq_ind = int(np.ceil(len(freqs) / 2.0))
    freqs = np.concatenate([freqs[mid_freq_ind:], freqs[:mid_freq_ind]])
    return freqs, time, fft_scalar

# Assume X is a 4D block tensor of the spectogram
def stitch_blocks_to_spectogram(X):
    X = np.squeeze(X)
    block_height , block_width = X.shape[1:3]
    num_blocks = X.shape[0]
    orig_height = X.shape[1]**2
    orig_width = (X.shape[2]**2) # + block_width
    stitched_image = np.zeros((orig_height,orig_width))

    i = 0
    for nh in range(0,orig_height,block_height):
        for nw in range(0 , orig_width , block_width):
            block=X[i,:,:]
            x_cor = nh
            y_cor = nw
            stitched_image[x_cor:x_cor+block_height , y_cor:y_cor + block_width] = block
            i=i+1
    return stitched_image


def load_iq_test_data(test_data_dir,weights_dir):
    scaler_path = os.path.join(weights_dir,"train_scaler.pkl")
    whiten_path = os.path.join(weights_dir ,"zca_scaler.pkl")
    test_data = load_raw_data(test_data_dir)
    if use_whitening:
        trf = load_object(whiten_path)
        test_data = trf.transform(test_data)

    test_data = scale_test_vectors(test_data, scaler_path)
    return test_data


def load_iq_train_data(train_data_dir , weights_dir):
    scaler_path = os.path.join(weights_dir,"train_scaler.pkl")
    whiten_path = os.path.join(weights_dir ,"zca_scaler.pkl")
    train_data = load_raw_data(train_data_dir)
    if use_whitening:
        train_data = whiten_train_data(train_data,whiten_path)
    #Transform data to [-1 - 1] range
    (train_data, _) = scale_train_vectors(train_data, scaler_path,rng=feature_range)
    return train_data


def persist_val_stat(val_errors, weights_dir):
    val_errors_path = os.path.join(weights_dir, "val_errors.pkl")
    persist_object(val_errors, val_errors_path)
    val_median_std_path = os.path.join(weights_dir, "val_median_std.pkl")
    error_median = np.median(val_errors)
    error_std = np.std(val_errors)
    val_dic = {'median': error_median, 'std': error_std}
    persist_object(val_dic, val_median_std_path)

def load_val_stat(weights_dir):
    val_median_std_path = os.path.join(weights_dir, "val_median_std.pkl")
    val_dic = load_object(val_median_std_path)
    return val_dic['median'], val_dic['std']



def load_fft_train_data(train_data_dir , rbw,weights_dir):
    iq_data = load_raw_data(train_data_dir)
    sample_rate = get_xhdr_sample_rate(train_data_dir)

    fft_train = compute_fft_train_data(iq_data, sample_rate, rbw, weights_dir)
    return fft_train


def compute_fft_train_data(iq_data, sample_rate, rbw, weights_path):
    scaler_path = os.path.join(weights_path, "train_scaler.pkl")
    whiten_path = os.path.join(weights_path, "zca_scaler.pkl")

    if use_whitening:
        iq_data = whiten_train_data(iq_data,whiten_path)

    freqs, time, fft_train = iq2fft(iq_data,sample_rate,rbw)
    if use_scaling:
        (fft_train, _) = scale_train_vectors(fft_train, scaler_path,rng=feature_range)

    return freqs, time, fft_train


def load_fft_test_data(test_data_dir , rbw,weights_dir):
    sample_rate = get_xhdr_sample_rate(test_data_dir)
    iq_data = load_raw_data(test_data_dir)
    freqs, time, fft_test = compute_fft_test_data(iq_data , sample_rate , rbw , weights_dir)

    return (freqs, time, fft_test)



def compute_fft_test_data(iq_data , sample_rate , rbw ,weights_dir):
    scaler_path = os.path.join(weights_dir, "train_scaler.pkl")
    whiten_path = os.path.join(weights_dir, "zca_scaler.pkl")

    if use_whitening:
        iq_data = whiten_test_data(iq_data,whiten_path)

    freqs, time, fft_test = iq2fft(iq_data,sample_rate,rbw)

    if use_scaling:
        scaler = load_object(scaler_path)
        fft_test = scale_test_vectors(fft_test , scaler)

    return freqs, time, fft_test




def scale_train_vectors(vectors, scaler_save_path, rng):
    vectors_shape_len = len(vectors.shape)
    scaler = MinMaxScaler(feature_range=rng)
    if vectors_shape_len == 1:
        vectors = vectors.reshape(-1, 1)
    scaled_vectors = scaler.fit_transform(vectors)
    persist_object(scaler, scaler_save_path)
    if vectors_shape_len == 1:
        scaled_vectors = np.squeeze(scaled_vectors)
    return (scaled_vectors, scaler)


def scale_test_vectors(vectors, scaler):
    vectors_shape_len = len(vectors.shape)
    if vectors_shape_len == 1:
        vectors = vectors.reshape(-1, 1)
    scaled_vectors = scaler.transform(vectors)
    if vectors_shape_len == 1:
        scaled_vectors = np.squeeze(scaled_vectors)
    return scaled_vectors