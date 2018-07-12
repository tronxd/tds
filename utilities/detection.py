__author__ = 's5806074'


import numpy as np
import os
import pyemd
from utilities.preprocessing import stitch_blocks_to_spectogram ,get_fft_by_iq,  reshape_to_blocks, load_val_stat, load_object, get_xhdr_sample_rate, load_raw_data
from utilities.config_handler import get_config
import matplotlib.patches as patches
from base_deep.ae_deep_model import AeDeepModel
from utilities.learning import predict_ae_error_vectors


import math

conf=get_config()
mode = conf['mode']
scores_sample_size = conf['detection']['rnn']['scores_sample_size']
sigma_rnn=conf['detection']['rnn']['sigma']
sigma_ae = conf['detection']['ae']['sigma']


assert mode in ['development','production']
if mode == 'production':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_emd_distributions(d1,d2,bins=50):
        return  pyemd.emd_samples(d1, d2, bins)

# compute the emd between |samples_size| size of sub arrays to the train scores
def compute_emd_split_samples(scores, train_scores, scores_sample_size=scores_sample_size):
    scores_split = np.array_split(scores, len(scores) / scores_sample_size)
    emd_scores_samples = [pyemd.emd_samples(split, train_scores, bins=50) for split in scores_split]
    return emd_scores_samples


def detect_emd_anomalies_median(val_emds,test_emds,sigma=sigma_rnn):
    val_median = np.median(val_emds)
    val_std = np.std(val_emds)
    test_median = np.median(test_emds)
    test_std = np.std(test_emds)

    threshold = val_median + val_std * sigma
    anomalies_percentage = sum(test_emds > threshold) / len(test_emds)
    print("Percent of anomaly emds: {:.2%}".format(anomalies_percentage))
    print("Test Median: {:.2}, Std.D: {:.2}".format(test_median,test_std))
    print("Val Median: {:.2}, Std.D: {:.2}  ||  val Sigma threshold: {:.2}, Threshold: {:.2}".format(val_median,val_std
                                                                                                     ,sigma,threshold))

def detect_reconstruction_anomalies_median(errors, error_median, error_std, sigma=sigma_ae):
    threshold = error_median + error_std * sigma
    return errors > threshold


def plot_spectogram_anomalies(X , anomalies_indices, freqs, time, weights_dir):
    plot_path = os.path.join(weights_dir,"spectogram_anomalies.png")
    X = np.squeeze(X,axis=-1)
    fig , ax = plt.subplots(1)
    X_spectogram = stitch_blocks_to_spectogram(X)
    orig_height , orig_width = X_spectogram.shape
    plt.sca(ax)
    # plt.imshow(X_spectogram, aspect='auto', origin='lower', extent=[freqs[0], freqs[-1], time[0], time[-1]])
    plt.imshow(X_spectogram, aspect='auto', origin='lower')
    (block_height,block_width) = X.shape[1:3]
    for error_ind in anomalies_indices:
        x_cord = (error_ind * block_width) % orig_width
        y_cord = int(error_ind * block_width / orig_width) * block_height
        rect = patches.Rectangle((x_cord,y_cord - block_height),block_width,block_height,facecolor='none',edgecolor='r',linewidth=0.1,fill='none')
        ax.add_patch(rect)
    plt.savefig(plot_path,dpi=2000,aspect='auto')

def predict_folder_by_ae(data_dir, model_weights_dir):
    sample_rate = get_xhdr_sample_rate(data_dir)
    data_iq = load_raw_data(data_dir)
    return predict_by_ae(data_iq, sample_rate, model_weights_dir)

def predict_by_ae(data_iq, sample_rate, model_weights_dir):
    gpus = conf['gpus']
    train_params = conf['learning']['ae']
    batch_size = conf['learning']['ae']['batch_size']
    rbw_set = conf['preprocessing']['ae']['rbw_set']

    found_anomaly_per_rbw = []
    for rbw in rbw_set:
        print('loading data and geting spectrogram...')
        freqs, time, data_spectro = get_fft_by_iq(data_iq, sample_rate, rbw, model_weights_dir)

        print('spliting to block and predicting AutoEncoders errors...')
        block_shape = load_object(os.path.join(model_weights_dir, 'block_shape.pkl'))
        block_indices, data_blocks = reshape_to_blocks(data_spectro, block_shape)
        conv_model = AeDeepModel(train_params, model_weights_dir, gpus, direct=True)
        conv_model.build_model(data_blocks.shape[1:])
        conv_model.load_weights()
        data_ae_errors = predict_ae_error_vectors(data_blocks, data_blocks, conv_model, batch_size)
        data_ae_errors = np.reshape(data_ae_errors, (block_indices.shape[0], block_indices.shape[1]))

        print('declaring anomaly block...')
        error_median, error_std = load_val_stat(model_weights_dir)
        anomalies_indices = detect_reconstruction_anomalies_median(data_ae_errors, error_median, error_std)

        print('voting between blocks...')
        has_anomaly = voting_anomalies(anomalies_indices, block_indices, time, freqs)

        found_anomaly_per_rbw.append(has_anomaly)
    print('finished!')

    return any(found_anomaly_per_rbw)

def voting_anomalies(anomalies_indices, block_indices, time, freqs):
    per_freq_anomaly_percent = np.sum(anomalies_indices, axis=0) / anomalies_indices.shape[0]
    print(np.sum(anomalies_indices))
    max_percent = np.max(per_freq_anomaly_percent)
    return max_percent>0.05
