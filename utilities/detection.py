__author__ = 's5806074'


import numpy as np
import os
import pyemd
from utilities.preprocessing import stitch_blocks_to_spectogram
from utilities.config_handler import get_config
import matplotlib.patches as patches
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
    anomalies_indices, = np.where(errors > threshold)
    return list(anomalies_indices)


def plot_spectogram_anomalies(X , anomalies_indices,weights_dir):
    plot_path = os.path.join(weights_dir,"spectogram_anomalies.png")
    X = np.squeeze(X,axis=-1)
    fig , ax = plt.subplots(1)
    X_spectogram = stitch_blocks_to_spectogram(X)
    orig_height , orig_width = X_spectogram.shape
    ax.imshow(X_spectogram,aspect='auto')
    (block_height,block_width) = X.shape[1:3]
    for error_ind in anomalies_indices:
        x_cord = (error_ind * block_width) % orig_width
        y_cord = int(error_ind * block_width / orig_width) * block_height
        rect = patches.Rectangle((x_cord,y_cord - block_height),block_width,block_height,facecolor='none',edgecolor='r',linewidth=0.1,fill='none')
        ax.add_patch(rect)
    plt.savefig(plot_path,dpi=2000,aspect='auto')
