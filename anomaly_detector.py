
# coding: utf-8

# In[1]:


import sys
import argparse
from base_model.ae_model import AeModel
from base_model.amir_model import AmirModel
from base_model.complex_gauss_model import ComplexGauss
from base_model.gaussian_cepstrum_model import GaussianCepstrum
from base_model.cepstrum_2dfft import Cepstrum2DFFT
from base_model.cepstrum_model import CepstrumModel
from utilities.config_handler import get_config, get_classes
from skimage.util import view_as_windows
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import os
import numpy as np

from utilities.preprocessing import  get_xhdr_sample_rate, load_raw_data, get_basic_block_len, persist_object, trim_iq_basic_block, path2list


# # Argument parsing
def main(sys_args):
    ModelClass_dic = get_classes()
    classes_names = list(ModelClass_dic.keys())

    parser = argparse.ArgumentParser()
    parser.prog = 'Spectrum Anomaly Detection'
    parser.description = 'Use this command parser for training or testing the anomaly detector'
    parser.add_argument('-m', '--mode', help='train or test mode', choices=['train', 'test', 'stat'])
    parser.add_argument('-M', '--model', help='chose model', choices=classes_names)
    parser.add_argument('-d', '--data-dir', help='I/Q recording directory')
    parser.add_argument('-w', '--weights-path', help='path for trained weights')


    namespace = parser.parse_args(sys_args)
    if not namespace.data_dir and namespace.mode == 'train':
        parser.error('the -d arg must be present when mode is train')

    if not namespace.data_dir and namespace.mode == 'test':
        parser.error('the -d arg must be present when mode is test')


    # # Hyper parameters

    # In[5]:

    data_dir = namespace.data_dir
    model_root = os.path.join('eval',path2list(data_dir)[1])
    model_path = namespace.weights_path
    mode = namespace.mode

    ModelClass = ModelClass_dic[namespace.model]

    ## loading data

    model = ModelClass(model_root=model_root)
    model_path = model.model_path

    if mode == 'train':
        sample_rate = get_xhdr_sample_rate(data_dir)
        iq_data = load_raw_data(data_dir)
        (time,fft_train) = model.preprocess_train_data(iq_data,sample_rate)
        model.train_data(fft_train)

    elif mode == 'test':
        sample_rate = get_xhdr_sample_rate(data_dir)
        data_iq = load_raw_data(data_dir)
        data_iq = trim_iq_basic_block(data_iq, sample_rate)
        model.plot_prediction(data_iq, sample_rate)

        data_name = os.path.basename(data_dir)
        f = plt.gcf()
        f.suptitle('using model "' + model.name +'" on file: ' + data_name)
        fig_path = os.path.join(model_path, data_name)
        f.savefig(fig_path+'.png')
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])