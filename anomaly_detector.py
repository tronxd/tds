
# coding: utf-8

# In[1]:


import sys
import argparse
from base_model.ae_model import AeModel
from base_model.amir_model import AmirModel
from skimage.util import view_as_windows
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import os
import numpy as np

from utilities.preprocessing import  get_xhdr_sample_rate, load_raw_data, get_basic_block_len, persist_object, trim_iq_basic_block


# # Argument parsing



parser = argparse.ArgumentParser()
parser.prog = 'Spectrum Anomaly Detection'
parser.description = 'Use this command parser for training or testing the anomaly detector'
parser.add_argument('-m', '--mode', help='train or test mode', choices=['train', 'test', 'stat'])
parser.add_argument('-M', '--model', help='chose model', choices=['ae', 'amir'])
parser.add_argument('-d', '--data-dir', help='I/Q recording directory')
parser.add_argument('-w', '--weights-path', help='path for trained weights')


namespace = parser.parse_args(sys.argv[1:])
if not namespace.data_dir and namespace.mode == 'train':
    parser.error('the -d arg must be present when mode is train')

if not namespace.data_dir and namespace.mode == 'test':
    parser.error('the -d arg must be present when mode is test')


# # Hyper parameters

# In[5]:

data_dir = namespace.data_dir
model_path = namespace.weights_path
mode = namespace.mode

ModelClass_dic = {'ae': AeModel,
                  'amir': AmirModel}
ModelClass = ModelClass_dic[namespace.model]

## loading data

model = ModelClass(model_path)
model_path = model.model_path

if mode == 'train':
    sample_rate = get_xhdr_sample_rate(data_dir)
    data_iq = load_raw_data(data_dir)
    model.preprocess_train(data_iq, sample_rate)


elif mode == 'test':
    sample_rate = get_xhdr_sample_rate(data_dir)
    data_iq = load_raw_data(data_dir)
    data_iq = trim_iq_basic_block(data_iq, sample_rate)
    model.plot_prediction(data_iq, sample_rate)

    data_name = os.path.basename(data_dir)
    f = plt.gcf()
    f.suptitle('useing model "' + model.name +'" on file: ' + data_name)
    fig_path = os.path.join(model_path, data_name)
    f.savefig(fig_path+'.png')
    plt.show()

elif mode == 'stat':
    normal_path = os.path.join(data_dir, 'normal')
    anomal_path = os.path.join(data_dir, 'anomal')

    normal_records = os.listdir(normal_path)
    anomal_records = os.listdir(anomal_path)
    num_records = len(normal_records) + len(anomal_records)

    records = []
    scores = []
    trues = []
    i = 0

    print('number of records is: '+str(num_records))
    print('started to work...')
    for r in normal_records:
        r_dir = os.path.join(normal_path, r)
        sample_rate = get_xhdr_sample_rate(r_dir)
        data_iq = load_raw_data(r_dir)

        basic_len = get_basic_block_len(sample_rate)
        basic_iq_shape = (basic_len, 2)
        data = view_as_windows(data_iq, basic_iq_shape, step=(basic_len // 4, 2))
        data = np.reshape(data, (-1, basic_len, 2))
        for i in range(data.shape[0]):
            records.append(data_dir)
            trues.append(False)
            score = model.predict_score(data[i,:,:], sample_rate)
            scores.append(score)
            i += 1

        print('\n####################\n@@@@@@@@@@@@@@@@@@@@\n')
        print('finished prediction on:\n'+r+' - {}/{}\n\n'.format(i, num_records))

    for r in anomal_records:
        r_dir = os.path.join(anomal_path, r)
        sample_rate = get_xhdr_sample_rate(r_dir)
        data_iq = load_raw_data(r_dir)

        basic_len = get_basic_block_len(sample_rate)
        basic_iq_shape = (basic_len, 2)
        data = view_as_windows(data_iq, basic_iq_shape, step=(basic_len // 4, 2))
        data = np.reshape(data, (-1, basic_len, 2))
        for i in range(data.shape[0]):
            records.append(data_dir)
            trues.append(True)
            score = model.predict_score(data[i,:,:], sample_rate)
            scores.append(score)
            i += 1

        print('\n####################\n@@@@@@@@@@@@@@@@@@@@\n')
        print('finished prediction on:\n'+r+' - {}/{}\n\n'.format(i, num_records))

    y_true = np.array(trues).astype('float')
    y_score = np.array(scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.plot(tpr, fpr)
    plt.show()

    for_pkl = {
        'records_path': records,
        'preds': preds,
        'is_anomal': trues
    }

    persist_object(for_pkl, os.path.join(model_path, 'evaluation_on_folders.pkl'))

