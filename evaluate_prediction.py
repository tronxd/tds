# coding: utf-8

# In[1]:


import sys
import os
import gc
import numpy as np
from utilities.config_handler import get_config
from utilities.detection import predict_by_ae, predict_folder_by_ae
from utilities.preprocessing import persist_object, load_object

# # Hyper parameters

conf = get_config()
gpus = conf['gpus']
train_params = conf['learning']['ae']
batch_size = conf['learning']['ae']['batch_size']
use_noise = conf['preprocessing']['ae']['use_noise']
rbw_set = conf['preprocessing']['ae']['rbw_set']

def print_record_pred(i, num_records, r, is_anomal, pred):
    print('\n####################\n@@@@@@@@@@@@@@@@@@@@\n\n')
    print('finished prediction on record: {}/{}\n'.format(i, num_records))
    print('file name is: '+r)
    print('it is: ', end='')
    if is_anomal:
        print('ANOMAL')
    else:
        print('NORMAL')
    print('prediction is: ', end='')
    if pred:
        print('ANOMAL')
    else:
        print('NORMAL')
    print('\n\n@@@@@@@@@@@@@@@@@@@@\n####################\n')
    gc.collect()


model_weight_path = 'model\\ae\\CELL_125000.0'

normal_path = 'iq_data\\CELL\\normal'
anomal_path = 'iq_data\\CELL\\anomal'

normal_records = os.listdir(normal_path)[1:] # discarding train record
anomal_records = os.listdir(anomal_path)

num_records = len(normal_records)+len(anomal_records)

records = []
preds = []
trues = []
i = 0

for r in normal_records:
    data_dir = os.path.join(normal_path, r)
    records.append(data_dir)
    pred = predict_folder_by_ae(data_dir, model_weight_path)
    preds.append(pred)
    is_anomal = False
    trues.append(is_anomal)
    i+=1
    print_record_pred(i, num_records, r, is_anomal, pred)

for r in anomal_records:
    data_dir = os.path.join(anomal_path, r)
    records.append(data_dir)
    pred = predict_folder_by_ae(data_dir, model_weight_path)
    preds.append(pred)
    is_anomal = True
    trues.append(is_anomal)
    i += 1
    print_record_pred(i, num_records, r, is_anomal, pred)

for_pkl = {
    'records_path': records,
    'preds': preds,
    'is_anomal': trues
}

persist_object(for_pkl, os.path.join(model_weight_path, 'evaluation_on_folders.pkl'))

