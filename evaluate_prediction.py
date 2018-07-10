# coding: utf-8

# In[1]:


import sys
import os
import numpy as np
from utilities.config_handler import get_config
from utilities.detection import predict_by_ae

# # Hyper parameters

conf = get_config()
gpus = conf['gpus']
train_params = conf['learning']['ae']
batch_size = conf['learning']['ae']['batch_size']
use_noise = conf['preprocessing']['ae']['use_noise']
rbw_set = conf['preprocessing']['ae']['rbw_set']


pred = predict_by_ae('iq_data\\CELL\\anomal1', 'model\\ae\\CELL_125000.0')
print('\n######################\n')
if pred:
    print('F O U N D   A N O M A L Y ! ! !')
else:
    print('N O   A N O M A L Y . . .')
print('\n######################')
