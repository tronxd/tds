# coding: utf-8

# In[1]:


import sys
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from utilities.anomal_gen import sweep, CW
from utilities.preprocessing import persist_object, load_object, get_xhdr_sample_rate, load_raw_data, trim_iq_basic_block, get_config, get_basic_block_len, iq2fft
from base_model.ae_model import AeModel
from base_model.amir_model import AmirModel
from base_model.complex_gauss_model import ComplexGauss
from base_model.cepstrum_model import CepstrumModel
from base_model.gaussian_cepstrum_model import GaussianCepstrum
from base_model.cepstrum_2dfft import Cepstrum2DFFT

from utilities.plots import save_fig_pickle, load_fig_pickle, save_fig


# # Hyper parameters

def save_roc_plot(iq_normal, sample_rate, dBs, num_ROC_samples=500, score_method=None, score_name='normal'):
    fprs, tprs, aucs = [], [], []

    normal_plot_saved = 0
    for ind,dB in enumerate(dBs):
        gc.collect()
        print('creating anomaly with {}dB...'.format(dB))
        sweep_params['dB'] = dB
        basic_len = get_basic_block_len(sample_rate)

        roc_start_indices_path = os.path.join('base_model','roc_start_indices.pkl')

        num_samples = num_ROC_samples

        if os.path.isfile(roc_start_indices_path) :
            a_starts, c_starts = load_object(roc_start_indices_path)
            if len(a_starts) != num_samples:
                print('wrong length of roc_start_indices.pkl...\nchanging to roc_start_indices.pkl len')
                num_samples = len(a_starts)
        else:
            a_starts = np.random.randint(0, iq_normal.shape[0] - basic_len, (num_samples,))
            c_starts = np.random.randint(0, iq_normal.shape[0] - basic_len, (num_samples,))

        tot_starts = np.concatenate([a_starts, c_starts])
        y_true = np.concatenate([np.ones((num_samples,)), np.zeros((num_samples,))]).astype(bool)
        y_score = np.zeros((2*num_samples,))

        for i in range(2*num_samples):
            if y_true[i]:
                basic_iq = trim_iq_basic_block(iq_normal, sample_rate, start=tot_starts[i])
                basic_iq = sweep(basic_iq, sample_rate, **sweep_params)
                if i < 2:
                    model.plot_prediction(basic_iq, sample_rate)
                    f = plt.gcf() #TODO check if figure exists
                    f.suptitle('Using model "' + model.name + '" on sweep with ISR: ' + str(dB)+'dB')
                    f.set_size_inches(8, 8, forward=True)
                    fig_path = os.path.join(plots_path, '{0:02d}_ISR_dB_{1}_sample_{2}'.format(ind, dB, i))
                    save_fig(f, fig_path)
                    plt.close()
            else:
                basic_iq = trim_iq_basic_block(iq_normal, sample_rate, start=tot_starts[i])
                if normal_plot_saved < 2:
                    model.plot_prediction(basic_iq, sample_rate)
                    f = plt.gcf()
                    f.suptitle('Using model "' + model.name + '" on normal data')
                    f.set_size_inches(8, 8, forward=True)
                    fig_path = os.path.join(plots_path, 'normal_sample_{0}'.format(normal_plot_saved))
                    save_fig(f, fig_path)
                    plt.close()
                    normal_plot_saved += 1


            if score_method:
                y_score[i] = score_method(basic_iq, sample_rate)
            else:
                y_score[i] = model.predict_basic_block_score(basic_iq,sample_rate)


        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc_score(y_true, y_score))

    # ploting
    f = plt.figure(0)
    for j in range(len(dBs)-1,-1,-1):
        plt.plot(fprs[j], tprs[j])

    plt.legend(['anomaly in {}dB, auc: {:.3f}'.format(dBs[j],aucs[j]) for j in range(len(dBs)-1,-1,-1)])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC for anomalies with different ISR [dB]', fontsize=20)
    plt.gca().grid(True)
    f.set_size_inches(8, 6.5, forward=True)

    fig_path = os.path.join(plots_path, 'All_ROCs_'+score_name)
    save_fig(f, fig_path)
    plt.close()

    persist_object({'dBs': dBs,
                    'aucs': aucs,
                    'name': model.name+'_score_'+score_name}, os.path.join(plots_path, 'roc_score_'+score_name+'.pkl'))

    f = plt.figure(1)
    plt.plot(dBs, aucs, '-o')
    plt.ylim([0,1])
    plt.xlabel('ISR [dB]', fontsize=18)
    plt.ylabel('AUC score', fontsize=18)
    plt.title('Sweep anomaly\nArea Under Curve as function of\nInterference Signal Ratio', fontsize=20)
    plt.gca().grid(True)
    f.set_size_inches(8, 6.5, forward=True)
    fig_path = os.path.join(plots_path, 'dB_vs_AUC_'+score_name)

    save_fig(f, fig_path)
    plt.close()


ModelClass_dic = {'ae': AeModel,
                  'amir': AmirModel,
                  'complex_gauss': ComplexGauss,
                  'cepstrum': CepstrumModel,
                  'gaussian_cepstrum':GaussianCepstrum,
                  'cepstrum_2dfft':Cepstrum2DFFT}

ModelClass = ModelClass_dic['cepstrum']
model = ModelClass()
plots_path = os.path.join(model.model_path, 'eval/ROC/sweep')
if not os.path.exists(plots_path):
    os.makedirs(plots_path)


normal_path = 'iq_data/CELL/normal/files_234'

sample_rate = get_xhdr_sample_rate(normal_path)
iq_normal = load_raw_data(normal_path)
dBs = np.arange(-10, 26, 5)

sweep_params = {'freq_band': [-5e6, 5e6],
                'delta_t': 50e-6,
                'dB': 0,}

i = 0
save_roc_plot(iq_normal, sample_rate, dBs, num_ROC_samples=250)

# score_meth = model.predict_basic_block_score_mean
# name = 'mean'
# save_roc_plot(iq_normal, sample_rate, dBs, num_ROC_samples=250, score_method=score_meth, score_name=name)
#
# score_meth = model.predict_basic_block_score_max
# name = 'max_per_time'
# save_roc_plot(iq_normal, sample_rate, dBs, num_ROC_samples=250, score_method=score_meth, score_name=name)
#
# score_meth = model.predict_basic_block_score_percent
# name = 'percent'
# save_roc_plot(iq_normal, sample_rate, dBs, num_ROC_samples=250, score_method=score_meth, score_name=name)