# coding: utf-8

# In[1]:


import sys
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, roc_auc_score

from utilities.config_handler import get_config, get_classes
from utilities.anomal_gen import sweep, CW
from utilities.preprocessing import persist_object, load_object, get_xhdr_sample_rate, load_raw_data, trim_iq_basic_block, get_config, get_basic_block_len, iq2fft, path2list
from utilities.plots import save_fig_pickle, load_fig_pickle, save_fig



def main(sys_args):
    def save_roc_plot(iq_normal, sample_rate, dBs, anomal_gen, num_ROC_samples=500, score_method=None, score_name='normal'):
        fprs, tprs, aucs = [], [], []

        normal_plot_saved = 0
        for ind,dB in enumerate(dBs):
            gc.collect()
            print('creating anomaly with {}dB...'.format(dB))
            anomal_params['dB'] = dB
            basic_len = get_basic_block_len(sample_rate)

            roc_start_indices_path = os.path.join(model_root,'roc_start_indices.pkl')

            num_samples = num_ROC_samples

            if os.path.isfile(roc_start_indices_path):
                a_starts, c_starts = load_object(roc_start_indices_path)
                if len(a_starts) != num_samples:
                    print('wrong length of roc_start_indices.pkl...\nchanging to roc_start_indices.pkl len')
                    num_samples = len(a_starts)
            else:
                a_starts = np.random.randint(0, iq_normal.shape[0] - basic_len, (num_samples,))
                c_starts = np.random.randint(0, iq_normal.shape[0] - basic_len, (num_samples,))
                persist_object([a_starts, c_starts], roc_start_indices_path)

            tot_starts = np.concatenate([a_starts, c_starts])
            y_true = np.concatenate([np.ones((num_samples,)), np.zeros((num_samples,))]).astype(bool)
            y_score = np.zeros((2*num_samples,))

            for i in range(2*num_samples):
                if y_true[i]:
                    basic_iq = trim_iq_basic_block(iq_normal, sample_rate, start=tot_starts[i])
                    basic_iq = anomal_dic[anomal_gen](basic_iq, sample_rate, **anomal_params)
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
        plt.title(anomal_gen+' anomaly\nArea Under Curve as function of\nInterference Signal Ratio', fontsize=20)
        plt.gca().grid(True)
        f.set_size_inches(8, 6.5, forward=True)
        fig_path = os.path.join(plots_path, 'dB_vs_AUC_'+score_name)

        save_fig(f, fig_path)
        plt.close()

    ModelClass_dic = get_classes()
    classes_names = list(ModelClass_dic.keys())

    parser = argparse.ArgumentParser()
    parser.prog = 'Spectrum Anomaly Detection'
    parser.description = 'Use this command parser for training or testing the anomaly detector'
    parser.add_argument('-M', '--model', help='chose model', choices=classes_names)
    parser.add_argument('-d', '--data', help='I/Q recording directory')
    parser.add_argument('-a', '--anomaly', help='anomaly type')


    namespace = parser.parse_args(sys_args)

    normal_path = namespace.data
    model_root = os.path.join('eval',path2list(normal_path)[1]) #TODO change 'eval' to somthing

    ModelClass = ModelClass_dic[namespace.model]
    model = ModelClass(model_root=model_root)


    sample_rate = get_xhdr_sample_rate(normal_path)
    iq_normal = load_raw_data(normal_path)
    dBs = np.arange(-20, 26, 5)

    anomal_params = {'freq_band': [-5e6, 5e6],
                    'delta_t': 50e-6,
                    'dB': 0,
                    'f_center': [-5e6, 5e6]}

    i = 0
    anomal_dic = {'CW': CW,
                  'sweep': sweep}
    anomal_gen = namespace.anomaly
    plots_path = os.path.join(model.model_path, 'eval/ROC/'+anomal_gen)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    score_methods = model.get_score_methods()

    for score_name in score_methods:
        score_meth = score_methods[score_name]
        save_roc_plot(iq_normal, sample_rate, dBs, anomal_gen, num_ROC_samples=250, score_method=score_meth, score_name=score_name)

if __name__ == '__main__':
    main(sys.argv[1:])