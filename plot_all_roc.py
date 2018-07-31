import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.preprocessing import persist_object, load_object, path2list
from utilities.plots import save_fig
from utilities.config_handler import get_config, get_classes
import argparse
import sys

def main(sys_args):
    def plot_all_roc(root_path, general_models, special_models, anomaly_name):
        mark_style = ['o', '^', 's', 'v']
        colors = ['purple', 'red', 'blue', 'green']
        ModelClass_dic = get_classes()

        f = plt.figure()
        for i,model_name in enumerate(special_models):
            model = ModelClass_dic[model_name](model_root=root_path)
            score_name = list(model.get_score_methods().keys())[0]
            d = load_object(os.path.join(model.model_path,'eval', 'ROC', anomaly_name, 'roc_score_' + score_name + '.pkl'))
            plt.plot(d['dBs'], d['aucs'], color=colors[-1], marker=mark_style[i], label=model_name+', score method - '+score_name)

        for i,model_name in enumerate(general_models):
            model = ModelClass_dic[model_name](model_root=root_path)
            score_methods_dic = model.get_score_methods()
            for j,score_name in enumerate(score_methods_dic.keys()):
                d = load_object(os.path.join(model.model_path,'eval', 'ROC', anomaly_name, 'roc_score_' + score_name + '.pkl'))
                plt.plot(d['dBs'], d['aucs'], color=colors[i], marker=mark_style[j], label=model_name+', score method - '+score_name)


        plt.ylim([0, 1])
        plt.xlabel('ISR in dB of anomaly', fontsize=18)
        plt.ylabel('AUC score', fontsize=18)
        plt.title(anomaly_name, fontsize=20)
        plt.legend()
        plt.gca().grid(True)
        f.set_size_inches(8, 6.5, forward=True)

        fig_path = os.path.join(root_path, '0_all_ROC', 'dB_vs_AUC_'+anomaly_name)

        save_fig(f, fig_path)
        plt.close()


    parser = argparse.ArgumentParser()
    parser.prog = 'Spectrum Anomaly Detection'
    parser.description = 'getting eval per data type'
    parser.add_argument('-r', '--root-path', help='root folder of data\'s models')

    namespace = parser.parse_args(sys_args)

    root_path = namespace.root_path

    general_models = ['ae', 'amir', 'complex_gauss']

    cep_path = os.path.join('cepstrum_125000', 'eval', 'ROC')
    gauss_cep_path = os.path.join('gaussian_cepstrum_125000', 'eval', 'ROC')
    cep_2d_path = os.path.join('cepstrum_2dfft_125000', 'eval', 'ROC')

    sweep_special_models = ['cepstrum', 'gaussian_cepstrum', 'cepstrum_2dfft']

    CW_special_models = ['CW_dedicated']

    plot_all_roc(root_path, general_models, sweep_special_models, 'sweep')

    plot_all_roc(root_path, general_models, CW_special_models, 'CW')

if __name__ == '__main__':
    main(sys.argv[1:])