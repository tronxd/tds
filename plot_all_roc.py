import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.preprocessing import persist_object, load_object
from utilities.plots import save_fig


amir_path = os.path.join('model','amir_63000', 'eval', 'ROC')
comp_gauss_path = os.path.join('model','ComplexGauss_63000', 'eval', 'ROC')
cep_path = os.path.join('model','cepstrum_125000', 'eval', 'ROC')
gauss_cep_path = os.path.join('model','gaussian_cepstrum_125000', 'eval', 'ROC')

special_scores = ['mean', 'max_per_time', 'percent']

mark_style = ['^','o','s','v']


## sweep
f = plt.figure()
score_name = 'normal'
d = load_object(os.path.join(gauss_cep_path, 'sweep', 'roc_score_'+score_name+'.pkl'))
plt.plot(d['dBs'], d['aucs'], color='green', marker='.', label='sweep dedicated model #1')

d = load_object(os.path.join(cep_path, 'sweep', 'roc_score_'+score_name+'.pkl'))
plt.plot(d['dBs'], d['aucs'], color='green', marker='.', label='sweep dedicated model #2')

for i,score_name in enumerate(special_scores):
    d = load_object(os.path.join(comp_gauss_path, 'sweep', 'roc_score_' + score_name + '.pkl'))
    plt.plot(d['dBs'], d['aucs'], color='blue', marker=mark_style[i], label='"complex gaussian" model, score method - '+score_name)

for i,score_name in enumerate(special_scores):
    d = load_object(os.path.join(amir_path, 'sweep', 'roc_score_' + score_name + '.pkl'))
    plt.plot(d['dBs'], d['aucs'], color='red', marker=mark_style[i], label='"log_power gaussian" model, score method - ' + score_name)

plt.title('sweep', fontsize=20)
plt.ylim([0,1])
plt.xlabel('ISR in dB of anomaly', fontsize=18)
plt.ylabel('AUC score', fontsize=18)
plt.legend()
plt.gca().grid(True)
f.set_size_inches(8, 6.5, forward=True)

fig_path = os.path.join('model', '0_eval_all_models', 'sweep_dB_vs_auc')
save_fig(f, fig_path)

plt.close()

## CW
f = plt.figure()
score_name = 'CW_dedicated'
d = load_object(os.path.join(comp_gauss_path, 'CW', 'roc_score_'+score_name+'.pkl'))
plt.plot(d['dBs'], d['aucs'], color='green', marker='.', label='CW dedicated model #1')

d = load_object(os.path.join(amir_path, 'CW', 'roc_score_'+score_name+'.pkl'))
plt.plot(d['dBs'], d['aucs'], color='green', marker='.', label='CW dedicated model #2')

for i,score_name in enumerate(special_scores):
    d = load_object(os.path.join(comp_gauss_path, 'CW', 'roc_score_' + score_name + '.pkl'))
    plt.plot(d['dBs'], d['aucs'], color='blue', marker=mark_style[i], label='"complex gaussian" model, score method - ' + score_name)

for i,score_name in enumerate(special_scores):
    d = load_object(os.path.join(amir_path, 'CW', 'roc_score_' + score_name + '.pkl'))
    plt.plot(d['dBs'], d['aucs'], color='red', marker=mark_style[i], label='"log_power gaussian" model, score method - ' + score_name)

plt.title('CW', fontsize=20)
plt.ylim([0,1])
plt.xlabel('ISR in dB of anomaly', fontsize=18)
plt.ylabel('AUC score', fontsize=18)
plt.legend()
plt.gca().grid(True)
f.set_size_inches(8, 6.5, forward=True)

fig_path = os.path.join('model', '0_eval_all_models', 'CW_dB_vs_auc')
save_fig(f, fig_path)
