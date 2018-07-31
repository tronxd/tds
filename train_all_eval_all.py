import anomaly_detector
import eval_pred_DB

data_type = [
    # {'train': 'iq_data/gsm/gsm_normal_1', 'test': 'iq_data/gsm/files_23'},
    # {'train': 'iq_data/GPS/GPS_NORM_0', 'test': 'iq_data/GPS/files_1234'},
    {'train': 'iq_data/FM/FM_normal_1', 'test': 'iq_data/FM/files_23'},
    {'train': 'iq_data/CELL/CELL_NORM_0', 'test': 'iq_data/CELL/files_234'},
    {'train': 'iq_data/radar/radar_normal_1', 'test': 'iq_data/radar/files_23'},
]

models = ['ae',
         'amir',
         'complex_gauss',
         'cepstrum',
         'gaussian_cepstrum',
         'cepstrum_2dfft',
         'CW_dedicated']

anomalys = ['sweep', 'CW']

for band in data_type:
    ## training
    comands = []
    for M in models:
        comands.append('-m train -M '+M+' -d '+band['train'])

    for c in comands:
        print('######################')
        print('######################\n')
        print(c)
        print('.\n.\n.\n')
        anomaly_detector.main(c.split(' '))

    ## eval
    comands = []
    for M in models:
        for a in anomalys:
            comands.append('-M ' + M + ' -d ' + band['test'] + ' -a '+ a)

    for c in comands:
        print('######################')
        print('######################\n')
        print(c)
        print('.\n.\n.\n')
        eval_pred_DB.main(c.split(' '))
