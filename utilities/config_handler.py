__author__ = 's5806074'

import json
import os
# from base_model.ae_model import AeModel
# from base_model.amir_model import AmirModel
# from base_model.complex_gauss_model import ComplexGauss
# from base_model.cepstrum_model import CepstrumModel
# from base_model.gaussian_cepstrum_model import GaussianCepstrum

_config_path = 'configuration.json'

def get_config():
    with open(_config_path,'r') as f:
        conf = json.load(f)

    return conf

def get_classes():
    d = {'ae': AeModel,
         'amir': AmirModel,
         'complex_gauss': ComplexGauss,
         'cepstrum': CepstrumModel,
         'gaussian_cepstrum': GaussianCepstrum}
    return d