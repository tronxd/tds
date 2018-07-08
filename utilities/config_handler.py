__author__ = 's5806074'

import json
_config_path = 'configuration.json'

def get_config():
    with open(_config_path,'r') as f:
        conf = json.load(f)
        return conf