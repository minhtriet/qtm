from omegaconf import OmegaConf
import os
import json


def _load_json(*dir):
    json_file = os.path.join(*dir)
    with open(json_file) as f:
        js = json.load(f)
    return js

OmegaConf.register_new_resolver("concat", lambda x, y: x+y)
OmegaConf.register_new_resolver("load_json", _load_json)
_ = OmegaConf.load("chem_config.yaml")
OmegaConf.resolve(_)
print(_)
