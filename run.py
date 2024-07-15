from omegaconf import OmegaConf
import os
import json 

def _load_json(*dirs):
    json_file = os.path.join(*dirs)
    with open(json_file) as f:
        js = json.load(f)
    return js


OmegaConf.register_new_resolver("concat", lambda x, y: x+y)
OmegaConf.register_new_resolver("load_json", _load_json)
OmegaConf.register_new_resolver("last_element", lambda x: x[-1])

_ = OmegaConf.load(os.path.join("qtm", "chem_config.yaml"))
_._set_flag("allow_objects", True)
OmegaConf.resolve(_)
print(_)
