import json
import os

from omegaconf import OmegaConf

from qtm.reaction import Reaction

def _load_json(*dirs):
    json_file = os.path.join(*dirs)
    with open(json_file) as f:
        js = json.load(f)
    return js

if __name__ == '__main__':
    OmegaConf.register_new_resolver("sum", lambda *numbers: sum(numbers))
    OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
    OmegaConf.register_new_resolver("load_json", _load_json)
    OmegaConf.register_new_resolver("last_element", lambda x: x[-1])

    conf = OmegaConf.load(os.path.join("qtm", "chem_config.yaml"))
    conf._set_flag("allow_objects", True)
    OmegaConf.resolve(conf)
    print(conf)

    step_to_run = conf.get('step_to_run', None)
    step_config = conf["steps"][step_to_run]

    reaction = Reaction(symbols=step_config["symbols"])
    # todo fix
    H, qubits = reaction.partial_h(symbols)
    n_qubits = len(H.wires)
    singles, doubles = qchem.excitations(active_electrons, n_qubits)
    return H, n_qubits, singles, doubles