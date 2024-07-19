import os
import time

import numpy as np
from omegaconf import OmegaConf
from pennylane import qchem
import pennylane as qml
import logging

from tqdm import tqdm

from bayes_opt import BayesianOptimization

from qtm.homogeneous_transformation import HomogenousTransformation
from qtm.reaction import Reaction

logging.basicConfig(level=logging.INFO)

def _load_json(*dirs):
    json_file = os.path.join(*dirs)
    with open(json_file) as f:
        js = json.load(f)
    return js
def black_box(reaction, molecules, coords, transform):
    """
    Bayesian optimization will use
    :param reaction:
    :param optimizable_coords:
    :param transform:
    :return: the negative of the energy, because bayesian optimization maximizes the result
    """
    new_coords = ht.transform(molecules, coords, transform)
    H, _ = reaction.build_hamiltonian(new_coords)
    # fixme now using eigen values, but later use theta for Double/Single excitation
    value, state = np.linalg.eig(qml.matrix(H))
    return -value


if __name__ == "__main__":
    OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
    OmegaConf.resolve_new_resolver("pi", lambda : np.pi)
    OmegaConf.resolve_new_resolver("divide", lambda x, y: x / y)
    OmegaConf.register_new_resolver("sum", lambda *numbers: sum(numbers))
    OmegaConf.register_new_resolver("load_json", _load_json)
    OmegaConf.register_new_resolver("last_element", lambda x: x[-1])
    chem_conf = OmegaConf.load("chem_config.yaml")
    chem_conf._set_flag("allow_objects", True)
    ml_conf = OmegaConf.load("ml_config.yaml")
    OmegaConf.resolve(chem_conf)
    ht = HomogenousTransformation(ml_conf['delta_angle'], ml_conf['delta_coords'])

    step_config = chem_conf["steps"][chem_conf.get('step_to_run', None)]
    reaction = Reaction(symbols=step_config["fixed"]["symbols"] + step_config["react"]["symbols"])

    optimzable_molecules = step_config["react"]["symbols"]
    optimzable_coords = step_config["react"]["coords"]

