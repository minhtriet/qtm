import json
import logging
import os

import numpy as np
import pennylane as qml
from omegaconf import OmegaConf

from smac import HyperparameterOptimizationFacade, Scenario

from qtm.homogeneous_transformation import HomogenousTransformation
from qtm.reaction import Reaction

logging.basicConfig(level=logging.INFO)


def _load_json(*dirs):
    json_file = os.path.join(*dirs)
    with open(json_file) as f:
        js = json.load(f)
    return js


def min_max(a: list):
    return min(a), max(a)


def black_box(reaction, molecules, coords, transform):
    """
    Bayesian optimization will use
    :param reaction:
    :param optimizable_coords:
    :param transform:
    :return: the negative of the energy, because bayesian optimization maximizes the result
    """
    new_coords = ht.mass_transform(molecules, coords, transform)
    H, _ = reaction.build_hamiltonian(new_coords)
    # fixme now using eigen values, but later use theta for Double/Single excitation
    value, state = np.linalg.eig(qml.matrix(H))
    return value


if __name__ == "__main__":
    chem_conf_path = os.path.join("qtm", "chem_config.yaml")
    ml_conf_path = os.path.join("qtm", "ml_config.yaml")

    OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    OmegaConf.register_new_resolver("sum", lambda *numbers: sum(numbers))
    OmegaConf.register_new_resolver("load_json", _load_json)
    OmegaConf.register_new_resolver("last_element", lambda x: x[-1])
    chem_conf = OmegaConf.load(chem_conf_path)
    chem_conf._set_flag("allow_objects", True)
    ml_conf = OmegaConf.load(ml_conf_path)
    OmegaConf.resolve(chem_conf)
    OmegaConf.resolve(ml_conf)
    ht = HomogenousTransformation

    step_config = chem_conf["steps"][chem_conf.get("step_to_run", None)]
    reaction = Reaction(
        symbols=step_config["fixed"]["symbols"] + step_config["react"]["symbols"],
        coords=step_config["fixed"]["coords"] + step_config["fixed"]["coords"],
    )

    optimizable_molecules = step_config["react"]["symbols"]
    optimizable_coords = step_config["react"]["coords"]

    # define boundaries for bayesian optimization
    x_bound = min_max(chem_conf["catalyst"]["coords"][::3])
    y_bound = min_max(chem_conf["catalyst"]["coords"][1::3])
    z_bound = (2, 3)
    angle_bound = (-np.pi, np.pi)

    configspace = ConfigurationSpace(
        {
            "x_bound": x_bound,
            "y_bound": y_bound,
            "z_bound": z_bound,
            "angle_bound": angle_bound,
        }
    )

    # Scenario object specifying the optimization environment
    scenario = Scenario(configspace, deterministic=True, n_trials=200)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, black_box)
    incumbent = smac.optimize()
