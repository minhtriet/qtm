import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf
from smac import Scenario, BlackBoxFacade

from qtm.optmizer.bayesian_optimizer import BayesianOptimizer

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


def add_tuple(t1: tuple, t2: tuple) -> tuple:
    return tuple(sum(x) for x in zip(t1,t2))


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
        fix_symbols=step_config["fixed"]["symbols"],
        react_symbols=step_config["react"]["symbols"],
        fix_coords=step_config["fixed"]["coords"],
        react_coords=step_config["react"]["coords"],
        charge=step_config["config"].get("charge"),
        mult=step_config["config"].get("mult"),
        active_electrons=step_config["config"].get("active_electrons"),
        active_orbitals=step_config["config"].get("active_orbitals"),
    )

    optimizable_molecules = step_config["react"]["symbols"]
    optimizable_coords = step_config["react"]["coords"]

    # define boundaries for bayesian optimization
    x_bound = add_tuple(min_max(chem_conf["catalyst"]["coords"][::3]), (-1,1))
    y_bound = add_tuple(min_max(chem_conf["catalyst"]["coords"][1::3]), (-1, 1))
    z_bound = (2., 3.)
    angle_bound = (-np.pi, np.pi)

    bound_config = {
        "x": x_bound,
        "y": y_bound,
        "z": z_bound,
        "theta_x": angle_bound,
        "theta_y": angle_bound,
        "theta_z": angle_bound,
    }


    bo = BayesianOptimizer(bound_config, reaction)
    scenario = Scenario(bo.cs, deterministic=True, n_trials=100, n_workers=2)
    smac = BlackBoxFacade(scenario, bo.black_box, overwrite=True, dask_client=None)
    incumbent = smac.optimize()
    print(incumbent)
