import numpy as np
from omegaconf import OmegaConf
import logging

from qtm.homogeneous_transformation import HomogenousTransformation

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
    OmegaConf.resolve_new_resolver("pi", lambda : np.pi)
    OmegaConf.resolve_new_resolver("divide", lambda x, y: x / y)
    chem_conf = OmegaConf.load("chem_config.yaml")
    ml_conf = OmegaConf.load("ml_config.yaml")
    OmegaConf.resolve(chem_conf)
    ht = HomogenousTransformation()
    transformations = [
        [ml_conf.ml_conf.delta_angle, 0, 0, 0, 0, 0],
        [-ml_conf.delta_angle, 0, 0, 0, 0, 0],
        [0, ml_conf.delta_angle, 0, 0, 0, 0],
        [0, -ml_conf.delta_angle, 0, 0, 0, 0],
        [0, 0, ml_conf.delta_angle, 0, 0, 0],
        [0, 0, -ml_conf.delta_angle, 0, 0, 0],
        [0, 0, 0, ml_conf.delta_coord, 0, 0],
        [0, 0, 0, -ml_conf.delta_coord, 0, 0],
        [0, 0, 0, 0, ml_conf.delta_coord, 0],
        [0, 0, 0, 0, -ml_conf.delta_coord, 0],
        [0, 0, 0, 0, 0, ml_conf.delta_coord],
        [0, 0, 0, 0, 0, -ml_conf.delta_coord],
    ]
    reaction = chem_conf.steps.two

    logging.info("== Preparing molecule first run")

