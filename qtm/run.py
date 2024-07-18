import numpy as np
from omegaconf import OmegaConf
from pennylane import qchem
import logging

from qtm.homogeneous_transformation import HomogenousTransformation
from qtm.reaction import Reaction

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
    OmegaConf.resolve_new_resolver("pi", lambda : np.pi)
    OmegaConf.resolve_new_resolver("divide", lambda x, y: x / y)
    chem_conf = OmegaConf.load("chem_config.yaml")
    ml_conf = OmegaConf.load("ml_config.yaml")
    OmegaConf.resolve(chem_conf)
    ht = HomogenousTransformation(ml_conf['delta_angle'], ml_conf['delta_coords'])

    step_config = chem_conf.get('step_to_run', None)
    reaction = Reaction(symbols=step_config["fixed"]["symbols"], fix=step_config["react"]["symbols"])  # react should have fixed and dynamic parts

    logging.info("== Preparing molecule first run")
    H, qubits = reaction.build_hamiltonian(reaction["coords"])
    n_qubits = len(H.wires)
    singles, doubles = qchem.excitations(step_config["active_electrons"], n_qubits)

