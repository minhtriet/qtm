import os

from smac import Scenario, BlackBoxFacade
from ConfigSpace import Configuration, ConfigurationSpace, Float
import pennylane as qml
import numpy as np
from ..homogeneous_transformation import HomogenousTransformation

class BayesianOptimizer:
    """
    Define the Bayesian optimizer for
    """
    def __init__(self, bound_config, reaction):
        self.cs = ConfigurationSpace()

        self.reaction = reaction
        self.param_names = []
        for i, m in enumerate(self.reaction.react_symbols):
            for name, bound in bound_config.items():
                self.param_names.append(f"{name}_{m}_{i}")
                if (len(m) == 1) and (name.startswith("theta")):
                    # just atom, no need to optimization for rotation
                    continue
                else:
                    self.cs.add([Float(f"{name}_{m}_{i}", bound)])

    def black_box(self, config: Configuration, seed):
        """
        Define the function to run the optimization
        :return:
        """
        ht = HomogenousTransformation
        params = [config.get(name, 0) for name in self.param_names]
        new_coords = ht.mass_transform(self.reaction.react_symbols,
                                       self.reaction.react_coords,
                                       params)
        H, _ = self.reaction.build_hamiltonian(new_coords)
        # fixme now using eigen values, but later use theta for Double/Single excitation
        value, state = np.linalg.eig(qml.matrix(H))
        return value
