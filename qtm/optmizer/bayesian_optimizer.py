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
                if (len(m) == 1) and (name.startswith("theta")):
                    # just atom, no need to optimization for rotation
                    continue
                else:
                    self.cs.add([Float(f"{name}_{m}_{i}", bound)])
                self.param_names.append(f"{name}_{m}_{i}")

    def black_box(self, seed):
        """
        Define the function to run the optimization
        :return:
        """
        ht = HomogenousTransformation
        params = [self.cs.get(name, 0) for name in self.param_names if self.cs[name]]
        new_coords = ht.mass_transform(self.reaction.react_symbols,
                                       self.reaction.react_coords,
                                       params)
        H, _ = self.reaction.build_hamiltonian(new_coords)
        # fixme now using eigen values, but later use theta for Double/Single excitation
        value, state = np.linalg.eig(qml.matrix(H))
        return value

    def train(self):
        """
        Bayesian optimization will use
        :param reaction:
        :param optimizable_coords:
        :param transform:
        :return: the negative of the energy, because bayesian optimization maximizes the result
        """
        scenario = Scenario(self.cs, deterministic=True, n_trials=4, n_workers=1)
        # scenario = Scenario(self.cs, deterministic=True, n_trials=4, n_workers=os.cpu_count()-4)
        smac = BlackBoxFacade(scenario, self.black_box, dask_client=None)
        incumbent = smac.optimize()
        print(incumbent)

