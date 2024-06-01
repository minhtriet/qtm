import time
import unittest

from pennylane import qchem
import numpy as np
from hamiltonian_builder import HamiltonianBuilder


class TestParallelBuildHamiltonian(unittest.TestCase):

    def test_hamiltonian_construction(self):
        symbols = ['H', 'H']
        hb = HamiltonianBuilder(symbols)

        # Known coordinates for H2 molecule
        np.random.seed(5)
        coordinates_list = np.random.random((18,6))

        # Call the parallel_build_hamiltonian function
        start_parallel = time.time()
        hamiltonians = hb.parallel_build_hamiltonian(coordinates_list)
        done_parallel = time.time()

        start_serial = time.time()
        expected_h1 = qchem.molecular_hamiltonian(symbols, coordinates_list[0])
        expected_h2 = qchem.molecular_hamiltonian(symbols, coordinates_list[1])
        done_serial = time.time()

        # Check the results
        assert done_parallel - start_parallel < 0.65 * (done_serial - start_serial)
        assert len(hamiltonians) == 2
        assert hamiltonians[0] == expected_h1[0]
        assert hamiltonians[1] == expected_h2[0]
        assert qubits[0] == expected_h1[1]
        assert qubits[1] == expected_h2[1]
