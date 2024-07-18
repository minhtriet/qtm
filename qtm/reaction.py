import os
from multiprocessing import Pool

from pennylane import qchem


class Reaction:

    def __init__(self, symbols, charge=0, mult=1, active_electrons=None, active_orbitals=None):
        self.symbols = symbols
        self.charge = charge
        self.mult = mult
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals

    def partial_h(self, coordinate):
        """
        Return a callable
        :param coordinate:
        :return:
        """
        return qchem.molecular_hamiltonian(
            symbols=self.symbols,
            charge=self.charge,
            mult=self.mult,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
            method="pyscf",
            coordinates=coordinate,
        )

    def parallel_build_hamiltonian(self, coordinates_list):
        # Function to be used with Pool.map to process coordinates in parallel
        # Initialize a multiprocessing Pool
        with Pool(os.cpu_count()) as pool:
            # Map the build_hamiltonian function to the list of coordinates
            hamiltonians = pool.map(self.partial_h, coordinates_list)
        return hamiltonians
