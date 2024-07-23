import os
from multiprocessing import get_context
from typing import Optional

from pennylane import qchem


class Reaction:
    def __init__(
        self,
        fix_symbols,
        react_symbols,
        fix_coords,
        react_coords,
        charge=0,
        mult=1,
        active_electrons=None,
        active_orbitals=None,
    ):
        self.symbols = fix_symbols + react_symbols
        self.coords = fix_coords + react_coords
        self.fix_symbols = fix_symbols
        self.fix_coords = fix_coords
        self.react_symbols = react_symbols
        self.react_coords = react_coords
        self.charge = charge
        self.mult = mult
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals

    def build_hamiltonian(self, coordinates: Optional[list] = None):
        """
        Return a callable
        :param coordinates:
        :return:
        """
        if coordinates is None:  # this is to for cases where optimizing a single atom
            return None, None
        return qchem.molecular_hamiltonian(
            symbols=self.symbols,
            charge=self.charge,
            mult=self.mult,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
            method="openfermion",
            coordinates=coordinates,
        )

    def parallel_build_hamiltonian(self, coordinates_list):
        # Function to be used with Pool.map to process coordinates in parallel
        # Initialize a multiprocessing Pool
        with get_context("spawn").Pool(os.cpu_count() - 4) as p:
            # Map the build_hamiltonian function to the list of coordinates
            hamiltonians = p.starmap(self.build_hamiltonian, coordinates_list)
        return hamiltonians
