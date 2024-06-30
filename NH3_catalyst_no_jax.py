import copy
import logging
from qtm.homogeneous_transformation import HomogenousTransformation
import json


logging.basicConfig(level = logging.INFO)

import time
import os
from multiprocessing import get_context

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

from ase import Atoms

from tqdm import tqdm

import qtm.chem_config as chem_config


np.random.seed(17)

# inside NH3
n2_x = np.random.uniform(0, 1.1, [2])  # one value for N2

h2_x = np.random.uniform(0, 1.1, [6])  # three values for H2
h2_y = np.random.uniform(-0.1, 1.1, [6])
h2_z = np.random.uniform(0.2, 0.6, [6])

molecule = chem_config.NH2
active_electrons = molecule["active_electrons"]
active_orbitals = molecule["active_orbitals"]
electrons = molecule["electrons"]


def create_pyscf_representation(symbols, coords):
    coords = np.reshape(coords, (-1, 3))
    return [[symbols[i], coords[i]] for i in range(len(symbols))]


symbols = chem_config.Fe['symbols'] + molecule["symbols"] + chem_config._3N_step1['symbols']


def hamiltonian_from_coords(coords):
    base_coords = chem_config.Fe["coords"]
    coordinates = np.append(base_coords, coords)
    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        method="openfermion",
        active_electrons=active_electrons + 1, # +1 for 3N
        active_orbitals=active_orbitals + 1, # +1 for 3N
        mult=1 + molecule["unpaired_e"] + 3,  # +3 for 3N
    )
    return H, qubits



def run_circuit(H, params=None, init_state=None):
    dev = qml.device("lightning.qubit", wires=len(H.wires))

    @qml.qnode(dev)
    def circuit_theta():
        state = qchem.hf_state(active_electrons, len(H.wires))
        singles, doubles = qchem.excitations(active_electrons, len(H.wires))
        qml.AllSinglesDoubles(
            hf_state=state,
            weights=params,
            wires=H.wires,
            singles=singles,
            doubles=doubles,
        )
        return qml.expval(H)

    @qml.qnode(dev)
    def circuit_state():
        qml.StatePrep(init_state, H.wires)
        return qml.expval(H)

    if init_state is not None:
        return circuit_state()
    elif params:
        logging.info("Optimizing for theta")
        return circuit_theta()


def prepare_H(coords):
    H, qubits = hamiltonian_from_coords(coords)
    n_qubits = len(H.wires)
    singles, doubles = qchem.excitations(active_electrons, n_qubits)
    return H, n_qubits, singles, doubles


# ## Some manual grad calculation

opt_theta = qml.GradientDescentOptimizer(stepsize=0.4)


def finite_diff(hs, theta, state, delta_theta=0.01, delta_xyz=0.01):
    """Compute the central-difference finite difference of a function
    x: coordinates, thetas is the rotational angles
    """
    theta_x_grad = (
        run_circuit(hs[0], init_state=state) - run_circuit(hs[1], init_state=state)
    ) * (0.5 * delta_theta**-1)
    theta_y_grad = (
        run_circuit(hs[2], init_state=state) - run_circuit(hs[3], init_state=state)
    ) * (0.5 * delta_theta**-1)
    theta_z_grad = (
        run_circuit(hs[4], init_state=state) - run_circuit(hs[5], init_state=state)
    ) * (0.5 * delta_theta**-1)
    x_grad = (
        run_circuit(hs[6], init_state=state) - run_circuit(hs[7], init_state=state)
    ) * (0.5 * delta_xyz**-1)
    y_grad = (
        run_circuit(hs[8], init_state=state) - run_circuit(hs[9], init_state=state)
    ) * (0.5 * delta_xyz**-1)
    z_grad = (
        run_circuit(hs[10], init_state=state) - run_circuit(hs[11], init_state=state)
    ) * (0.5 * delta_xyz**-1)

    return np.array([theta_x_grad, theta_y_grad, theta_z_grad, x_grad, y_grad, z_grad])


def loss_f(thetas, coords):
    H, _ = hamiltonian_from_coords(coords)
    return run_circuit(H, thetas)


# #### Optimize
if __name__ == "__main__":
    # prepare for the 1st run
    adsorbate_coords = np.array(molecule["coords"]+chem_config._3N_step1['coords'])
    logging.info("Preparing molecule first run")

    _, __, singles, doubles = prepare_H(adsorbate_coords)
    total_single_double_gates = len(singles) + len(doubles)
    lr = 1e-4
    logging.info(f"New coordinates {adsorbate_coords}")

    # store the values of the cost function
    thetas = np.random.normal(0, np.pi, total_single_double_gates)
    max_iterations = 20
    delta_angle = 0.01

    # store the values of the circuit parameter
    angle = []
    coords = []
    energies = []

    # debug the moving away
    # init cooridation
    # optimize theta
    # print the ground state this hamiltonian 1
    # eigenvalue of the Hamiltonian 2
    # comapre (1) and (2)

    # todo to speed up (print number of qubits)

    for _ in tqdm(range(max_iterations)):
        [os.remove(hdf5) for hdf5 in os.listdir(".") if hdf5.endswith(".hdf5")]
        # Optimize the circuit parameters
        start = time.time()
        # thetas.requires_grad = True
        #  adsorbate_coords.requires_grad = False
        H, _ = hamiltonian_from_coords(adsorbate_coords)
        # fixme re-enable the optimize later
        value, state = np.linalg.eig(qml.matrix(H))
        smallest_i = np.argmin(value)
        g_energy, g_state = value[smallest_i], state[smallest_i]
        g_state /= np.linalg.norm(g_state)
        energies.append(float(run_circuit(H, init_state=g_state)))
        # early stopping
        if len(energies) > 2 and np.abs(energies[-1] - energies[-2]) < 1e-5:
            break
        # thetas, _ = opt_theta.step(loss_f, thetas, adsorbate_coords)
        logging.info(f"Done theta, starting coordinates {time.time()- start}")

        # Optimize the nuclear coordinates
        # adsorbate_coords.requires_grad = True
        ht = HomogenousTransformation()
        thetas.requires_grad = False
        delta_angle = np.pi / 90
        delta_coord = 0.1
        # all possible transformations
        transformations = [
            [delta_angle, 0, 0, 0, 0, 0],
            [-delta_angle, 0, 0, 0, 0, 0],
            [0, delta_angle, 0, 0, 0, 0],
            [0, -delta_angle, 0, 0, 0, 0],
            [0, 0, delta_angle, 0, 0, 0],
            [0, 0, -delta_angle, 0, 0, 0],
            [0, 0, 0, delta_coord, 0, 0],
            [0, 0, 0, -delta_coord, 0, 0],
            [0, 0, 0, 0, delta_coord, 0],
            [0, 0, 0, 0, -delta_coord, 0],
            [0, 0, 0, 0, 0, delta_coord],
            [0, 0, 0, 0, 0, -delta_coord],
        ]
        shifted_coords = [
            (ht.transform(adsorbate_coords, *transformation))
            for transformation in transformations
        ]

        # .unique()   # unique to reduce the rotation for single atoms, since it doesn't change
        start = time.time()
        with get_context("spawn").Pool(6) as p:
            hs = p.map(hamiltonian_from_coords, shifted_coords)
            # Each hs[i] contains coordinates and the corresponding H
            logging.info(f"Energy level {run_circuit(hs[0][0], init_state=g_state)}")
        grad_x = finite_diff([h[0] for h in hs], thetas, g_state, delta_angle)

        logging.info(f"gradients {grad_x}")
        transform_matrix = np.zeros(6)
        transform_matrix -= lr * grad_x
        adsorbate_coords = ht.transform(adsorbate_coords, *transform_matrix)
        logging.info(f"New coordinates {adsorbate_coords}")
        # angle.append(thetas)
        coords.append(adsorbate_coords.tolist())
    with open("coords.txt", "w") as filehandle:
        json.dump(coords, filehandle)
    with open("energies.txt", "w") as filehandle:
        json.dump(energies, filehandle)
