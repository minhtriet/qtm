import json
import logging
import os
import time
from multiprocessing import get_context
from functools import reduce
from itertools import repeat
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
from tqdm import tqdm

import qtm.chem_config as chem_config
from qtm.homogeneous_transformation import HomogenousTransformation
from qtm.itertools_helper import batched

logging.basicConfig(level = logging.INFO)

np.random.seed(17)

molecule = chem_config.NH2
active_electrons = molecule["active_electrons"]
active_orbitals = molecule["active_orbitals"]
electrons = molecule["electrons"]


def create_pyscf_representation(symbols, coords):
    coords = np.reshape(coords, (-1, 3))
    return [[symbols[i], coords[i]] for i in range(len(symbols))]


def hamiltonian_from_coords(symbols, coords):
    if coords is None:
        return None, None
    base_coords = np.hstack([chem_config.Fe["coords"], *[x['coords'] for x in chem_config.step2_fix]])
    base_symbols = np.hstack([chem_config.Fe["symbols"], *[x['symbols'] for x in chem_config.step2_fix]]).tolist()
    coordinates = np.append(base_coords, coords)

    H, qubits = qchem.molecular_hamiltonian(
        base_symbols +  symbols,
        coordinates,
        method="openfermion",
        active_electrons=active_electrons + 1, # +1 for 3N
        active_orbitals=active_orbitals + 1, # +1 for 3N
        mult=1 + molecule["unpaired_e"] + 3,  # +3 for 3N
    )
    return H, qubits


def run_circuit(H, params=None, init_state=None):
    if H is None:
        return 0
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


def prepare_H(symbols, coords):
    H, qubits = hamiltonian_from_coords(symbols, coords)
    n_qubits = len(H.wires)
    singles, doubles = qchem.excitations(active_electrons, n_qubits)
    return H, n_qubits, singles, doubles


opt_theta = qml.GradientDescentOptimizer(stepsize=0.4)


def finite_diff(hamiltonians, theta, state, delta_theta=0.01, delta_xyz=0.01):
    """Compute the central-difference finite difference of a function
    hs: all the Hamiltonians of the molecules
    x: coordinates, thetas is the rotational angles
    """
    batched_hs = list(batched(hamiltonians, 6*2))
    molecular_grads = []
    for hs in batched_hs:
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
        molecular_grads.extend([theta_x_grad, theta_y_grad, theta_z_grad, x_grad, y_grad, z_grad])

    return np.array(molecular_grads)


# #### Optimize
if __name__ == "__main__":
    delta_angle = np.pi / 90
    delta_coord = 0.1
    max_iterations = 70
    lr = 1e-4
    ht = HomogenousTransformation()
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
    molecules = chem_config.step2
    adsorbate_coords = reduce(lambda x,y: x+y, [x['coords'] for x in molecules])
    symbols = reduce(lambda x, y: x+y, [x['symbols'] for x in molecules])

    logging.info("== Preparing molecule first run")
    _, __, singles, doubles = prepare_H(symbols, adsorbate_coords)
    total_single_double_gates = len(singles) + len(doubles)
    logging.info(f"New coordinates {adsorbate_coords}")

    # store the values of the cost function
    thetas = np.random.normal(0, np.pi, total_single_double_gates)

    # store the values of the circuit parameter
    angle = []
    coords = []
    energies = []

    for _ in tqdm(range(max_iterations)):
        [os.remove(hdf5) for hdf5 in os.listdir(".") if hdf5.endswith(".hdf5")]
        # Optimize the circuit parameters
        start = time.time()
        H, _ = hamiltonian_from_coords(symbols, adsorbate_coords)
        # fixme now using eigen values, but later use theta for Double/Single excitation
        value, state = np.linalg.eig(qml.matrix(H))
        smallest_i = np.argmin(value)
        g_energy, g_state = value[smallest_i], state[smallest_i]
        g_state /= np.linalg.norm(g_state)
        energies.append(float(run_circuit(H, init_state=g_state)))

        # early stopping
        if len(energies) > 2 and np.abs(energies[-1] - energies[-2]) < 1e-5:
            break
        logging.info(f"Done theta, starting coordinates {time.time()- start}")

        # Optimize the nuclear coordinates
        thetas.requires_grad = False
        # all possible transformations
        shifted_coords = [ht.transform(molecules, i, adsorbate_coords, *transformation) for i, molecule in enumerate(molecules) for transformation in transformations]
        start = time.time()
        with get_context("spawn").Pool(os.cpu_count()-4) as p:
            hs = p.starmap(hamiltonian_from_coords, zip(repeat(symbols), shifted_coords))
            # Each hs[i] contains coordinates and the corresponding H
            logging.info(f"Energy level {run_circuit(hs[0][0], init_state=g_state)}")

        logging.info("== Calculate the gradients for coordinates")
        assert len(hs) == 2*6*len(molecules)  # each molecule has 6 dof. Need two more each for gradients
        grad_x = finite_diff([h[0] for h in hs], thetas, g_state, delta_angle)
        logging.info(f"gradients {grad_x}")
        transform_params = np.zeros(len(grad_x))
        transform_params -= lr * grad_x

        logging.info("== Transforming the coordinates")
        new_coords = []
        for i in range(len(molecules)):
            new_coords.extend(ht.transform(molecules, i, adsorbate_coords, *transform_params[6*i:6*(i+1)], pad=False))
        logging.info(f"New coordinates {adsorbate_coords}")
        # angle.append(thetas)
        coords.append(new_coords)
        adsorbate_coords = new_coords
        logging.info(f"All coords: {coords}")
        logging.info(f"All energies: {energies}")
        with open("coords.txt", "w") as filehandle:
            json.dump(coords, filehandle)
        with open("energies.txt", "w") as filehandle:
            json.dump(energies, filehandle)
