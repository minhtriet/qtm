import copy
import logging
from qtm.homogeneous_transformation import HomogenousTransformation
import json

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

# ## Structure generation

# In[1]:

import time
import os
from multiprocessing import get_context

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

from ase import Atoms

from tqdm import tqdm

import qtm.chem_config as chem_config

# According to [Fig 1. in this paper](https://pubs.rsc.org/en/content/getauthorversionpdf/c9cp01611b), the config of $Fe$ lattice is
#
# > Fe coordinate
# > - the top site corresponds to (1.0, 0.0, 0.50)
# > - the bottom site corresponds to (0.69, 0.14, 0.36)
# > - the climbing site corresponds to (0.63, 0.58, 0.44)
# > - the bridge site corresponds to (0.7, 1.0, 0.44)
# > - the trough site corresponds to (0.59, 0.50, 0.31)
# >
# > Binding preference location
# > - H prefers the bridge (0.71 eV) and top sites (0.62-0.70eV),
# > - N prefers the bottom sites (1.06 eV) and the trough sites (1.53 eV),
# > - NH prefers the bottom site with the H opposite the first layer Fe atoms to minimize vdW repulsion (0.92 eV),
# > - NH2 prefers the climbing site (0.70 eV), and
# > - NH3 prefers the top site (0.49 eV)
#
# Always 4 $N$ in Fig 3


# in case we have to load from SDMol
# sd_supplier = SDMolSupplier("Structure2D_COMPOUND_CID_123329.sdf")
#
# for mol in sd_supplier:
#     logging.info(rdmolfiles.MolToXYZBlock(mol))


np.random.seed(17)

# inside NH3
n2_x = np.random.uniform(0, 1.1, [2])  # one value for N2

h2_x = np.random.uniform(0, 1.1, [6])  # three values for H2
h2_y = np.random.uniform(-0.1, 1.1, [6])
h2_z = np.random.uniform(0.2, 0.6, [6])

fe_top = [1.0, 0.0, 0.50]
fe_bottom = [0.69, 0.14, 0.36]
fe_climbing = [0.63, 0.58, 0.44]
fe_bridge = [0.7, 1.0, 0.44]
fe_trough = [0.59, 0.5, 0.31]


molecule = chem_config.NH2
active_electrons = molecule["active_electrons"]
active_orbitals = molecule["active_orbitals"]
electrons = molecule["electrons"]
orbitals = molecule["orbitals"]
# Let's visualize!

fe_lattice = Atoms(
    f"FeFeFeFeFe{''.join(molecule['symbols'])}",
    np.concatenate(
        [
            [fe_top, fe_bottom, fe_climbing, fe_bridge, fe_trough],
            np.reshape(molecule["coords"], (-1, 3)),
        ],
    ),
)


# # ## ! New define Hamitonian

# Fe:
# - $4s2:\uparrow \downarrow$
# - $3d6:\uparrow \downarrow,\uparrow,\uparrow,\uparrow,\uparrow $
#
# Therefore there are four active oribitals
#
# > Defining a chemically meaningful active space depends on the problem and usually requires some experience.
# [Source](https://discuss.pennylane.ai/t/co2-active-electrons-orbitals/1589/2)

# NH2 and Fe


def create_pyscf_representation(symbols, coords):
    coords = np.reshape(coords, (-1, 3))
    return [[symbols[i], coords[i]] for i in range(len(symbols))]


symbols = ["Fe", "Fe", "Fe", "Fe", "Fe"] + molecule["symbols"]


def hamiltonian_from_coords(coords):
    base_coords = fe_top + fe_bottom + fe_climbing + fe_bridge + fe_trough
    coordinates = np.append(base_coords, coords)
    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        method="openfermion",
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mult=1 + molecule["unpaired_e"],
    )
    return H, qubits


# ### Geometry optimization
#
# I am planning to use a pure quantum method rather than the classical way.
# - Build the parametrized electronic Hamiltonian H(x) of the molecule.
# - Design the variational quantum circuit to prepare the electronic trial state $|\Psi(\theta)⟩$
# - Define the cost function $⟨\Psi(\theta)|H|\Psi(\theta)⟩$
# - Optimize for $\theta$


# #### Create circuit
def run_circuit(H, params=None, init_state=None):
    dev = qml.device("lightning.qubit", wires=len(H.wires))

    @qml.qnode(dev)
    def circuit_theta():
        state = qchem.hf_state(active_electrons, len(H.wires))
        singles, doubles = qchem.excitations(active_electrons, len(H.wires))
        qml.AllSinglesDoubles(
            hf_state=state, weights=params, wires=H.wires, singles=singles, doubles=doubles
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
    theta_x_grad = (run_circuit(hs[0], init_state=state) - run_circuit(hs[1], init_state=state)) * (0.5*delta_theta**-1)
    theta_y_grad = (run_circuit(hs[2], init_state=state) - run_circuit(hs[3], init_state=state)) * (0.5*delta_theta**-1)
    theta_z_grad = (run_circuit(hs[4], init_state=state) - run_circuit(hs[5], init_state=state)) * (0.5*delta_theta**-1)
    x_grad = (run_circuit(hs[6], init_state=state) - run_circuit(hs[7], init_state=state)) * (0.5*delta_xyz**-1)
    y_grad = (run_circuit(hs[8], init_state=state) - run_circuit(hs[9], init_state=state)) * (0.5*delta_xyz**-1)
    z_grad = (run_circuit(hs[10], init_state=state) - run_circuit(hs[11], init_state=state)) * (0.5*delta_xyz**-1)

    return np.array([theta_x_grad, theta_y_grad, theta_z_grad, x_grad, y_grad, z_grad])


def loss_f(thetas, coords):
    H, _ = hamiltonian_from_coords(coords)
    return run_circuit(H, thetas)


# #### Optimize
if __name__ == "__main__":
    # prepare for the 1st run
    adsorbate_coords = np.array(molecule["coords"])
    _, __, singles, doubles = prepare_H(adsorbate_coords)
    total_single_double_gates = len(singles) + len(doubles)
    lr = 5e-2
    logging.info(f"New coordinates {adsorbate_coords}")

    # store the values of the cost function
    thetas = np.random.normal(0, np.pi, total_single_double_gates)
    max_iterations = 100
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
        if len(energies) > 2 and np.abs(energies[-1]-energies[-2]) < 1e-5:
            break
        # thetas, _ = opt_theta.step(loss_f, thetas, adsorbate_coords)
        logging.info(f"Done theta, starting coordinates {time.time()- start}")

        # Optimize the nuclear coordinates
        # adsorbate_coords.requires_grad = True
        ht = HomogenousTransformation()
        thetas.requires_grad = False
        delta_angle = np.pi / 180
        delta_coord = 0.1
        # all possible transformations
        transformations = [[delta_angle, 0, 0, 0, 0, 0],
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
                          [0, 0, 0, 0, 0, -delta_coord]]
        shifted_coords = [(ht.transform(adsorbate_coords, *transformation)) for transformation in transformations]
            
        # .unique()   # unique to reduce the rotation for single atoms, since it doesn't change
        start = time.time()
        with get_context("spawn").Pool() as p:
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
    with open('coords.txt', 'w') as filehandle:
        json.dump(coords, filehandle)
    with open('energies.txt', 'w') as filehandle:
        json.dump(energies, filehandle)
