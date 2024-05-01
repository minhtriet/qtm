#!/usr/bin/env python
# coding: utf-8

# In[7]:


#from pyscf import gto, scf, ci
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem 


# In[3]:


symbols = ["Fe"]
coordinates = np.array([0.0, 0.0, 0.0])
H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, method='pyscf')


# In[8]:


qubits


# In[8]:


dataset_N2 = qml.data.load("qchem", molname="N2", bondlength=1.12, basis="STO-3G")
dataset_N2[0]


# In[9]:


dataset_N2[0].hamiltonian.wires

