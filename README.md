# Quantum simulation for transition metal

## How to run
1. Run `poetry install`
2. `python qtm/run.py`, the `step_to_run` parameter in `chem_conf.yaml` below will be executed

## How to customize
We have two main configurations files. `chem_conf.yaml` is for the chemical related configuration and `ml_conf.yaml` is for the machine learning related

### `chem_conf.yaml`
This file is a YAML configuration file that uses OmegaConf logic for defining chemical simulation steps. Here's a brief explanation of its structure and content:
`catalyst` definition:
1. `catalyst` anchor is defined with symbols (8 iron atoms) and their coords (3D coordinates).
2. `steps`: Two steps are defined: `one` and `two`, corresponding to the first three steps in the reference paper
    - Step `one`:
      - `fixed`: References the catalyst structure using the YAML anchor.
      - `react`: Defines additional atoms (N and H) with their coordinates, they will be the reactant of this step
      - `config`: Specifies simulation parameters like active electrons, orbitals, charge, and basis set.
    - Step `two`:
      - `fixed`: References to the fixed coordinates of the previous steps
      - `react`: Defines two hydrogen atoms with their coordinates, they will be the reactant of this step
3. `step_to_run`: Specifies which step to execute (in this case, `two`).



