# N: 1s2 2s2 2p3
NH2 = {
    "electrons": 1 + 1 + 7,  # H + H + N
    "orbitals": 1 + 1 + 1 + 1 + 3,  # (1s1) + (1s1) + (1s2 + 2s2 + 2p3)
    "active_electrons": 1 + 1 + 3,  # (1s1) + (1s1) + (2p3)
    "active_orbitals": 1 + 1 + 3,  # (1s1) + (1s1) + (2p3)
    "unpaired_e": 1,
    "coords": [
        # 3N_NH2. NH2 is a radical, short lived https://pubchem.ncbi.nlm.nih.gov/compound/123329#section=2D-Structure
        2.5369, -0.1550, 0.0000,   # nh2_n
        3.0739, 0.1550, 0.0000,    # nh2_h1
        2.0000, 0.1550, 0.0000 # nh2_h2
    ],
    "symbols": ["N", "H", "H"]
}

NH = {   # todo not finished
    "electrons": 1 + 7,  # H + N
    "orbitals": 1 + 1 + 1 + 3,  # (1s1) + (1s2 + 2s2 + 2p3)
    "active_electrons": 1 + 3,  # (1s1) + (1s1) + (2p3)
    "active_orbitals": 1 + 1 + 3,  # (1s1) + (1s1) + (2p3)
    "unpaired_e": 1,
    "coords": [
        # 3N_NH2. NH2 is a radical, short lived https://pubchem.ncbi.nlm.nih.gov/compound/123329#section=2D-Structure
        2.5369, -0.1550, 0.0000,  # nh_n
        3.0739, 0.1550, 0.0000,  # nh_h1
    ],
    "symbols": ["N", "H"]
}

H2 = {
    "electrons": 1 + 1,  # H + H
    "orbitals": 1 + 1,  # (1s1) + (1s1)
    "active_electrons": 1 + 1,  # (1s1) + (1s1)
    "active_orbitals": 1 + 1,  # (1s1) + (1s1)
    "unpaired_e": 0,
    "coords": [0,0,-1.5,
               0,0,1.5],
    "symbols": ["H", "H"]
}