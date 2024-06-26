# N: 1s2 2s2 2p3
NH2 = {
    "electrons": 1 + 1 + 7,  # H + H + N
    "orbitals": 1 + 1 + 1 + 1 + 3,  # (1s1) + (1s1) + (1s2 + 2s2 + 2p3)
    "active_electrons": 1 + 1 + 3,  # (1s1) + (1s1) + (2p3)
    "active_orbitals": 1 + 1 + 3,  # (1s1) + (1s1) + (2p3)
    "unpaired_e": 1,
    "coords": [
        # 3N_NH2. NH2 is a radical, short-lived https://pubchem.ncbi.nlm.nih.gov/compound/123329#section=2D-Structure
        2.5369-1.5, -0.1550+3, 0.0000+0.5,   # nh2_n
        3.0739-1.5,  0.1550+3, 0.0000+0.5,    # nh2_h1
        2.0000-1.5,  0.1550+3, 0.0000+0.5     # nh2_h2
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
        2.5369, -0.1550, 0.0000,  # nh_n
        3.0739, 0.1550, 0.0000,  # nh_h1
    ],
    "symbols": ["N", "H"]
}

# todo make 3d
H2 = {
    "electrons": 1 + 1,  # H + H
    "orbitals": 1 + 1,  # (1s1) + (1s1)
    "active_electrons": 1 + 1,  # (1s1) + (1s1)
    "active_orbitals": 1 + 1,  # (1s1) + (1s1)
    "unpaired_e": 0,
    "coords": [0, 0, -1.5,
               0, 0, 1.5],
    "symbols": ["H", "H"],
}

# catalyst
# slab = fcc211('Fe', (3, 3, 1), a=2.586)
Fe = {
    "coords": [2.98605559e+00, 9.14289068e-01, 1.05573008e+00,
               2.98605559e+00, 2.74286720e+00, 1.05573008e+00,
               2.98605559e+00, 4.57144534e+00, 1.05573008e+00,
               1.49302780e+00, 1.82857814e+00, 5.27865040e-01,
               1.49302780e+00, 3.65715627e+00, 5.27865040e-01,
               0, 9.14289068e-01, 0,
               0, 2.74286720e+00, 0,
               0, 4.57144534e+00, 0],
    "symbols": ["Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "Fe"]
}

fe_top = [1.0, 0.0, 0.50]
fe_bottom = [0.69, 0.14, 0.36]
fe_climbing = [0.63, 0.58, 0.44]
fe_bridge = [0.7, 1.0, 0.44]
fe_trough = [0.59, 0.5, 0.31]
