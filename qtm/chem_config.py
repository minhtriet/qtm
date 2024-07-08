# step1
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

_N1_step1 = {
    "coords": [2,1,2],
    "symbols": ["N"]
}

_N2_step1 = {
    "coords": [2,2,2],
    "symbols": ["N"]
}

_N3_step1 = {
    "coords": [2, 3, 2],
    "symbols": ["N"]
}

# step2, position are copied from step1's result (artifact/step1/coords.txt)
step2 = [
     { # NH2
        "electrons": 1 + 1 + 7,  # H + H + N
        "orbitals": 1 + 1 + 1 + 1 + 3,  # (1s1) + (1s1) + (1s2 + 2s2 + 2p3)
        "active_electrons": 1 + 1 + 3,  # (1s1) + (1s1) + (2p3)
        "active_orbitals": 1 + 1 + 3,  # (1s1) + (1s1) + (2p3)
        "unpaired_e": 1,
        "coords": [
            -0.21998835018675736, 2.432765666922257, 2.136527664404811,
            0.25380019628872463, 2.8006621861643435, 2.2935167224891395,
            -0.7975590295362942, 2.657918869630763, 2.1275970616360285,
        ],
        "symbols": ["N", "H", "H"]
    },
    {  # N
        "coords": [1.757634721691183, 0.35316930601879926, 2.4381569021834557],
        "symbols": ["N"]
    },
    {  # N
        "coords": [1.925108101630958, 1.9323910501924726, 2.6599059859169203],
        "symbols": ["N"]
    },
    {  # N
        "coords": [1.953091237622412, 3.5147644488263454, 2.6802743012682475],
        "symbols": ["N"]
    },
    {  # H
        "coords": [1, 1, 2],
        "symbols": ["H"]
    },
    {  # H
        "coords": [2, 4, 2],
        "symbols": ["H"]
    },
]

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

