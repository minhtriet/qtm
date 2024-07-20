import numpy as np
import pytest

from qtm.homogeneous_transformation import HomogenousTransformation


@pytest.mark.parametrize(
    "angle_x, angle_y, angle_z, t_x, t_y, t_z, point, expected",
    [
        (0, 0, 0, 1, 2, 3, [1, 2, 3, 1], [2, 4, 6, 1]),
        (0, 0, 0, 0, 0, 0, [99, 99, 99, 1], [99, 99, 99, 1]),
        (np.pi, 0, 0, 1, 0, 0, [0, 0, 1, 1], [1, 0, -1, 1]),
        (np.pi, np.pi, np.pi, 1, 2, 3, [0, 0, 1, 1], [1, 2, 4, 1]),
        (
            np.pi * 0.5,
            0,
            0,
            0,
            0,
            0,
            [np.sqrt(2), np.sqrt(2), 0, 1],
            [np.sqrt(2), 0, np.sqrt(2), 1],
        ),
        (
            0,
            np.pi * 0.5,
            0,
            0,
            0,
            0,
            [np.sqrt(2), 0, np.sqrt(2), 1],
            [np.sqrt(2), 0, -np.sqrt(2), 1],
        ),
        (np.pi * 0.5, np.pi * 0.5, 0, 0, 0, 0, [1, 1, 0, 1], [0, 1, 1, 1]),
    ],
)
def test_translation(angle_x, angle_y, angle_z, t_x, t_y, t_z, point, expected):
    ht = HomogenousTransformation

    # Define angles for rotation
    transformation_matrix = ht._generate_transform_matrix(
        angle_x, angle_y, angle_z, t_x, t_y, t_z
    )
    transformed_point = transformation_matrix @ point

    np.testing.assert_array_almost_equal(transformed_point, expected, decimal=6)


def test_convert_to_homogeneous():
    descartes = list(range(15))
    homogeneous = [
        [0, 3, 6, 9, 12],
        [1, 4, 7, 10, 13],
        [2, 5, 8, 11, 14],
        [1, 1, 1, 1, 1],
    ]
    ht = HomogenousTransformation
    received = ht._convert_to_homogeneous(descartes)
    assert (received == homogeneous).all()


def test_convert_to_descartes():
    h_coords = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [1, 1, 1, 1],
        ]
    )
    ht = HomogenousTransformation

    received_coords = ht._convert_to_descartes(h_coords)
    expected_coords = np.array([1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12])
    assert (received_coords == expected_coords).all()


@pytest.mark.parametrize(
    "symbol, length",
    [
        ("NH3", 12),
        ("H2", 6),
        ("NH2", 9),
        ("CaCO3", 15),
    ],
)
def test_symbol_to_length_coords(symbol, length):
    ht = HomogenousTransformation
    received_length = ht._symbol_to_length_coords(symbol)
    assert received_length == length
