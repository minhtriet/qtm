import numpy as np
import pytest

from qtm.homogenous_transformation import HomogenousTransformation


@pytest.mark.parametrize(
    "angle_x, angle_y, angle_z, t_x, t_y, t_z, point, expected",
    [
        (0, 0, 0, 1, 2, 3, [1, 2, 3, 1], [2, 4, 6, 1]),
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
    ht = HomogenousTransformation()

    # Define angles for rotation
    transformation_matrix = ht.transform(angle_x, angle_y, angle_z, t_x, t_y, t_z)
    transformed_point = transformation_matrix @ point

    np.testing.assert_array_almost_equal(transformed_point, expected, decimal=6)
