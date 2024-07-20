import re

import numpy as np


class HomogenousTransformation:
    def __init__(
        self,
    ):
        pass

    def __int__(self, delta_angle, delta_coord):
        self.transformations = [
            [delta_angle, 0, 0, 0, 0, 0],
            [-delta_angle, 0, 0, 0, 0, 0],
            [0, delta_angle, 0, 0, 0, 0],
            [0, -delta_angle, 0, 0, 0, 0],
            [0, 0, delta_angle, 0, 0, 0],
            [0, 0, -delta_angle, 0, 0, 0],
            # the angle part is not needed for the single atom
            [0, 0, 0, delta_coord, 0, 0],
            [0, 0, 0, -delta_coord, 0, 0],
            [0, 0, 0, 0, delta_coord, 0],
            [0, 0, 0, 0, -delta_coord, 0],
            [0, 0, 0, 0, 0, delta_coord],
            [0, 0, 0, 0, 0, -delta_coord],
        ]

    @staticmethod
    def _symbol_to_length_coords(symbol):
        """
        :return: The length of the array needed to represent its coordinates
        e.g NH3 -> 12
        H2 -> 6
        NH2 -> 9
        CaCO3 -> 15
        """
        if isinstance(symbol, list) or isinstance(symbol, np.ndarray):
            return 3 * len(symbol)
        if isinstance(symbol, str):
            l = 0
            numbers = [
                int(num) for num in re.split(r"[a-zA-Z]", symbol) if num.isnumeric()
            ]
            for number in numbers:
                l += 3 * (number - 1)
            upper_cases = [c for c in symbol if c.isupper()]
            l += 3 * len(upper_cases)
            return l

    @staticmethod
    def mass_transform(molecules, coordinates, transform_params):
        """
        :param molecules:
        :param coordinates:
        :param transform_params:
        :return:
        """
        new_coords = []
        for i, molecule in range(len(molecules)):
            new_coords.append(
                HomogenousTransformation.transform(
                    molecules, i, coordinates, *transform_params[i * 6 : (i + 1) * 6]
                )
            )
        return new_coords

    @staticmethod
    def transform(
        molecules,
        order,
        coordinates,
        theta_x,
        theta_y,
        theta_z,
        t_x,
        t_y,
        t_z,
        pad=True,
    ):
        """
        Transform the n_th molecule in the reactants, used in gradient descent
        Example: Reactants H2ONN has one molecule H2O and two Nitrogen
        The transformation would zero padding the `order`th molecule to generate
        0, 0, 0, ... (x_N1,y_N1,z_N1), 0, 0, 0 ..., 0, 0, 0

        If the `order`th molecule has just one atom, then skip the rotation and return None.
        None will be used in calculations of the Hamiltionian
        """
        if (len(molecules[order]["symbols"]) == 1) and ((t_x, t_y, t_z) == (0, 0, 0)):
            return None
        # concat all prefix/suffix symbols
        prefix_symbols = (
            np.hstack([_["symbols"] for _ in molecules[:order]]) if order > 0 else ""
        )
        suffix_symbols = (
            np.hstack([_["symbols"] for _ in molecules[order + 1 :]])
            if order < len(molecules) - 1
            else ""
        )
        # prefix/suffix: coordinates of molecules before/after `order`
        prefix = HomogenousTransformation._symbol_to_length_coords(prefix_symbols)
        suffix = HomogenousTransformation._symbol_to_length_coords(suffix_symbols)

        unpadded_transformed = HomogenousTransformation._transform_a_molecule(
            coordinates[prefix : None if suffix == 0 else -suffix],
            theta_x,
            theta_y,
            theta_z,
            t_x,
            t_y,
            t_z,
        )
        if pad:
            result = np.hstack(
                [
                    coordinates[:prefix],
                    unpadded_transformed,
                    coordinates[prefix + len(unpadded_transformed) :],
                ]
            )
            assert len(result) == len(coordinates)
            return result
        return unpadded_transformed

    @staticmethod
    def _transform_a_molecule(coordinates, theta_x, theta_y, theta_z, t_x, t_y, t_z):
        # transform a single atom/molecule with the given coordinates and rotation angles
        homogeneous = HomogenousTransformation._convert_to_homogeneous(coordinates)
        trans_matrix = HomogenousTransformation._generate_transform_matrix(
            theta_x, theta_y, theta_z, t_x, t_y, t_z
        )
        homogeneous = trans_matrix @ homogeneous
        return HomogenousTransformation._convert_to_descartes(homogeneous)

    @staticmethod
    def _generate_transform_matrix(
        theta_x, theta_y, theta_z, t_x, t_y, t_z, rotation_origin=(0, 0, 0)
    ):
        """
        Construct a homogenous transformation
        """
        # Bring the away to the origin

        # Create a rotation matrix for a rotation around the x-axis
        c, s = np.cos(theta_x), np.sin(theta_x)
        homo_x = np.array(
            [
                [1, 0, 0, 0],
                [0, c, -s, 0],
                [0, s, c, 0],
                [0, 0, 0, 1],
            ]
        )

        # Create a rotation matrix for a rotation around the y-axis
        c, s = np.cos(theta_y), np.sin(theta_y)
        homo_y = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

        # Create a rotation matrix for a rotation around the z-axis."""
        c, s = np.cos(theta_z), np.sin(theta_z)
        homo_z = np.array(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        result = homo_x @ homo_y @ homo_z  # result is purely a rotation matrix
        # https://math.stackexchange.com/a/4397766
        move_to_coordinate_for_rotation = [*rotation_origin, 1] - result @ [
            *rotation_origin,
            1,
        ]
        result[0][3] = (
            move_to_coordinate_for_rotation[0] + t_x
        )  # result is now affine transformation matrix
        result[1][3] = move_to_coordinate_for_rotation[1] + t_y
        result[2][3] = move_to_coordinate_for_rotation[2] + t_z
        return result

    @staticmethod
    def _convert_to_homogeneous(coordinates):
        assert len(coordinates) % 3 == 0
        coordinates = np.reshape(coordinates, (3, -1), "F")
        all_ones = np.ones(coordinates.shape[1])
        return np.vstack((coordinates, all_ones))

    @staticmethod
    def _convert_to_descartes(coordinates):
        assert (coordinates[3] == 1).all()
        return np.ravel(coordinates[:3], order="F")
