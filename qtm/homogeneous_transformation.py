import numpy as np
import re
from functools import reduce

class HomogenousTransformation:

    def __init__(self, ):
        pass

    def _symbol_to_length_coords(symbol):
        """
        :return: The length of the array needed to represent its coordinates
        e.g NH3 -> 12
        H2 -> 6
        NH2 -> 9
        CaCO3 -> 15
        """
        if not symbol:
            return 0
        l = 0
        numbers = [int(num) for num in re.split(r'[a-zA-Z]', symbol) if num.isnumeric()]
        for number in numbers:
            l += 3*(number - 1)
        upper_cases = [c for c in symbol if c.isupper()]
        l += 3*len(upper_cases)
        return l

    @staticmethod
    def transform(molecules, order, coordinates, theta_x, theta_y, theta_z, t_x, t_y, t_z):
        """
        Transform the n_th molecule in the reactants
        Example: Reactants H2ONN has one molecule H2O and two Nitrogen
        The transformation would zero padding the `order`th molecule to generate
        0, 0, 0, ... (x_N1,y_N1,z_N1), 0, 0, 0 ..., 0, 0, 0
        """
        # concat all prefix/suffix symbols
        prefix_symbols = np.array([_['symbols'] for _ in molecules[:order]]).flatten()
        suffix_symbols = np.array([_['symbols'] for _ in molecules[order+1:]]).flatten()
        # prefix/suffix: coordinates of molecules before/after `order`
        prefix = HomogenousTransformation._symbol_to_length_coords(prefix_symbols)
        suffix = HomogenousTransformation._symbol_to_length_coords(suffix_symbols)

        unpadded_transformed = HomogenousTransformation._transform(coordinates[prefix:suffix], theta_x, theta_y, theta_z, t_x, t_y, t_z)
        return np.pad(unpadded_transformed, (prefix, suffix), 'constant', constant_values=[0, 0])

    @staticmethod
    def _transform(coordinates, theta_x, theta_y, theta_z, t_x, t_y, t_z):
        homogeneous = HomogenousTransformation._convert_to_homogeneous(coordinates)
        trans_matrix = HomogenousTransformation._generate_transform_matrix(theta_x, theta_y, theta_z, t_x, t_y, t_z)
        homogeneous = trans_matrix @ homogeneous
        return HomogenousTransformation._convert_to_descartes(homogeneous)

    @staticmethod
    def _generate_transform_matrix(theta_x, theta_y, theta_z, t_x, t_y, t_z):
        """
        Construct a homogenous transformation
        """
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
        homo_y = np.array(
            [

                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1]
            ]
        )

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
        result = homo_x @ homo_y @ homo_z
        result[0][3], result[1][3], result[2][3] = t_x, t_y, t_z
        return result

    @staticmethod
    def _convert_to_homogeneous(coordinates):
        assert len(coordinates) % 3 == 0
        coordinates = np.reshape(coordinates, (3, -1), 'F')
        all_ones = np.ones(coordinates.shape[1])
        return np.vstack((coordinates, all_ones))

    @staticmethod
    def _convert_to_descartes(coordinates):
        assert (coordinates[3] == 1).all()
        return np.ravel(coordinates[:3], order='F')
