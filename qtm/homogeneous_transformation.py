import numpy as np


class HomogenousTransformation:

    def __init__(self, ):
        pass

    @staticmethod
    def transform(coordinates, theta_x, theta_y, theta_z, t_x, t_y, t_z):
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
