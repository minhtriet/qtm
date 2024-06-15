import numpy as np


class HomogenousTransformation:

    def __init__(self, ):
        pass

    @staticmethod
    def transform(theta_x, theta_y, theta_z, t_x, t_y, t_z):
        """
        Construct a homogenous transformation
        """
        # Create a rotation matrix for a rotation around the x-axis
        c, s = np.cos(theta_x), np.sin(theta_x)
        homo_x = np.array(
            [
                [1, 0, 0, t_x],
                [0, c, -s, 1],
                [0, s, c, 1],
                [0, 0, 0, 1],
            ]
        )

        # Create a rotation matrix for a rotation around the y-axis
        c, s = np.cos(theta_y), np.sin(theta_y)
        homo_y = np.array(
            [

                [c, 0, s, 1],
                [0, 1, 0, t_y],
                [-s, 0, c, 1],
                [0, 0, 0, 1]
            ]
        )

        # Create a rotation matrix for a rotation around the z-axis."""
        c, s = np.cos(theta_z), np.sin(theta_z)
        homo_z = np.array(
            [
                [c, -s, 0, 1],
                [s, c, 0, 1],
                [0, 0, 1, t_z],
                [0, 0, 0, 1],
            ]
        )

        return homo_x @ homo_y @ homo_z
        """Apply the transformation matrix to a vector."""
        return 1
