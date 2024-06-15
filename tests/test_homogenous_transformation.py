import numpy as np

from qtm.homogenous_transformation import HomogenousTransformation

def test_transformation():
    ht = HomogenousTransformation()
    # Define a vector
    vector = np.array([1, 1, 1])

    # Define angles for rotation
    angle_x = np.pi / 4  # 45 degrees
    angle_y = np.pi / 6  # 30 degrees
    angle_z = np.pi / 3  # 60 degrees

    # Define translation vector
    translation_vector = np.array([1, 2, 3])

    # Example usage
    angle_x = np.pi / 2  # 90 degrees

    transformation_matrix = ht.transform(angle_x, angle_y, angle_z, 1, 2, 3)
    point = [1, 2, 3, 1]
    transformed_point = transformation_matrix @ point
    print("Original point:", point)
    print("Transformed point:", transformed_point)

    expected_transformed_point = np.array([2, 3, 4, 1])

    np.testing.assert_array_almost_equal(transformed_point, expected_transformed_point, decimal=6)

