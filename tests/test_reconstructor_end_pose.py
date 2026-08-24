import unittest

import numpy as np

from reconstructor import Reconstructor


class BuildEndPoseTests(unittest.TestCase):
    def setUp(self):
        self.T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.Y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.Z = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    def test_without_rotation_returns_terminal_t_y_z_axes(self):
        end_pose = Reconstructor._build_end_pose(T=self.T, Y=self.Y, Z=self.Z)

        np.testing.assert_array_equal(
            end_pose,
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
        )

    def test_explicit_rotation_remains_backward_compatible(self):
        rot_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )

        end_pose = Reconstructor._build_end_pose(
            T=self.T,
            Y=self.Y,
            Z=self.Z,
            rot_matrix=rot_matrix,
        )

        np.testing.assert_array_equal(
            end_pose,
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
