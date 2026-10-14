"""
Module 4: Kalman Box Filter
8-state Kalman Filter for bounding box tracking:
State: [cx, cy, a, h, v_cx, v_cy, v_a, v_h]
Standard constant-velocity motion model used in ByteTrack / SORT.
"""

from typing import Tuple
import numpy as np


class KalmanBoxFilter:
    """
    Standard constant-velocity Kalman filter for 2D bounding boxes in image plane.
    State vector: [cx, cy, a, h, vx, vy, va, vh]
    - (cx, cy): box center
    - a: aspect ratio (w / h)
    - h: box height
    - (vx, vy, va, vh): respective velocities
    """

    def __init__(self):
        ndim = 4
        dt = 1.0

        # Motion matrix F (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Update / Measurement matrix H (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Standard deviation weights for state uncertainties
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initializes track state from an unobserved measurement (cx, cy, a, h).

        Args:
            measurement: [cx, cy, a, h] array.

        Returns:
            mean: (8,) state vector.
            covariance: (8, 8) state covariance matrix.
        """
        mean_pos = measurement.astype(np.float32)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2.0 * self._std_weight_position * measurement[3],
            2.0 * self._std_weight_position * measurement[3],
            1e-2,
            2.0 * self._std_weight_position * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            1e-5,
            10.0 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Kalman filter prediction step to project state forward one time step.

        Args:
            mean: (8,) current state mean.
            covariance: (8, 8) current state covariance.

        Returns:
            predicted_mean: (8,) predicted state mean.
            predicted_covariance: (8, 8) predicted state covariance.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        predicted_mean = np.dot(self._motion_mat, mean)
        predicted_cov = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov

        return predicted_mean, predicted_cov

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects state distribution to measurement space.

        Args:
            mean: (8,) state mean.
            covariance: (8, 8) state covariance.

        Returns:
            proj_mean: (4,) projected measurement mean.
            proj_cov: (4, 4) projected measurement covariance.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        proj_mean = np.dot(self._update_mat, mean)
        proj_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + innovation_cov

        return proj_mean, proj_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Kalman filter correction step using an observed measurement.

        Args:
            mean: (8,) predicted state mean.
            covariance: (8, 8) predicted state covariance.
            measurement: (4,) observed measurement [cx, cy, a, h].

        Returns:
            updated_mean: (8,) corrected state mean.
            updated_covariance: (8, 8) corrected state covariance.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain K = P * H^T * S^-1
        kalman_gain = np.linalg.solve(
            projected_cov,
            np.dot(covariance, self._update_mat.T).T
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))

        return new_mean, new_covariance
