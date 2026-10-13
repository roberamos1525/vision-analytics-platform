"""Linear Kalman filter and the SORT constant-velocity bounding-box model."""

from __future__ import annotations

import numpy as np

from vap.errors import InvalidInputError
from vap.geometry import BBox


class KalmanFilter:
    """Textbook linear Kalman filter with Joseph-form covariance update."""

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        n = F.shape[0]
        k = H.shape[0]
        if F.shape != (n, n) or H.shape != (k, n) or Q.shape != (n, n) or R.shape != (k, k):
            raise InvalidInputError("inconsistent Kalman matrix shapes")
        if x0.shape != (n,) or P0.shape != (n, n):
            raise InvalidInputError("initial state shapes do not match the model")
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x = x0.astype(np.float64)
        self.P = P0.astype(np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z: np.ndarray) -> np.ndarray:
        y, S = self.innovation(np.asarray(z, dtype=np.float64))
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.x = self.x + K @ y
        identity = np.eye(self.P.shape[0])
        A = identity - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        return self.x

    def mahalanobis(self, z: np.ndarray) -> float:
        y, S = self.innovation(np.asarray(z, dtype=np.float64))
        return float(y @ np.linalg.solve(S, y))


def bbox_to_z(box: BBox) -> np.ndarray:
    cx, cy = box.center
    area = max(box.area, 1e-6)
    ratio = box.width / max(box.height, 1e-6)
    return np.array([cx, cy, area, ratio], dtype=np.float64)


def x_to_bbox(x: np.ndarray) -> BBox:
    area = max(float(x[2]), 1e-6)
    ratio = max(float(x[3]), 1e-6)
    w = float(np.sqrt(area * ratio))
    h = area / w if w > 0 else 0.0
    return BBox.from_cxcywh(float(x[0]), float(x[1]), w, h)


class KalmanBoxModel:
    """SORT state ``[cx, cy, s, r, vcx, vcy, vs]`` with constant velocity per frame."""

    def __init__(
        self, box: BBox, *, position_noise: float = 1.0, velocity_noise: float = 0.01
    ) -> None:
        F = np.eye(7)
        F[0, 4] = F[1, 5] = F[2, 6] = 1.0
        H = np.zeros((4, 7))
        H[:4, :4] = np.eye(4)
        R = np.diag([1.0, 1.0, 10.0, 0.01])
        P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        Q = np.diag([1.0, 1.0, 1.0, 0.0001, velocity_noise, velocity_noise, velocity_noise * 0.01])
        Q[:4, :4] *= position_noise
        x0 = np.zeros(7)
        x0[:4] = bbox_to_z(box)
        self.kf = KalmanFilter(F, H, Q, R, x0, P0)

    def predict(self) -> BBox:
        # Prevent the area from going negative when shrinking quickly.
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        return x_to_bbox(self.kf.x)

    def update(self, box: BBox) -> BBox:
        self.kf.update(bbox_to_z(box))
        return x_to_bbox(self.kf.x)

    @property
    def bbox(self) -> BBox:
        return x_to_bbox(self.kf.x)

    @property
    def velocity(self) -> tuple[float, float]:
        """Centre velocity in pixels per frame."""
        return (float(self.kf.x[4]), float(self.kf.x[5]))
