"""Positive, per-WL strain gains for the four-core common-response model."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FourCoreGainConfig:
    """TRF settings for the four-core objective.

    Distances returned by the predictor share one length unit. The ridge and
    smoothness coefficients are in that same unit. A zero evaluation limit
    runs to convergence, and None uses SciPy's default evaluation limit.
    """

    weight_bounds: tuple = (0.5, 1.5)
    max_nfev: int = None
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    trajectory_weight: float = 0.4
    endpoint_weight: float = 0.6
    ridge: float = 0.0
    smooth: float = 0.0


def _distances(values, label):
    distances = np.asarray(values, dtype=float)
    if distances.ndim != 1 or not distances.size:
        raise ValueError(f"{label} distances must be a nonempty 1-D array")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError(f"{label} distances must be finite and nonnegative")
    return distances


def fit_strain_gains(predict_errors, initial, config=FourCoreGainConfig()):
    """Fit gains using the weighted mean point and endpoint distances.

    The predictor receives the (number of WLs, 4) gain matrix and returns
    (path point distances, endpoint distances) as two nonempty 1-D arrays
    with fixed lengths. SciPy's OptimizeResult keeps its flattened x and
    also exposes the fitted matrix as gains.
    """
    initial = np.array(initial, dtype=float, copy=True)
    if (
        initial.ndim != 2
        or initial.shape[0] == 0
        or initial.shape[1] != 4
        or not np.isfinite(initial).all()
        or np.any(initial <= 0)
    ):
        raise ValueError("Initial gains must be a finite positive (number of WLs, 4) matrix")

    bounds = np.asarray(config.weight_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError("Gain bounds must be finite, positive, and strictly increasing")
    if np.any(initial < bounds[0]) or np.any(initial > bounds[1]):
        raise ValueError("Initial gains must lie inside the current bounds")

    tolerances = (config.ftol, config.xtol, config.gtol)
    if not all(np.isfinite(t) and t > np.finfo(float).eps for t in tolerances):
        raise ValueError("TRF tolerances must be finite and greater than machine epsilon")
    max_nfev = config.max_nfev
    if max_nfev is not None and (
        isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 0
    ):
        raise ValueError("max_nfev must be None or a nonnegative integer")

    weights = np.array([config.trajectory_weight, config.endpoint_weight], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Trajectory and endpoint weights must be nonnegative and sum to 1")
    penalties = np.array([config.ridge, config.smooth], dtype=float)
    if not np.isfinite(penalties).all() or np.any(penalties < 0):
        raise ValueError("Ridge and smoothness penalties must be finite and nonnegative")

    shape = initial.shape
    ridge_scale = np.sqrt(config.ridge / initial.size)
    smooth_scale = np.sqrt(config.smooth / max(1, initial[1:].size))
    distance_sizes = None

    def residual(flat):
        nonlocal distance_sizes
        gains = flat.reshape(shape)
        path, endpoint = predict_errors(gains)
        path = _distances(path, "Path")
        endpoint = _distances(endpoint, "Endpoint")
        sizes = (path.size, endpoint.size)
        if distance_sizes is None:
            distance_sizes = sizes
        elif sizes != distance_sizes:
            raise ValueError("Predictor must return a fixed number of path and endpoint distances")

        # SciPy's cost is half the squared residual norm. These residuals
        # give weighted means of Euclidean distance rather than squared distance.
        return np.concatenate(
            (
                np.sqrt(path * (2 * weights[0] / path.size)),
                np.sqrt(endpoint * (2 * weights[1] / endpoint.size)),
                ridge_scale * (flat - 1),
                smooth_scale * np.diff(gains, axis=0).ravel(),
            )
        )

    result = least_squares(
        residual,
        initial.ravel(),
        bounds=tuple(bounds),
        method="trf",
        max_nfev=np.inf if max_nfev == 0 else max_nfev,
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
    )
    result.gains = result.x.reshape(shape).copy()
    return result
