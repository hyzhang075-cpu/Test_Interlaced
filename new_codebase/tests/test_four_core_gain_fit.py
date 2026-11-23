import numpy as np
import pytest
from scipy.optimize import OptimizeResult


def test_fit_uses_source_mean_distances_and_accepts_gain_matrix():
    from shape_sensing.optimization.four_core import FourCoreGainConfig, fit_strain_gains

    initial = np.ones((2, 4))

    def predict_errors(gains):
        assert gains.shape == initial.shape
        return np.array([1.0, 5.0]), np.array([8.0])

    result = fit_strain_gains(predict_errors, initial, FourCoreGainConfig(max_nfev=1))

    assert isinstance(result, OptimizeResult)
    assert result.cost == pytest.approx(0.4 * 3.0 + 0.6 * 8.0)
    np.testing.assert_array_equal(result.gains, initial)
    assert result.x.shape == (initial.size,)


def test_fit_uses_positive_trf_bounds_for_every_core():
    from shape_sensing.optimization.four_core import FourCoreGainConfig, fit_strain_gains

    target = np.array([[0.2, 0.7, 1.15, 1.9], [0.9, 0.8, 1.2, 1.4]])
    initial = np.ones_like(target)

    def predict_errors(gains):
        distances = np.abs(gains - target)
        return distances.ravel(), distances[-1]

    result = fit_strain_gains(predict_errors, initial, FourCoreGainConfig())

    assert result.success
    np.testing.assert_allclose(
        result.gains,
        [[0.5, 0.7, 1.15, 1.5], [0.9, 0.8, 1.2, 1.4]],
        atol=2e-5,
    )
    np.testing.assert_array_equal(initial, np.ones_like(target))


def test_fit_rejects_invalid_seed_and_negative_distances():
    from shape_sensing.optimization.four_core import FourCoreGainConfig, fit_strain_gains

    config = FourCoreGainConfig()
    with pytest.raises(ValueError, match="bounds"):
        fit_strain_gains(
            lambda _: pytest.fail("infeasible seed reached callback"), [[2] * 4], config
        )
    with pytest.raises(ValueError, match="positive"):
        fit_strain_gains(lambda _: pytest.fail("invalid seed reached callback"), [[0] * 4], config)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_strain_gains(lambda _: ([-1], [1]), [[1] * 4], config)


def test_source_regularizers_use_unit_gain_and_adjacent_wl_differences():
    from shape_sensing.optimization.four_core import FourCoreGainConfig, fit_strain_gains

    initial = np.array([[1.25] * 4, [0.75] * 4])
    config = FourCoreGainConfig(max_nfev=1, ridge=0.02, smooth=0.02)
    result = fit_strain_gains(lambda _: ([0.0], [0.0]), initial, config)

    assert result.cost == pytest.approx(0.5 * 0.02 * (0.25**2 + 0.5**2))
