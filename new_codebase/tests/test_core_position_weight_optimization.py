from types import SimpleNamespace

import numpy as np
import pytest

from shape_sensing.contracts import CalibrationParameters, CurvatureConfig, OptimizationConfig
from shape_sensing.optimization.service import OptimizationService


@pytest.mark.parametrize("objective", ["geometry", "real_points"])
def test_core_position_optimizes_legacy_weight_levels_with_frozen_seed_and_no_slopes(objective):
    block = {
        "core_position": [{"wl_id": 2, "cores": {"ch1": {"r_um": 1.0}}}],
        "core_position_metadata": {"schema_version": 6},
        "rot_ang": 13.0,
        "weights": {0: [1.1, 1.0], 120: [0.8, 1.3]},
    }
    seed = CalibrationParameters({"full": block})
    dataset = SimpleNamespace(
        data={"circ": {0: {200: np.zeros((1, 4))}}},
        radii={0: {"circ": [200]}},
        design=SimpleNamespace(sampling_distance_m=np.array([0.1]), source_indices=np.array([1])),
    )

    def predict(parameters, kind, direct, ideal):
        candidate = parameters.parameters_by_region["full"]
        assert kind == "circ" and direct == 0 and not ideal
        assert "slopes" not in candidate
        assert candidate["core_position"] == block["core_position"]
        assert candidate["core_position_metadata"] == block["core_position_metadata"]
        assert candidate["rot_ang"] == 13.0
        weights = candidate["weights"][0]
        assert weights[0] == 1.1
        return weights[1], dataset.design.sampling_distance_m

    def evaluate(distance, curves, direct):
        return SimpleNamespace(
            metrics={
                "avg_point_err": {"calib": (curves["circ"] - 1.25) ** 2},
                "avg_end_err": {"calib": 0.0},
                "avg_pitch_err": 0.0,
                "avg_radius_err": 0.0,
            }
        )

    def real_curvature(values, parameters, direct, kind):
        candidate = parameters.parameters_by_region["full"]
        assert "slopes" not in candidate
        assert direct == 0 and kind == "circ"
        return np.array([5.0 * candidate["weights"][0][1] / 1.25]), np.array([0.7])

    config = OptimizationConfig(
        max_iter=2,
        freeze=(0,),
        weight_bounds=(0.5, 1.5),
        real_point_calib=objective == "real_points",
    )
    curvature = CurvatureConfig(thresholds=(5.0,), init_weight=1.0, segmentation=None)
    service = OptimizationService(
        config,
        curvature,
        dataset,
        seed,
        region="full",
        predict_shapes=predict,
        evaluate_shapes=evaluate,
        compute_curvature=real_curvature,
        score_inputs=None,
    )
    parameters, report = service.optimize(verbose=False)
    result = parameters.parameters_by_region["full"]
    assert report["objective"] == objective
    assert set(report["stages"]) == set(report["loss_definitions"]) == {"weights"}
    assert report["stages"]["weights"]["final_loss"] < 1e-12
    np.testing.assert_allclose(result["weights"][0], [1.1, 1.25])
    np.testing.assert_allclose(result["weights"][120], [0.8, 1.3])
    assert not hasattr(service, "slopes") and "slopes" not in result
    assert result["core_position"] == block["core_position"]
    assert result["core_position_metadata"] == block["core_position_metadata"]
    assert block["weights"][0] == [1.1, 1.0]


def test_core_position_weight_round_retries_and_restores_last_accepted_weights():
    from shape_sensing.optimization._kernel import OptimizationKernel

    class Search(OptimizationKernel):
        core_position_mode = True
        direct = 0
        max_iter = 2
        min_loss_diff = 1e-6

        def __init__(self):
            self.weights = {0: np.array([1.0, 1.0])}
            self.rounds = []

        def _optim_coeff_iter(self, *, i, **kwargs):
            assert "slopes_normalised" not in kwargs
            self.rounds.append(i)
            if i == 0:
                self.weights[0][:] = [0.9, 1.2]
                return 0.75, 1.0
            np.testing.assert_array_equal(self.weights[0], [0.9, 1.2])
            self.weights[0][:] = [0.5, 1.5]
            raise SystemError("unknown opcode")

    search = Search()
    loss, status, start, recoveries = search._optim_one_coeff_until_stop("weights", None, None)
    assert (loss, status, start) == (0.75, "runtime_error_fallback", 1.0)
    assert search.rounds == [0, 1, 1]
    assert recoveries[0]["outcome"] == "fallback_previous_iteration"
    np.testing.assert_array_equal(search.weights[0], [0.9, 1.2])
