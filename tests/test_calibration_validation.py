from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_validation import (
    boundary_diagnostics,
    compute_curve_metrics,
    straight_prefix_circular_tail,
)


def test_curve_metrics_include_tip_tangent_and_pose_error():
    ideal = np.column_stack((np.linspace(0, .1, 6), np.zeros((6, 2))))
    result = ideal.copy()
    result[:, 1] += np.linspace(0, .002, 6)
    metrics = compute_curve_metrics(
        result, ideal, result_tangent=[1, .1, 0], ideal_tangent=[1, 0, 0],
        distances=np.linspace(0, .1, 6), evaluation_start_m=.06,
    )
    assert metrics['mean_point_error_mm'] > 0
    assert metrics['tip_position_error_mm'] == pytest.approx(2.0)
    assert metrics['tip_tangent_error_deg'] > 0
    assert metrics['pose_error_mm'] > metrics['tip_position_error_mm']
    assert metrics['evaluated_region_point_error_mm'] > metrics['mean_point_error_mm']


def test_tail_ideal_is_continuous_at_region_start():
    distances = np.array([0, .05, .1, .15, .2])
    curve, tangent = straight_prefix_circular_tail(distances, .1, .03)
    np.testing.assert_allclose(curve[2], [.1, 0, 0])
    np.testing.assert_allclose(tangent, [np.cos(.1/.03), np.sin(.1/.03), 0])
    assert np.all(np.isfinite(curve))


def test_boundary_diagnostics_reports_each_region_transition():
    diagnostics = {
        'final_weight': [1, 1, 1, 2, 2, 3],
        'slope_map': [10, 10, 10, 20, 20, 30],
        'base_magnitude': [0, 1, 2, 3, 4, 5],
        'final_magnitude': [0, 1, 2, 4, 5, 7],
        'final_angle': [0, 0, 0, .1, .1, .3],
        'final_theta_dynamic': [[0, 0, 0, .2, .2, .4]],
    }
    result = boundary_diagnostics(diagnostics, [3, 5])
    assert list(result) == ['3->4', '5->6']
    assert result['3->4']['weight_jump'] == 1
    assert result['5->6']['slope_jump'] == 10
    assert result['5->6']['curvature_vector_jump_m_inv'] > 0
