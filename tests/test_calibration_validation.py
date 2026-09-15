from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration_validation import (
    build_phase_transport_evidence,
    boundary_diagnostics,
    circular_angle_error_deg,
    compute_curve_metrics,
    integrate_phase_to_point,
    straight_prefix_circular_tail,
    validate_score_evidence,
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


def test_circular_phase_transport_wraps_and_end5cm_is_non_blocking():
    assert circular_angle_error_deg(np.deg2rad(179), np.deg2rad(-179)) == pytest.approx(2)
    distances = [0.0, 0.1, 0.2, 0.3]
    slopes = [0.0, 1.0, 2.0, 3.0]
    phase_at_two = integrate_phase_to_point(0.5, distances, slopes, 2)
    assert phase_at_two == pytest.approx(0.8)

    evidence = build_phase_transport_evidence(
        0.5, distances, slopes,
        local_phases={'end15cm': 0.8 + np.deg2rad(29.9),
                      'end5cm': 1.1 + np.deg2rad(30.1)},
        start_indices={'end15cm': 2, 'end5cm': 3}, tolerance_deg=30,
    )
    assert evidence['end15cm']['within_30_deg'] is True
    assert evidence['end15cm']['blocking'] is True
    assert evidence['end5cm']['within_30_deg'] is False
    assert evidence['end5cm']['blocking'] is False


def _score_package():
    return {
        'sensor_id': 'sensor-A',
        'regions': {
            'full': {'thresholds': [1.0, 2.0]},
            'end15cm': {'thresholds': [1.0, 3.0]},
            'end5cm': {'thresholds': [1.0, 3.0, 45.0]},
        },
    }


def _score_report():
    def region(thresholds, values):
        return {
            'status': 'complete',
            'score_distribution': {
                'thresholds': thresholds,
                'sigmoid_10_90_width_m_inv': .9,
                'samples': [{
                    'keys': [15],
                    'score_magnitude': values,
                    'quantiles': {'min': min(values), 'median': 2.0,
                                  'p90': max(values), 'max': max(values)},
                }],
            },
        }
    return {
        'sensor_id': 'sensor-A', 'task': 'score', 'direct': 0,
        'regions': {
            'full': region([1.0, 2.0], [0.0, 3.0]),
            'end15cm': region([1.0, 3.0], [0.0, 4.0]),
            'end5cm': region([1.0, 3.0, 45.0], [0.0, 50.0]),
        },
    }


def test_score_evidence_is_shared_and_requires_empirical_threshold_crossing():
    evidence = validate_score_evidence(_score_package(), _score_report(), 'score/results.json')
    assert evidence['score_thresholds_verified'] is True
    assert evidence['source_path'] == 'score/results.json'
    assert evidence['regions']['end5cm']['observed_max_m_inv'] == 50.0

    report = _score_report()
    report['regions']['end5cm']['score_distribution']['samples'][0]['score_magnitude'] = [0, 40]
    failed = validate_score_evidence(_score_package(), report, 'score/results.json')
    assert failed['score_thresholds_verified'] is False
    assert any('45.0' in reason for reason in failed['failures'])


@pytest.mark.parametrize('mutation, match', [
    ('sensor', 'sensor_id'),
    ('threshold', 'thresholds'),
    ('missing', 'end15cm'),
    ('nonfinite', '非有限'),
])
def test_score_evidence_rejects_incompatible_reports(mutation, match):
    report = _score_report()
    if mutation == 'sensor':
        report['sensor_id'] = 'sensor-B'
    elif mutation == 'threshold':
        report['regions']['full']['score_distribution']['thresholds'] = [1.0, 2.5]
    elif mutation == 'missing':
        del report['regions']['end15cm']
    else:
        report['regions']['full']['score_distribution']['samples'][0]['score_magnitude'][0] = float('nan')
    evidence = validate_score_evidence(_score_package(), report, 'score/results.json')
    assert evidence['score_thresholds_verified'] is False
    assert any(match in reason for reason in evidence['failures'])
