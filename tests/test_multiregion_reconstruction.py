from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconstructor import Reconstructor


def package():
    directions = [0, 120, 240]

    def region(thresholds, segmentation, support, owned, segment_ids, base):
        levels = len(thresholds) + 1
        return {
            'thresholds': thresholds,
            'sharpness': 8.0,
            'score_window': 1,
            'score_boundary_mode': 'constant',
            'local_segmentation': segmentation,
            'global_segment_ids': segment_ids,
            'support_point_ids': support,
            'owned_point_ids': owned,
            'weights': {str(d): [base + level for level in range(levels)] for d in directions},
            'slopes': {str(d): [
                [base * 10 + level + segment for segment in range(len(segmentation))]
                for level in range(levels)
            ] for d in directions},
        }

    return {
        'multi_region': {'region_order': ['full', 'end15cm', 'end5cm'],
                         'directions': directions},
        'regions': {
            'full': region([1.0], [2, 2, 2], [1, 2, 3, 4, 5, 6], [1, 2], [0, 1, 2], 1),
            'end15cm': region([2.0, 3.0], [2, 2], [3, 4, 5, 6], [3, 4], [1, 2], 10),
            'end5cm': region([3.0, 4.0, 5.0], [2], [5, 6], [5, 6], [2], 100),
        },
    }


def bare_reconstructor(config=None):
    obj = Reconstructor.__new__(Reconstructor)
    obj.total_points = 6
    obj.global_point_ids = np.arange(1, 7)
    obj.base_segments = [(0, 2), (2, 4), (4, 6)]
    obj.soft = True
    obj.discont = np.pi
    obj.turn_threshold = 2.0
    obj.sharpness = 4.8
    obj.min_consecutive = 3
    obj.multi_region_parameters = obj._normalise_multiregion_parameters(config or package())
    obj._calculate_dynamic_angles = lambda **kwargs: np.zeros((3, obj.total_points))

    calls = []
    def curvature(epsilon, alpha_dynamic):
        calls.append(np.asarray(epsilon).copy())
        return np.abs(np.asarray(epsilon)[0]), np.zeros(obj.total_points)
    obj._calculate_curvature_and_angle = curvature
    return obj, calls


def test_each_region_uses_its_own_threshold_count_weights_and_slopes():
    obj, calls = bare_reconstructor()
    strain = np.vstack((np.arange(6, dtype=float), np.zeros((2, 6))))

    magnitude, angle, final_weight, slope_map, diagnostics = (
        obj._assemble_multiregion_parameters(strain, 120)
    )

    assert magnitude.shape == angle.shape == final_weight.shape == slope_map.shape == (6,)
    assert [diagnostics[name]['mass_shape'][0] for name in ('full', 'end15cm', 'end5cm')] == [2, 3, 4]
    assert np.all(final_weight[:2] < 10)
    assert np.all((final_weight[2:4] >= 10) & (final_weight[2:4] < 100))
    assert np.all(final_weight[4:] >= 100)
    np.testing.assert_allclose(calls[-1], strain * final_weight)


def test_fixed_phase_mode_pins_only_bridges_in_base_and_final_maps():
    config = package()
    config['multi_region']['use_full_transported_phase'] = True
    config['multi_region']['phase_reference_directions'] = [0]
    obj, _ = bare_reconstructor(config)
    strain = np.vstack((np.arange(6, dtype=float), np.zeros((2, 6))))
    *_, slopes, diagnostics = obj._assemble_multiregion_parameters(strain, 0)
    assert diagnostics['fixed_full_level0_bridges'] is True
    assert diagnostics['base_slope_map'][2] == slopes[2] == 11.
    assert diagnostics['base_slope_map'][4] == slopes[4] == 12.
    assert slopes[3] >= 100. and slopes[5] >= 1000.
    *_, other, diagnostics = obj._assemble_multiregion_parameters(strain, 120)
    assert diagnostics['fixed_full_level0_bridges'] is False
    assert other[2] >= 100. and other[4] >= 1000.


def test_scores_are_computed_on_support_before_owned_points_are_extracted():
    config = package()
    config['regions']['end15cm']['owned_point_ids'] = [3, 4, 5]
    config['regions']['end5cm']['owned_point_ids'] = [6]
    config['regions']['end5cm']['score_window'] = 2
    obj, _ = bare_reconstructor(config)
    strain = np.vstack(([0., 0., 0., 0., 10., 10.], np.zeros((2, 6))))

    *_, diagnostics = obj._assemble_multiregion_parameters(strain, 0)

    assert diagnostics['end5cm']['support_point_ids'] == [5, 6]
    assert diagnostics['end5cm']['owned_point_ids'] == [6]
    assert diagnostics['end5cm']['mass_shape'] == [4, 2]
    # constant-window averaging on the two-point support attenuates both edges;
    # it would be different if each owned point were scored separately.
    assert diagnostics['end5cm']['scores'][0] == pytest.approx([.5, 1.0], rel=1e-3)


def test_region_validation_rejects_gaps_overlap_and_direction_indices():
    config = package()
    config['regions']['end5cm']['owned_point_ids'] = [4, 5, 6]
    with pytest.raises(ValueError, match='owned'):
        bare_reconstructor(config)

    config = package()
    config['multi_region']['directions'] = [0, 1, 2]
    with pytest.raises(ValueError, match='0、120、240'):
        bare_reconstructor(config)


def test_global_segment_mapping_drives_region_base_and_final_slopes():
    obj, _ = bare_reconstructor()
    strain = np.vstack((np.arange(6, dtype=float), np.zeros((2, 6))))

    *_, slope_map, diagnostics = obj._assemble_multiregion_parameters(strain, 0)

    assert diagnostics['base_slope_map'][:2] == [10.0, 10.0]
    assert diagnostics['base_slope_map'][2:4] == [100.0, 100.0]
    assert diagnostics['base_slope_map'][4:] == [1000.0, 1000.0]
    assert np.all(slope_map[:2] < 100)
    assert np.all((slope_map[2:4] >= 100) & (slope_map[2:4] < 1000))
    assert np.all(slope_map[4:] >= 1000)


def test_strain_to_curvature_uses_assembler_when_multiregion_is_configured():
    obj, _ = bare_reconstructor()
    strain = np.vstack((np.arange(6, dtype=float), np.zeros((2, 6))))
    obj.processed = strain

    magnitude, angle = obj._Strain_2_Cur(
        keys=None, calib_shape='circ', calib_Data=strain.T,
        weight=None, slope=None, direct=240,
    )

    assert magnitude.shape == angle.shape == (6,)
    assert obj.last_parameter_assembly['direct'] == 240
