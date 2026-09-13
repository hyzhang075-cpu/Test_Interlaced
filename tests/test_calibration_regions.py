"""Physical cropping and global segment compatibility for nested calibration."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRESET = [510, 521, 532, 543, 554, 588, 622, 656, 707, 758, 809,
          860, 911, 962, 1013, 1064, 1115, 1166, 1217, 1268, 1319, 1370]
FULL_SEGMENTS = [2, 3, 5, 2, 3, 3, 2, 2]


def resolve(**overrides):
    assert importlib.util.find_spec('calibration_regions'), 'missing region resolver'
    from calibration_regions import resolve_calibration_region
    profiles = {
        'full': dict(length_m=None, folder_path='body', radii={'circ': [0, 200]},
                     thresholds=[0.8, 4], segmentation=FULL_SEGMENTS),
        'end15cm': dict(length_m=.15, folder_path='15cm', radii={'circ': [0, 30]},
                        thresholds=[0.8, 4, 16.7], segmentation=[3, 2, 2]),
        'end5cm': dict(length_m=.05, folder_path='5cm', radii={'circ': [0, 15]},
                       thresholds=[0.8, 4, 58.3], segmentation=[2, 2]),
    }
    kwargs = dict(start_value=7, end_value=10, samp_dist_preset=PRESET,
                  full_segmentation=FULL_SEGMENTS, profiles=profiles,
                  optimization_region='full', multi_region_calib=False)
    kwargs.update(overrides)
    return resolve_calibration_region(**kwargs)


@pytest.mark.parametrize('region,ids,segments,trim,local,tip', [
    ('full', list(range(1, 23)), FULL_SEGMENTS, 0.0, [7, 867], .877),
    ('end15cm', list(range(16, 23)), [3, 2, 2], -.15, [28, 140], .15),
    ('end5cm', list(range(19, 23)), [2, 2], -.05, [7, 40], .05),
])
def test_selected_region_matches_physical_geometry(region, ids, segments, trim, local, tip):
    result = resolve(optimization_region=region, multi_region_calib=True)
    assert result['ideal_len'] == pytest.approx(.877)
    assert result['global_point_ids'] == ids
    assert result['segmentation'] == segments
    assert result['straight_trim'] == trim
    assert np.asarray(result['sample_dist_m'])[[0, -1]] == pytest.approx(np.array(local)/1000)
    assert result['reconstruction_length_m'] == pytest.approx(tip)


def test_disabled_region_mode_rejects_local_selection_but_accepts_full():
    assert resolve()['global_point_ids'] == list(range(1, 23))
    with pytest.raises(ValueError, match='multi_region_calib'):
        resolve(optimization_region='end5cm')


def test_crop_inside_global_segment_is_rejected_instead_of_truncated():
    # Old [2,2,3] tail splits the four-point 5 cm region through a segment.
    with pytest.raises(ValueError, match='分段'):
        resolve(full_segmentation=[2, 3, 5, 2, 3, 2, 2, 3],
                optimization_region='end5cm', multi_region_calib=True)


def test_local_segmentation_must_match_global_partition_not_just_point_count():
    # All four points are a full global segment, but profile requests two segments.
    with pytest.raises(ValueError, match='分段'):
        resolve(full_segmentation=[2, 3, 5, 2, 3, 3, 4],
                optimization_region='end5cm', multi_region_calib=True)


@pytest.mark.parametrize('segments', [[2, 3], [22, 1], [18, 0, 4], [18, 2.5, 1.5]])
def test_invalid_full_partition_is_not_silently_repaired(segments):
    with pytest.raises(ValueError, match='分段'):
        resolve(full_segmentation=segments)


@pytest.mark.parametrize('overrides', [
    dict(start_value=-1), dict(end_value=float('nan')),
    dict(samp_dist_preset=[510, 510, 520]),
    dict(samp_dist_preset=[510, 500]),
])
def test_invalid_geometry_fails_before_data_read(overrides):
    with pytest.raises(ValueError):
        resolve(**overrides)


def test_exact_crop_boundary_keeps_real_point_and_full_segment():
    profiles = {
        'full': dict(length_m=None, folder_path='body', radii={'circ': [0, 200]},
                     thresholds=[1], segmentation=[2, 2]),
        'tail': dict(length_m=.020, folder_path='tail', radii={'circ': [0, 15]},
                     thresholds=[1], segmentation=[2]),
    }
    result = resolve(samp_dist_preset=[100, 110, 120, 130],
                     full_segmentation=[2, 2], profiles=profiles,
                     optimization_region='tail', multi_region_calib=True)
    assert result['global_point_ids'] == [3, 4]
    assert result['sample_dist_m'] == pytest.approx([0, .01])
    assert result['global_segment_ids'] == [1]


def test_selected_dataset_and_levels_travel_together():
    result = resolve(optimization_region='end5cm', multi_region_calib=True)
    assert result['folder_path'] == '5cm'
    assert result['radii'] == {'circ': [0, 15]}
    assert result['thresholds'] == [.8, 4, 58.3]
    assert result['global_segment_ids'] == [6, 7]


def test_parameter_shape_rejects_old_local_slopes_for_full_calibration():
    from calibration_regions import validate_calibration_parameters
    with pytest.raises(ValueError, match='slopes'):
        validate_calibration_parameters(None, {0: np.ones((6, 2))},
                                        [.8, 1.2, 2, 3, 4], FULL_SEGMENTS, [0])


@pytest.mark.parametrize('region,enabled,count,segments,last_radius', [
    ('full', False, 22, FULL_SEGMENTS, 200),
    ('end15cm', True, 7, [3, 2, 2], 30),
    ('end5cm', True, 4, [2, 2], 15),
])
def test_main_region_selection_reaches_optimizer(monkeypatch, region, enabled, count, segments, last_radius):
    import main

    class ConfigReached(Exception):
        pass

    # Stop before the real optimizer reads CSV files or builds any GUI.
    received = {}
    def capture(**kwargs):
        received.update(kwargs)
        raise ConfigReached
    monkeypatch.setattr(main, 'Optimiser', capture)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--task', 'init',
                                    '--multi_region_calib', str(enabled),
                                    '--optimization_region', region])
    with pytest.raises(ConfigReached):
        main.main()
    assert received['ideal_len'] == pytest.approx(.877)
    assert received['segmentation'] == segments
    assert sum(received['segmentation']) == count
    assert received['radii']['circ'][-1] == last_radius
    assert Path(received['folder_path']).is_dir()
    assert received['weights'] is None and received['slopes'] is None
    assert 'multi_region_calib' not in received


def test_main_geometry_overrides_recompute_length_before_optimizer(monkeypatch):
    import main
    class ConfigReached(Exception):
        pass
    def capture(**kwargs):
        assert kwargs['ideal_len'] == pytest.approx(.890)
        assert kwargs['straight_trim'] == 0.0
        raise ConfigReached
    monkeypatch.setattr(main, 'Optimiser', capture)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--task', 'init',
                                    '--start_value', '10', '--end_value', '20'])
    with pytest.raises(ConfigReached):
        main.main()


def test_main_rejects_cross_boundary_segments_before_optimizer(monkeypatch):
    import main
    def forbidden(**kwargs):
        pytest.fail('invalid spatial partition reached optimizer')
    monkeypatch.setattr(main, 'Optimiser', forbidden)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--task', 'optim',
                                    '--multi_region_calib', 'True',
                                    '--optimization_region', 'end5cm',
                                    '--full_segmentation', '[2,3,5,2,3,2,2,3]'])
    with pytest.raises(ValueError, match='分段'):
        main.main()


@pytest.mark.parametrize('trim,ids,samples,tip', [
    (0.0, list(range(1, 23)), [7, 867], .877),
    (-.15, list(range(16, 23)), [28, 140], .15),
    (-.05, list(range(19, 23)), [7, 40], .05),
])
def test_sensor_geometry_matches_resolver_even_with_stale_ideal_length(trim, ids, samples, tip):
    from sensor import Sensor
    sensor = Sensor.__new__(Sensor)  # geometry only; no CSV or hardware construction
    sensor.interrogator = 'FiberTecc'
    sensor.ideal_len = .878
    sensor.generate_samp_dist(False, 7, 10, trim, PRESET, None, False)
    assert sensor.ideal_len == pytest.approx(.877)
    assert sensor.global_point_ids.tolist() == ids
    assert sensor.samp_dist[[0, -1]] == pytest.approx(np.array(samples)/1000)
    assert sensor.ad_samp_dist[-1] == pytest.approx(tip)


def test_sensor_exact_boundary_and_reverse_preserve_global_ids():
    from sensor import Sensor
    sensor = Sensor.__new__(Sensor)
    sensor.interrogator = 'FiberTecc'
    sensor.ideal_len = 100
    sensor.generate_samp_dist(True, 7, 10, -.02, [100, 110, 120, 130], None, False)
    assert sensor.global_point_ids.tolist() == [4, 3]
    assert sensor.samp_dist == pytest.approx([.01, .02])


def test_main_preloaded_lists_support_reconstructor_two_axis_indexing(monkeypatch):
    import main
    class ConfigReached(Exception):
        pass
    def capture(**kwargs):
        # _Strain_2_Cur indexes slope[level, segment], which plain nested lists reject.
        assert kwargs['slopes'][0][5, 1] == 1.0
        assert kwargs['weights'][0].shape == (6,)
        raise ConfigReached
    slopes = {d: [[1.0, 1.0] for _ in range(6)] for d in [0, 120, 240]}
    weights = {d: [1.0] * 6 for d in [0, 120, 240]}
    monkeypatch.setattr(main, 'Optimiser', capture)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--task', 'recon',
                                    '--multi_region_calib', 'True', '--optimization_region', 'end5cm',
                                    '--weights', repr(weights), '--slopes', repr(slopes)])
    with pytest.raises(ConfigReached):
        main.main()
