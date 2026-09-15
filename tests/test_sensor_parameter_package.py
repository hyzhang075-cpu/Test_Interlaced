import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sensor_parameter_package import (
    apply_validation_results,
    build_sensor_package,
    load_calibration_results,
    load_sensor_package,
    package_parameter_destinations,
    package_deployment_ready,
    package_parser_defaults,
    render_calibration_comparison,
    write_sensor_package,
)


REGIONS = {
    'full': dict(thresholds=[.8, 1.2, 2, 3, 4], segmentation=[2, 2],
                 point_ids=[1, 2, 3, 4], segment_ids=[0, 1], length=None),
    'end15cm': dict(thresholds=[.8, 1.2, 2, 3, 4, 8, 20], segmentation=[2],
                    point_ids=[3, 4], segment_ids=[1], length=.15),
    'end5cm': dict(thresholds=[.8, 1.2, 2, 3, 4, 8, 20, 35], segmentation=[2],
                   point_ids=[3, 4], segment_ids=[1], length=.05),
}


def _parser():
    import main
    return main.build_argument_parser()


def _entry(region, direct, *, stage_status='converge'):
    cfg = REGIONS[region]
    levels = len(cfg['thresholds']) + 1
    segments = len(cfg['segmentation'])
    phase = {'full': -.7, 'end15cm': -.2, 'end5cm': .7}[region]
    return {
        'status': 'complete',
        'context': {
            'optimization_region': region,
            'length_m': cfg['length'],
            'folder_path': region,
            'radii': {'circ': [0, 200]},
            'thresholds': cfg['thresholds'],
            'segmentation': cfg['segmentation'],
            'global_point_ids': cfg['point_ids'],
            'global_segment_ids': cfg['segment_ids'],
            'global_sample_dist_m': [index / 100 for index in cfg['point_ids']],
            'sample_dist_m': [index / 100 for index in range(len(cfg['point_ids']))],
            'crop_origin_m': 0.0,
            'reconstruction_length_m': .4 if region == 'full' else cfg['length'],
            'ideal_len': .4,
            'straight_trim': 0 if region == 'full' else -cfg['length'],
        },
        'parameters': {
            'weights': {str(direct): [float(direct + level + 1) for level in range(levels)]},
            'slopes': {str(direct): [
                [float(direct + level + segment) for segment in range(segments)]
                for level in range(levels)
            ]},
            'global_phases_set': phase,
            'rot_ang': 0.0,
        },
        'errors': {
            'normal_weighted_loss': .01,
            'avg_point_err': {'calib': .005},
            'avg_end_err': {'calib': .01},
            'avg_pitch_err': 0.0,
            'avg_radius_err': 0.0,
            'point_errors_by_sample': {'circ': {'200': .005}},
            'end_errors_by_sample': {'circ': {'200': .01}},
        },
        'optimization': {'stages': {
            'slopes': {'status': stage_status, 'start_loss': .02, 'final_loss': .01},
            'weights': {'status': stage_status, 'start_loss': .02, 'final_loss': .01},
        }},
    }


def _report(direct, *, sensor_id='sensor-A', stage_status='converge'):
    return {
        'sensor_id': sensor_id,
        'task': 'optim',
        'direct': direct,
        'regions': {
            region: _entry(region, direct, stage_status=stage_status)
            for region in REGIONS
        },
    }


def test_package_merges_literal_direction_angles_in_stable_order():
    package = build_sensor_package(
        _parser(), [_report(240), _report(0), _report(120)],
        source_commit='abc123', generated_at='2026-09-14T00:00:00Z',
    )

    assert list(package) == [
        'sensor_id', 'format_version', 'shared_parameters',
        'multi_region', 'regions', 'provenance',
    ]
    assert package['multi_region']['directions'] == [0, 120, 240]
    assert list(package['regions']) == ['full', 'end15cm', 'end5cm']
    assert list(package['regions']['end5cm']['weights']) == ['0', '120', '240']
    assert len(package['regions']['full']['weights']['0']) == 6
    assert len(package['regions']['end15cm']['weights']['0']) == 8
    assert len(package['regions']['end5cm']['weights']['0']) == 9
    assert package['provenance']['source_commit'] == 'abc123'
    assert package['provenance']['generated_at'] == '2026-09-14T00:00:00Z'


def test_parser_and_exported_package_use_project_relative_paths():
    parser = _parser()
    assert parser.get_default('zero_reference_path') == (
        './params/06B6_NeTi_v2_ftube_zero_reference.json'
    )

    reports = [_report(direct) for direct in (0, 120, 240)]
    for report in reports:
        report['_source_path'] = str(
            ROOT / 'calibration_results' / f'run-direct{report["direct"]}' / 'results.json'
        )
        for region in REGIONS:
            report['regions'][region]['context']['folder_path'] = str(
                ROOT / '0908-06B6' / region
            )

    package = build_sensor_package(parser, reports)

    # Region folders are relative to calibration_data_root, avoiding a duplicated
    # ``0908-06B6/0908-06B6`` prefix when main resolves them at runtime.
    assert package['regions']['full']['folder_path'] == 'full'
    assert package['provenance']['source_results'][0] == (
        './calibration_results/run-direct0/results.json'
    )
    assert not any(':' in path for path in package['provenance']['source_results'])


def test_every_parser_destination_has_one_authoritative_package_location():
    parser = _parser()
    expected = [action.dest for action in parser._actions if action.dest != 'help']
    package = build_sensor_package(parser, [_report(0), _report(120), _report(240)])

    actual = package_parameter_destinations(package)

    assert sorted(actual) == sorted(expected)
    assert len(actual) == len(set(actual))
    assert list(package['shared_parameters']) == [
        'hardware_parameters', 'calibration_experiment_parameters',
        'optimization_hyperparameters', 'dynamic_display_parameters',
        'reconstruction_parameters', 'initialization_parameters',
        'angle_optimization_parameters', 'runtime_control_parameters',
    ]


def test_package_rejects_direction_indices_and_sensor_mismatch():
    with pytest.raises(ValueError, match='0、120、240'):
        build_sensor_package(_parser(), [_report(0), _report(1), _report(2)])
    with pytest.raises(ValueError, match='sensor_id'):
        build_sensor_package(_parser(), [_report(0), _report(120), _report(240, sensor_id='sensor-B')])


def test_region_parameter_shape_is_validated_independently():
    reports = [_report(0), _report(120), _report(240)]
    reports[0]['regions']['end5cm']['parameters']['slopes']['0'] = [[1.0, 2.0]]

    with pytest.raises(ValueError, match='end5cm.*slopes'):
        build_sensor_package(_parser(), reports)


def test_unverified_phase_and_diverged_stage_produce_candidate_only():
    package = build_sensor_package(
        _parser(), [_report(0), _report(120), _report(240, stage_status='diverge')]
    )

    assert package['multi_region']['deployment_ready'] is False
    directions = package['multi_region']['direction_deployment']
    assert directions['0']['quality_gates']['optimization_usable'] is True
    assert directions['120']['quality_gates']['optimization_usable'] is True
    assert directions['240']['quality_gates']['optimization_usable'] is False
    assert directions['0']['quality_gates']['end15cm_phase_verified'] is False
    assert any('240/full slopes diverged' in failure for failure in directions['240']['failures'])


def test_no_improvement_stage_is_not_optimization_complete():
    package = build_sensor_package(
        _parser(), [_report(0), _report(120), _report(240, stage_status='no_improvement')]
    )
    directions = package['multi_region']['direction_deployment']
    assert directions['0']['quality_gates']['optimization_usable'] is True
    assert directions['120']['quality_gates']['optimization_usable'] is True
    assert directions['240']['quality_gates']['optimization_usable'] is False
    assert any('240/full slopes no_improvement' in failure for failure in directions['240']['failures'])


def test_json_round_trip_preserves_order_and_rejects_strict_deployment(tmp_path):
    package = build_sensor_package(_parser(), [_report(0), _report(120), _report(240)])
    path = tmp_path / 'sensor.json'

    write_sensor_package(package, path)
    loaded = load_sensor_package(path, expected_sensor_id='sensor-A')

    assert list(json.loads(path.read_text(encoding='utf-8'))) == list(package)
    assert loaded == package
    with pytest.raises(ValueError, match='deployment'):
        load_sensor_package(path, require_deployable=True)


def test_load_calibration_results_reads_json_and_rejects_duplicate_direct(tmp_path):
    paths = []
    for direct in (0, 120, 240):
        path = tmp_path / f'{direct}.json'
        path.write_text(json.dumps(_report(direct)), encoding='utf-8')
        paths.append(path)
    assert [report['direct'] for report in load_calibration_results(paths)] == [0, 120, 240]

    duplicate = tmp_path / 'duplicate.json'
    duplicate.write_text(json.dumps(_report(240)), encoding='utf-8')
    with pytest.raises(ValueError, match='重复'):
        load_calibration_results([*paths, duplicate])


def test_load_calibration_results_accepts_result_directories(tmp_path):
    paths = []
    for direct in (0, 120, 240):
        directory = tmp_path / f'run-direct{direct}'
        directory.mkdir()
        (directory / 'results.json').write_text(
            json.dumps(_report(direct)), encoding='utf-8'
        )
        paths.append(directory)

    reports = load_calibration_results(paths)

    assert [report['direct'] for report in reports] == [0, 120, 240]
    assert all(Path(report['_source_path']).name == 'results.json' for report in reports)


def test_comparison_report_exposes_parameters_errors_and_failed_stages():
    reports = [_report(0), _report(120), _report(240, stage_status='diverge')]

    markdown = render_calibration_comparison(reports)

    assert '| 240 | full |' in markdown
    assert 'diverged' in markdown
    assert 'weights: `[241.0, 242.0' in markdown
    assert 'point=5.000 mm' in markdown
    assert 'local global phase' in markdown


def _package_path(tmp_path, **shared_overrides):
    package = build_sensor_package(_parser(), [_report(0), _report(120), _report(240)])
    for name, value in shared_overrides.items():
        matches = [group for group in package['shared_parameters'].values() if name in group]
        assert len(matches) == 1
        matches[0][name] = value
    path = tmp_path / 'sensor.json'
    write_sensor_package(package, path)
    return path, package


def test_package_parser_defaults_restore_region_profiles_and_literal_directions(tmp_path):
    path, package = _package_path(tmp_path)

    defaults = package_parser_defaults(load_sensor_package(path))

    assert defaults['sensor_id'] == 'sensor-A'
    assert defaults['global_phases_set'] == -.7
    assert defaults['rot_ang'] == 0.0
    assert defaults['calibration_regions']['end5cm']['thresholds'] == REGIONS['end5cm']['thresholds']
    assert list(defaults['calibration_regions']['end5cm']['weights']) == [0, 120, 240]
    assert defaults['calibration_regions']['end5cm']['slopes'][120][0][0] == 120.0


def test_parse_main_args_loads_package_and_cli_explicit_value_wins(tmp_path):
    import main
    path, _ = _package_path(tmp_path, move_filter=11)

    loaded = main.parse_main_args([
        '--task', 'recon', '--sensor_params', str(path), '--move_filter', '5',
    ])

    assert loaded.sensor_id == 'sensor-A'
    assert loaded.move_filter == 5
    assert loaded.sensor_params == str(path)


def test_parse_main_args_keeps_task_required_and_rejects_explicit_sensor_conflict(tmp_path):
    import main
    path, package = _package_path(tmp_path)
    package['shared_parameters']['runtime_control_parameters']['task'] = 'recon'
    write_sensor_package(package, path)

    with pytest.raises(SystemExit):
        main.parse_main_args(['--sensor_params', str(path)])
    with pytest.raises(ValueError, match='sensor_id'):
        main.parse_main_args([
            '--task', 'recon', '--sensor_params', str(path), '--sensor_id', 'sensor-B',
        ])


def test_parse_main_args_rejects_unknown_package_version_before_runtime(tmp_path):
    import main
    path, package = _package_path(tmp_path)
    package['format_version'] = 999
    write_sensor_package(package, path)

    with pytest.raises(ValueError, match='format_version'):
        main.parse_main_args(['--task', 'recon', '--sensor_params', str(path)])


def test_validation_evidence_keeps_failed_candidate_non_deployable():
    package = build_sensor_package(_parser(), [_report(0), _report(120), _report(240)])
    package['shared_parameters']['dynamic_display_parameters']['direction_reference_direct'] = 240
    validation = {
        'shared_quality_gates': {'score_thresholds_verified': True},
        'directions': {
            str(direct): {
                'quality_gates': {
                    'end15cm_phase_verified': direct != 240,
                    'full_geometry_verified': direct != 240,
                },
                'failures': [] if direct != 240 else ['240-degree full reconstruction failed'],
                'phase_transport': {'end5cm': {'blocking': False, 'within_30_deg': False}},
            }
            for direct in (0, 120, 240)
        },
    }

    updated = apply_validation_results(package, validation)

    assert updated['multi_region']['deployment_ready'] is False
    assert updated['provenance']['full_geometry_validation'] == validation
    assert updated['multi_region']['available_deployment_directions'] == [0, 120]
    assert updated['multi_region']['direction_deployment']['0']['deployment_ready'] is True
    assert updated['multi_region']['direction_deployment']['240']['deployment_ready'] is False
    assert updated['multi_region']['direction_deployment']['240']['phase_transport']['end5cm']['blocking'] is False


def test_package_readiness_uses_reference_direction_when_blending_is_disabled():
    package = build_sensor_package(_parser(), [_report(0), _report(120), _report(240)])
    package['shared_parameters']['dynamic_display_parameters'].update({
        'direction_blend_enabled': False,
        'direction_reference_direct': 0,
    })
    package['multi_region']['shared_quality_gates']['score_thresholds_verified'] = True
    for direct, ready in ((0, True), (120, False), (240, False)):
        gates = package['multi_region']['direction_deployment'][str(direct)]['quality_gates']
        gates['end15cm_phase_verified'] = ready
        gates['full_geometry_verified'] = ready

    assert package_deployment_ready(package) is True

    package['shared_parameters']['dynamic_display_parameters']['direction_blend_enabled'] = True
    assert package_deployment_ready(package) is False
