"""Run the four-core model through the same YAML tasks and run snapshots."""
import csv
import json

import numpy as np
import pytest

from shape_sensing import cli
from shape_sensing.contracts import CalibrationParameters, PreparedFrame
from shape_sensing.curvature.four_core import compute_four_core
from shape_sensing.pipelines.tasks import build_session, execute_task


def _run(config, output, task, *extra):
    args = cli.parse_main_args(['--config', str(config), '--task', task, *extra])
    args.calibration_output_dir = str(output / 'calibration_results')
    result = execute_task(
        args, optimiser_factory=None, visualizer_factory=None, app_factory=None,
        configure_args=cli.configure_calibration_args, parameter_builder=cli.optimizer_parameters,
        prepare_jobs=cli.prepare_calibration_jobs, run_jobs=cli.run_calibration_jobs,
        compute_error_ang=None, optim_ang=None, project_root=output, artifact_root=output,
    )
    run, = (output / 'fixture').glob(f'*-{task}')
    return result, run


def test_schema6_init_optim_recon_use_standard_tasks(tmp_path):
    channels = (7, 9, 2, 4)
    root = tmp_path / 'input'
    config = tmp_path / 'sensor.yaml'
    config.write_text(json.dumps(dict(
        sensor_id='fixture', R=1e-4, core_position_calib=True, max_iter=1,
        thresholds=[5.0, 15.0], soft=False, min_consecutive=1, hyper_params=[1, 1e-6, 1e-6],
        start_value=-10.0, end_value=6.0, samp_dist_preset=[100, 150, 300],
        active_channels=list(channels), folder_path=str(root), radii={'circ': [100]},
        interp_num=20, curvature_interp='vector_pchip', move_filter=1, frame_rule='TNN',
    )), encoding='utf-8')

    positions_um = np.array([[0, 0], [110, 0], [-55, 55*np.sqrt(3)], [-55, -55*np.sqrt(3)]])
    for direction, common_pm in [(0, 73), (120, -19), (240, 42)]:
        folder = root / f'{direction}-100'
        folder.mkdir(parents=True)
        angle = np.deg2rad(-direction)
        responses = positions_um @ (10 * np.array([np.cos(angle), np.sin(angle)]))
        for channel, response in zip(channels, responses):
            with (folder / f'Channel_{channel}_fixture.csv').open('w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'SensorIDs', 'WL_3', 'WL_1', 'WL_2'])
                for row in range(10):
                    writer.writerow([row, 'fixture', response+common_pm, common_pm, response+common_pm])

    runs = tmp_path / 'runs'
    initialized, init_run = _run(config, runs, 'init')
    core_plot = init_run / 'outputs' / 'full' / 'init_core_positions_full.png'
    assert core_plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    init_plots = list((init_run / 'outputs' / 'full').glob('*/reconstruction_circ.png'))
    assert {path.parent.name for path in init_plots} == {'0', '120', '240'}
    assert all(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for path in init_plots)
    initial = initialized['core_position_parameters']['full']
    calibration = initial['core_position_metadata']
    assert calibration['schema_version'] == 6
    assert calibration['calibration']['wl_ids'] == [2, 3]
    assert calibration['channel_configuration']['response_channels'] == list(channels)
    assert 'slopes' not in initial
    assert np.asarray(initial['weights'][0]).shape == (3,)
    section = initial['core_position'][0]
    assert section['cores']['ch7']['alpha_rad'] is None
    assert section['cores']['ch9']['r_um'] == pytest.approx(110)
    assert [row['b_pm'] for row in section['calibration_common_response_pm']] == pytest.approx([73, -19, 42])

    engine = build_session(cli.parse_main_args(['--config', str(config), '--task', 'init']),
                           parameter_builder=cli.optimizer_parameters,
                           configure_args=cli.configure_calibration_args)
    gains = np.array([[0.8, 1.2, 0.9, 1.1], [1.1, 0.7, 1.25, 0.85]])
    target = np.array([3.0, -4.0])
    matrix_m = positions_um * 1e-6
    drift = np.array([0.00021, -0.00035])
    signals = (matrix_m @ target) / gains + drift[:, None]
    order = [2, 0, 3, 1]
    frame = PreparedFrame.from_design(engine.design, signals[:, order], tuple(channels[i] for i in order))
    solved = compute_four_core(frame, engine.design, {**calibration, 'sections': initial['core_position']}, gains, 1.0)
    np.testing.assert_allclose(solved.diagnostics['k_y_per_m'], [3, 3])
    np.testing.assert_allclose(solved.diagnostics['k_z_per_m'], [-4, -4])
    np.testing.assert_allclose(solved.diagnostics['common_response_b_strain'], drift)
    graded_frame = PreparedFrame.from_design(engine.design, 10 * matrix_m[:, 0] + drift[:, None], channels)
    graded_parameters = CalibrationParameters({'full': {**initial, 'weights': {0: [1.0, 0.3, 0.8]}}})
    graded = engine.curvature.compute(graded_frame, engine.design, graded_parameters)
    np.testing.assert_allclose(graded.curvature_per_m, [3, 3])
    np.testing.assert_allclose(graded.diagnostics['effective_weights'], [0.3, 0.3])
    np.testing.assert_allclose(graded.diagnostics['common_response_b_strain'], drift)

    recon, recon_run = _run(config, runs, 'recon', '--resume', str(init_run), '--direct', '120')
    assert not list((recon_run / 'outputs').rglob('init_core_positions_*.png'))
    recon_plot = recon_run / 'outputs' / 'full' / '120' / 'reconstruction_circ.png'
    assert recon_plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    points = recon['reports']['full']['point_curvature']['100']
    np.testing.assert_allclose(points['curvature_per_m'], [10, 10], atol=1e-10)
    np.testing.assert_allclose(points['common_response_b_strain'], [-19e-6, -19e-6])
    np.testing.assert_allclose(points['direction_rad'], [0, 0], atol=1e-10)
    assert recon['reports']['full']['curves_m'][100].shape == (20, 3)
    np.testing.assert_allclose(points['effective_weights'], [1, 1])

    (_, optimized), opt_run = _run(config, runs, 'optim', '--resume', str(init_run))
    block = optimized['core_position_parameters']['full']
    assert block['core_position'] == initial['core_position']
    assert 'slopes' not in block and 'strain_gains' not in block
    assert np.asarray(block['weights'][0]).shape == (3,)
    assert np.all(np.asarray(block['weights'][0]) > 0)
    saved = json.loads((opt_run / 'optim.json').read_text(encoding='utf-8'))
    report = saved['result'][1]
    assert report['sensor_id'] == 'fixture' and report['task'] == 'optim'
    region = report['regions']['full']
    assert region['configuration']['segmentation'] == region['context']['segmentation'] == [2]
    assert set(region['optimization']['stages']) == {'weights'}
    assert region['parameters']['slopes'] is None
    assert region['errors']['avg_point_err']['calib'] >= 0
    assert region['errors']['avg_end_err']['calib'] >= 0
    np.testing.assert_allclose(block['weights'][120], initial['weights'][120])
    replay, _ = _run(config, tmp_path / 'optimized-recon', 'recon', '--resume', str(opt_run))
    for metric in ('avg_point_err', 'avg_end_err'):
        assert saved[metric] == region['errors'][metric]
        assert region['errors'][metric] == replay['reports']['full']['metrics'][metric]
    np.testing.assert_allclose(replay['core_position_parameters']['full']['weights']['0'],
                               block['weights'][0])

    # A previous-version init carries only the source schema 6 payload.
    snapshot = init_run / 'initialization.json'
    legacy = json.loads(snapshot.read_text(encoding='utf-8'))
    legacy['result']['core_position_parameters']['full'] = {
        'core_calibration': {**calibration, 'sections': initial['core_position']},
    }
    snapshot.write_text(json.dumps(legacy), encoding='utf-8')
    resumed, _ = _run(config, tmp_path / 'legacy-init-recon', 'recon', '--resume', str(init_run))
    migrated = resumed['core_position_parameters']['full']
    assert migrated['weight_layout'] == initial['weight_layout']
    np.testing.assert_allclose(migrated['weights'][0], initial['weights'][0])
