"""Run the four-core model through the same YAML tasks and run snapshots."""
import csv
import json

import numpy as np
import pytest

from shape_sensing import cli
from shape_sensing.contracts import PreparedFrame
from shape_sensing.curvature.four_core import compute_four_core
from shape_sensing.pipelines.tasks import build_session, execute_task


def _run(config, output, task, *extra):
    args = cli.parse_main_args(['--config', str(config), '--task', task, *extra])
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
        sensor_id='fixture', R=1e-4, core_position_calib=True, core_gain_max_nfev=2,
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
    calibration = initialized['core_position_parameters']['full']['core_calibration']
    assert calibration['schema_version'] == 6
    assert calibration['calibration']['wl_ids'] == [2, 3]
    assert calibration['channel_configuration']['response_channels'] == list(channels)
    section = calibration['sections'][0]
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
    solved = compute_four_core(frame, engine.design, calibration, gains, 1.0)
    np.testing.assert_allclose(solved.diagnostics['k_y_per_m'], [3, 3])
    np.testing.assert_allclose(solved.diagnostics['k_z_per_m'], [-4, -4])
    np.testing.assert_allclose(solved.diagnostics['common_response_b_strain'], drift)

    recon, _ = _run(config, runs, 'recon', '--resume', str(init_run), '--direct', '120')
    points = recon['reports']['full']['point_curvature']['100']
    np.testing.assert_allclose(points['curvature_per_m'], [10, 10], atol=1e-10)
    np.testing.assert_allclose(points['common_response_b_strain'], [-19e-6, -19e-6])
    np.testing.assert_allclose(points['direction_rad'], [0, 0], atol=1e-10)
    assert recon['reports']['full']['curves_m'][100].shape == (20, 3)

    optimized, opt_run = _run(config, runs, 'optim', '--resume', str(init_run))
    block = optimized['core_position_parameters']['full']
    assert block['core_calibration'] == calibration
    assert np.asarray(block['strain_gains']).shape == (2, 4)
    assert np.all(np.asarray(block['strain_gains']) > 0)
    replay, _ = _run(config, tmp_path / 'optimized-recon', 'recon', '--resume', str(opt_run))
    np.testing.assert_allclose(replay['core_position_parameters']['full']['strain_gains'],
                               block['strain_gains'])
