"""Analytic geometry and shared static/live entry regression for HT."""
from pathlib import Path
import numpy as np
import pytest
from shape_sensing.reconstruction.integration import ShapeIntegrator
from shape_sensing.reconstruction.service import ReconstructionService
from shape_sensing.contracts import CurvatureResult, DesignGeometry, ReconstructionConfig
from shape_sensing.cli import parse_main_args, configure_calibration_args, optimizer_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.pipelines.tasks import module_parameters

ROOT = Path(__file__).resolve().parents[1]


def test_ht_matches_integrated_reference():
    import json
    fixture = json.loads((Path(__file__).parent/'fixtures/ht_reference.json').read_text(encoding='utf-8'))
    actual = ShapeIntegrator('ht').integrate(fixture['curvature'], fixture['angle'], fixture['distance'], [1.,0.,0.])
    for value, expected in zip(actual, fixture['expected']):
        np.testing.assert_allclose(value, expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize('kappa', [0., 1e-13, 2.])
@pytest.mark.parametrize('angle', [0., .7, -2.])
def test_ht_analytic_arc_and_frame(kappa, angle):
    s = np.linspace(0., 1., 101) ** 1.3
    r, t, y, z = ShapeIntegrator('ht').integrate(np.full(s.size, kappa), np.full(s.size, angle), s, [1., 0., 0.])
    phi = kappa*s
    lateral = .5*kappa*s*s*np.sinc(phi/(2*np.pi))**2
    expected = np.column_stack((s*np.sinc(phi/np.pi), lateral*np.cos(angle), lateral*np.sin(angle)))
    np.testing.assert_allclose(r, expected, atol=2e-13)
    frames = np.stack((t, y, z), axis=-1)
    np.testing.assert_allclose(frames.transpose(0,2,1) @ frames, np.broadcast_to(np.eye(3), frames.shape), atol=2e-14)
    np.testing.assert_allclose(np.linalg.det(frames), 1., atol=2e-14)


def test_ht_static_service_preserves_state_and_diagnostics():
    s = np.linspace(0., 1., 12)
    cur = CurvatureResult(s, 1.+s, .4+2*s, np.arange(s.size))
    design = DesignGeometry('test', .0002, s, (1,3,4), {})
    service = ReconstructionService(ReconstructionConfig(frame_rule='ht', zero_init=False, end_point=False))
    state = None
    for _ in range(3):
        prepared = service.prepare(cur, design, [1.,0.,0.], state)
        result = service.integrate(prepared)
        assert result.frame_rule == 'ht'
        assert np.all(np.isfinite(result.position_m))
        np.testing.assert_array_equal(result.diagnostics['cur_interp'], prepared.curvature_per_m)
        np.testing.assert_array_equal(result.diagnostics['ang_interp'], prepared.direction_rad)
        state = prepared.state
        assert state is not None


@pytest.mark.parametrize('task', ['recon', 'demo'])
@pytest.mark.parametrize('rule', ['TNN', 'ht'])
def test_package_cli_override_and_live_frames(task, rule):
    args = parse_main_args(['--sensor_params', str(ROOT/'packages/09B7_NeTi.json'), '--task', task, '--frame_rule', rule])
    configure_calibration_args(args)
    assert args.frame_rule == rule
    if task == 'demo':
        session = TaskSession(module_parameters(args, optimizer_parameters), task='demo')
        signals = np.zeros((len(session.design.sampling_distance_m), len(args.active_channels)))
        state = None
        for _ in range(3):
            shape, state = session.process_frame(signals, state=state, acquired=True, direct=args.direction_reference_direct)
            assert shape.frame_rule == rule
            assert shape.position_m.shape == (args.interp_num, 3)
            assert np.all(np.isfinite(shape.position_m))
            assert state is not None


def test_ht_yaml_choice():
    from shape_sensing.configuration.loader import validate_working_config
    config = validate_working_config(dict(sensor_id='S', R=.001, samp_dist_preset=[0,10], frame_rule='ht'))
    assert config['frame_rule'] == 'ht'
