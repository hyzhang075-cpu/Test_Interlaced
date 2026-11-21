"""Stored zero references are optional until explicitly enabled for demo."""
import pytest
import yaml
from shape_sensing.cli import build_argument_parser, parse_main_args, configure_calibration_args


def config_file(tmp_path):
    data = {
        'sensor_id': 'ZERO-SCOPE', 'R': .0001,
        'samp_dist_preset': [0., 20., 110., 130., 190., 290., 390., 490.],
        'start_value': 0., 'end_value': 10., 'full_segmentation': [2, 2, 2, 2],
        'use_stored_zero': True, 'zero_reference_path': None,
        'calibration_regions': {'full': {'length_m': None, 'folder_path': 'data',
            'radii': {'circ': [0, 30]}, 'thresholds': [1.], 'end_weights': False,
            'end_start': -1, 'weights': None, 'slopes': None}},
    }
    path = tmp_path / 'sensor.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def test_stored_zero_is_opt_in_by_default():
    parser = build_argument_parser()
    assert parser.get_default('use_stored_zero') is False
    assert parser.get_default('zero_reference_path') is None


@pytest.mark.parametrize('task', ['init', 'optim', 'score', 'recon', 'ang'])
def test_calibration_tasks_accept_unprepared_demo_reference(tmp_path, task):
    args = parse_main_args(['--config', str(config_file(tmp_path)), '--task', task])
    configure_calibration_args(args)
    assert args.use_stored_zero is False
    assert args.zero_reference_path is None


def test_demo_requires_reference_only_when_enabled(tmp_path):
    path = config_file(tmp_path)
    args = parse_main_args(['--config', str(path), '--task', 'demo'])
    with pytest.raises(ValueError, match='zero_reference_path'):
        configure_calibration_args(args)
    args = parse_main_args(['--config', str(path), '--task', 'demo', '--use_stored_zero', 'False'])
    configure_calibration_args(args)
    assert args.use_stored_zero is False
