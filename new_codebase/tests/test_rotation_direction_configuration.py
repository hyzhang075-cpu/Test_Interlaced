"""Direction mappings and historical shape lists must rotate the intended curve."""
import numpy as np
import pytest
import yaml
from shape_sensing.cli import parse_main_args
from shape_sensing.reconstruction.compat import Reconstructor


@pytest.mark.parametrize('rotation,sign', [
    ({0: -1, 120: 1, 240: -1}, 1),
    ({'0': -1, '120': 1, '240': -1}, 1),
    ([-1, 1], 1),
    (-1, -1),
])
def test_reconstruct_uses_direction_mapping_or_shape_list(rotation, sign):
    engine = Reconstructor.__new__(Reconstructor)
    engine.calib_shapes = ['circ', 'S']
    engine.processed_Data = {'S': {120: {100: np.zeros((2, 3))}}}
    engine.weights = {120: [1.]}
    engine.slopes = {120: [[0.]]}
    engine.radii = {120: {'S': [100]}}
    engine.rot_dir = rotation
    engine.rot_ang = 0.
    points = np.array([[0., 1., 0.], [0., 0., 1.]])
    distance = np.array([0., 1.])
    engine._reconstruct_frame = lambda **kwargs: (None, None, None, distance, points, None, None, None, None)
    actual, _ = engine.reconstruct_shapes('S', 120)
    expected = points @ engine._get_rot_x(120, rot_dir=sign).T
    np.testing.assert_allclose(actual[100], expected)


@pytest.mark.parametrize('rotation', [{'0': -1, '120': 1}, [-1, 1], -1])
def test_rotation_yaml_and_cli_parse_same_representation(tmp_path, rotation):
    path = tmp_path / 'sensor.yaml'
    path.write_text(yaml.safe_dump({'sensor_id': 'ROT', 'R': .0001,
        'samp_dist_preset': [0., 10.], 'rot_dir': rotation}), encoding='utf-8')
    configured = parse_main_args(['--config', str(path), '--task', 'init'])
    explicit = parse_main_args(['--task', 'init', '--rot_dir', repr(rotation)])
    expected = {0: -1, 120: 1} if isinstance(rotation, dict) else rotation
    assert configured.rot_dir == expected
    assert explicit.rot_dir == rotation
