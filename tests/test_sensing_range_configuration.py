"""Channel-key loading and interrogator-specific sensing ranges."""
import numpy as np
import pytest
import yaml
from shape_sensing.configuration import load_working_config
from shape_sensing.input_preprocessing.sensor import Sensor


def load_config(tmp_path, ranges):
    path = tmp_path / 'sensor.yaml'
    path.write_text(yaml.safe_dump({'sensor_id': 'RANGES', 'R': .0001,
        'samp_dist_preset': [0., 10., 20.], 'sensing_ranges': ranges}), encoding='utf-8')
    return load_working_config(path)


def test_yaml_physical_channel_keys_are_integers(tmp_path):
    config = load_config(tmp_path, {'1': [0, 8], 3: [2, 8]})
    assert config['sensing_ranges'] == {1: [0, 8], 3: [2, 8]}


def test_yaml_rejects_duplicate_normalized_physical_channels(tmp_path):
    with pytest.raises(ValueError, match='sensing_ranges'):
        load_config(tmp_path, {'1': [0, 8], 1: [2, 8]})


def fake_fbg(self, **kwargs):
    self.Data = {'circ': {0: {0: np.zeros((3, 3))}}}


def fake_ofdr(self, **kwargs):
    self.processed_Data = {}
    self.plot_Data = {}


@pytest.mark.parametrize('interrogator', ['FiberTecc', 'Hippo'])
def test_fbg_does_not_require_luna_ranges(tmp_path, monkeypatch, interrogator):
    monkeypatch.setattr(Sensor, 'read_data_FBG', fake_fbg)
    sensor = Sensor(sensor_id='RANGES', start_value=0., end_value=0., folder_path=str(tmp_path),
        centre_fibre=False, active_channels=[2, 5, 7], interrogator=interrogator,
        samp_dist_preset=[0., 10., 20.], sensing_ranges={})
    assert sensor.processed_Data['circ'][0][0].shape == (3, 3)
    assert sensor.sensing_ranges == {}


def test_luna_uses_normalized_ranges_and_common_length(tmp_path, monkeypatch):
    config = load_config(tmp_path, {'1': [0, 8], '3': [2, 8]})
    monkeypatch.setattr(Sensor, 'read_data_OFDR', fake_ofdr)
    sensor = Sensor(sensor_id='RANGES', start_value=0., end_value=0., folder_path=str(tmp_path),
        centre_fibre=False, active_channels=[1, 3], interrogator='LUNA',
        sensing_ranges=config['sensing_ranges'])
    assert sensor.min_len == 6
    assert sensor.sensing_ranges == {1: (0, 6), 3: (2, 8)}


def test_luna_still_reports_missing_channel_ranges(tmp_path):
    with pytest.raises(ValueError, match='sensing_ranges'):
        Sensor(sensor_id='RANGES', start_value=0., end_value=0., folder_path=str(tmp_path),
            active_channels=[1, 3], interrogator='LUNA', sensing_ranges={1: (0, 8)})
