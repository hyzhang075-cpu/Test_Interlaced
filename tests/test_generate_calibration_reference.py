"""Export contracts: real Sensor CSV loading and independently checked geometry."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_calibration_reference import generate, main


def write_samples(root, name, channels=(1, 2, 3, 4)):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for channel in channels:
        # Sensor reverses physical FBG positions; one negative point is trimmed.
        header = 'Timestamp,SensorIDs,' + ','.join(f'WL_{i}' for i in range(1, 23))
        values = ','.join(str(channel * 100 + i) for i in range(22))
        (folder / f'Channel_{channel}_sample.csv').write_text(
            header + '\n' + '\n'.join(f'{i},ids,{values}' for i in range(12)) + '\n',
            encoding='utf-8',
        )
    return folder


@pytest.fixture
def inputs(tmp_path):
    package = json.loads((ROOT / 'params/08B5_NeTi.json').read_text(encoding='utf-8'))
    package['regions']['full']['radii']['circ'] = [0, 1000]
    package['shared_parameters']['calibration_experiment_parameters']['active_channels'] = [4, 2, 1, 3]
    params = tmp_path / 'params.json'
    params.write_text(json.dumps(package), encoding='utf-8')
    return tmp_path / 'data', params


def test_legacy_flat_export_preserves_channel_order_and_negative_trim(inputs, tmp_path):
    data, params = inputs
    for radius in (0, 1000):
        write_samples(data, f'0-{radius}')
    result = generate(data, params, tmp_path / 'legacy.json')
    assert result['format'] == 'static_calibration_raw_and_ideal_v1'
    assert result['direct'] == 0
    assert result['calibration_shape'] == 'circ'
    assert result['raw_channel_order'] == [4, 2, 1, 3]
    assert result['raw_point_ids_1based'] == list(range(2, 23))
    assert result['raw_point_count'] == 21
    curved = result['radii'][1]
    np.testing.assert_allclose(curved['raw_data'][0], np.array([401, 201, 101, 301]) * 1e-6)
    xyz = np.array(curved['ideal_coordinates_m'])
    assert xyz.shape == (120, 3)
    np.testing.assert_allclose(xyz[-1], [np.sin(.895), 1 - np.cos(.895), 0], atol=1e-12)


def test_all_angles_and_both_end_lengths_use_correct_physical_radii(inputs, tmp_path):
    data, params = inputs
    package = json.loads(params.read_text(encoding='utf-8'))
    circ_radii = [0, 2000, 1500, 1000, 800, 600, 400, 300, 200]
    package['regions']['full']['radii']['circ'] = circ_radii
    params.write_text(json.dumps(package), encoding='utf-8')
    # 60 is intentionally absent from the parameter package's exp_directs.
    for direct in (0, 60, 120, 240):
        root = data / f'{direct}度'
        for radius in circ_radii:
            write_samples(root, f'{direct}-{radius}')
        for group, radii in ((1500, (150, 100, 50)), (2000, (30, 20, 15))):
            for radius in radii:
                write_samples(root, f'{direct}-{group}-{radius}')
    result = generate(data, params, tmp_path / 'mixed.json')
    assert len(result['radii']) == 60
    assert result['directs'] == [0, 60, 120, 240]
    assert result['skipped_empty_directories'] == []
    records = {(r['direct'], r['sample_id']): r for r in result['radii']}
    for record in records.values():
        assert np.asarray(record['raw_data']).shape == (21, 4)
        assert np.asarray(record['ideal_coordinates_m']).shape == (120, 3)
    assert records[60, '60-1000']['source_folder'].endswith('60-1000')
    np.testing.assert_allclose(records[60, '60-1000']['raw_data'][0],
                               np.array([401, 201, 101, 301]) * 1e-6)
    c0 = np.array(records[0, '0-1000']['ideal_coordinates_m'])
    assert c0[-1, 1] == pytest.approx(0, abs=1e-12)
    assert c0[-1, 2] > 0
    for direct, y_sign, z_factor in ((60, -1, .5), (120, -1, -.5), (240, 1, -.5)):
        xyz = np.array(records[direct, f'{direct}-1000']['ideal_coordinates_m'])
        assert np.sign(xyz[-1, 1]) == y_sign
        assert xyz[-1, 2] == pytest.approx(z_factor * c0[-1, 2])
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), np.linalg.norm(c0, axis=1))
    # Analytic endpoints: L=.895m; directory tags give R=1.5m / R=2m.
    # These literals also detect a 6mm error from anchoring at the last real FBG.
    end1500 = records[0, '0-1500-100']
    end2000 = records[0, '0-2000-20']
    assert end1500['end_length_m'] == .15
    assert end1500['base_radius_mm'] == 1500
    assert end2000['end_length_m'] == .05
    assert end2000['base_radius_mm'] == 2000
    np.testing.assert_allclose(end1500['ideal_coordinates_m'][-1],
                               [0.7581646003523581, 0, 0.31046527783359507], atol=1e-12)
    np.testing.assert_allclose(end2000['ideal_coordinates_m'][-1],
                               [0.8162298407143568, 0, 0.21362986120566307], atol=1e-12)
    for sample, x, lateral in (
        ('60-1500-100', .7581646003523581, .31046527783359507),
        ('60-2000-20', .8162298407143568, .21362986120566307),
    ):
        np.testing.assert_allclose(records[60, sample]['ideal_coordinates_m'][-1],
                                   [x, -np.sqrt(3) / 2 * lateral, .5 * lateral], atol=1e-12)
    # Every end curve must match its corresponding full-radius circ up to
    # the junction, for all tail radii and all four bending directions.
    arc = np.asarray(result['ideal_arc_distances_m'])
    for record in records.values():
        if record['calibration_shape'] != 'end':
            continue
        base = record['directory_radius_mm']
        assert record['base_radius_mm'] == base
        circ = records[record['direct'], f"{record['direct']}-{base}"]
        prefix = arc <= record['junction_arc_distance_m']
        np.testing.assert_allclose(np.asarray(record['ideal_coordinates_m'])[prefix],
                                   np.asarray(circ['ideal_coordinates_m'])[prefix], atol=1e-12)
    json.dumps(result, allow_nan=False)


def test_cli_output_defaults_to_overridden_data_directory(inputs):
    data, params = inputs
    for radius in (0, 1000):
        write_samples(data, f'0-{radius}')
    assert main(['--data-dir', str(data), '--sensor-params', str(params)]) == 0
    assert (data / 'calibration_reference_data.json').is_file()


def test_cli_reports_new_60_degree_and_uploaded_direction_counts(inputs, capsys):
    data, params = inputs
    for direct in (60, 120, 240):
        for radius in (0, 1000):
            write_samples(data / f'{direct}度', f'{direct}-{radius}')
    assert main(['--data-dir', str(data), '--sensor-params', str(params)]) == 0
    output = capsys.readouterr().out
    result = json.loads((data / 'calibration_reference_data.json').read_text(encoding='utf-8'))
    assert result['directs'] == [60, 120, 240]
    assert result['skipped_empty_directories'] == []
    for direct in (60, 120, 240):
        assert f'方向 {direct}°: circ=2; end=0' in output


@pytest.mark.parametrize('problem', ['missing', 'unexpected', 'missing_channel'])
def test_radius_errors_identify_input_and_radius(inputs, tmp_path, problem):
    data, params = inputs
    write_samples(data, '0-0')
    if problem != 'missing':
        write_samples(data, '0-1000', channels=(1, 2, 3) if problem == 'missing_channel' else (1, 2, 3, 4))
    if problem == 'unexpected':
        write_samples(data, '0-999')
    with pytest.raises(ValueError) as caught:
        generate(data, params, tmp_path / 'bad.json')
    assert str(data) in str(caught.value)
    assert ('999' if problem == 'unexpected' else '1000') in str(caught.value)
    assert not (tmp_path / 'bad.json').exists()


def test_nonfinite_config_rejected_before_overwriting_output(inputs, tmp_path):
    data, params = inputs
    for radius in (0, 1000):
        write_samples(data, f'0-{radius}')
    package = json.loads(params.read_text(encoding='utf-8'))
    package['shared_parameters']['hardware_parameters']['end_value'] = float('inf')
    params.write_text(json.dumps(package), encoding='utf-8')
    output = tmp_path / 'existing.json'
    output.write_text('original', encoding='utf-8')
    with pytest.raises(ValueError, match='有限|finite'):
        generate(data, params, output)
    assert output.read_text(encoding='utf-8') == 'original'


def test_default_reports_completely_empty_directions_without_inventing_data(inputs, tmp_path):
    data, params = inputs
    for direct in (0, 120, 240):
        for radius in (0, 1000):
            folder = data / f'{direct}度' / f'{direct}-{radius}'
            folder.mkdir(parents=True)
            if direct == 0:
                write_samples(folder.parent, folder.name)
    result = generate(data, params, tmp_path / 'available.json')
    assert result['directs'] == [0]
    assert len(result['radii']) == 2
    assert result['skipped_empty_directories'] == [str(data / '120度'), str(data / '240度')]
    with pytest.raises(ValueError, match='120'):
        generate(data, params, tmp_path / 'empty.json', directs=[120])
    # A partially recorded direction MUST fail rather than being skipped.
    write_samples(data / '120度', '120-0')
    with pytest.raises(ValueError, match='120-1000'):
        generate(data, params, tmp_path / 'partial.json')


@pytest.mark.parametrize('problem', ['nonfinite', 'dimension'])
def test_invalid_optimiser_output_is_rejected_with_source_radius(inputs, tmp_path, monkeypatch, problem):
    from optimiser import Optimiser
    data, params = inputs
    for radius in (0, 1000):
        write_samples(data, f'0-{radius}')
    construct = Optimiser._construct_ideals

    def broken_geometry(self, *args, **kwargs):
        curves = construct(self, *args, **kwargs)
        if problem == 'nonfinite':
            curves[1000][0, 0] = np.nan
        else:
            curves[1000] = curves[1000][:-1]
        return curves

    monkeypatch.setattr(Optimiser, '_construct_ideals', broken_geometry)
    with pytest.raises(ValueError, match='0-1000'):
        generate(data, params, tmp_path / 'invalid.json')
    assert not (tmp_path / 'invalid.json').exists()


def test_interp_num_comes_from_package_and_wrong_raw_count_fails(inputs, tmp_path):
    data, params = inputs
    for radius in (0, 1000):
        write_samples(data, f'0-{radius}')
    package = json.loads(params.read_text(encoding='utf-8'))
    package['shared_parameters']['reconstruction_parameters']['interp_num'] = 60
    params.write_text(json.dumps(package), encoding='utf-8')
    result = generate(data, params, tmp_path / '60.json')
    assert np.asarray(result['radii'][1]['ideal_coordinates_m']).shape == (60, 3)
    package['shared_parameters']['hardware_parameters']['start_value'] = 0
    params.write_text(json.dumps(package), encoding='utf-8')
    with pytest.raises(ValueError, match='21'):
        generate(data, params, tmp_path / 'wrong-count.json')


def test_end_only_package_radii_override_and_legacy_profile(inputs, tmp_path):
    data, params = inputs
    package = json.loads(params.read_text(encoding='utf-8'))
    package['regions']['full']['radii']['end'] = {'1500': [100]}
    params.write_text(json.dumps(package), encoding='utf-8')
    write_samples(data, '0-1500-100')
    result = generate(data, params, tmp_path / 'legacy-end.json',
                      calibration_shape='end', end_profile='legacy')
    record, = result['radii']
    assert result['calibration_shape'] == 'end'
    assert record['base_radius_mm'] == 1500
    assert record['end_length_m'] == .05
    np.testing.assert_allclose(record['ideal_coordinates_m'][-1],
                               [0.835008416159492, 0.26773214670181344, 0], atol=1e-12)
