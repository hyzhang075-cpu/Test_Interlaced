"""Four-core demo region ownership, zeroing and wire compatibility."""
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from shape_sensing.contracts import InputConfig
from shape_sensing.input_preprocessing._kernel import InputKernel
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.runtime.demo import LiveFrameProcessor
from shape_sensing.runtime.service import RuntimeService
from shape_sensing.runtime.source import InterrogatorSource
from shape_sensing.runtime.transport import UdpTransport


CHANNELS = (1, 2, 3, 4)
GRID = np.arange(1, 7) * 0.01
POSITIONS = np.array([[0., 0.], [100., 0.], [-50., 85.], [-50., -85.]]) * 1e-6


def _block(first, scale, weight):
    indices = list(range(first, 6))
    origin = first * 0.01
    cores = {
        f'ch{channel}': dict(r_um=scale*np.linalg.norm(position)*1e6,
                            alpha_rad=float(np.arctan2(position[0], position[1])))
        for channel, position in zip(CHANNELS, POSITIONS)
    }
    return {
        'core_position': [dict(wl_id=i+1, cores=cores, response_model='four_core_common')
                          for i in indices],
        'core_position_metadata': {
            'schema_version': 6, 'channel_configuration': {'response_channels': list(CHANNELS)},
            'calibration': dict(wl_ids=[i+1 for i in indices], strain_sensitivity_pm_per_ue=1.,
                                strain_projection_sign=1),
            'scope': dict(crop_origin_m=origin, global_sample_dist_m=GRID[indices].tolist(),
                          local_sample_dist_m=(GRID[indices]-origin).tolist()),
        },
        'weights': {'240': [weight]},
        'weight_layout': dict(thresholds=[], end_weights=False, end_start=-5, soft=True,
                              sharpness=4.8, min_consecutive=1, weight_group_ids=None,
                              independent_weight_score=False),
    }


def _parameters(multi=True, core=True):
    return dict(sensor_id='fixture', start_value=10., end_value=10., ideal_len=.07,
                samp_dist_preset=list(range(0, 60, 10)), active_channels=list(CHANNELS),
                R=.0001, thresholds=[], min_consecutive=1, end_start=-5,
                core_position_calib=core, multi_region_calib=multi, direction_reference_direct=240,
                interp_num=24, move_filter=1, frame_rule='TNN', curvature_interp='vector_pchip',
                segmentation=[6], weights={240: [1.]}, slopes={240: [[0.]]}, global_phases_set=0.,
                core_position_parameters={
                    'full': _block(0, 1., 1.), 'end15cm': _block(2, 2., 4.), 'end5cm': _block(4, 3., 9.),
                })


@pytest.mark.parametrize('multi', [False, True])
def test_demo_pairs_core_positions_and_weights_by_global_wl(multi):
    session = TaskSession(_parameters(multi), task='demo')
    vector = np.array([1.2, 0.5])
    signals = np.tile(POSITIONS @ vector + 3e-5, (6, 1))
    shape, _ = session.process_frame(signals, acquired=True, direct=240)
    factors = np.repeat([1, 2, 3], 2) if multi else np.ones(6)
    np.testing.assert_allclose(shape.diagnostics['k_y_per_m'], vector[0] * factors)
    np.testing.assert_allclose(shape.diagnostics['k_z_per_m'], vector[1] * factors)
    np.testing.assert_allclose(shape.diagnostics['effective_weights'], factors**2)
    np.testing.assert_allclose(shape.diagnostics['common_response_b_strain'], 3e-5)
    expected = ['full'] * 6 if not multi else ['full'] * 2 + ['end15cm'] * 2 + ['end5cm'] * 2
    assert list(shape.diagnostics['source_region_by_wl']) == expected


def test_demo_rejects_missing_region_or_weight_before_frames():
    parameters = _parameters()
    parameters['core_position_parameters']['end5cm']['weights'] = {}
    with pytest.raises(ValueError, match='end5cm.*240'):
        TaskSession(parameters, task='demo')
    del parameters['core_position_parameters']['end5cm']
    with pytest.raises(ValueError, match='end5cm'):
        TaskSession(parameters, task='demo')


def test_region_probe_uses_merged_responses_for_seven_channel_acquisition():
    from dataclasses import replace
    from shape_sensing.curvature.core_regions import CoreDemoCurvature

    session = TaskSession(_parameters(), task='demo')
    design = replace(session.design, active_channels=(1, 2, 7, 4, 3, 6, 5),
                     interlaced_grating=True)
    assembler = CoreDemoCurvature(session.curvature, design, session.parameters,
                                 ('full', 'end15cm', 'end5cm'), 240)
    signals = np.tile(POSITIONS @ np.array([1.2, .5]), (6, 1))
    result = assembler.compute(session.frame(signals), session.parameters, direct=240)
    np.testing.assert_allclose(result.curvature_per_m,
                               np.linalg.norm([1.2, .5]) * np.repeat([1, 2, 3], 2))


def test_spliced_region_scores_the_entire_overlap_before_selecting_owned_points():
    parameters = _parameters()
    middle = parameters['core_position_parameters']['end15cm']
    middle['weights']['240'] = [2., 4.]
    middle['weight_layout'].update(thresholds=[0.6], soft=False, min_consecutive=3)
    session = TaskSession(parameters, task='demo')
    signals = np.tile(POSITIONS @ np.array([1.2, .5]), (6, 1))
    shape, _ = session.process_frame(signals, acquired=True, direct=240)
    # Both owned points reach the high bin only when the overlapping support is retained.
    np.testing.assert_allclose(shape.diagnostics['effective_weights'], [1, 1, 4, 4, 9, 9])


def _packet(samples):
    body = json.dumps(samples).encode()
    return struct.pack('<I', len(body)) + body


class Socket:
    def __init__(self, packets):
        self.packets = iter(packets)
        self.read_count = 0

    def connect(self, address):
        pass

    def shutdown(self, how):
        pass

    def close(self):
        pass

    def settimeout(self, value):
        pass

    def recv(self, size):
        self.read_count += 1
        return next(self.packets, b'')

    def sendto(self, data, address):
        pass


@pytest.mark.parametrize('stored', [False, True])
def test_fibertecc_waits_for_full_snapshot_and_uses_matching_zero(tmp_path, monkeypatch, stored):
    values = [[100*ch+wl, 1550.1+ch/10+wl/1000] for ch in CHANNELS for wl in (1, 2, 3)]
    later = [[sensor, wave+0.02] for sensor, wave in values]
    # Update the first point before the snapshot completes. Per-point zeroing would retain its old value.
    first = [[values[0][0], values[0][1]-0.01]]
    packets = [_packet(first), _packet(values), _packet(later)]
    socket = Socket(packets)
    monkeypatch.setattr('shape_sensing.runtime.source.socket.socket', lambda *args: socket)
    monkeypatch.setattr(InputKernel, '_parse_packet', lambda self, data: json.loads(data))
    ticks = iter([1., 2., 3., 4.])
    monkeypatch.setattr('shape_sensing.runtime.source.time.monotonic', lambda: next(ticks))
    zero = tmp_path / 'zero.json'
    if stored:
        zero.write_text(json.dumps(dict(format_version=1, sensor_id='fixture',
            active_channels=list(CHANNELS), point_count=3, channels={
                str(ch): {'wavelengths_nm': [wave for sensor, wave in values if sensor//100==ch]}
                for ch in CHANNELS
            })), encoding='utf-8')
    source = InterrogatorSource(InputConfig(
        sensor_id='fixture', start_value=10., end_value=10., samp_dist_preset=[0, 10, 20],
        core_position_calib=True, active_channels=list(CHANNELS), use_stored_zero=stored,
        zero_reference_path=str(zero) if stored else None,
    ))
    source.open()
    np.testing.assert_allclose(source.read(), np.zeros((3,4)), atol=1e-13)
    assert socket.read_count == 2
    np.testing.assert_allclose(source.read(), np.full((3,4), 2e-5), atol=1e-13)


@pytest.mark.parametrize('stored', [False, True])
def test_hippo_zero_uses_the_same_cropped_wls_as_the_live_frame(tmp_path, monkeypatch, stored):
    channels = [4, 1, 3, 2]
    wave = np.zeros((8, 16))
    wave[:4, :3] = 1550. + np.arange(12).reshape(4, 3)*.01
    incomplete = wave.copy()
    incomplete[2, 2] = 0.
    shifted = wave.copy()
    shifted[:4, 1:3] += np.array([.02, .03, .04, .05])[:, None]
    frames = iter([incomplete, wave, shifted])
    monkeypatch.setattr('shape_sensing.runtime.source.socket.socket', lambda *args: Socket([b'ack']))
    monkeypatch.setattr(InputKernel, '_recv_010c_once', lambda self, connection: next(frames))
    zero = tmp_path / 'zero.json'
    if stored:
        zero.write_text(json.dumps(dict(format_version=1, sensor_id='fixture',
            active_channels=channels, point_count=3, channels={
                str(ch): {'wavelengths_nm': wave[ch-1, :3].tolist()} for ch in channels
            })), encoding='utf-8')
    source = InterrogatorSource(InputConfig(
        sensor_id='fixture', start_value=-5., end_value=10., samp_dist_preset=[0, 10, 20],
        interrogator='Hippo', core_position_calib=True, active_channels=channels,
        use_stored_zero=stored, zero_reference_path=str(zero) if stored else None,
    ))
    source.open()
    np.testing.assert_allclose(source.read(), np.zeros((2,4)), atol=1e-13)
    np.testing.assert_allclose(source.read(), np.tile([5e-5, 2e-5, 4e-5, 3e-5], (2,1)), atol=1e-13)


@pytest.mark.parametrize('enabled', [False, True])
def test_udp_schema_is_identical_for_slope_and_spliced_core_demo(enabled):
    packets = []
    signals = np.tile(POSITIONS @ np.array([1.2, 0.5]) + 3e-5, (6, 1))
    for core in (False, True):
        parameters = _parameters(multi=core, core=core)
        if not core:
            parameters.update(thresholds=[2.], weights={240: [1., 1.]}, slopes={240: [[0.], [0.]]})
        session = TaskSession(parameters, task='demo')
        options = SimpleNamespace(direction_blend_enabled=False, demo_kalman_enabled=enabled,
                                  demo_kalman_Rk=1e-4, demo_kalman_Bk=1e-5)
        runtime = RuntimeService(LiveFrameProcessor(session, options), direct=240, moving_window=1,
                                 anchor_enabled=enabled, anchor_offset_m=.03)
        transport = UdpTransport(runtime)
        transport.socket = SimpleNamespace(sendto=lambda data, address: packets.append(json.loads(data)))
        runtime.publisher = transport.publish
        frame, = runtime.replay([signals])
        packet = packets[-1]
        np.testing.assert_allclose(packet['data'], frame.shape.position_m * 1000)
        np.testing.assert_array_equal(packet['raw_data'], signals)
    def signature(value):
        if isinstance(value, dict):
            return {key: signature(item) for key, item in value.items()}
        if isinstance(value, list):
            return [signature(item) for item in value]
        return type(value)
    assert signature(packets[0]) == signature(packets[1])

@pytest.mark.parametrize('stored', [False, True])
@pytest.mark.parametrize('trim', [0., -.03])
@pytest.mark.parametrize('interrogator', ['FiberTecc', 'Hippo'])
def test_interlaced_live_zero_merge_and_crop_match_static(tmp_path, monkeypatch, stored, trim, interrogator):
    from shape_sensing.input_preprocessing import InputService
    channels = [1, 2, 7, 4, 3, 6, 5]
    absolute = 1550. + np.arange(21).reshape(3, 7) * .001
    delta = (np.arange(21).reshape(3, 7) + 1) * .00001
    packets = [_packet([[100*ch+wl+1, values[wl, col]]
                        for col, ch in enumerate(channels) for wl in range(3)])
               for values in (absolute, absolute+delta)]
    connection = Socket(packets if interrogator == 'FiberTecc' else [b'ack'])
    monkeypatch.setattr('shape_sensing.runtime.source.socket.socket', lambda *args: connection)
    monkeypatch.setattr(InputKernel, '_parse_packet', lambda self, data: json.loads(data))
    monkeypatch.setattr('shape_sensing.runtime.source.time.monotonic', iter([1., 2.]).__next__)
    waves = []
    for values in (absolute, absolute+delta):
        wave = np.zeros((8, 16))
        wave[np.asarray(channels)-1, :3] = values.T
        waves.append(wave)
    iterator = iter(waves)
    monkeypatch.setattr(InputKernel, '_recv_010c_once', lambda self, sock: next(iterator))
    zero = tmp_path / 'interlaced_zero.json'
    zero.write_text(json.dumps(dict(format_version=1, sensor_id='interlaced',
        active_channels=channels, point_count=3, channels={
            str(ch): {'wavelengths_nm': absolute[:, col].tolist()}
            for col, ch in enumerate(channels)})), encoding='utf-8')
    config = InputConfig(sensor_id='interlaced', interrogator=interrogator,
        start_value=10., end_value=10., samp_dist_preset=[0, 20, 40],
        samp_dist_preset_sub=[-10, 10, 30], interlaced_grating=True,
        core_position_calib=True, active_channels=channels, straight_trim=trim,
        use_stored_zero=stored, zero_reference_path=str(zero) if stored else None)
    source = InterrogatorSource(config)
    source.open()
    service = InputService(config)
    expected = service.preprocess((absolute+delta-absolute)*1000e-6,
                                  channel_ids=tuple(channels))
    np.testing.assert_allclose(source.read(), np.zeros_like(expected.signals), atol=1e-13)
    assert source.kernel.current_snapshot.shape == (3, 7)
    np.testing.assert_allclose(source.read(), expected.signals, atol=1e-13)
    assert expected.channel_ids == (1, 2, 4, 6)
    source.close()
