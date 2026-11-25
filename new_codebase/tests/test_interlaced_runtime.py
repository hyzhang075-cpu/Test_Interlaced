"""Seven-channel raw acquisition must share the static spatial preprocessing."""
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shape_sensing.contracts import InputConfig
from shape_sensing.input_preprocessing import InputService
from shape_sensing.input_preprocessing._kernel import InputKernel
from shape_sensing.runtime.source import InterrogatorSource


def config(**kwargs):
    return InputConfig(sensor_id='interlaced-live', core_position_calib=True,
        active_channels=[4, 2, 1, 5, 3, 7, 6], start_value=10, end_value=8,
        samp_dist_preset=list(range(0, 200, 10)),
        samp_dist_preset_sub=list(range(5, 205, 10)), interlaced_grating=True, **kwargs)


@pytest.mark.parametrize('trim', [0., -0.05])
def test_live_raw_frame_matches_static_merge_and_crop(trim):
    service = InputService(config(straight_trim=trim))
    raw = np.arange(140, dtype=float).reshape(20, 7) * 1e-6
    before = raw.copy()
    live = service.preprocess(raw, acquired=True)
    static = service.preprocess(raw[:, np.argsort(service.config.active_channels)])
    np.testing.assert_allclose(live.signals, static.signals)
    np.testing.assert_array_equal(live.distance_m, static.distance_m)
    np.testing.assert_array_equal(live.source_indices, static.source_indices)
    np.testing.assert_array_equal(raw, before)
    assert live.channel_ids == (4, 2, 5, 7)
    assert live.signals.shape == ((40 if trim == 0 else len(live.distance_m)), 4)
    with pytest.raises(ValueError, match='差分帧'):
        service.preprocess(np.zeros((40, 4)), acquired=True)


@pytest.mark.parametrize('stored', [False, True])
def test_fragmented_seven_channel_stream_zero_and_freshness(tmp_path, monkeypatch, stored):
    cfg = config(straight_trim=-.05)
    absolute = 1550 + np.arange(140).reshape(20, 7) * .001
    channels = cfg.active_channels
    def samples(matrix, cols):
        return [[100*channels[c]+r+1, matrix[r, c]] for c in cols for r in range(20)]
    def packet(values):
        body = json.dumps(values).encode()
        return struct.pack('<I', len(body)) + body
    initial = packet(samples(absolute, range(7)))
    shifted = absolute + np.arange(1, 8)[None, :] * .01
    # Channel 7's update arrives later; never emit a partially updated frame.
    chunks = iter([initial[:9], initial[9:], packet(samples(shifted, range(6))),
                   packet(samples(shifted, [6])), b''])
    class Socket:
        def settimeout(self, value): pass
        def connect(self, address): pass
        def recv(self, size): return next(chunks)
        def shutdown(self, value): pass
        def close(self): pass
    monkeypatch.setattr('shape_sensing.runtime.source.socket.socket', lambda *a: Socket())
    monkeypatch.setattr(InputKernel, '_parse_packet', lambda self, body: json.loads(body))
    if stored:
        path = tmp_path / 'zero.json'
        path.write_text(json.dumps(dict(format_version=1, sensor_id=cfg.sensor_id,
            active_channels=channels, point_count=20, channels={str(ch):
            {'wavelengths_nm': absolute[:, c].tolist()} for c, ch in enumerate(channels)})))
        cfg = replace(cfg, use_stored_zero=True, zero_reference_path=str(path))
    source = InterrogatorSource(cfg)
    source.open()
    source.last_push = -1e10
    first = source.read()
    assert source.kernel.total_points < 40  # Trim applies only after merging.
    assert source.kernel.current_snapshot.shape == (20, 7)
    np.testing.assert_allclose(first, 0, atol=1e-12)
    source.last_push = -1e10
    later = source.read()
    np.testing.assert_allclose(later, np.broadcast_to(np.arange(1, 8)*1e-5, (20, 7)), atol=1e-12)
    assert source.read() is None
    source.close()


@pytest.mark.parametrize('rule', ['ht', 'TNN'])
def test_11b8_package_raw_frame_reaches_reconstruction(rule):
    from shape_sensing.cli import parse_main_args, configure_calibration_args
    from shape_sensing.optimization.batch import optimizer_parameters
    from shape_sensing.pipelines.tasks import build_session
    root = Path(__file__).parents[1]
    args = parse_main_args(['--task', 'demo',
        '--sensor_params', str(root/'packages/11B8_CoreCalib-full-v1.json'), '--frame_rule', rule])
    session = build_session(args, parameter_builder=optimizer_parameters, configure_args=configure_calibration_args)
    raw = np.broadcast_to(np.arange(7)*1e-6, (20, 7)).copy()
    response = session.input.preprocess(raw, acquired=True, design=session.design)
    assert response.signals.shape == (40, 4)
    shape, _ = session.process_frame(raw, acquired=True, direct=args.direction_reference_direct)
    assert shape.frame_rule == rule
    assert np.isfinite(shape.position_m).all()
    assert shape.position_m.shape[1] == 3


def test_interlaced_hippo_fails_before_socket_open(monkeypatch):
    monkeypatch.setattr('shape_sensing.runtime.source.socket.socket',
                        lambda *a: pytest.fail('unsupported geometry opened device'))
    with pytest.raises(ValueError, match='FiberTecc'):
        InterrogatorSource(replace(config(), interrogator='Hippo')).open()
