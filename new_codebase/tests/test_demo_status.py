from types import SimpleNamespace

from shape_sensing.runtime.demo import _DemoStatusPrinter


def test_per_frame_preprocessing_does_not_repeat_geometry(capsys):
    import numpy as np
    from shape_sensing.contracts import InputConfig
    from shape_sensing.input_preprocessing import InputService
    service = InputService(InputConfig(sensor_id='quiet', active_channels=(1,2,3,4),
        core_position_calib=True, samp_dist_preset=(0,10,20), start_value=-5, end_value=2))
    design = service.design()
    assert '初始采样距离' in capsys.readouterr().out
    signals = np.zeros((len(design.sampling_distance_m), 4))
    for _ in range(3):
        service.preprocess(signals, acquired=True, design=design)
    assert capsys.readouterr().out == ''


def test_demo_status_throttles_counts_effective_frames_and_uses_actual_receiver(capsys):
    ticks = iter([0., .5, 1., 1.5, 2.5])
    transport = SimpleNamespace(receiver=('192.168.1.10', 8888))
    status = _DemoStatusPrinter(transport, clock=lambda: next(ticks))
    def frame(index, insertion=None):
        return SimpleNamespace(frame_index=index, insert_value_mm=insertion,
                               shape=SimpleNamespace(frame_rule='ht'))
    status(frame(9))
    assert capsys.readouterr().out == ''
    status(frame(19))
    line = capsys.readouterr().out
    assert '算法 ht | 已处理 20 帧 | 处理帧率 20.0 FPS' in line
    assert '当前推进量 -- | 发送ip：192.168.1.10' in line
    status(frame(24))
    assert capsys.readouterr().out == ''
    transport.receiver = ('192.168.1.20', 9999)
    status(frame(34, 12.34))
    line = capsys.readouterr().out
    assert '已处理 35 帧 | 处理帧率 10.0 FPS' in line
    assert '当前推进量 12.3 mm | 发送ip：192.168.1.20' in line
