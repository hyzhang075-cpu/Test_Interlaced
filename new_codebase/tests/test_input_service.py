import numpy as np
import pytest

from shape_sensing.contracts import InputConfig
from shape_sensing.input_preprocessing import InputService


def test_input_service_selects_physical_channels_without_mutating_inputs():
    """输入模块按物理通道号重排，并维持真实点距离与信号的行对应。"""
    signals = np.arange(15, dtype=float).reshape(5, 3)
    original = signals.copy()
    distance = np.arange(5, dtype=float) * 0.01
    indices = np.arange(10, 15)
    config = InputConfig(sensor_id="test", active_channels=(4, 1))

    frame = InputService(config).prepare(
        signals, distance, indices, channel_ids=(1, 3, 4)
    )

    np.testing.assert_array_equal(frame.signals, original[:, [2, 0]])
    np.testing.assert_array_equal(frame.distance_m, distance)
    np.testing.assert_array_equal(frame.source_indices, indices)
    assert frame.channel_ids == (4, 1)
    np.testing.assert_array_equal(signals, original)


def test_input_service_rejects_missing_channel_and_nonmonotonic_distance():
    """缺失物理通道或错误点序时应在模块1边界明确失败。"""
    service = InputService(InputConfig(sensor_id="test", active_channels=(4, 1)))
    signals = np.ones((3, 2))

    with pytest.raises(ValueError, match="缺少激活物理通道"):
        service.prepare(signals, [0.0, 0.1, 0.2], [0, 1, 2], channel_ids=(1, 3))

    with pytest.raises(ValueError, match="严格递增"):
        service.prepare(signals, [0.0, 0.2, 0.1], [0, 1, 2], channel_ids=(1, 4))
