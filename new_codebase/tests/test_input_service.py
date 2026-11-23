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


def test_core_calibration_retains_zero_referenced_signals_in_wl_and_channel_order(tmp_path):
    folder = tmp_path / "0-200"
    folder.mkdir()
    for channel in (1, 2, 3, 4):
        (folder / f"Channel_{channel}_data.csv").write_text(
            "Timestamp,SensorIDs,WL_3,WL_1,WL_2\n"
            f"0,ids,{channel * 10 + 3},{channel * 10 + 1},{channel * 10 + 2}\n",
            encoding="utf-8",
        )
    dataset = InputService(InputConfig(
        sensor_id="core-test", core_position_calib=True, active_channels=[4, 2, 1, 3],
        start_value=-5, end_value=2, samp_dist_preset=[0, 10, 20], reverse=True,
        exp_directs=[0], radii={"circ": [200]},
    )).load_data(tmp_path)

    assert dataset.channel_ids == (4, 2, 1, 3)
    np.testing.assert_array_equal(dataset.design.source_indices, [2, 1])
    np.testing.assert_allclose(dataset.data["circ"][0][200],
                               np.array([[43, 23, 13, 33], [42, 22, 12, 32]]) * 1e-6)


def test_core_calibration_merges_seven_interlaced_inputs_with_effective_channel_ids(tmp_path):
    folder = tmp_path / "0-200"
    folder.mkdir()
    for channel in range(1, 8):
        (folder / f"Channel_{channel}_data.csv").write_text(
            f"Timestamp,SensorIDs,WL_1,WL_2\n0,ids,{channel * 10},{channel * 10 + 10}\n",
            encoding="utf-8",
        )
    dataset = InputService(InputConfig(
        sensor_id="interlaced-core", core_position_calib=True,
        active_channels=[4, 2, 1, 5, 3, 7, 6], start_value=0, end_value=0,
        samp_dist_preset=[0, 10], samp_dist_preset_sub=[5, 15], interlaced_grating=True,
        exp_directs=[0], radii={"circ": [200]},
    )).load_data(tmp_path)

    assert dataset.channel_ids == (4, 2, 5, 7)
    np.testing.assert_array_equal(dataset.design.source_indices, [0, 1, 2, 3])
    np.testing.assert_allclose(dataset.data["circ"][0][200], np.array([
        [40, 20, 50, 70], [45, 10, 30, 60], [50, 30, 60, 80], [55, 20, 40, 70],
    ]) * 1e-6)


def test_core_calibration_preserves_center_column_in_live_frame():
    service = InputService(InputConfig(
        sensor_id="core-live", core_position_calib=True, active_channels=[4, 2, 1, 3],
        start_value=0, end_value=0, samp_dist_preset=[0, 10],
    ))
    signals = np.arange(8, dtype=float).reshape(2, 4)
    frame = service.preprocess(signals, acquired=True)

    assert frame.channel_ids == (4, 2, 1, 3)
    np.testing.assert_array_equal(frame.signals, signals)


def test_core_calibration_uses_selected_circle_states_and_reports_missing_bends(tmp_path):
    for directory in ("0-0", "0-200", "0-300", "120-200", "240-200", "0-200-S", "0-200-OTHER"):
        folder = tmp_path / directory
        folder.mkdir()
        for channel in (1, 2, 3, 4):
            (folder / f"Channel_{channel}_data.csv").write_text(
                "Timestamp,SensorIDs,WL_1,WL_2\n0,ids,10,20\n", encoding="utf-8",
            )
    config = dict(sensor_id="core-test", core_position_calib=True, active_channels=[1, 2, 3, 4],
                  start_value=0, end_value=0, samp_dist_preset=[0, 10], exp_directs=[0, 120])
    selected = InputService(InputConfig(**config, radii={"circ": [0, 200]})).load_data(tmp_path)

    assert {direction: set(cases) for direction, cases in selected.data["circ"].items()} == {
        0: {0, 200}, 120: {200},
    }
    assert selected.radii[120]["circ"] == [200]
    assert not selected.data["S"]

    service = InputService(InputConfig(**config, radii={"circ": [0, 200, 300]}))
    with pytest.raises(ValueError, match="120-300"):
        service.load_data(tmp_path)
