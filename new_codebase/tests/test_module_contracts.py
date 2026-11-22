from pathlib import Path
import numpy as np
from shape_sensing.configuration import load_working_config
from shape_sensing.contracts import (
    CalibrationParameters, DesignGeometry, ModuleConfiguration,
    PreparedFrame, ReconstructionState,
)


def test_09b7_config_projection_and_real_point_geometry():
    config_path = Path(__file__).parents[1] / "configs" / "09B7_NeTi.yaml"
    raw = load_working_config(config_path)
    projected = ModuleConfiguration.from_mapping(raw)
    assert projected.input.sensor_id == raw["sensor_id"]
    assert projected.input.active_channels == raw["active_channels"]

    distance = np.linspace(0.0, 0.86, 22)
    geometry = DesignGeometry(
        sensor_id=raw["sensor_id"],
        radius_m=raw["R"],
        sampling_distance_m=distance,
        active_channels=tuple(raw["active_channels"]),
        channel_angles_rad={1: 0.0, 3: 2.0, 4: 4.0},
        region_point_indices={"full": tuple(range(22))},
    )
    signals = np.zeros((22, 3))
    frame = PreparedFrame(distance, np.arange(22), signals, (1, 3, 4))
    assert geometry.sampling_distance_m.shape == (22,)
    assert frame.signals.shape == (22, 3)
    assert frame.distance_m.shape == (22,)
    assert frame.source_indices[0] == 0 and frame.source_indices[-1] == 21


def test_region_direct_parameters_have_one_authoritative_mapping():
    """从同一映射读取区域斜率 rad/m 和方向相位 rad。"""
    values = {
        "full": {"mode": "region", "slopes": [[1.0, 2.0], [3.0, 4.0]]},
        "direct": {"mode": "direct", "phases_by_direction_rad": {0: 0.1, 120: 0.2}},
    }
    fitted = CalibrationParameters(parameters_by_region=values)
    assert fitted.parameters_by_region["full"]["slopes"][1] == [3.0, 4.0]
    assert fitted.parameters_by_region["direct"]["phases_by_direction_rad"][120] == 0.2
    assert "direct_parameters" not in fitted.__dataclass_fields__
    assert "region_parameters" not in fitted.__dataclass_fields__


def test_reconstruction_state_holds_previous_angles():
    """方向展开历史包含两组 [Q] 弧度数组，无标定参数。"""
    state = ReconstructionState(np.array([0.1, 0.2]), np.array([0.1, 6.48]))
    assert state.previous_raw_angles_rad.shape == (2,)
    assert state.previous_unwrapped_angles_rad.shape == (2,)
    assert "calibration_seed" not in state.__dataclass_fields__
