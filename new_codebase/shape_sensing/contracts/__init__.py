"""模块间显式数据契约。

距离均以 m 表示，曲率以 1/m 表示，角度以 rad 表示。数组字段均为 NumPy
数组；生产者不得原地修改调用方传入的数据。旧契约类保留供兼容适配层使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class ShapeResult:
    """积分输出；位置和标架为 [N,3]，位置 m，三个标架向量无量纲。"""
    frame_rule: str
    position_m: np.ndarray
    tangent: np.ndarray | None = None
    normal: np.ndarray | None = None
    binormal: np.ndarray | None = None
    distance_m: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DesignGeometry:
    """静态几何设计。radius_m为m；sampling_distance_m为全局真实点弧长(P,)m。"""
    sensor_id: str
    radius_m: float
    sampling_distance_m: np.ndarray
    active_channels: tuple[int, ...]
    channel_angles_rad: Mapping[int, float]
    physical_start_m: float | None = None
    physical_end_m: float | None = None
    region_point_indices: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    segment_mapping: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    source_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    straight_trim: float | int = 0.0
    active_angles_deg: tuple[float, ...] | None = None
    interlaced_grating: bool = False
    interlaced_offset_rad: float = 0.0
    samp_dist_preset_mm: tuple[float, ...] = ()
    samp_dist_preset_sub_mm: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PreparedFrame:
    """模块1输出。signals为(P,C)，channel_ids[j]标识第j列物理通道；source_indices是全局真实点0-based索引。"""
    distance_m: np.ndarray
    source_indices: np.ndarray
    signals: np.ndarray
    channel_ids: tuple[int, ...]

    @classmethod
    def from_design(cls, design: DesignGeometry, signals: np.ndarray,
                    channel_ids: tuple[int, ...]) -> "PreparedFrame":
        """绑定静态设计与已预处理的 [P,C] 信号，保留其历史量纲。

        校验点数及物理通道列，帧拥有生产者交入的信号数组。距离 [P] m 与
        真实点全局索引 [P] 来自设计，帧只持有其只读视图。
        """
        distance = design.sampling_distance_m.view()
        indices = design.source_indices.view()
        if distance.ndim != 1 or indices.shape != distance.shape:
            raise ValueError('设计距离与真实点索引必须是等长 [P] 数组')
        if signals.shape != (distance.size, len(channel_ids)) or len(channel_ids) != len(set(channel_ids)):
            raise ValueError('预处理信号必须为 [P,C]，通道列标签必须唯一且与列数一致')
        distance.flags.writeable = False
        indices.flags.writeable = False
        return cls(distance, indices, signals, channel_ids)


@dataclass(frozen=True)
class CalibrationParameters:
    """模块3、4和实时方向选择共用的区域参数。

    region块内以实验方向（deg）索引 weights 和 slopes。
    weights 按组优先、组内曲率等级排列（无量纲），slopes 为 [等级, 分段] rad/m。
    source 表示内存中这组参数的来源，发布状态仍由任务报告和产物清单决定。
    """
    parameters_by_region: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source: str = "configured"


@dataclass(frozen=True)
class CurvatureResult:
    """模块3真实采样点曲率；三数组长度相同且与输入点位逐点对应。"""
    distance_m: np.ndarray
    curvature_per_m: np.ndarray
    direction_rad: np.ndarray
    source_indices: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PreparedCurvature:
    """模块5供积分使用的曲率场；允许重构专用滤波、展开与虚拟点插值。"""
    distance_m: np.ndarray
    curvature_per_m: np.ndarray
    direction_rad: np.ndarray
    tangent_seed: np.ndarray
    source_indices: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    state: "ReconstructionState | None" = None

@dataclass(frozen=True)
class EvaluationReference:
    """模块6误差评估的显式参考；坐标单位m、曲率1/m、方向rad、角度deg。"""
    ideal_position_m: Any = None
    ideal_curvature_per_m: Any = None
    direction_rad: Any = None
    radius_m: float | None = None
    shape_kind: str | None = None
    angle_deg: float | None = None
    sample_coordinates_m: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ReconstructionState:
    """模块5跨帧展开状态；raw/unwrapped各按含虚拟点的重构点顺序(Q,) rad，不存标定参数。"""
    previous_raw_angles_rad: np.ndarray | None = None
    previous_unwrapped_angles_rad: np.ndarray | None = None
    distance_m: np.ndarray | None = None

@dataclass(frozen=True)
class EvaluationResult:
    """模块6评估值；metrics键为指标名，值为标量或逐点数组。"""
    metrics: Mapping[str, Any]
    details: Mapping[str, Any] = field(default_factory=dict)

# Explicit module configuration projections. Fields retain the external YAML names.
@dataclass(frozen=True)
class InputConfig:
    """模块1读取与几何设置；距离预设mm，R/ideal_len/ss为m，信号保持历史量纲。"""
    sensor_id: str = None
    start_value: float = None
    end_value: float = None
    folder_path: str | None = None
    centre_fibre: bool = True
    core_position_calib: bool = False
    active_channels: list[int] = field(default_factory=lambda: [1, 3, 4])
    rot_dir: int | list[int] | dict[int, int] = -1
    exp_directs: list[int] = field(default_factory=lambda: [0, 120, 240])
    interrogator: str = 'FiberTecc'
    R: float = 0.000235
    ideal_len: float = 1.35
    radii: dict = field(default_factory=lambda: {'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200], 'S': [0, 250, 200, 150], 'end': {2000: [30, 20, 15], 1500: [150, 100, 50]}, 'rot0': {100: [10, 40, 70]}, 'up': {200: [0, 25, 50, 100], 100: [0, 25, 50, 100]}, 'down': {200: [0, 25, 50, 100], 100: [0, 25, 50, 100]}})
    samp_dist_preset: list[float] | None = field(default_factory=lambda: [510, 540, 570, 630, 710, 810, 910, 1010, 1110, 1210, 1310, 1410, 1510, 1610, 1710, 1815])
    active_angles: list[float] | None = None
    reverse: bool = False
    straight_trim: float = 0.0
    ss: float = 0.0026
    N: int = 100
    sensing_ranges: dict = field(default_factory=lambda: {1: (583, 1048), 3: (678, 1146), 4: (749, 1215)})
    smoothing_params: dict = field(default_factory=lambda: {'window_length': 21, 'polyorder': 3})
    IP: str = '192.168.1.75'
    PORT: int = 26546
    data_freq: int = 100
    maxsize: int = 100
    samp_dist_preset_sub: list[float] | None = None
    interlaced_grating: bool = False
    channel_change: bool = False
    channel_layout: dict | None = None
    interlaced_offset: float = 0.0
    use_stored_zero: bool = False
    zero_reference_path: str | None = None
    wavelength_jump_threshold_pm: float = 12000.0
    wavelength_jump_guard_enabled: bool = True

@dataclass(frozen=True)
class InitializationConfig:
    discont: float = 3.141592653589793
    segmentation: tuple[int, ...] = (16,)
    connect: bool = True
    global_phases_set: float | None = None
    interlaced_offset: float = 0.0
    core_position_calib: bool = False
    core_diameter_um: float = 350.0
    core_radius_um: float = 50.0
    core_sensitivity_pm_per_ue: float = 1.0
    core_bend_offset_deg: float = -90.0

@dataclass(frozen=True)
class CurvatureConfig:
    thresholds: tuple[float, ...] = (0.8, 1.5, 2.0, 3.0, 4.0)
    soft: bool = True
    segmentation: tuple[int, ...] = (16,)
    discont: float = 3.141592653589793
    turn_threshold: float = 2.0
    min_consecutive: int = 3
    sharpness: float = 4.8
    end_start: int = -6
    end_weights: bool = False
    init_weight: float = 1.0
    weight_group_ids: tuple[int, ...] | None = None
    end5cm_independent_weight_score: bool = False

@dataclass(frozen=True)
class ReconstructionConfig:
    frame_rule: str = "TNB"
    rot_ang: float = 0.0
    interp_num: int = 120
    move_filter: int = 3
    curvature_interp: str = "hermite"
    nonnegative_curvature: bool = False
    TNB_threshold: float = 0.15
    move_filter_for_vector: bool = False
    kalman_params: Mapping[str, Mapping[str, float]] = field(default_factory=lambda: {
        'angle': {'Rk': 0.001, 'Bk': 0.001},
        'magnitude': {'Rk': 0.0001, 'Bk': 0.01}})
    zero_init: bool = True
    end_point: bool = False
    discont: float = np.pi
    unwrap_threshold: float = 0.65
    align_threshold: float = 0.15

@dataclass(frozen=True)
class OptimizationConfig:
    direct: int = 0
    end_len: float = 0.05
    max_iter: int = 5
    min_loss_diff: float = 1e-6
    method: str = "Powell"
    hyper_params: tuple[float, ...] = (10000.0, 1e-6, 1e-6)
    freeze: tuple[int, ...] | None = None
    straight_threshold: float = 2.0
    rot_threshold: float = 4.0
    k_point: float = 1.0
    k_end: float = 0.0
    k_pitch: float = 1.0
    k_radius: float = 1.0
    weight_bounds: tuple[float, float] = (0.5, 1.0)
    slope_upper_bound: float = 20.0
    end5cm_point_calib: bool = False
    real_point_calib: bool = False
    optim_weights_only: bool = False
    end5cm_coverage_aware_weights: bool = False
    end5cm_min_weight_effective_samples: float = 1.0
    end5cm_min_weight_max_mass: float = 0.05
    end5cm_uncovered_weight_policy: str = 'interpolate_neutral_edges'
    source_sensor_params: str | None = None

@dataclass(frozen=True)
class EvaluationConfig:
    ideal_curvature: bool = False
    end_len: float = 0.05
    straight_threshold: float = 2.0
    rot_threshold: float = 4.0

@dataclass(frozen=True)
class RuntimeConfig:
    """模块8会话与设备状态设置；距离m、阈值1/m、方向deg、噪声沿旧尺度。"""
    imu: bool = False
    imu_com: str = "COM6"
    imu_maxlen: int = 10
    demo_recorder: bool = False
    direction_blend_enabled: bool = False
    direction_reference_direct: int = 0
    direction_min_curvature: float = 0.5
    direction_min_consecutive: int = 3
    direction_consistency_threshold: float = 0.6
    direction_smooth_alpha: float = 0.15
    demo_kalman_enabled: bool = True
    demo_kalman_Rk: float = 0.0001
    demo_kalman_Bk: float = 0.01
    anchor_constraint_enabled: bool = False
    anchor_translation_enabled: bool = False
    anchor_rotation_enabled: bool = False
    anchor_offset_m: float = 0.06

@dataclass(frozen=True)
class ViewConfig:
    total_scale: float = 1000.0
    title: str = "Viewer"
    body_model_path: str = "./model.stl"
    max_curvature: float = 2.3
    back_num: int = 8

@dataclass(frozen=True)
class ModuleConfiguration:
    """按模块显式投影外部工作配置；未知/未归属参数仍由既有loader校验。"""
    input: InputConfig
    initialization: InitializationConfig
    curvature: CurvatureConfig
    reconstruction: ReconstructionConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    view: ViewConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModuleConfiguration":
        """由已通过loader验证的原名配置构建模块视图；不修改输入映射。"""
        def selected(config_type, names):
            return config_type(**{name: data[name] for name in names if name in data})
        return cls(**{
            name: selected(kind, [item.name for item in fields(kind)])
            for name, kind in (
                ('input', InputConfig), ('initialization', InitializationConfig),
                ('curvature', CurvatureConfig), ('reconstruction', ReconstructionConfig),
                ('optimization', OptimizationConfig), ('evaluation', EvaluationConfig),
                ('runtime', RuntimeConfig), ('view', ViewConfig))
        })

@dataclass(frozen=True)
class CalibrationDataset:
    """模块1静态数据；data[shape][direct][radius][case]叶子(P,C)，末层case可省略。"""
    data: Mapping[str, Mapping[int, dict]]
    radii: Mapping[int, Mapping[str, list | dict]]
    design: DesignGeometry
    channel_ids: tuple[int, ...]
    plot_data: Mapping[str, Mapping[int, dict]] = field(default_factory=dict)

@dataclass(frozen=True)
class InitializationResult:
    """模块2输出初值及拟合诊断；所有诊断容器属于此次fit调用。"""
    global_phases_set: float | None
    slopes_init: Mapping[int, list[float]]
    magnitude_results: dict
    angle_results_raw: dict
    angle_results_unwrap: dict
    global_fit_direct: dict
    individual_fits_direct: dict
    core_position: list[dict] | None = None
    core_position_metadata: dict | None = None
