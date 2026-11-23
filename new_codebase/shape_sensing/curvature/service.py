"""模块3真实点物理模型；不读取文件、不执行初始化拟合或重构滤波。"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d
from .model import CurvatureModelMixin
from .geometry import StrainGeometryMixin
from ._segments import SegmentGeometry
from ..contracts import CurvatureResult, CurvatureConfig, DesignGeometry, CalibrationParameters, PreparedFrame

class CurvatureKernel(CurvatureModelMixin, StrainGeometryMixin, SegmentGeometry):
    """只含模块3物理公式与局部分级状态；不继承任何任务引擎。"""
    def __init__(self, config, design, phase_rad, channel_count, multi_region_parameters=None):
        """建立局部物理模型；设计距离为 m，phase_rad 为 rad，channel_count 为通道数。"""
        self.samp_dist = np.array(design.sampling_distance_m, copy=True)
        self.total_points = len(self.samp_dist)
        self.R = design.radius_m
        self.num_channels_recon = channel_count
        self.active_angles = design.active_angles_deg
        self.global_phases_set = phase_rad
        self.interlaced_grating = design.interlaced_grating
        self.interlaced_offset = design.interlaced_offset_rad
        self.samp_dist_preset = design.samp_dist_preset_mm
        self.samp_dist_preset_sub = design.samp_dist_preset_sub_mm
        self.segmentation = list(config.segmentation)
        self.num_segments = len(self.segmentation)
        self.base_segments = self._calculate_segments(self.total_points)
        self.discont = config.discont
        self.thresholds = config.thresholds
        self.num_levels = len(config.thresholds) + 1
        self.soft = config.soft
        self.turn_threshold = config.turn_threshold
        self.min_consecutive = config.min_consecutive
        self.sharpness = config.sharpness
        self.end_start = config.end_start
        self.end_weights = config.end_weights
        self.end_mask = np.arange(self.total_points) >= self.total_points + config.end_start
        self.weight_group_ids = config.weight_group_ids
        self.end5cm_independent_weight_score = config.end5cm_independent_weight_score
        self.global_point_ids = design.source_indices.copy() + 1 if len(design.source_indices) else np.arange(1, self.total_points + 1)
        self.multi_region_parameters = (self._normalise_multiregion_parameters(multi_region_parameters)
                                        if multi_region_parameters is not None else None)
        self.last_parameter_assembly = None

    def _moving_average_scipy(self, data, window_size, mode):
        """分级评分窗口；仅平滑评分，绝不平滑输出曲率。"""
        if window_size < 1:
            raise ValueError("Window size must be a positive integer.")
        if window_size > len(data):
            raise ValueError("Window size cannot be larger than the data length.")
        return uniform_filter1d(data, size=window_size, mode=mode, cval=0.0, origin=0)

class CurvatureService:
    """接收真实点帧和显式标定参数，保持输入点数、顺序与坐标。"""
    def __init__(self, config: CurvatureConfig):
        """保存模块3分级与物理配置，不读取数据或执行曲率计算。"""
        self.config = config

    def compute(self, frame: PreparedFrame, design: DesignGeometry,
                parameters: CalibrationParameters, *, direct=0, region='full',
                shape_kind='circ', magnitude_ratio=1.0) -> CurvatureResult:
        """输入signals(P,C)历史应变尺度；返回(P,)曲率1/m、方向rad；不修改输入。

        按region->direct查参数，先计算基础物理曲率，再选等级权重/分段扭转重新求解。
        缺少参数立即ValueError；构造和计算均不读文件、不调用模块2/5。
        """
        block = parameters.parameters_by_region.get(region)
        if block is None:
            raise ValueError(f'缺少区域标定参数: {region}')
        if 'core_calibration' in block:
            from .four_core import compute_four_core
            return compute_four_core(frame, design, block['core_calibration'],
                                     block.get('strain_gains'), magnitude_ratio)
        def direction_value(name):
            """取当前 region/direct 的标定数组副本；slopes 为 [level,segment]。"""
            mapping = block.get(name)
            if mapping is None:
                raise ValueError(f'缺少{name}；请先运行init并通过resume提供初值')
            result = mapping.get(direct, mapping.get(str(direct))) if isinstance(mapping, dict) else mapping
            if result is None:
                raise ValueError(f'{name}缺少direct={direct}')
            return np.array(result, dtype=float, copy=True)
        kernel = CurvatureKernel(self.config, design, float(block.get('global_phases_set', 0.0)),
                                 frame.signals.shape[1], block.get('multi_region_parameters'))
        if not np.array_equal(frame.distance_m, design.sampling_distance_m):
            raise ValueError('帧距离必须与真实点设计逐点一致')
        magnitude, angle = kernel._Strain_2_Cur(
            keys=None, calib_shape=shape_kind, calib_Data=np.array(frame.signals, copy=True),
            weight=direction_value('weights'), slope=direction_value('slopes'),
            magnitude_ratio=magnitude_ratio, direct=direct)
        return CurvatureResult(np.array(frame.distance_m, copy=True), np.array(magnitude, copy=True),
                               np.array(angle, copy=True), np.array(frame.source_indices, copy=True),
                               deepcopy(kernel.last_parameter_assembly or {}))
    def base(self, frame, design, parameters, *, direct=0, region='full'):
        """返回原始分级依据：应变(C,P)、基础曲率(P,)1/m、方向rad和转折区间。

        所有输入保持不变；不加虚拟点、不滤波输出曲率、不拟合初值。
        """
        block = parameters.parameters_by_region[region]
        slopes = block['slopes']
        slopes = slopes.get(direct, slopes.get(str(direct))) if isinstance(slopes, dict) else slopes
        kernel = CurvatureKernel(self.config, design, block['global_phases_set'], frame.signals.shape[1])
        strain, magnitude, angle, segments = kernel._get_base(None, np.array(frame.signals, copy=True), np.array(slopes, copy=True)[0])
        return strain.copy(), np.array(magnitude, copy=True), np.array(angle, copy=True), tuple(segments)

    def scores(self, magnitude, *, thresholds=None, sharpness=None, min_consecutive=None, boundary_mode='constant'):
        """输入基础曲率(P,)1/m，返回阈值×真实点(L-1,P)分级分数。"""
        kernel = _ScoringKernel(self.config)
        return kernel._get_scores(np.array(magnitude, copy=True), thresholds=thresholds,
            sharpness=sharpness, min_consecutive=min_consecutive, boundary_mode=boundary_mode)

    def allocate(self, scores):
        """输入(L-1,P)分数，返回(L,P)等级质量，不改变输入。"""
        return _ScoringKernel(self.config)._alloc_mass(np.array(scores, copy=True))

    def groupwise_scores(self, magnitude, group_ids, **kwargs):
        """输入(P,)基础曲率与(P,)权重组索引；输出分数及组内基础曲率诊断。"""
        return _ScoringKernel(self.config)._get_groupwise_weight_scores(np.array(magnitude, copy=True), group_ids, **kwargs)

class _ScoringKernel(CurvatureModelMixin):
    """只保存模块3分级参数，不依赖其他模块对象。"""
    def __init__(self, config):
        """复制评分所需阈值和连续性配置，不持有输入曲率数组。"""
        self.thresholds = config.thresholds
        self.num_levels = len(config.thresholds) + 1
        self.sharpness = config.sharpness
        self.min_consecutive = config.min_consecutive

    _moving_average_scipy = CurvatureKernel._moving_average_scipy


class GeometryService:
    """模块3基础物理几何API；模块2可在尚无标定结果时调用。"""
    def __init__(self, design: DesignGeometry, channel_count: int):
        """保存静态几何与通道数；不拟合、不读取文件。"""
        self.design = design
        self.channel_count = channel_count

    def dynamic_angles(self, *, phase_rad, mode='init', segments=None, slopes=None):
        """输入相位rad、分段索引/斜率rad/m；返回通道×真实点(C,P)动态角度rad。"""
        kernel = CurvatureKernel(CurvatureConfig(), self.design, phase_rad, self.channel_count)
        return kernel._calculate_dynamic_angles(mode, segments, slopes)

    def curvature(self, strain, angles_rad):
        """输入应变(C,P)与角度(C,P)rad；返回曲率(P,)1/m和方向(P,)rad。"""
        kernel = CurvatureKernel(CurvatureConfig(), self.design, 0., self.channel_count)
        magnitude, angle = kernel._calculate_curvature_and_angle(np.array(strain, copy=True), np.array(angles_rad, copy=True))
        return np.array(magnitude, copy=True), np.array(angle, copy=True)
