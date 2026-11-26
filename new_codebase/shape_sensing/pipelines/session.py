"""八模块任务组合；任务对象只编排公开service，无跨模块算法继承。"""
from copy import deepcopy
from dataclasses import replace
import numpy as np
from ..contracts import (ModuleConfiguration, CalibrationParameters, PreparedFrame,
                         ShapeResult, EvaluationReference, ReconstructionState)
from ..input_preprocessing import InputService
from ..initialization.service import InitializationService
from ..curvature.service import CurvatureService
from ..reconstruction.service import ReconstructionService
from ..evaluation.service import EvaluationService
from ..common.utils import dict_recur_paths, dict_recur_unflatten

def validate_seed(parameters, *, region='full'):
    """校验任务所需显式初值，在文件读取和模块2拟合前失败并指导续算。"""
    missing = [name for name in ('weights', 'slopes', 'global_phases_set')
               if parameters.get(name) is None]
    if missing:
        raise ValueError(f'区域{region}缺少标定初值 ' + ', '.join(missing)
                         + '；请先运行该区域init，再用--resume载入初值，或显式提供这些参数')


def _initial_weights(directions, config):
    levels = len(config.thresholds) + 1
    groups = config.weight_group_ids
    split_end = config.end_weights and groups is None
    group_count = max(groups) + 1 if groups is not None else (2 if split_end else 1)
    initial_weight = 1.0 if split_end else config.init_weight
    return {direct: initial_weight * np.ones(levels * group_count) for direct in directions}


def _initial_parameters(result, dataset, config, region, previous):
    """由单区域拟合结果生成完整运行参数，并保留显式 weights/slopes 覆盖。

    拟合相位为 rad，slopes_init 按方向（deg）与分段给出 rad/m。
    输出各方向的无量纲 [组数×等级数] 权重及 [等级, 分段] 扭转斜率。
    S 形布局复制拟合分段以覆盖两段；旧末端双组布局以单位权重开始。
    """
    levels = len(config.curvature.thresholds) + 1
    block = {
        'global_phases_set': result.global_phases_set,
        'rot_ang': config.reconstruction.rot_ang,
        'weights': _initial_weights(dataset.radii, config.curvature),
        'slopes': {},
    }
    shapes = dataset.radii[next(iter(dataset.radii))]
    for direct, slopes in result.slopes_init.items():
        segments = np.concatenate((slopes, slopes)) if 'S' in shapes else slopes
        block['slopes'][direct] = np.array([segments for _ in range(levels)])
    for name in ('weights', 'slopes'):
        if previous.get(name) is not None:
            block[name] = deepcopy(previous[name])
    return CalibrationParameters({region: block}, source='initialization')


class TaskSession:
    """一个任务的显式数据与服务所有者；初始化拟合仅在initialize调用发生。"""
    def __init__(self, parameters, *, task='recon', region='full', dataset=None):
        """从已校验参数建立单任务会话；region/direct 分层持有标定值，不自动读取 CSV。"""
        self.config = ModuleConfiguration.from_mapping(parameters)
        self.task = task
        self.region = region
        self.input = InputService(self.config.input)
        self.design = self.input.design()
        self.curvature = CurvatureService(self.config.curvature)
        self.config = replace(self.config, reconstruction=replace(self.config.reconstruction,
            zero_init=self.design.physical_start_m is not None,
            end_point=self.design.physical_end_m is not None))
        self.configuration = self.config
        self.reconstruction = ReconstructionService(self.config.reconstruction)
        self.evaluation = EvaluationService(self.config.evaluation)
        self.dataset = dataset
        self.initialization_result = None
        self.optimization = None
        self.calibration_context = {}
        self.phase_reference = None
        self._core_demo_curvature = None
        self.parameters = CalibrationParameters({region: {
            'weights': deepcopy(parameters.get('weights')), 'slopes': deepcopy(parameters.get('slopes')),
            'global_phases_set': parameters.get('global_phases_set'),
            'rot_ang': parameters.get('rot_ang', 0.0),
            'multi_region_parameters': deepcopy(parameters.get('multi_region_parameters'))}})
        if self.config.input.core_position_calib:
            source = parameters.get('core_position_parameters', {})
            regions = (region,)
            if task == 'demo':
                from ..curvature.core_regions import CORE_REGION_ORDER, CoreDemoCurvature
                if parameters.get('multi_region_calib', False):
                    if region != 'full':
                        raise ValueError('四芯区域拼接需要 full 实时采样网格')
                    regions = CORE_REGION_ORDER
            blocks = {name: deepcopy(source.get(name, {})) for name in regions}
            block = blocks[region]
            if task != 'demo' and block.get('core_position') is not None and 'weights' not in block:
                directions = block['core_position_metadata']['calibration']['directions_deg']
                block['weights'] = _initial_weights(directions, self.config.curvature)
                block['weight_layout'] = self._weight_layout()
            self.parameters = CalibrationParameters(blocks, source='four_core_common')
            self.point_curvature = {}
        self.cur_interp = np.array([])
        self.metrics = None
        if task != 'init':
            self.require_seed()
        if task == 'demo' and self.config.input.core_position_calib:
            self._core_demo_curvature = CoreDemoCurvature(self.curvature, self.design, self.parameters,
                regions, self.config.runtime.direction_reference_direct)

    def require_seed(self):
        """任务开始校验参数，缺少初值明确指导init+resume，不偷偷执行模块2。"""
        block = self.parameters.parameters_by_region[self.region]
        if self.config.input.core_position_calib:
            if (not block.get('core_position')
                    or block.get('core_position_metadata', {}).get('schema_version') != 6):
                raise ValueError(f'区域 {self.region} 需要 schema 6 芯位参数，请先运行 init 并通过 --resume 加载')
            if 'weight_layout' in block and block['weight_layout'] != self._weight_layout():
                raise ValueError(f'区域 {self.region} 的曲率阈值或权重分组与快照不一致，请重新运行 init')
            return
        validate_seed(block, region=self.region)

    def load_data(self):
        """仅静态任务显式读取标定数据；实时构造与replay不读标定CSV。"""
        if self.dataset is None:
            self.dataset = self.input.load_data()
        self.design = self.dataset.design
        return self.dataset

    def initialize(self, verbose=False):
        """拟合静态 [P,C] 信号并保存模块2诊断，展开本区域各方向的等级初值。"""
        dataset = self.load_data()
        if self.config.input.core_position_calib:
            service = InitializationService(self.config.initialization,
                                            input_config=self.config.input, region=self.region)
        else:
            service = InitializationService(self.config.initialization)
        result = service.fit(dataset, verbose=verbose)
        if self.config.input.core_position_calib:
            block = dict(core_position=result.core_position,
                         core_position_metadata=result.core_position_metadata,
                         weights=_initial_weights(dataset.radii, self.config.curvature),
                         weight_layout=self._weight_layout())
            self.parameters = CalibrationParameters({self.region: block}, source='four_core_common')
        else:
            previous = self.parameters.parameters_by_region[self.region]
            self.parameters = _initial_parameters(result, dataset, self.config, self.region, previous)
        self.initialization_result = result
        return result

    def _weight_layout(self):
        config = self.config.curvature
        groups = None if config.weight_group_ids is None else list(config.weight_group_ids)
        return dict(thresholds=list(config.thresholds), end_weights=config.end_weights,
                    soft=config.soft, sharpness=config.sharpness, min_consecutive=config.min_consecutive,
                    independent_weight_score=config.end5cm_independent_weight_score,
                    end_start=config.end_start, weight_group_ids=groups)

    def frame(self, values):
        """输入已预处理的静态信号 [P,C]，返回带物理通道及设计几何的真实点帧。

        信号保持数据集的历史应变尺度。本帧拥有信号副本，距离 [P] m 和索引来自设计。
        """
        channels = self.dataset.channel_ids if self.dataset is not None else self.design.active_channels
        return PreparedFrame.from_design(self.design, np.array(values, copy=True), channels)

    def process_frame(self, signals, state=None, direct=0, tangent_seed=None, temporal_filter=None, acquired=False, parameter_selector=None):
        """输入动态或回放信号 [P,C]（历史差分尺度），沿模块 1→3→5 计算。

        返回形状坐标 [N,3] m 和下帧的方向展开状态。模块 1 与模块 3 共用设计几何。
        """
        frame = self.input.preprocess(signals, acquired=acquired, design=self.design)
        parameters = self.parameters if parameter_selector is None else parameter_selector(frame, self.parameters)
        if self._core_demo_curvature is None:
            curvature = self.curvature.compute(frame, self.design, parameters, direct=direct, region=self.region)
        else:
            curvature = self._core_demo_curvature.compute(frame, parameters, direct=direct)
        prepared = self.reconstruction.prepare(curvature, self.design, np.array([1., 0., 0.]) if tangent_seed is None else tangent_seed, state=state)
        if temporal_filter is not None:
            prepared = temporal_filter(prepared)
        shape = self.reconstruction.integrate(prepared)
        return shape, prepared.state

    def _predict_shapes(self, parameters, calib_shape, direct, force_ideal_magnitude=False,
                        record_points=False):
        """按指定形状和方向遍历静态数据，返回位置 m 的曲线树及积分距离 m。"""
        dataset = self.load_data()
        tree = dataset.data[calib_shape][direct]
        paths = dict_recur_paths(tree)
        output = {}
        for path in paths:
            values = tree
            for key in path:
                values = values[key]
            curvature = self.curvature.compute(self.frame(values), self.design, parameters,
                direct=direct, region=self.region, shape_kind=calib_shape)
            if self.config.input.core_position_calib:
                core = parameters.parameters_by_region[self.region]['core_position_metadata']['calibration']
                fixture_angle = core['core2_azimuth_deg'] + core['bend_side_offset_deg']
                angle = curvature.direction_rad - np.deg2rad(fixture_angle + core['direction_sign'] * direct)
                angle = np.arctan2(np.sin(angle), np.cos(angle))
                if record_points:
                    self.point_curvature[path] = {
                        'wl_ids': curvature.source_indices + 1,
                        'distance_m': curvature.distance_m,
                        'curvature_per_m': curvature.curvature_per_m,
                        'direction_rad': angle,
                        **curvature.diagnostics,
                    }
                curvature = replace(curvature, direction_rad=angle)
            ideal = None
            if force_ideal_magnitude:
                reference = EvaluationReference(radius_m=float(path[0])/1000., shape_kind=calib_shape,
                    sample_coordinates_m=self.design.sampling_distance_m,
                    metadata={'idx_m': float(path[1])/1000.} if len(path)>1 else {})
                ideal = lambda distance, reference=reference: self.evaluation.ideal_curvature(distance, reference)
            prepared = self.reconstruction.prepare(curvature, self.design, np.array([1.,0.,0.]), ideal_curvature=ideal)
            result = self.reconstruction.integrate(prepared)
            self.cur_interp = prepared.curvature_per_m.copy()
            if self.config.input.core_position_calib:
                output[path] = result.position_m
                continue
            direction = self.config.input.rot_dir
            if isinstance(direction, dict):
                direction = direction.get(direct, direction.get(str(direct)))
            elif not np.isscalar(direction):
                direction = direction[self.calib_shapes.index(calib_shape)]
            transformed = self.reconstruction.transform_shape(result, direct=direct, rot_dir=direction,
                rot_ang=parameters.parameters_by_region[self.region].get('rot_ang', 0.))
            output[path] = transformed.position_m
        return dict_recur_unflatten(output), result.distance_m

    def reconstruct_shapes(self, calib_shape, direct, verbose=False, force_ideal_magnitude=False):
        """公开静态重构入口；叶子坐标(N,3)m，距离(N,)m。"""
        return self._predict_shapes(self.parameters, calib_shape, direct, force_ideal_magnitude,
                                    record_points=self.config.input.core_position_calib)

    def evaluate(self, interp_dist, rs, direct=None):
        """只调用模块6并保存此次快照，供报告/图形读取。"""
        dataset = self.load_data()
        self.metrics = self.evaluation.evaluate_batch(interp_dist, rs, dataset.radii,
            getattr(self, 'direct', self.config.optimization.direct) if direct is None else direct,
            self.design.sampling_distance_m, self.cur_interp)
        return self.metrics

    def _compute_error(self, interp_dist, rs):
        """兼容优化回调：评估插值距离 m 上的形状树并返回 EvaluationResult。"""
        return self.evaluate(interp_dist, rs)

    def _score_inputs(self, keys, data, slopes, parameters, **kwargs):
        """将优化评分回调分派到模块3；数据叶子为真实点 [P,C] 信号。"""
        if 'group_ids' in kwargs:
            return self.curvature.groupwise_scores(**kwargs)
        if 'magnitude' in kwargs:
            return self.curvature.scores(**kwargs)
        if 'scores' in kwargs:
            return self.curvature.allocate(kwargs['scores'])
        values = data
        for key in keys or ():
            values = values[key]
        return self.curvature.base(self.frame(values), self.design, parameters,
                                   direct=self.config.optimization.direct, region=self.region)

    def _get_base(self, keys, calib_Data, base_slopes, verbose=False):
        """兼容模块4的基础评分回调，返回模块3的真实点基础量。"""
        return self._score_inputs(keys, calib_Data, base_slopes, self.parameters)

    def _get_scores(self, magnitude, **kwargs):
        """兼容模块4评分回调；magnitude 为真实点曲率 [P]，单位 1/m。"""
        return self._score_inputs(None, None, None, self.parameters, magnitude=magnitude, **kwargs)

    def _alloc_mass(self, scores):
        """兼容模块4质量分配回调，输入等级分数并返回权重质量。"""
        return self._score_inputs(None, None, None, self.parameters, scores=scores)

    def _get_groupwise_weight_scores(self, magnitude, group_ids, **kwargs):
        """兼容模块4分组评分回调；曲率和组索引均按真实点排列。"""
        return self.curvature.groupwise_scores(magnitude, group_ids, **kwargs)

    def validate_calibration_data_radii(self):
        """校验当前 direct 的配置半径与静态 Dataset 中实际加载半径一致。"""
        from ..optimization.service import validate_dataset
        return validate_dataset(self.load_data(), self.config.optimization.direct)

    def optim_coeff(self, verbose=True):
        """模块4通过独立参数候选回调3+5、6，禁止模块2隐式初始化。"""
        if 'channel_weights' in self.parameters.parameters_by_region.get(self.region, {}):
            raise ValueError('逐通道分级参数须使用通道校准优化器；普通weights优化只适用于标量权重')
        from ..optimization.service import OptimizationService
        dataset = self.load_data()
        def real_curvature(values, parameters, direct, kind):
            """将真实点信号 [P,C] 交模块3，返回曲率 [P] 1/m 与方向 [P] rad。"""
            result = self.curvature.compute(self.frame(values), self.design, parameters,
                direct=direct, region=self.region, shape_kind=kind)
            return result.curvature_per_m, result.direction_rad
        self.optimization = OptimizationService(self.config.optimization, self.config.curvature,
            dataset, self.parameters, region=self.region, predict_shapes=self._predict_shapes,
            evaluate_shapes=lambda dist, results, direct: self.evaluate(dist, results, direct),
            compute_curvature=real_curvature, score_inputs=self._score_inputs)
        parameters, report = self.optimization.optimize(verbose)
        self.parameters = parameters
        return report

    def _print_errors(self):
        """输出最近一次模块6评估快照；尚无结果时输出空映射。"""
        print(self.metrics.metrics if self.metrics is not None else {})

    def __getattr__(self, name):
        """旧报告字段投影；只暴露显式契约，不导入或创建旧引擎。"""
        if name in ('weights', 'slopes', 'global_phases_set', 'rot_ang'):
            return self.parameters.parameters_by_region[self.region][name]
        if name in ('processed_Data', 'radii', 'calib_shapes', 'plot_Data'):
            dataset = self.load_data()
            return {'processed_Data':dataset.data, 'radii':dataset.radii,
                    'calib_shapes':list(dataset.radii[next(iter(dataset.radii))]), 'plot_Data':dataset.plot_data}[name]
        if name == 'global_point_ids':
            return self.design.source_indices + 1
        if name == 'num_levels':
            return len(self.config.curvature.thresholds) + 1
        if name in ('samp_dist', 'total_points'):
            return self.design.sampling_distance_m if name=='samp_dist' else len(self.design.sampling_distance_m)
        if self.optimization is not None and hasattr(self.optimization, name):
            return getattr(self.optimization, name)
        if self.metrics is not None and name in self.metrics.metrics:
            return self.metrics.metrics[name]
        for config in (self.config.curvature, self.config.optimization, self.config.reconstruction, self.config.input):
            if hasattr(config, name):
                return getattr(config, name)
        raise AttributeError(name)
