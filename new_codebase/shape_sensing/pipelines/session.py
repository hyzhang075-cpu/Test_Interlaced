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
        self.parameters = CalibrationParameters({region: {
            'weights': deepcopy(parameters.get('weights')), 'slopes': deepcopy(parameters.get('slopes')),
            'global_phases_set': parameters.get('global_phases_set'),
            'rot_ang': parameters.get('rot_ang', 0.0),
            'multi_region_parameters': deepcopy(parameters.get('multi_region_parameters'))}})
        self.cur_interp = np.array([])
        self.metrics = None
        if task != 'init':
            self.require_seed()

    def require_seed(self):
        """任务开始校验参数，缺少初值明确指导init+resume，不偷偷执行模块2。"""
        validate_seed(self.parameters.parameters_by_region[self.region], region=self.region)

    def load_data(self):
        """仅静态任务显式读取标定数据；实时构造与replay不读标定CSV。"""
        if self.dataset is None:
            self.dataset = self.input.load_data()
        self.design = self.dataset.design
        return self.dataset

    def initialize(self, verbose=False):
        """模块1->2，展开原权重/slopes初值；不触发优化。"""
        dataset = self.load_data()
        result = InitializationService(self.config.initialization).fit(dataset, region=self.region, verbose=verbose)
        block = deepcopy(result.seed.parameters_by_region[self.region])
        levels = len(self.config.curvature.thresholds) + 1
        groups = self.config.curvature.weight_group_ids
        count = levels * (max(groups) + 1 if groups is not None else (2 if self.config.curvature.end_weights else 1))
        block['weights'] = {direct: self.config.curvature.init_weight * np.ones(count) for direct in dataset.radii}
        if self.config.curvature.end_weights and groups is None:
            block['weights'] = {direct: np.ones(count) for direct in dataset.radii}
        block['slopes'] = {direct: np.array([np.concatenate((slopes, slopes)) if 'S' in self.calib_shapes else slopes
                                            for _ in range(levels)]) for direct, slopes in block['slopes_init'].items()}
        previous = self.parameters.parameters_by_region[self.region]
        for name in ('weights', 'slopes'):
            if previous.get(name) is not None:
                block[name] = deepcopy(previous[name])
        block['rot_ang'] = self.config.reconstruction.rot_ang
        self.parameters = CalibrationParameters({self.region: block})
        self.initialization_result = result
        return result

    def frame(self, values):
        """将真实点信号叶子 values[P,C] 包装为 PreparedFrame，复制设计距离 m 与索引。"""
        """静态已预处理(P,C)叶子包装，保留真实点及通道顺序。"""
        channels = self.dataset.channel_ids if self.dataset is not None else tuple(self.design.active_channels)
        return PreparedFrame(self.design.sampling_distance_m.copy(), self.design.source_indices.copy(),
                             np.array(values, copy=True), channels)

    def process_frame(self, signals, state=None, direct=0, tangent_seed=None, temporal_filter=None, acquired=False, parameter_selector=None):
        """动态与replay同帧入口：1预处理->3真实点物理模型->5重构，无窗口。"""
        frame = self.input.preprocess(signals, acquired=acquired)
        parameters = self.parameters if parameter_selector is None else parameter_selector(frame, self.parameters)
        curvature = self.curvature.compute(frame, self.design, parameters, direct=direct, region=self.region)
        prepared = self.reconstruction.prepare(curvature, self.design, np.array([1., 0., 0.]) if tangent_seed is None else tangent_seed, state=state)
        if temporal_filter is not None:
            prepared = temporal_filter(prepared)
        shape = self.reconstruction.integrate(prepared)
        return shape, prepared.diagnostics['state_out']

    def _predict_shapes(self, parameters, calib_shape, direct, force_ideal_magnitude=False):
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
            ideal = None
            if force_ideal_magnitude:
                reference = EvaluationReference(radius_m=float(path[0])/1000., shape_kind=calib_shape,
                    sample_coordinates_m=self.design.sampling_distance_m,
                    metadata={'idx_m': float(path[1])/1000.} if len(path)>1 else {})
                ideal = lambda distance, reference=reference: self.evaluation.ideal_curvature(distance, reference)
            prepared = self.reconstruction.prepare(curvature, self.design, np.array([1.,0.,0.]), ideal_curvature=ideal)
            result = self.reconstruction.integrate(prepared)
            self.cur_interp = prepared.curvature_per_m.copy()
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
        return self._predict_shapes(self.parameters, calib_shape, direct, force_ideal_magnitude)

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
