"""模块4候选参数搜索；预测和评估通过显式回调提供。"""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from ._kernel import OptimizationKernel
from ..contracts import CalibrationParameters

class OptimizationService(OptimizationKernel):
    """只继承本模块搜索核，不继承重构器；候选不污染输入seed或最优结果。"""
    def __init__(self, config, curvature_config, dataset, parameters, *, region,
                 predict_shapes, evaluate_shapes, compute_curvature, score_inputs):
        """显式接收标定初值和3+5预测、6评估回调；构造不拟合、不读文件。"""
        for name, value in asdict(config).items():
            setattr(self, name, deepcopy(value))
        block = deepcopy(parameters.parameters_by_region[region])
        if block.get('slopes') is None or block.get('weights') is None:
            raise ValueError('优化需要完整初值；请先运行init并通过--resume载入结果')
        self.weights = {int(key): np.array(value, dtype=float, copy=True) for key, value in block['weights'].items()}
        self.slopes = {int(key): np.array(value, dtype=float, copy=True) for key, value in block['slopes'].items()}
        self.global_phases_set = float(block.get('global_phases_set', 0.0))
        self.rot_ang = float(block.get('rot_ang', 0.0))
        self.region = region
        self.processed_Data = deepcopy(dataset.data)
        self.radii = deepcopy(dataset.radii)
        self.calib_shapes = list(dataset.radii[next(iter(dataset.radii))])
        self.total_points = len(dataset.design.sampling_distance_m)
        self.samp_dist = dataset.design.sampling_distance_m.copy()
        self.global_point_ids = dataset.design.source_indices.copy() + 1
        self.end_weights = curvature_config.end_weights
        self.end_start = curvature_config.end_start
        self.weight_group_ids = curvature_config.weight_group_ids
        self.init_weight = curvature_config.init_weight
        self.thresholds = curvature_config.thresholds
        self.num_levels = len(self.thresholds) + 1
        self.end5cm_independent_weight_score = curvature_config.end5cm_independent_weight_score
        self.num_segments = len(curvature_config.segmentation)
        self.slopes_shape = self.slopes[self.direct].shape
        self.weight_bounds = [(config.weight_bounds[0] * self.init_weight,
                               config.weight_bounds[1] * self.init_weight)]
        self._coverage_support_map = None
        self._predict = predict_shapes
        self._evaluate = evaluate_shapes
        self._curvature = compute_curvature
        self._score_inputs = score_inputs
        self._get_freeze_idx()

    def parameters(self):
        """返回独立region->direct参数副本，角度rad、斜率rad/m、权重无量纲。"""
        return CalibrationParameters({self.region: {
            'weights': deepcopy(self.weights), 'slopes': deepcopy(self.slopes),
            'global_phases_set': self.global_phases_set, 'rot_ang': self.rot_ang}})

    def reconstruct_shapes(self, calib_shape, direct, verbose=False, force_ideal_magnitude=False):
        """用当前 region/direct 参数调用预测器；返回形状树及其距离数组（m）。"""
        return self._predict(self.parameters(), calib_shape, direct, force_ideal_magnitude)

    def _compute_error(self, interp_dist, rs):
        """评估指定插值距离（m）上的曲线树，并缓存此次返回的指标。"""
        result = self._evaluate(interp_dist, rs, self.direct)
        for name, value in result.metrics.items():
            setattr(self, name, deepcopy(value))
        return result

    def _Strain_2_Cur(self, keys, calib_shape, calib_Data, weight, slope, **kwargs):
        """按 keys 取出真实点信号叶子，委托显式回调计算曲率及方向。"""
        values = calib_Data
        for key in keys or ():
            values = values[key]
        return self._curvature(values, self.parameters(), self.direct, calib_shape)

    def _get_base(self, keys, calib_Data, base_slopes, verbose=False):
        """计算权重评分所需基础量；keys 指向标定数据树中的叶子。"""
        return self._score_inputs(keys, calib_Data, base_slopes, self.parameters())

    def _get_scores(self, magnitude, **kwargs):
        """将真实点基础曲率（1/m）转换为分级权重评分。"""
        return self._score_inputs(None, None, None, self.parameters(), magnitude=magnitude, **kwargs)

    def _get_groupwise_weight_scores(self, magnitude, group_ids, **kwargs):
        """按真实点权重组计算分级分数；group_ids 与曲率点数对应。"""
        return self._score_inputs(None, None, None, self.parameters(), magnitude=magnitude, group_ids=group_ids, **kwargs)

    def _alloc_mass(self, scores):
        """把分级评分分配为各等级权重质量并返回新数组。"""
        return self._score_inputs(None, None, None, self.parameters(), scores=scores)

    def _print_errors(self):
        """输出最近一次预测评估产生的逐点与末端误差。"""
        print(f"逐点误差: {self.avg_point_err}; 末端误差: {self.avg_end_err}")

    def optimize(self, verbose=True):
        """运行原两阶段搜索/回滚规则，返回参数契约和阶段报告。"""
        report = self.optim_coeff(verbose=verbose)
        return self.parameters(), report


def validate_dataset(dataset, direct):
    """公开数据校验：配置半径与模块1实际加载半径必须一致；不修改输入。"""
    problems = []
    for shape, configured in dataset.radii.get(direct, {}).items():
        if not isinstance(configured, (list, tuple, np.ndarray)):
            continue
        expected = [OptimizationKernel._normalise_radius_key(value) for value in configured]
        loaded = dataset.data.get(shape, {}).get(direct, {})
        actual = [OptimizationKernel._normalise_radius_key(value) for value in loaded]
        missing = [value for value in expected if value not in set(actual)]
        unexpected = [value for value in actual if value not in set(expected)]
        if missing or unexpected:
            problems.append(f'direct={direct}, shape={shape}: missing configured radii={missing}; unexpected data radii={unexpected}')
    if problems:
        raise ValueError('定标数据半径与配置不一致；请显式设置 --radii。' + ' | '.join(problems))
