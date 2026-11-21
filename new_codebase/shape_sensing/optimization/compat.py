# 历史类兼容入口集中于本模块；新生产入口请使用 shape_sensing.optimization.service.OptimizationService。
"""旧Optimiser构造边界；搜索公式统一使用模块4私有核。"""
import numpy as np
from scipy.optimize import minimize
from ._kernel import OptimizationKernel
from ..reconstruction.compat import Reconstructor
from ..evaluation.metrics import EvaluationMixin

class Optimiser(OptimizationKernel, EvaluationMixin, Reconstructor):
    def __init__(self, direct=0, end_len=0.05, max_iter=5, min_loss_diff=1e-6, hyper_params=[10000, 1e-6, 1e-6], freeze=None, method='Powell', straight_threshold=2.0, rot_threshold=4.0, k_point=1.0, k_end=0.0, k_pitch=1.0, k_radius=1.0, weight_bounds=(0.5, 1.0), slope_upper_bound=20.0, *args, end5cm_point_calib=False, real_point_calib=False, **kwargs):
        self.optim_weights_only = bool(kwargs.pop('optim_weights_only', False))
        self.end5cm_coverage_aware_weights = bool(kwargs.pop('end5cm_coverage_aware_weights', False))
        self.end5cm_min_weight_effective_samples = float(
            kwargs.pop('end5cm_min_weight_effective_samples', 1.0))
        self.end5cm_min_weight_max_mass = float(
            kwargs.pop('end5cm_min_weight_max_mass', 0.05))
        self.end5cm_uncovered_weight_policy = kwargs.pop(
            'end5cm_uncovered_weight_policy', 'interpolate_neutral_edges')
        self.source_sensor_params = kwargs.pop('source_sensor_params', None)
        super().__init__(*args, **kwargs)
        self.direct = direct
        self.end5cm_point_calib = end5cm_point_calib
        self.real_point_calib = bool(real_point_calib or end5cm_point_calib)
        self.end_len = end_len
        self.max_iter = max_iter
        self.min_loss_diff = min_loss_diff
        self.hyper_params = hyper_params

        self.method = method
        self.straight_threshold = straight_threshold
        self.rot_threshold = rot_threshold
        self.k_point = k_point
        self.k_end = k_end
        self.k_pitch = k_pitch
        self.k_radius = k_radius
        self.weight_bounds = [(weight_bounds[0] * self.init_weight, weight_bounds[1] * self.init_weight)]
        self.slope_upper_bound = slope_upper_bound
        self.slopes_shape = np.asarray(self.slopes[self.direct], dtype=float).shape

        self.freeze = freeze
        self._coverage_support_map = None
        self._get_freeze_idx()


    def _minimize(self, **kwargs):
        """兼容旧入口对scipy搜索函数的替换；生产服务使用模块4默认实现。"""
        return minimize(**kwargs)
