"""模块6理想曲线和原标定误差指标的独立入口。"""
from copy import deepcopy
import numpy as np
from .metrics import EvaluationMixin
from .ideal_curvature import IdealCurvatureMixin
from ..contracts import EvaluationResult, ShapeResult, EvaluationReference, EvaluationConfig


class _EvaluationKernel(EvaluationMixin, IdealCurvatureMixin):
    """同模块原公式的显式数值上下文，无引擎、IO、拟合依赖。"""

    def __init__(self, config, sample_distance, interp_num):
        """输入配置、真实点(P,)距离m及插值点数N，创建独立评估上下文。"""
        self.samp_dist = np.array(sample_distance, dtype=float, copy=True)
        self.end_len = config.end_len
        self.straight_threshold = config.straight_threshold
        self.rot_threshold = config.rot_threshold
        self.interp_num = interp_num


class EvaluationService:
    """显式接收参考曲线/形状参数，返回原有归一化误差与诊断数据。"""

    def __init__(self, config: EvaluationConfig):
        """保存模块6评估配置，构造不读取文件。"""
        self.config = config

    def ideal_curvature(self, distance_m, reference: EvaluationReference):
        """输入重采样距离(N,)m及理论半径m，返回原公式曲率(N,)1/m。

        end分界依赖真实点范围sample_coordinates_m；第二参数metadata['idx_m']单位m。
        """
        distance = np.array(distance_m, dtype=float, copy=True)
        sample = distance if reference.sample_coordinates_m is None else reference.sample_coordinates_m
        kernel = _EvaluationKernel(self.config, sample, len(distance))
        if reference.ideal_curvature_per_m is not None:
            values = np.array(reference.ideal_curvature_per_m, dtype=float, copy=True)
            if values.shape != distance.shape:
                raise ValueError('显式理想曲率须与请求距离等长')
            return values
        if reference.radius_m is None or reference.shape_kind is None:
            raise ValueError('理想曲率需要shape_kind和radius_m')
        keys = [reference.radius_m * 1000.0]
        if reference.shape_kind in {'end', 'rot0'}:
            keys.append(float(reference.metadata['idx_m']) * 1000.0)
        return kernel._construct_ideal_magnitude_for_recon(keys, reference.shape_kind, distance)

    def ideal_shape(self, distance_m, reference: EvaluationReference):
        """输入(N,)距离m与原理论形状参数，输出(M,3)m，S形超有效长度会裁至M<=N。"""
        distance = np.array(distance_m, dtype=float, copy=True)
        if reference.ideal_position_m is not None:
            return np.array(reference.ideal_position_m, dtype=float, copy=True)
        if reference.radius_m is None or reference.shape_kind is None:
            raise ValueError('理想曲线需要显式ideal_position_m或shape_kind/radius_m')
        sample = distance if reference.sample_coordinates_m is None else reference.sample_coordinates_m
        kernel = _EvaluationKernel(self.config, sample, len(distance))
        relative = distance - distance[0]
        val_len = float(relative[-1])
        if reference.shape_kind == 'S':
            val_len = 2 * np.pi * reference.radius_m + 0.1
        idx = reference.metadata.get('idx_m')
        if reference.shape_kind in {'end', 'rot0', 'up', 'down'} and idx is None:
            raise ValueError('此理论形状需要metadata[idx_m]，单位m')
        return kernel._construct_ideal_shape(reference.shape_kind, relative, val_len,
                                             reference.radius_m, idx, reference.angle_deg)

    def evaluate(self, shape: ShapeResult, reference: EvaluationReference):
        """输入位置(N,3)m、距离(N,)m及显式参考，返回原归一化逐点/末端和螺距/半径误差。

        不用RMSE替换原指标；point_errors为距离误差除有效评估长度的无量纲数组。
        rot0识别所用实际插值曲率通过reference.metadata['curvature_per_m']传入。
        """
        positions = np.array(shape.position_m, dtype=float, copy=True)
        distance = np.array(shape.distance_m, dtype=float, copy=True)
        if (positions.ndim != 2 or positions.shape[1] != 3 or distance.shape != (len(positions),)
                or len(distance) < 2 or not np.all(np.isfinite(positions))
                or not np.all(np.isfinite(distance)) or not np.all(np.diff(distance) > 0)):
            raise ValueError('形状须有限(N,3)、距离须为至少两点严格递增(N,)')
        relative = distance - distance[0]
        val_len = float(relative[-1])
        sample = distance if reference.sample_coordinates_m is None else reference.sample_coordinates_m
        kernel = _EvaluationKernel(self.config, sample, len(distance))
        if reference.shape_kind == 'rot0':
            if reference.radius_m is None:
                raise ValueError('rot0需要radius_m')
            curvature = np.array(reference.metadata['curvature_per_m'], dtype=float, copy=True)
            if curvature.shape != distance.shape:
                raise ValueError('rot0实际曲率须与形状距离等长')
            pitch, radius = kernel._compute_rot0_err(positions, curvature, reference.radius_m)
            return EvaluationResult({'rot0_pitch_error_m': float(pitch),
                                     'rot0_radius_error_m': float(radius)},
                                    {'rot0_segment_indices': kernel._find_rot_segment(curvature)})
        ideal = self.ideal_shape(distance, reference)
        errors = kernel._compute_point_err(positions, ideal, relative, val_len)
        count = len(errors)
        absolute = np.linalg.norm(positions[:count] - ideal[:count], axis=1)
        metrics = {'point_errors': errors.copy(), 'avg_point_err': float(np.nanmean(errors)),
                   'avg_end_err': float(errors[-1]), 'point_error_mean_m': float(np.nanmean(absolute)),
                   'point_error_max_m': float(np.nanmax(absolute)), 'end_error_m': float(absolute[-1])}
        return EvaluationResult(metrics, {'ideal_position_m': ideal.copy(),
                                         'point_error_m': absolute, 'distance_m': distance})

    def evaluate_batch(self, interp_dist, results, radii, direct, sample_distance_m,
                       curvature_per_m):
        """输入旧形状树results、radii标定树、方向deg和显式曲率(N,)1/m，返回批指标快照。

        每叶位置(N,3)m，interp_dist(N,)m；返回全部原逐点/末端/螺距/半径字典及汇总。
        """
        kernel = _EvaluationKernel(self.config, sample_distance_m, len(interp_dist))
        kernel.radii = deepcopy(radii)
        kernel.direct = direct
        kernel.calib_shapes = list(results)
        kernel.cur_interp = np.array(curvature_per_m, dtype=float, copy=True)
        kernel._compute_error(np.array(interp_dist, copy=True), deepcopy(results))
        names = ('point_errors', 'point_errors_avg', 'end_errors', 'avg_point_err',
                 'avg_end_err', 'rot0_pitch_errors', 'rot0_radius_errors',
                 'avg_pitch_err', 'avg_radius_err')
        relative_distance = np.array(interp_dist, dtype=float, copy=True)
        relative_distance -= relative_distance[0]
        ideal_curves = {kind: kernel._construct_ideals(kind, relative_distance, float(relative_distance[-1]))
                        for kind in results}
        return EvaluationResult({name: deepcopy(getattr(kernel, name)) for name in names},
                                {'ideals': deepcopy(ideal_curves)})

    def evaluate_dataset(self, dataset, direct, interp_dist, results, curvature_per_m):
        """输入CalibrationDataset、方向deg和重构树，返回原批评估指标，不修改数据集。

        dataset.design.sampling_distance_m(P,)m，interp_dist(N,)m，各位置叶(N,3)m。
        """
        return self.evaluate_batch(interp_dist, results, dataset.radii, direct,
                                   dataset.design.sampling_distance_m, curvature_per_m)
