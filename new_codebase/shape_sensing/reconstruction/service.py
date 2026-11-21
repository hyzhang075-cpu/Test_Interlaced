"""模块5：曲率预处理与形状积分，不读取标定数据。"""
import numpy as np
from .preparation import CurvaturePreparationKernel
from .integration import ShapeIntegrator
from ..contracts import (CurvatureResult, PreparedCurvature, ShapeResult,
                         ReconstructionConfig, DesignGeometry, ReconstructionState)


class _ConfiguredPreparation(CurvaturePreparationKernel):
    """显式保存本模块数值核所需配置与距离，构造不执行拟合或IO。"""

    def __init__(self, config, distance, adjusted_distance):
        """输入config和真实(P,)/补点(Q,)距离m，输出独立数值上下文。"""
        self.samp_dist = distance.copy()
        self.ad_samp_dist = adjusted_distance.copy()
        self.zero_init = config.zero_init
        self.end_point = config.end_point
        self.discont = config.discont
        self.unwrap_threshold = config.unwrap_threshold
        self.align_threshold = config.align_threshold
        self.move_filter = config.move_filter
        self.move_filter_for_vector = config.move_filter_for_vector
        self.kalman_params = config.kalman_params
        self.interp_num = config.interp_num
        self.curvature_interp = self._normalise_curvature_interp(config.curvature_interp)
        self.nonnegative_curvature = config.nonnegative_curvature


class ReconstructionService:
    """模块5独立入口，真实点输入保持不变，重构补点仅保存在返回值内。"""

    def __init__(self, config: ReconstructionConfig):
        """保存显式重构配置，不加载数据或初始化模块2。"""
        self.config = config

    def prepare(self, curvature: CurvatureResult, design: DesignGeometry,
                tangent_seed, state: ReconstructionState | None = None,
                ideal_curvature=None):
        """输入真实曲率(P,)1/m、方向rad、几何m及初始切向(3,)，返回积分场(N,)和诊断。

        ideal_curvature是距离数组(m)->理论曲率数组(1/m)的显式回调，模块5不依赖模块6。
        state存上一帧补点后的方向(Q,)rad；返回diagnostics['state_out']显式传给下一帧。
        """
        distance = np.array(curvature.distance_m, dtype=float, copy=True)
        magnitude = np.array(curvature.curvature_per_m, dtype=float, copy=True)
        angle = np.array(curvature.direction_rad, dtype=float, copy=True)
        seed = np.array(tangent_seed, dtype=float, copy=True)
        if (distance.ndim != 1 or distance.size < 2 or magnitude.shape != distance.shape
                or angle.shape != distance.shape or not np.all(np.isfinite(distance))
                or not np.all(np.diff(distance) > 0) or not np.all(np.isfinite(magnitude))
                or not np.all(np.isfinite(angle)) or np.any(magnitude < 0)):
            raise ValueError('真实曲率、方向、距离须有限等长且距离严格递增，曲率非负')
        if seed.shape != (3,) or not np.all(np.isfinite(seed)) or np.linalg.norm(seed) == 0:
            raise ValueError('初始切向量须为有限非零三维向量')
        adjusted = distance.copy()
        if self.config.zero_init:
            if design.physical_start_m is None:
                raise ValueError('zero_init需要design.physical_start_m显式提供虚拟起点')
            adjusted = np.insert(adjusted, 0, design.physical_start_m)
        if self.config.end_point:
            if design.physical_end_m is None:
                raise ValueError('end_point需要design.physical_end_m显式提供虚拟末点')
            adjusted = np.append(adjusted, design.physical_end_m)
        kernel = _ConfiguredPreparation(self.config, distance, adjusted)
        previous_raw = None if state is None else state.previous_raw_angles_rad
        previous_unwrapped = None if state is None else state.previous_unwrapped_angles_rad
        cur, ang, grid, diagnostics = kernel.prepare_arrays(
            magnitude, angle, previous_raw, previous_unwrapped, ideal_curvature)
        next_state = ReconstructionState(diagnostics['angle_prepared'].copy(),
                                         diagnostics['angle_unwrap'].copy())
        diagnostics.update({'angle_raw': diagnostics['angle'].copy(),
                            'angle_filter': diagnostics['ang_filter'].copy(),
                            'curvature_raw_per_m': diagnostics['magnitude'].copy(),
                            'source_distance_m': distance.copy(),
                            'adjusted_distance_m': adjusted.copy(),
                            'state_out': next_state})
        source = np.interp(grid, distance, curvature.source_indices).astype(int)
        return PreparedCurvature(grid.copy(), cur.copy(), ang.copy(), seed, source, diagnostics, state=next_state)

    def integrate(self, prepared: PreparedCurvature):
        """输入插值曲率(N,)1/m、方向rad、距离m；返回位置(N,3)m与原始单位标架。

        两种标架均保持旧tuple布局(position,T,N或Y,B或Z)，named字段逐一对应。
        """
        integrator = ShapeIntegrator(self.config.frame_rule, self.config.TNB_threshold,
                                     self.config.rot_ang)
        values = integrator.integrate(prepared.curvature_per_m, prepared.direction_rad,
                                      prepared.distance_m, prepared.tangent_seed)
        position, tangent, normal, binormal = values
        diagnostics = dict(prepared.diagnostics)
        diagnostics.update({'cur_interp': np.array(prepared.curvature_per_m, copy=True),
                            'ang_interp': np.array(prepared.direction_rad, copy=True)})
        return ShapeResult(values, self.config.frame_rule, position, tangent, normal,
                           binormal, np.array(prepared.distance_m, copy=True), diagnostics)

    def transform_shape(self, shape, *, direct, rot_dir, rot_ang):
        """输入ShapeResult(N,3)m、方向/附加旋转deg和旋转符号；返回旋转后的独立形状。

        使用原绕x轴公式，位置与所有单位标架统一旋转，不修改输入。
        """
        from dataclasses import replace
        integrator = ShapeIntegrator(self.config.frame_rule, self.config.TNB_threshold, rot_ang)
        matrix = integrator._get_rot_x(direct, rot_dir)
        position = np.array(shape.position_m, copy=True) @ matrix.T
        tangent = None if shape.tangent is None else np.array(shape.tangent, copy=True) @ matrix.T
        normal = None if shape.normal is None else np.array(shape.normal, copy=True) @ matrix.T
        binormal = None if shape.binormal is None else np.array(shape.binormal, copy=True) @ matrix.T
        return replace(shape, position_m=position, tangent=tangent, normal=normal,
                       binormal=binormal, values=(position, tangent, normal, binormal))
