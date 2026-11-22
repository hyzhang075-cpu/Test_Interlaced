"""模块5的纯时间滤波；状态由模块8逐会话传入并保存。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

@dataclass(frozen=True)
class TemporalCurvatureState:
    """局部曲率向量估计x(Q,2)单位1/m，协方差p(Q,2)单位1/m²。"""
    x: np.ndarray | None = None
    p: np.ndarray | None = None
    distance_m: np.ndarray | None = None


def filter_curvature_vectors(prepared, state=None, *, measurement_noise=1e-4, process_noise=1e-5):
    """输入PreparedCurvature及可选状态，输出新曲率场和状态；Q点曲率1/m、方向rad。

    先将幅角转成局部两分量，再沿用恒定曲率向量Kalman预测/更新公式，最后转回幅角。
    不修改输入数组；积分距离网格变化时重新捕获初态。
    """
    if measurement_noise <= 0 or process_noise < 0:
        raise ValueError('temporal noise must be positive/nonnegative')
    cur = np.asarray(prepared.curvature_per_m, dtype=float)
    ang = np.asarray(prepared.direction_rad, dtype=float)
    if cur.ndim != 1 or ang.shape != cur.shape:
        raise ValueError('cur_interp and ang_interp must be equal-length one-dimensional arrays')
    z = cur[:, None] * np.column_stack((np.cos(ang), np.sin(ang)))
    distance = np.asarray(prepared.distance_m)
    if (state is None or state.x is None or state.x.shape != z.shape
            or (state.distance_m is not None and not np.array_equal(state.distance_m, distance))):
        x = z.copy()
        p = np.full_like(z, process_noise)
    else:
        # 预测：恒定曲率向量模型，状态不变，协方差累加过程噪声。
        p_pred = state.p + process_noise
        gain = p_pred / (p_pred + measurement_noise)
        x = state.x + gain * (z - state.x)
        p = (1.0 - gain) * p_pred
    magnitude = np.hypot(x[:, 0], x[:, 1])
    direction = np.arctan2(x[:, 1], x[:, 0])
    diagnostics = dict(prepared.diagnostics)
    diagnostics.update({'cur_interp_before_temporal': cur.copy(),
                        'ang_interp_before_temporal': ang.copy()})
    result = replace(prepared, curvature_per_m=magnitude,
                     direction_rad=direction, diagnostics=diagnostics)
    return result, TemporalCurvatureState(x, p, distance.copy())


def average_curve(curve_m, history, window):
    """输入曲线(N,3)m、历史tuple和窗口int，输出平滑曲线及新历史；不修改输入。

    沿用原展示逻辑：积满窗口后取均值，积满之前直接使用当前帧；点数变化时清空历史。
    """
    if window < 1:
        raise ValueError('window must be at least one')
    curve = np.asarray(curve_m, dtype=float)
    compatible = history if all(item.shape == curve.shape for item in history) else ()
    updated = (*compatible, curve.copy())[-window:]
    result = np.mean(np.asarray(updated), axis=0) if len(updated) == window else curve.copy()
    return result, updated
