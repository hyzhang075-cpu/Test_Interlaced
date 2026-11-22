"""模块7显式绘图输入与延迟绘图服务；不读取或绑定算法引擎。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..contracts import CalibrationDataset, InitializationResult, ShapeResult, EvaluationResult

@dataclass(frozen=True)
class VisualizationInputs:
    """模块1/2显式诊断；distance_m(P,)m，角度诊断rad，segmentation为点数。"""
    dataset: CalibrationDataset
    initialization: InitializationResult | None = None
    segmentation: tuple[int, ...] = ()
    discont_rad: float = np.pi
    connect: bool = True
    rotation_deg: float = 0.0
    output_root: Path | None = None

class VisualizationService:
    """接收DTO；只有绘图方法调用时加载matplotlib/VTK，不拟合、不重构。"""
    def __init__(self, inputs: VisualizationInputs | None = None):
        """输入可选VisualizationInputs，输出无I/O服务实例。"""
        self.inputs = inputs

    def init_visualise(self, direct):
        """输入方向int(deg)，输出None；按模块2现有诊断绘图并保存原格式文件。"""
        if self.inputs is None or self.inputs.initialization is None:
            raise ValueError('initialization diagnostics are required')
        from .plots import Visualisor
        return Visualisor(inputs=self.inputs).init_visualise(direct)

    def plot_shape(self, shape: ShapeResult, *, ideal_position_m=None, evaluation: EvaluationResult | None = None, ax=None):
        """输入重构(N,3)m、可选理想(M,3)m/误差DTO/坐标轴；输出matplotlib Figure。

        仅展示已有输出，不生成理想曲线或重新计算误差。
        """
        import matplotlib.pyplot as plt
        if ax is None:
            figure = plt.figure()
            ax = figure.add_subplot(111, projection='3d')
        curve = np.asarray(shape.position_m)
        ax.plot(*curve.T, label='reconstruction')
        if ideal_position_m is not None:
            ax.plot(*np.asarray(ideal_position_m).T, '--', label='ideal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        if evaluation is not None:
            ax.set_title(', '.join(f'{name}: {value:.4g}' for name, value in evaluation.metrics.items() if np.isscalar(value)))
        return ax.figure

    def show_reconstruction(self, direct, rs, interp_dist, showanime=False, *, ideals=None):
        """输入方向deg、曲线树叶子(N,3)m与距离(N,)m；输出None并按原目录保存MP4动画。

        ideals按同样路径显式传入，来自模块6；保留原旋转动画、ffmpeg保存和可选静态图。
        """
        if self.inputs is None:
            raise ValueError('explicit visualization inputs are required for saved plots')
        if ideals is None:
            raise ValueError('ideal curves must be supplied explicitly by module 6')
        from .plots import Visualisor
        return Visualisor(inputs=self.inputs).show_reconstruction(
            direct, rs, interp_dist, showanime, ideals=ideals)
