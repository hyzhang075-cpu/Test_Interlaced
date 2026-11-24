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
    static_reconstruction: bool = False

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

    def plot_core_positions(self, region):
        """保存模块2的逐 WL 芯位诊断 PNG，每个区域一张。"""
        inputs = self.inputs
        if inputs is None or inputs.initialization is None or not inputs.initialization.core_position:
            raise ValueError('core position initialization results are required')
        if inputs.output_root is None:
            raise ValueError('an output directory is required for core position plots')
        from .plots import save_core_position_plot
        path = inputs.output_root / f'init_core_positions_{region}.png'
        save_core_position_plot(inputs.initialization.core_position,
                                sensor_id=inputs.dataset.design.sensor_id,
                                region=region, output_path=path)
        return path

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
        """输入方向deg、曲线树叶子(N,3)m与距离(N,)m；保存MP4动画或静态PNG。

        ideals按相同路径从模块6传入。static_reconstruction=True 时保存静态PNG。
        """
        if self.inputs is None:
            raise ValueError('explicit visualization inputs are required for saved plots')
        if ideals is None:
            raise ValueError('ideal curves must be supplied explicitly by module 6')
        from .plots import Visualisor
        return Visualisor(inputs=self.inputs).show_reconstruction(
            direct, rs, interp_dist, showanime, ideals=ideals)
