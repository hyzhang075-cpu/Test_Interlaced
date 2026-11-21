"""输入文件解析、真实采样帧准备和旧API边界适配。"""
import numpy as np

from ..contracts import PreparedFrame, InputConfig
def __getattr__(name):
    if name in ('Sensor', 'WavelengthContinuityGuard'):
        from . import compat
        return getattr(compat, name)
    raise AttributeError(name)


class InputService:
    """将已解析的原始通道矩阵映射为模块1帧契约，不读文件、不接设备。

    config只描述通道选择；distance_m、source_indices由静态设计显式传入。
    原始信号单位沿用解调仪输入（通常为pm），不在本服务内做应变换算。
    """

    def __init__(self, config: InputConfig):
        """保存模块1配置。构造时不执行文件、设备或窗口I/O。"""
        self.config = config

    def prepare(self, signals, distance_m, source_indices, channel_ids=None) -> PreparedFrame:
        """复制并校验单帧矩阵，按物理通道号选列。

        signals形状为(P,C)，列对应channel_ids；distance_m与source_indices为(P,)，
        分别以m和全局真实点0-based索引表示。返回矩阵仍为原始信号单位。
        所有输入均保持不变。
        """
        values = np.array(signals, dtype=float, copy=True)
        distance = np.array(distance_m, dtype=float, copy=True)
        indices = np.array(source_indices, dtype=int, copy=True)
        available = tuple(channel_ids if channel_ids is not None else self.config.active_channels)
        if values.ndim != 2 or distance.ndim != 1 or indices.ndim != 1:
            raise ValueError("signals须为(P,C)，distance_m与source_indices须为(P,)数组")
        if values.shape[0] != distance.size or indices.shape != distance.shape:
            raise ValueError("输入帧点数、距离和真实点索引必须一致")
        if values.shape[1] != len(available) or len(set(available)) != len(available):
            raise ValueError("channel_ids必须唯一且与signals列数一致")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(distance)):
            raise ValueError("输入帧信号与距离必须为有限值")
        if distance.size > 1 and np.any(np.diff(distance) <= 0):
            raise ValueError("真实采样距离必须严格递增")
        missing = [channel for channel in self.config.active_channels if channel not in available]
        if missing:
            raise ValueError(f"输入帧缺少激活物理通道: {missing}")
        columns = [available.index(channel) for channel in self.config.active_channels]
        return PreparedFrame(distance, indices, values[:, columns].copy(), tuple(self.config.active_channels))

    def preprocess(self, signals, *, channel_ids=None, acquired=False):
        """统一静态/动态预处理：通道重排、交错插值、裁剪、中心补偿及反向。

        输入原始差分信号(Praw,C)，输出PreparedFrame(P,Couter)；保持历史尺度，
        不执行第二次1e-6换算。波长到该尺度的转换在读取/解包边界完成。
        """
        from dataclasses import asdict
        from ._kernel import InputKernel
        kernel = InputKernel(**asdict(self.config))
        values = np.array(signals, dtype=float, copy=True)
        if acquired:
            values = values.copy()
        else:
            available = tuple(channel_ids or sorted(self.config.active_channels))
            values = values[:, [available.index(channel) for channel in sorted(self.config.active_channels)]]
            kernel.Data = {'frame': values}
            if kernel.channel_change and kernel.channel_layout:
                kernel._reorder_channels_by_layout()
            else:
                kernel._reorder_channels_by_active()
            if kernel.interlaced_grating:
                kernel._apply_interlaced_grating(kernel.samp_dist_preset, kernel.samp_dist_preset_sub)
            values = kernel.Data['frame'][kernel.raw_trim_idx:, :]
        channels = tuple(self.config.active_channels)
        if kernel.centre_fibre:
            values = kernel._process_centreData(values)[2]
            channels = channels[1:]
        indices = np.arange(kernel.raw_trim_idx, kernel.raw_trim_idx + len(values))
        if kernel.reverse:
            values = values[::-1, :]
            indices = indices[::-1]
        return PreparedFrame(np.array(kernel.samp_dist, copy=True), indices.copy(), values.copy(), channels)

    def load_data(self, folder_path=None):
        """显式读CSV并返回CalibrationDataset；叶子(P,C)沿用旧1e-6信号尺度。"""
        from dataclasses import asdict
        from copy import deepcopy
        from ._kernel import InputKernel
        from ..contracts import CalibrationDataset
        values = asdict(self.config)
        kernel = InputKernel(**values)
        kernel.load_data(folder_path if folder_path is not None else self.config.folder_path)
        channels = tuple(self.config.active_channels)
        if self.config.centre_fibre:
            channels = channels[1:]
        return CalibrationDataset(deepcopy(kernel.processed_Data), deepcopy(kernel.radii),
                                  self._design(kernel), channels, deepcopy(kernel.plot_Data))

    def design(self):
        """只计算静态设计，不读取标定CSV；返回真实点m坐标和物理通道角。"""
        from dataclasses import asdict
        from ._kernel import InputKernel
        return self._design(InputKernel(**asdict(self.config)))

    def _design(self, kernel):
        """把本模块局部几何映射为显式设计契约，数组均复制。"""
        from ..contracts import DesignGeometry
        return DesignGeometry(
            sensor_id=self.config.sensor_id, radius_m=kernel.R, straight_trim=kernel.straight_trim,
            sampling_distance_m=np.array(kernel.samp_dist, copy=True),
            active_channels=tuple(self.config.active_channels), channel_angles_rad={},
            source_indices=kernel.global_point_ids.copy() - 1,
            physical_start_m=0.0 if kernel.zero_init else None,
            physical_end_m=float(kernel.ad_samp_dist[-1]) if kernel.end_point else None,
            active_angles_deg=None if kernel.active_angles is None else tuple(kernel.active_angles),
            interlaced_grating=kernel.interlaced_grating,
            interlaced_offset_rad=kernel.interlaced_offset,
            samp_dist_preset_mm=tuple(kernel.samp_dist_preset),
            samp_dist_preset_sub_mm=None if kernel.samp_dist_preset_sub is None else tuple(kernel.samp_dist_preset_sub))


class InputPreprocessor:
    """旧任务上下文适配器，仅供兼容入口使用；新模块间传递PreparedFrame。"""
    def __init__(self, context): self.context=context
    @property
    def engine(self): return self.context.engine
    def read(self,*args,**kwargs):
        mode=getattr(self.engine,"interrogator",None)
        method_name={"FiberTecc":"read_data_FBG","Hippo":"read_data_FBG","LUNA":"read_data_OFDR"}.get(mode)
        if method_name is None: raise ValueError(f"不支持的解调仪: {mode!r}")
        return getattr(self.engine,method_name)(*args,**kwargs)
    def receive(self,*args,**kwargs):
        mode=getattr(self.engine,"interrogator",None)
        method_name={"FiberTecc":"receive_data_FiberTecc","Hippo":"receive_data_Hippo"}.get(mode)
        if method_name is None: raise ValueError(f"{mode!r} 不支持实时接收")
        return getattr(self.engine,method_name)(*args,**kwargs)
    def preprocess(self,data,*args,**kwargs):
        for name in ("_process_centreData","_process_Data","_process_data"):
            method=getattr(self.engine,name,None)
            if method: return method(data,*args,**kwargs)
        raise AttributeError("当前 Sensor 没有可调用的数据预处理实现")

__all__=["Sensor","WavelengthContinuityGuard","InputService","InputPreprocessor"]
