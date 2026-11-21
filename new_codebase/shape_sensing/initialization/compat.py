# 历史类兼容入口集中于本模块；新生产入口请使用 shape_sensing.initialization.service.InitializationService。
"""旧Initialiser构造边界，数值拟合统一复用模块2私有核。"""
from ._kernel import InitializationKernel
from ..curvature.geometry import StrainGeometryMixin
from ..input_preprocessing.compat import Sensor

class Initialiser(StrainGeometryMixin, InitializationKernel, Sensor):
    def __init__(self, discont, segmentation, connect=True, global_phases_set=None, interlaced_offset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.discont = discont
        self.segmentation = segmentation # 对应原本的segments
        self.num_segments = len(segmentation)
        self.connect = connect
        self.global_phases_set = global_phases_set
        self.interlaced_offset = interlaced_offset
        if self.global_phases_set is None:
            self.global_phases_set = 0.0
            self.rot_seg_fit(verbose=False, update_phase=True) # 初始化正确的全局相位与全局扭转系数
