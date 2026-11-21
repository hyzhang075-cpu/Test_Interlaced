"""曲率计算接口。"""
from .geometry import StrainGeometryMixin

__all__ = ["StrainGeometryMixin"]

from .geometry import StrainGeometryMixin
from .model import CurvatureModelMixin
class CurvatureEstimator:
    """通过任务上下文执行已迁移的曲率模型实现。"""
    def __init__(self,context): self.context=context
    def estimate(self,*args,**kwargs): return self.context.engine._Strain_2_Cur(*args,**kwargs)
__all__=["StrainGeometryMixin","CurvatureModelMixin","CurvatureEstimator"]
