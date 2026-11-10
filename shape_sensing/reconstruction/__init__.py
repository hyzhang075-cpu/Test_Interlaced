"""重构器与纯曲率积分公开接口。"""
from .engine import Reconstructor
from .integration import ShapeIntegrator

class ShapeReconstructor:
    """将任务上下文引擎暴露为重构接口。"""
    def __init__(self, context): self.context = context
    def reconstruct_shapes(self, *args, **kwargs): return self.context.engine.reconstruct_shapes(*args, **kwargs)
    def integrate(self, cur_interp, ang_interp, interp_dist, init_T):
        """使用当前重构器的积分实现和对象状态。"""
        return self.context.engine._reconstruct_from_interpolated_curvature(
            cur_interp, ang_interp, interp_dist, init_T)


__all__ = ["Reconstructor", "ShapeIntegrator", "ShapeReconstructor"]
