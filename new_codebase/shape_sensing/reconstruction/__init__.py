"""模块5公开服务；旧引擎延迟导入，仅供兼容调用。"""
from .integration import ShapeIntegrator
from .service import ReconstructionService


class ShapeReconstructor:
    """重构兼容入口，接受 TaskSession 或旧引擎上下文。"""

    def __init__(self, context):
        """输入 TaskSession 或 LegacyContext，保存引用供旧签名转发。"""
        self.context = context

    def reconstruct_shapes(self, *args, **kwargs):
        """输入旧形状/方向参数，输出重构树(N,3)m与距离(N,)m。"""
        from ..context import LegacyContext
        if isinstance(self.context, LegacyContext):
            return self.context.engine.reconstruct_shapes(*args, **kwargs)
        return self.context.reconstruct_shapes(*args, **kwargs)

    def integrate(self, cur_interp, ang_interp, interp_dist, init_T):
        """输入曲率(N,)1/m、方向rad、距离m与切向(3,)，输出旧位置/标架tuple。"""
        from ..context import LegacyContext
        if isinstance(self.context, LegacyContext):
            return self.context.engine._reconstruct_from_interpolated_curvature(
                cur_interp, ang_interp, interp_dist, init_T)
        config = self.context.configuration.reconstruction
        return ShapeIntegrator(config.frame_rule, config.TNB_threshold,
                               config.rot_ang).integrate(cur_interp, ang_interp, interp_dist, init_T)


def __getattr__(name):
    """仅显式请求旧Reconstructor符号时加载兼容引擎。"""
    if name == 'Reconstructor':
        from .compat import Reconstructor
        return Reconstructor
    raise AttributeError(name)


__all__ = ['Reconstructor', 'ShapeIntegrator', 'ShapeReconstructor', 'ReconstructionService']
