"""初始化接口和基于任务上下文的薄适配器。"""
def __getattr__(name):
    if name == 'Initialiser':
        from .compat import Initialiser
        return Initialiser
    raise AttributeError(name)

class Initializer:
    """调用当前任务 Initialiser 实现的轻量入口。"""
    def __init__(self, context): self.context = context
    def fit(self, verbose=False, update_phase=True):
        return self.context.engine.rot_seg_fit(verbose=verbose, update_phase=update_phase)

__all__ = ["Initialiser", "Initializer"]

from .service import InitializationService
