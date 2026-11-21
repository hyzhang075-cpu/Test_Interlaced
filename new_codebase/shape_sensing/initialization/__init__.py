"""初始化接口和基于任务上下文的薄适配器。"""
from .fitting import Initialiser

class Initializer:
    """调用当前任务 Initialiser 实现的轻量入口。"""
    def __init__(self, context): self.context = context
    def fit(self, verbose=False, update_phase=True):
        return self.context.engine.rot_seg_fit(verbose=verbose, update_phase=update_phase)

__all__ = ["Initialiser", "Initializer"]
