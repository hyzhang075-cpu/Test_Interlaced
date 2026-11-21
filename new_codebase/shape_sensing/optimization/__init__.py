"""优化引擎与任务上下文入口。"""
def __getattr__(name):
    if name == 'Optimiser':
        from .compat import Optimiser
        return Optimiser
    raise AttributeError(name)

class ParameterOptimizer:
    """通过当前任务优化引擎执行系数优化。"""
    def __init__(self, context): self.context = context
    def optimize(self, verbose=True): return self.context.session.optim_coeff(verbose=verbose)

__all__ = ["Optimiser", "ParameterOptimizer"]

from .service import OptimizationService
