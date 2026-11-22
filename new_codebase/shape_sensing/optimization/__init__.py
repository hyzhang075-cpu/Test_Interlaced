"""优化引擎与任务上下文入口。"""
def __getattr__(name):
    if name == 'Optimiser':
        from .compat import Optimiser
        return Optimiser
    raise AttributeError(name)

class ParameterOptimizer:
    """通过当前任务优化引擎执行系数优化。"""
    def __init__(self, session):
        """输入 TaskSession，保存引用供兼容优化入口使用。"""
        self.session = session

    def optimize(self, verbose=True):
        """用任务的方向、等级和分段参数运行优化，返回搜索报告。"""
        return self.session.optim_coeff(verbose=verbose)

__all__ = ["Optimiser", "ParameterOptimizer"]

from .service import OptimizationService
