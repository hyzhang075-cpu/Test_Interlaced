"""旧参数优化适配。"""
class ParameterOptimizer:
    """复用 optim_coeff 调整本任务方向 weights/slopes，不共享其他任务候选状态。"""
    def __init__(self,context): self.context=context
    def optimize(self,verbose=True):
        """运行旧 Powell 等配置的优化；返回旧损失/状态并更新 engine 参数。"""
        return self.context.engine.optim_coeff(verbose=verbose)
