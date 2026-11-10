"""初始化拟合适配。"""
class Initializer:
    """调用旧 rot_seg_fit 初始化每个方向的分段相位/斜率。

    context.engine 必须是初始化后的 Optimiser/Initialiser；拟合可能更新
    global_phases、slopes 等任务状态并输出诊断信息。数组单位和形状沿旧对象。
    """
    def __init__(self, context): self.context=context
    def fit(self, verbose=False, update_phase=True):
        """拟合并返回旧方法结果；verbose 控制诊断输出，update_phase 控制相位写回。"""
        return self.context.engine.rot_seg_fit(verbose=verbose,update_phase=update_phase)
