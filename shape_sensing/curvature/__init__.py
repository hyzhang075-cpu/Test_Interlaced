"""应变到曲率的旧算法适配。"""
class CurvatureEstimator:
    """调用 Optimiser._Strain_2_Cur；输入为旧标定数据/权重/斜率结构。

    输出 magnitude 单位 1/m、angle 单位 rad，维度与旧 calib_Data 对齐；
    不改写或缓存算法状态，异常沿旧方法传播。
    """
    def __init__(self,context): self.context=context
    def estimate(self,*args,**kwargs):
        """执行旧应变曲率换算；参数名与 _Strain_2_Cur 一致。"""
        return self.context.engine._Strain_2_Cur(*args,**kwargs)
