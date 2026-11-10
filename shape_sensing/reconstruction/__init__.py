"""形状重构和曲率积分适配。"""
class ShapeReconstructor:
    """复用 Reconstructor 的插值重构和数值积分，保持弧长 m、曲率 1/m。

    reconstruct_shapes 处理原标定数据；integrate 接收等长一维数组，单位分别为
    曲率 1/m、平面角 rad、弧长 m 和单位切向量。调用会执行既有标架积分。
    """
    def __init__(self,context): self.context=context
    def reconstruct_shapes(self,*args,**kwargs): return self.context.engine.reconstruct_shapes(*args,**kwargs)
    def integrate(self,cur_interp,ang_interp,interp_dist,init_T):
        """使用与正式重构相同的插值曲率构造位置/标架数组。"""
        return self.context.engine._reconstruct_from_interpolated_curvature(cur_interp,ang_interp,interp_dist,init_T)
