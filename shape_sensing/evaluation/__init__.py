"""理想曲线与误差评估适配。"""
from copy import deepcopy
class Evaluator:
    """调用旧 Optimiser 指标实现并快照副作用状态；弧长、曲线坐标单位沿旧实现。"""
    def __init__(self,context): self.context=context
    def ideal_shape(self,calib_shape,interp_dist,val_len,radius_m,idx_m,angle_deg=None):
        """构造标定理想曲线；interp_dist/radius_m/idx_m 为 m，angle_deg 为度。"""
        return self.context.engine._construct_ideal_shape(calib_shape,interp_dist,val_len,radius_m,idx_m,angle_deg)
    def evaluate(self,interp_dist,rs):
        """运行误差计算并返回独立结果快照；旧引擎状态仍会更新。"""
        engine=self.context.engine
        engine._compute_error(interp_dist=interp_dist,rs=rs)
        names=('point_errors','point_errors_avg','end_errors','avg_point_err','avg_end_err','rot0_pitch_errors','rot0_radius_errors','avg_pitch_err','avg_radius_err')
        return {name:deepcopy(getattr(engine,name)) for name in names if hasattr(engine,name)}
