"""评估和理想曲线接口。"""
from copy import deepcopy
from .metrics import EvaluationMixin
from .ideal_curvature import IdealCurvatureMixin
class Evaluator:
    """对当前任务引擎复用包内理想曲线和误差实现。"""
    def __init__(self, context): self.context=context
    def ideal_shape(self,calib_shape,interp_dist,val_len,radius_m,idx_m,angle_deg=None):
        return self.context.engine._construct_ideal_shape(calib_shape,interp_dist,val_len,radius_m,idx_m,angle_deg)
    def evaluate(self,interp_dist,rs):
        engine=self.context.engine
        engine._compute_error(interp_dist=interp_dist,rs=rs)
        names=('point_errors','point_errors_avg','end_errors','avg_point_err','avg_end_err','rot0_pitch_errors','rot0_radius_errors','avg_pitch_err','avg_radius_err')
        return {name:deepcopy(getattr(engine,name)) for name in names if hasattr(engine,name)}
__all__=["EvaluationMixin","IdealCurvatureMixin","Evaluator"]
