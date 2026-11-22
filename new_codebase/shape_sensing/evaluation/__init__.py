"""模块6公开理想曲线与评估接口。"""
from .metrics import EvaluationMixin
from .ideal_curvature import IdealCurvatureMixin
from .service import EvaluationService


class Evaluator:
    """评估兼容入口，接受 TaskSession 或旧引擎上下文。"""

    def __init__(self, context):
        """输入 TaskSession 或 LegacyContext，保存引用供旧签名转发。"""
        self.context = context

    def ideal_shape(self, calib_shape, interp_dist, val_len, radius_m, idx_m, angle_deg=None):
        """输入旧形状参数及距离(N,)m，输出理论位置(M,3)m。"""
        from ..context import LegacyContext
        if isinstance(self.context, LegacyContext):
            return self.context.engine._construct_ideal_shape(
                calib_shape, interp_dist, val_len, radius_m, idx_m, angle_deg)
        from ..contracts import EvaluationReference
        session = self.context
        reference = EvaluationReference(radius_m=radius_m, shape_kind=calib_shape,
            angle_deg=angle_deg, sample_coordinates_m=session.dataset.design.sampling_distance_m,
            metadata={'idx_m': idx_m})
        return EvaluationService(session.configuration.evaluation).ideal_shape(interp_dist, reference)

    def evaluate(self, interp_dist, rs):
        """输入距离(N,)m与重构位置树，输出原9组评估指标字典。"""
        from ..context import LegacyContext
        if isinstance(self.context, LegacyContext):
            from copy import deepcopy
            engine = self.context.engine
            engine._compute_error(interp_dist=interp_dist, rs=rs)
            names = ('point_errors', 'point_errors_avg', 'end_errors', 'avg_point_err',
                     'avg_end_err', 'rot0_pitch_errors', 'rot0_radius_errors',
                     'avg_pitch_err', 'avg_radius_err')
            return {name: deepcopy(getattr(engine, name)) for name in names if hasattr(engine, name)}
        return self.context.evaluate(interp_dist, rs).metrics


__all__ = ['EvaluationMixin', 'IdealCurvatureMixin', 'Evaluator', 'EvaluationService']
