"""数组与结果结构契约。"""
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class CurveInput:
    """插值曲线输入：curvature/angle/distance 均为一维 N 数组，单位 1/m、rad、m。"""
    curvature: Any
    angle: Any
    distance_m: Any
    initial_tangent: Any

@dataclass(frozen=True)
class ShapeResult:
    """旧标架积分输出；position 与 tangent/frame 行数为 N，空间坐标单位 m。"""
    values: tuple
    frame_rule: str

@dataclass(frozen=True)
class EvaluationSnapshot:
    """误差评估的一次深拷贝快照，不与 Optimiser 后续状态共享容器。"""
    metrics: dict
