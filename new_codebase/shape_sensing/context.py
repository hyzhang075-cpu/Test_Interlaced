"""任务数据组合的显式上下文；旧engine槽只供兼容入口。"""
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .pipelines.session import TaskSession

@dataclass
class TaskContext:
    session: 'TaskSession'

@dataclass
class LegacyContext:
    engine: object
