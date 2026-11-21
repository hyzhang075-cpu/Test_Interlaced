"""独立任务的旧算法上下文。"""
from dataclasses import dataclass
from typing import Any

@dataclass
class LegacyContext:
    """持有一个任务专属的旧算法对象，避免候选状态跨任务共享。

    engine 是 Optimiser 等兼容对象；不复制算法状态。上下文自身不产生 I/O。
    """
    engine: Any
