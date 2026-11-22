"""旧引擎兼容入口的数据上下文。"""
from dataclasses import dataclass


@dataclass
class LegacyContext:
    engine: object
