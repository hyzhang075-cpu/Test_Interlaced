"""兼容旧导入路径；实现位于 shape_sensing.runtime.demonstrator。"""
import importlib as _importlib
import sys as _sys
_module = _importlib.import_module("shape_sensing.runtime.demonstrator")
_sys.modules[__name__] = _module
