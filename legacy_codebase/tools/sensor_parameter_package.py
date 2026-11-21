"""旧工具脚本入口；实际实现位于 shape_sensing.artifacts.legacy_package。"""
from pathlib import Path
import importlib as _importlib
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_module = _importlib.import_module("shape_sensing.artifacts.legacy_package")
_sys.modules[__name__] = _module
if __name__ == "__main__":
    _module.main()
