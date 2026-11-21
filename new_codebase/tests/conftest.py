"""Test-only aliases for the former top-level imports.

Production code never installs these aliases; every target is a module inside
the standalone project package.
"""
import importlib
import sys


_MODULES = {
    "main": "shape_sensing.cli",
    "calibration_batch": "shape_sensing.optimization.batch",
    "calibration_regions": "shape_sensing.configuration.regions",
    "calibration_phase_reference": "shape_sensing.initialization.phase_reference",
    "reconstructor": "shape_sensing.reconstruction.engine",
    "optimiser": "shape_sensing.optimization.engine",
    "sensor": "shape_sensing.input_preprocessing.sensor",
    "initialiser": "shape_sensing.initialization.fitting",
    "entrance_anchor": "shape_sensing.runtime.anchor",
    "demonstrator": "shape_sensing.runtime.demonstrator",
}

for _alias, _target in _MODULES.items():
    sys.modules.setdefault(_alias, importlib.import_module(_target))
