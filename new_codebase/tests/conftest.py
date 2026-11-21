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
    "reconstructor": "shape_sensing.reconstruction.compat",
    "optimiser": "shape_sensing.optimization.compat",
    "sensor": "shape_sensing.input_preprocessing.compat",
    "initialiser": "shape_sensing.initialization.compat",
    "entrance_anchor": "shape_sensing.runtime.anchor",
    "demonstrator": "shape_sensing.runtime.compat",
}

for _alias, _target in _MODULES.items():
    sys.modules.setdefault(_alias, importlib.import_module(_target))
