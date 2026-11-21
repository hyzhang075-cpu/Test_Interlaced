"""Command line entry point for the standalone shape_sensing project."""
from pathlib import Path
import os
import sys

# Allow direct execution from the project root or this file's own directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import shape_sensing.cli as _cli


if __name__ == "__main__":
    os.chdir(_PROJECT_ROOT)
    _cli.main()
else:
    # Preserve historical helper access and monkeypatch behavior for callers.
    sys.modules[__name__] = _cli
