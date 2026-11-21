"""Direct-file entry point; supports ``python shape_sensing/main.py``."""
from pathlib import Path
import os
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import shape_sensing.cli as _cli


if __name__ == "__main__":
    if (Path(_PROJECT_ROOT) / "pyproject.toml").is_file():
        os.chdir(_PROJECT_ROOT)
    _cli.main()
else:
    sys.modules[__name__] = _cli
