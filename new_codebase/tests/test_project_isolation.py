"""The standalone project owns its imports and all supported source launchers."""
import ast
from pathlib import Path
import subprocess
import sys
import pytest

PROJECT = Path(__file__).resolve().parents[1]
OLD_MODULES = {'main', 'sensor', 'initialiser', 'reconstructor', 'optimiser',
    'visualisor', 'demonstrator', 'proto_demonstrator', 'IMU', 'calibration_batch',
    'calibration_regions', 'calibration_phase_reference', 'entrance_anchor', 'utils', 'zero_reference'}


def test_package_and_tools_have_no_legacy_module_imports():
    found = []
    for directory in ('shape_sensing', 'tools'):
        for path in (PROJECT / directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or '']
                for name in names:
                    if name.split('.')[0] in OLD_MODULES:
                        found.append((str(path.relative_to(PROJECT)), node.lineno, name))
    assert not found, found


@pytest.mark.parametrize('directory, arguments', [
    ('', ['-m', 'shape_sensing', '--help']),
    ('', ['shape_sensing/main.py', '--help']),
    ('shape_sensing', ['main.py', '--help']),
])
def test_source_entrypoints_work_without_legacy_path(directory, arguments):
    result = subprocess.run([sys.executable, *arguments], cwd=PROJECT / directory,
        capture_output=True, text=True, errors='replace', timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()
