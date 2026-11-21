"""主窗口独立验收：真实点边界、初始化基线和优化模块调用约束。"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from shape_sensing.cli import parse_main_args, configure_calibration_args
from shape_sensing.optimization.batch import optimizer_parameters
from shape_sensing.contracts import CalibrationDataset
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.input_preprocessing import InputService
from shape_sensing.initialization.service import InitializationService
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.reconstruction.service import ReconstructionService
from shape_sensing.evaluation.service import EvaluationService

FIXTURE = Path(__file__).with_name('fixtures') / 'module_split_baseline'


@pytest.fixture
def baseline():
    """读取独立golden，不读取真实校准CSV或旧代码目录。"""
    with np.load(FIXTURE / 'real_09B7.npz') as source:
        arrays = {key: source[key].copy() for key in source.files}
    metadata = json.loads((FIXTURE / 'metadata.json').read_text(encoding='utf-8'))
    args = parse_main_args(['--config', str(FIXTURE / 'working_config.yaml'), '--task', 'init'])
    configure_calibration_args(args)
    parameters = optimizer_parameters(args)
    initial = TaskSession(parameters, task='init')
    data = {}
    radii = {}
    for key, values in arrays.items():
        found = re.fullmatch(r'(\d+)_([A-Za-z0-9]+)_(\d+)_input', key)
        if found:
            direction, kind, radius = int(found[1]), found[2], int(found[3])
            data.setdefault(kind, {}).setdefault(direction, {})[radius] = values.copy()
            radii.setdefault(direction, {}).setdefault(kind, []).append(radius)
    channels = tuple(parameters['active_channels'][1:])
    dataset = CalibrationDataset(data, radii, initial.design, channels, deepcopy(data))
    return parameters, dataset, arrays, metadata


def test_session_construction_has_no_io_or_implicit_fit(baseline, monkeypatch):
    """模块构造允许纯几何计算，但不能读CSV、拟合或启动设备。"""
    parameters, _, _, _ = baseline
    def forbidden(*args, **kwargs):
        raise AssertionError('constructor executed file loading or fitting')
    monkeypatch.setattr(InputService, 'load_data', forbidden)
    monkeypatch.setattr(InitializationService, 'fit', forbidden)
    session = TaskSession(parameters, task='init')
    assert session.dataset is None and session.initialization_result is None
    assert session.design.sampling_distance_m.shape == (22,)
    with pytest.raises(ValueError, match='init.*resume'):
        TaskSession(parameters, task='optim')


def test_true_point_curvature_and_reconstruction_match_real_baseline(baseline, monkeypatch):
    """逐点检查模块3原位对应，模块5允许扩网格但不改原数组。"""
    parameters, dataset, arrays, metadata = baseline
    original = deepcopy(dataset.data)
    session = TaskSession(parameters, task='init', dataset=dataset)
    monkeypatch.setattr(InputService, 'load_data', lambda *a, **k: pytest.fail('unexpected CSV read'))
    session.initialize()
    assert session.global_phases_set == pytest.approx(metadata['global_phases_set'], abs=1e-13)
    for direction in (0, 120, 240):
        np.testing.assert_allclose(session.slopes[direction], metadata['slopes'][str(direction)], rtol=1e-12, atol=1e-12)
        results, distance = session.reconstruct_shapes('circ', direction)
        for radius, positions in results.items():
            label = f'{direction}_circ_{radius}'
            values = dataset.data['circ'][direction][radius]
            raw_frame = session.frame(values)
            computed = session.curvature.compute(raw_frame, session.design, session.parameters, direct=direction)
            np.testing.assert_array_equal(computed.distance_m, raw_frame.distance_m)
            np.testing.assert_array_equal(computed.source_indices, raw_frame.source_indices)
            assert computed.curvature_per_m.shape == (22,)
            np.testing.assert_allclose(computed.curvature_per_m, arrays[label + '_magnitude'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(computed.direction_rad, arrays[label + '_angle'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(positions, arrays[label + '_shape'], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(values, original['circ'][direction][radius])
        evaluation = session.evaluate(distance, {'circ': results}, direction)
        np.testing.assert_allclose(evaluation.metrics['avg_point_err']['calib'], metadata['directions'][str(direction)]['avg_point_err']['calib'], rtol=1e-12, atol=1e-12)


def test_optimization_calls_physical_reconstruction_evaluation_without_fit(baseline, monkeypatch):
    """单次候选评估仍走原目标函数，模块2若被调用则立即失败。"""
    parameters, dataset, _, metadata = baseline
    parameters = deepcopy(parameters)
    parameters.update(weights={int(k): np.array(v) for k, v in metadata['weights'].items()},
        slopes={int(k): np.array(v) for k, v in metadata['slopes'].items()},
        global_phases_set=metadata['global_phases_set'], max_iter=1)
    original = deepcopy(parameters)
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('optimization implicitly initialized phase')
    monkeypatch.setattr(InitializationService, 'fit', forbidden)
    def loader(*args, **kwargs):
        calls.append(1)
        return dataset
    monkeypatch.setattr(InputService, 'load_data', loader)
    for owner, name, number in [(CurvatureService, 'compute', 3),
        (ReconstructionService, 'prepare', 5), (EvaluationService, 'evaluate_batch', 6)]:
        original_method = getattr(owner, name)
        def traced(self, *args, _original=original_method, _number=number, **kwargs):
            calls.append(_number)
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(owner, name, traced)
    def one_candidate(fun, x0, *args, **kwargs):
        values = np.array(x0, dtype=float, copy=True)
        return OptimizeResult(x=values, fun=fun(values), success=True, nit=1, message='acceptance candidate')
    import shape_sensing.optimization._kernel as search
    monkeypatch.setattr(search, 'minimize', one_candidate)
    session = TaskSession(parameters, task='optim')
    session.optim_coeff(verbose=False)
    assert {1, 3, 5, 6}.issubset(calls)
    for direction in original['weights']:
        np.testing.assert_array_equal(parameters['weights'][direction], original['weights'][direction])
        np.testing.assert_array_equal(parameters['slopes'][direction], original['slopes'][direction])


def test_production_import_graph_does_not_load_compatibility_engines():
    """在独立进程中导入生产入口；兼容继承链和GUI不应进入模块依赖图。"""
    import subprocess
    import sys
    script = """
import sys
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.runtime.demo import run_demo
from shape_sensing.visualization.service import VisualizationService
for name in ('shape_sensing.input_preprocessing.compat',
             'shape_sensing.initialization.compat',
             'shape_sensing.reconstruction.compat',
             'shape_sensing.optimization.compat', 'PyQt5', 'vtk'):
    assert name not in sys.modules, name
print('PRODUCTION_IMPORT_GRAPH_OK')
"""
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRODUCTION_IMPORT_GRAPH_OK' in result.stdout
