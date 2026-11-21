"""主聊天独立验收：公共接口必须真正接入旧算法，并保持数值行为。"""
from types import SimpleNamespace

import numpy as np

from shape_sensing.context import LegacyContext
from shape_sensing.input_preprocessing import InputPreprocessor
from shape_sensing.reconstruction import ShapeReconstructor
from shape_sensing.evaluation import Evaluator


def test_acceptance_receiver_uses_selected_interrogator():
    calls = []
    engine = SimpleNamespace(
        interrogator='Hippo',
        receive_data_FiberTecc=lambda: calls.append('FiberTecc'),
        receive_data_Hippo=lambda: calls.append('Hippo'),
    )
    InputPreprocessor(LegacyContext(engine)).receive()
    assert calls == ['Hippo']


def test_acceptance_integration_matches_actual_legacy_curvature_entry():
    from reconstructor import Reconstructor
    engine = Reconstructor.__new__(Reconstructor)
    engine.frame_rule = 'TNN'
    curvature = np.array([0., 1., 5., 10.])
    angles = np.array([0., .2, .4, -.3])
    distance = np.array([0., .01, .025, .04])
    tangent = np.array([1., 0., 0.])
    expected = engine._reconstruct_from_interpolated_curvature(
        curvature, angles, distance, tangent
    )
    actual = ShapeReconstructor(LegacyContext(engine)).integrate(
        curvature, angles, distance, tangent
    )
    for old, new in zip(expected, actual):
        np.testing.assert_allclose(new, old, atol=1e-12, rtol=1e-12)


def test_acceptance_ideal_curve_matches_actual_legacy_entry():
    from optimiser import Optimiser
    engine = Optimiser.__new__(Optimiser)
    values = dict(calib_shape='circ', interp_dist=np.linspace(0., .1, 7),
                  val_len=.1, radius_m=.03, idx_m=0.)
    expected = engine._construct_ideal_shape(**values)
    actual = Evaluator(LegacyContext(engine)).ideal_shape(**values)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_acceptance_store_true_yaml_rejects_string_boolean(tmp_path):
    import pytest
    import yaml
    from shape_sensing.configuration import load_working_config
    values = dict(sensor_id='CHECK', R=.0001, samp_dist_preset=[1., 2.],
                  nonnegative_curvature='false')
    path = tmp_path / 'bad-switch.yaml'
    path.write_text(yaml.safe_dump(values), encoding='utf-8')
    with pytest.raises(ValueError, match='nonnegative_curvature'):
        load_working_config(path)


def test_acceptance_resume_uses_saved_parameters_not_error_summary(tmp_path, monkeypatch):
    """真实 init 输出同时含误差摘要和参数，续算必须读取参数。"""
    import runpy
    from pathlib import Path
    from shape_sensing.artifacts.runs import create_run
    from shape_sensing.pipelines import execute_task
    helpers = runpy.run_path(str(Path(__file__).with_name('test_task_pipelines.py')))
    run = create_run('RESUME', 'init', {'sensor_id': 'RESUME'}, tmp_path)
    run.write_result({
        'task': 'init', 'result': {'0': .01},
        'weights': {'0': [1., 2.]},
        'slopes': {'0': [[3.], [4.]]},
        'global_phases_set': 0.,
    }, 'initialization')
    run.finish('succeeded')
    args = helpers['args']('optim', sensor_id='RESUME', resume=str(run.path))
    import shape_sensing.pipelines.tasks as tasks
    # Exercise the real artifact/resume merge before numerical work. Numerical
    # services and the full init->optim path are independently golden-tested.
    seen = {}
    def dispatch(resumed_args, **kwargs):
        seen.update(weights=resumed_args.weights, slopes=resumed_args.slopes)
        return {'stage': 'parameters_loaded'}
    monkeypatch.setattr(tasks, '_execute_task_body', dispatch)
    execute_task(args, **helpers['deps'](monkeypatch, artifact_root=tmp_path/'continued'))
    assert args.global_phases_set == 0.
    assert 0 in args.weights
    np.testing.assert_array_equal(args.weights[0], [1., 2.])
    np.testing.assert_array_equal(args.slopes[0], [[3.], [4.]])


def test_acceptance_package_keeps_regions_and_can_configure_reconstruction(tmp_path):
    """同方向三个区域导出后，必须能够执行实际区域参数解析。"""
    from copy import deepcopy
    import main
    from shape_sensing.artifacts.runs import create_run
    from shape_sensing.artifacts.packages import export_package
    base = vars(main.parse_main_args(['--task', 'optim', '--direct', '0']))
    base = {k: deepcopy(v) for k, v in base.items() if not k.startswith('_')}
    base.update(sensor_id='REGION-CHECK', start_value=0., end_value=10.,
                samp_dist_preset=[0., 20., 110., 130., 190., 290., 390., 490.],
                full_segmentation=[2, 2, 2, 2],
                segmentation=None, thresholds=None, weights=None, slopes=None)
    # 旧几何函数将预设点间距倒序，以上预设对应实际位置0/100/200/300/360/380/470/490 mm。
    profiles = {}
    for region, length, segmentation in [('full', None, [2,2,2,2]),
                                          ('end15cm', .15, [2,2]),
                                          ('end5cm', .05, [2])]:
        profiles[region] = dict(length_m=length, folder_path=region,
                                radii={'circ': [0, 30]}, thresholds=[1.],
                                segmentation=segmentation, end_weights=False,
                                end_start=-1, weights=None, slopes=None)
    base['calibration_regions'] = profiles
    base.update(zero_reference_path=None, use_stored_zero=False)
    paths = []
    for direct in (0, 120, 240):
        for marker, region in enumerate(profiles, start=1):
            config = deepcopy(base)
            config.update(optimization_region=region, direct=direct)
            run = create_run(base['sensor_id'], 'optim', config, tmp_path/'runs')
            segments = len(profiles[region]['segmentation'])
            parameters = {'weights':{str(direct):[float(marker)]*2},
                          'slopes':{str(direct):[[float(marker)]*segments]*2},
                          'global_phases_set':0., 'rot_ang':0.}
            report = {'sensor_id':base['sensor_id'], 'direct':direct, 'regions':{
                region:{'status':'complete', 'parameters':parameters}}}
            run.write_result({'task':'optim', 'result':['legacy-results.json',report],
                              **parameters}, 'optim')
            run.finish('succeeded')
            paths.append(run.path)
    output = tmp_path/'published.json'
    export_package(paths, output)
    for marker, region in enumerate(profiles, start=1):
        args = main.parse_main_args(['--task','recon','--sensor_params',str(output),
                                     '--optimization_region',region,'--direct','0'])
        main.configure_calibration_args(args)
        np.testing.assert_array_equal(args.weights[0], [float(marker)]*2)
        assert args.slopes[0].shape == (2, len(profiles[region]['segmentation']))
        assert args.global_phases_set == 0.
    args = main.parse_main_args(['--task','recon','--sensor_params',str(output),
                                 '--multi_region_calib','True','--direct','0'])
    context = main.configure_calibration_args(args)
    assert set(args.multi_region_parameters['regions']) == set(profiles)
    from reconstructor import Reconstructor
    engine = Reconstructor.__new__(Reconstructor)
    engine.global_point_ids = np.asarray(context['global_point_ids'])
    engine.sharpness = args.sharpness
    engine.min_consecutive = args.min_consecutive
    normalized = engine._normalise_multiregion_parameters(args.multi_region_parameters)
    assert normalized['directions'] == [0, 120, 240]
    assert sorted(p for r in normalized['regions'].values() for p in r['owned_point_ids']) == list(range(1, 9))


def test_acceptance_explicit_zero_phase_does_not_refit(monkeypatch):
    """合法的零相位必须与未提供相位区分，加载标定后不能隐式重拟合。"""
    from initialiser import Initialiser
    from sensor import Sensor
    calls = []
    monkeypatch.setattr(Sensor, '__init__', lambda self, *a, **kw: None)
    monkeypatch.setattr(Initialiser, 'rot_seg_fit', lambda self, **kw: calls.append(kw))
    result = Initialiser(discont=np.pi, segmentation=[2], global_phases_set=0.)
    assert result.global_phases_set == 0.
    assert calls == []


def test_acceptance_demo_closes_runtime_on_normal_and_failed_exit(tmp_path, monkeypatch):
    """假数据源与窗口验证真实demo装配退出清理，不连接设备。"""
    import pytest
    import shape_sensing.runtime.demo as demo
    import shape_sensing.runtime.transport as transport
    events = []
    class Source:
        def open(self): events.append('source-open')
        def read(self): events.append('receive')
        def close(self): events.append('source-close')
    class Runtime:
        def __init__(self):
            self.source = Source()
            self.resources = []
        def start(self):
            self.source.open()
            self.source.read()
        def close(self):
            for resource in self.resources:
                resource.close()
            self.source.close()
            events.append('runtime-close')
    class Transport:
        def __init__(self, runtime): pass
        def open(self): events.append('transport-open')
        def publish(self, frame): pass
        def close(self): events.append('transport-close')
    class App:
        def __init__(self, fail): self.fail = fail
        def exec_(self):
            if self.fail:
                raise RuntimeError('event-loop-failure')
            return 7
    def create(session, args):
        return Runtime(), SimpleNamespace(show=lambda: events.append('show'),
                                          close=lambda: events.append('window-close'))
    monkeypatch.setattr(demo, 'create_demo', create)
    monkeypatch.setattr(transport, 'UdpTransport', Transport)
    args = SimpleNamespace(imu=False, demo_recorder=False)
    for fail in (False, True):
        factory = lambda argv: App(fail)
        if fail:
            with pytest.raises(RuntimeError, match='event-loop-failure'):
                demo.run_demo(object(), args, factory)
        else:
            assert demo.run_demo(object(), args, factory) == 7
        assert events[-2:] == ['source-close', 'runtime-close']
        assert 'window-close' in events
    assert events.count('receive') == 2


def test_acceptance_algorithm_modules_import_without_gui():
    """配置与算法接口导入不依赖Qt/VTK，不产生设备操作。"""
    import subprocess
    import sys
    code = '''import sys
sys.modules['PyQt5'] = None
sys.modules['vtk'] = None
import shape_sensing.configuration
import shape_sensing.input_preprocessing
import shape_sensing.initialization
import shape_sensing.curvature
import shape_sensing.optimization
import shape_sensing.reconstruction
import shape_sensing.evaluation
import shape_sensing.visualization
import shape_sensing.runtime
import shape_sensing.pipelines
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors='replace')


def test_acceptance_weights_only_accepts_validated_resume_slopes():
    import main
    args=main.parse_main_args(['--task','optim','--optimization_region','end5cm',
        '--multi_region_calib','False','--end5cm_point_calib','True','--optim_weights_only','True'])
    args._resume_seed_parameters=['slopes']
    args.slopes={d:np.ones((10,2)) for d in (0,120,240)}
    args.weights=None
    context=main.configure_calibration_args(args)
    assert context['optimization_region']=='end5cm'


def test_acceptance_legacy_constructor_receives_no_internal_control_keys(monkeypatch):
    """走真实继承构造链，验证Sensor参数后停止，避免读取实际数据。"""
    import inspect
    import pytest
    import main
    from sensor import Sensor
    from optimiser import Optimiser
    from calibration_batch import optimizer_parameters
    signature = inspect.signature(Sensor.__init__)
    class StopBeforeReadingData(Exception): pass
    def check_constructor(self, *args, **kwargs):
        signature.bind(self, *args, **kwargs)
        raise StopBeforeReadingData
    monkeypatch.setattr(Sensor, '__init__', check_constructor)
    args = main.parse_main_args(['--task', 'init'])
    main.configure_calibration_args(args)
    args._resume_provenance = {'source':'verified-run'}
    args._resume_seed_parameters = ['slopes']
    with pytest.raises(StopBeforeReadingData):
        Optimiser(**optimizer_parameters(args))
