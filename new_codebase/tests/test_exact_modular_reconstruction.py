"""模块5/6使用拆分前真实数据及原独立公式作回归。"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from shape_sensing.contracts import (CurvatureResult, DesignGeometry, ReconstructionConfig,
                                     ReconstructionState, EvaluationConfig, EvaluationReference)
from shape_sensing.reconstruction.service import ReconstructionService
from shape_sensing.evaluation.service import EvaluationService
from shape_sensing.reconstruction.integration import ShapeIntegrator

BASE = Path(__file__).parent / 'fixtures/module_split_baseline/real_09B7.npz'


def baseline():
    """读取拆分前保存的真实09B7数组，距离m、曲率1/m、角度rad。"""
    return np.load(BASE)


def geometry(z):
    """根据真实点(P,)与含虚拟点(Q,)基线构造显式几何m。"""
    return DesignGeometry('09B7_NeTi', .0002, z['samp_dist'].copy(), (1,3,4), {},
                          float(z['ad_samp_dist'][0]), float(z['ad_samp_dist'][-1]))


def old_methods():
    """从固定拆分前提交加载参考方法；独立于新生产核，用于数值比较。"""
    source = BASE.with_name('reference_reconstruction.txt').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    nodes = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name != '__init__']
    ns = {'np': np, 'uniform_filter1d': uniform_filter1d,
          'CubicHermiteSpline': CubicHermiteSpline, 'PchipInterpolator': PchipInterpolator}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<baseline>', 'exec'), ns)
    return ns


@pytest.mark.parametrize('frame_rule', ['TNB', 'TNN'])
@pytest.mark.parametrize('mode', ['hermite', 'scalar_pchip', 'vector_linear', 'vector_pchip'])
@pytest.mark.parametrize('virtual_points', [False, True])
@pytest.mark.parametrize('vector_filter', [False, True])
def test_real_inputs_all_preparation_branches(frame_rule, mode, vector_filter, virtual_points):
    """真实曲率输入，四插值/双标架/方向滤波及上一帧均对照固定旧代码。"""
    z = baseline()
    config = ReconstructionConfig(frame_rule=frame_rule, curvature_interp=mode,
                                   end_point=virtual_points, zero_init=virtual_points, move_filter_for_vector=vector_filter)
    design = geometry(z)
    service = ReconstructionService(config)
    methods = old_methods()
    prefix = '0_circ_2000'
    cur = CurvatureResult(z['samp_dist'], z[prefix+'_magnitude'], z[prefix+'_angle'], np.arange(22))
    before = [a.copy() for a in (cur.distance_m, cur.curvature_per_m, cur.direction_rad)]
    old = SimpleNamespace(**config.__dict__)
    old.samp_dist = z['samp_dist'].copy()
    old.ad_samp_dist = z['ad_samp_dist'].copy() if virtual_points else z['samp_dist'].copy()
    for name, fn in methods.items():
        if callable(fn) and name.startswith('_'):
            setattr(old, name, fn.__get__(old) if not isinstance(fn, staticmethod) else fn.__func__)
    old._vector_curvature_method = lambda x: {'vector_linear':'linear', 'vector_pchip':'pchip'}[x]
    old._Strain_2_Cur = lambda **kw: (cur.curvature_per_m.copy(), cur.direction_rad.copy())
    integrator = ShapeIntegrator(frame_rule, config.TNB_threshold)
    old._F_S_frame = integrator._F_S_frame
    old._parallel_frame = integrator._parallel_frame
    state = None
    for repeat in range(2):
        reference = old._reconstruct_frame(None, 'circ', None, None, None, 1., [1.,0.,0.],
                       angle_prev=None if state is None else state.previous_raw_angles_rad,
                       angle_unwrap_prev=None if state is None else state.previous_unwrapped_angles_rad)
        prepared = service.prepare(cur, design, [1.,0.,0.], state)
        shape = service.integrate(prepared)
        for actual, expected in [(prepared.diagnostics['angle_prepared'],reference[0]),
                                 (prepared.diagnostics['angle_unwrap'],reference[1]),
                                 (prepared.curvature_per_m,reference[2]), (prepared.distance_m,reference[3])]:
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(shape.values, reference[4:8]):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        for key in reference[8]:
            np.testing.assert_allclose(prepared.diagnostics[key], reference[8][key], rtol=0, atol=0)
        state = prepared.diagnostics['state_out']
    for actual, expected in zip((cur.distance_m, cur.curvature_per_m, cur.direction_rad), before):
        np.testing.assert_array_equal(actual, expected)


def test_saved_real_baseline_all_frames():
    """所有已保存09B7 rawcurv恢复原帧数组，不重新计算曲率或拟合。"""
    z = baseline()
    service = ReconstructionService(ReconstructionConfig(end_point=True, frame_rule="TNN",
        curvature_interp="vector_pchip", move_filter_for_vector=True, TNB_threshold=0.0))
    design = geometry(z)
    prefixes = [k[:-10] for k in z.files if k.endswith('_magnitude')]
    for prefix in prefixes:
        cur = CurvatureResult(z['samp_dist'], z[prefix+'_magnitude'], z[prefix+'_angle'], np.arange(22))
        prepared = service.prepare(cur, design, [1.,0.,0.])
        shape = service.integrate(prepared)
        arrays = (prepared.diagnostics['angle_prepared'], prepared.diagnostics['angle_unwrap'],
                  prepared.curvature_per_m, prepared.distance_m, *shape.values)
        for i, actual in enumerate(arrays):
            np.testing.assert_allclose(actual, z[f'{prefix}_frame_{i}'], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('kind,idx,angle', [('circ',None,None), ('S',None,None),
    ('end',.03,None), ('end3',None,120), ('rot0',.03,None), ('up',.04,None), ('down',.04,None)])
def test_all_ideal_shapes_match_original(kind, idx, angle):
    """所有七类理想曲线按原公式比较，单位均为m。"""
    from shape_sensing.evaluation.metrics import EvaluationMixin
    distance = np.linspace(0,.4,120)
    radius = .1
    reference = EvaluationReference(radius_m=radius, shape_kind=kind, angle_deg=angle,
                                    sample_coordinates_m=distance, metadata={'idx_m':idx})
    service = EvaluationService(EvaluationConfig())
    expected = EvaluationMixin._construct_ideal_shape
    original = SimpleNamespace(samp_dist=distance, end_len=.05)
    for name in dir(EvaluationMixin):
        if name.startswith('_construct_'):
            setattr(original,name,getattr(EvaluationMixin,name).__get__(original))
    val_len = 2*np.pi*radius+.1 if kind == 'S' else distance[-1]
    np.testing.assert_array_equal(service.ideal_shape(distance,reference),
                                  expected(original,kind,distance,val_len,radius,idx,angle))



def test_real_evaluation_batch_matches_saved_metrics():
    """真实三方向批量逐点与末端汇总对照拆分前metadata，指标保持无量纲。"""
    import json
    z = baseline()
    metadata = json.loads(BASE.with_name('metadata.json').read_text(encoding='utf-8'))
    service = EvaluationService(EvaluationConfig())
    for direct in (0,120,240):
        prefixes = [key[:-6] for key in z.files if key.startswith(f'{direct}_circ_') and key.endswith('_shape')]
        results = {'circ': {int(p.split('_')[-1]): z[p+'_shape'] for p in prefixes}}
        radii = {direct: {'circ': sorted(results['circ'])}}
        result = service.evaluate_batch(z[prefixes[0]+'_frame_3'],
            results, radii, direct, z['samp_dist'], z[prefixes[0]+'_frame_2'])
        for key in ('avg_point_err','avg_end_err'):
            for kind, expected in metadata['directions'][str(direct)][key].items():
                np.testing.assert_allclose(result.metrics[key][kind], expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('kind,idx', [('circ',None), ('S',None), ('end',.03), ('end3',None), ('rot0',.03)])
@pytest.mark.parametrize('mode', ['hermite','scalar_pchip','vector_linear','vector_pchip'])
def test_explicit_ideal_curvature_preserves_original(kind,idx,mode):
    """5接收6显式理想曲率回调，四插值路径与原forceideal各网格一致。"""
    from shape_sensing.evaluation.ideal_curvature import IdealCurvatureMixin
    z=baseline()
    config=ReconstructionConfig(end_point=True,curvature_interp=mode)
    evaluation=EvaluationService(EvaluationConfig())
    reference=EvaluationReference(radius_m=.1,shape_kind=kind,sample_coordinates_m=z['samp_dist'],
                                   metadata={'idx_m':idx})
    cb=lambda distance: evaluation.ideal_curvature(distance,reference)
    service=ReconstructionService(config)
    cur=CurvatureResult(z['samp_dist'],z['0_circ_2000_magnitude'],z['0_circ_2000_angle'],np.arange(22))
    prepared=service.prepare(cur,geometry(z),[1.,0.,0.],ideal_curvature=cb)
    original=SimpleNamespace(**config.__dict__,samp_dist=z['samp_dist'],ad_samp_dist=z['ad_samp_dist'])
    methods=old_methods()
    for name,fn in methods.items():
        if callable(fn) and name.startswith('_'):
            setattr(original,name,fn.__get__(original) if not isinstance(fn,staticmethod) else fn.__func__)
    original._vector_curvature_method=lambda x: {'vector_linear':'linear','vector_pchip':'pchip'}[x]
    original._Strain_2_Cur=lambda **kw:(cur.curvature_per_m.copy(),cur.direction_rad.copy())
    original._construct_ideal_magnitude_for_recon=IdealCurvatureMixin._construct_ideal_magnitude_for_recon.__get__(original)
    original.end_len=.05
    original._F_S_frame=ShapeIntegrator()._F_S_frame
    keys=[100.] if idx is None else [100.,idx*1000]
    expected=original._reconstruct_frame(keys,kind,None,None,None,1.,[1.,0.,0.],force_ideal_magnitude=True)
    np.testing.assert_array_equal(prepared.curvature_per_m,expected[2])
    np.testing.assert_array_equal(prepared.direction_rad,expected[8]['ang_interp'])


def test_rot0_original_segment_and_normalized_error_definitions():
    """旋转段阈值识别和原PCA指标不以非零点替换；逐点保持有效长度归一化。"""
    from shape_sensing.evaluation.metrics import EvaluationMixin
    from shape_sensing.contracts import ShapeResult
    distance=np.linspace(0,.5,120)
    reference=EvaluationReference(radius_m=.05,shape_kind='rot0',metadata={'idx_m':.08})
    service=EvaluationService(EvaluationConfig())
    ideal=service.ideal_shape(distance,reference)
    cur=np.where((distance>=.08)&(distance<=.08+2*np.pi*.05),20.,0.)
    shape=ShapeResult((ideal,), 'TNN',position_m=ideal,distance_m=distance)
    ref=replace(reference,metadata={'idx_m':.08,'curvature_per_m':cur})
    report=service.evaluate(shape,ref)
    old=SimpleNamespace(interp_num=120,straight_threshold=2.,rot_threshold=4.)
    for name in ('_compute_rot0_err','_find_rot_segment','_find_rot_axis'):
        setattr(old,name,getattr(EvaluationMixin,name).__get__(old))
    pitch,radius=old._compute_rot0_err(ideal,cur,.05)
    assert report.metrics['rot0_pitch_error_m']==pitch
    assert report.metrics['rot0_radius_error_m']==radius
    shifted=ideal+np.array([.001,0,0])
    shape=replace(shape,position_m=shifted)
    report=service.evaluate(shape,EvaluationReference(ideal_position_m=ideal))
    np.testing.assert_allclose(report.metrics['point_errors'], .001/.5,rtol=1e-13)
    assert report.metrics['avg_end_err']==report.metrics['point_errors'][-1]


def test_temporal_vector_filter_matches_original_and_keeps_inputs():
    """时间Kalman与原动态方法逐数组一致，状态显式传出且不改输入。"""
    from shape_sensing.reconstruction.temporal import filter_curvature_vectors, average_curve
    from shape_sensing.contracts import PreparedCurvature
    source = BASE.with_name('reference_temporal.txt').read_text(encoding='utf-8')
    method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='_apply_curvature_vector_kalman')
    ns={'np':np}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<original-temporal>','exec'),ns)
    obj=SimpleNamespace(kalman_x=None,kalman_p=None,demo_kalman_Rk=.0001,demo_kalman_Bk=.00001)
    state=None
    for length in (10,10,10,7,7):
        magnitude=np.linspace(.1,10.,length)
        angle=np.linspace(-3.2,3.2,length)
        expected=ns['_apply_curvature_vector_kalman'](obj,magnitude,angle)
        prepared=PreparedCurvature(np.linspace(0,.5,length),magnitude,angle,np.array([1.,0.,0.]),np.arange(length))
        result,state=filter_curvature_vectors(prepared,state,measurement_noise=.0001,process_noise=.00001)
        np.testing.assert_array_equal(result.curvature_per_m,expected[0])
        np.testing.assert_array_equal(result.direction_rad,expected[1])
        np.testing.assert_array_equal(state.x,obj.kalman_x)
        np.testing.assert_array_equal(state.p,obj.kalman_p)
        np.testing.assert_array_equal(prepared.curvature_per_m,magnitude)
        np.testing.assert_array_equal(prepared.direction_rad,angle)
        np.testing.assert_array_equal(result.diagnostics['cur_interp'],expected[0])
    history=()
    frames=[np.full((10,3),float(i)) for i in range(5)]
    for i,frame in enumerate(frames):
        curve,history=average_curve(frame,history,3)
        np.testing.assert_array_equal(curve,frame if i<2 else np.mean(frames[i-2:i+1],axis=0))
