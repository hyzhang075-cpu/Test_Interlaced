import sys
import types
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from shape_sensing.contracts import ShapeResult
from shape_sensing.runtime.service import RuntimeService


def processor(signals, state=None, direct=0, tangent_seed=None):
    """模拟处理输入信号 [P,C]，返回五点曲线 m、单位标架和帧计数。"""
    offset = float(signals[0, 0])
    points = np.column_stack((np.linspace(0., 1., 5), np.full(5, offset), np.zeros(5)))
    tangent = np.tile([1., 0., 0.], (5, 1))
    normal = np.tile([0., 1., 0.], (5, 1))
    binormal = np.tile([0., 0., 1.], (5, 1))
    shape = ShapeResult('TNB', points, tangent, normal, binormal, np.linspace(0.,1.,5))
    return shape, (state or 0) + 1


def test_runtime_final_anchor_result_is_shared_by_record_publish_and_render():
    """记录、通信和显示共享同一 [N,3] m 锚定结果。"""
    recorded, published = [], []
    runtime = RuntimeService(processor, recorder=recorded.append, publisher=published.append,
        moving_window=2, anchor_enabled=True, anchor_offset_m=.2)
    raw = np.zeros((3, 2))
    original = raw.copy()
    first = runtime.process_frame(raw, timestamp=1.)
    raw2 = np.full((3, 2), .3)
    runtime.submit(raw2)
    raw2[:] = 999
    final = runtime.poll()
    assert final is recorded[-1] is published[-1] is runtime.state.last_frame
    np.testing.assert_allclose(final.shape.position_m, first.shape.position_m)
    assert final.anchor.curve_m is final.shape.position_m
    np.testing.assert_array_equal(raw, original)
    assert runtime.state.processor_state == 2
    assert runtime.poll() is None


def test_sessions_and_replay_are_independent_and_reset_clears_state():
    one, two = RuntimeService(processor), RuntimeService(processor)
    first = list(one.replay([np.ones((2, 2)), np.ones((2, 2))*2]))
    two.submit(np.ones((2, 2)))
    np.testing.assert_array_equal(two.poll().shape.position_m, first[0].shape.position_m)
    assert one.state.frame_index == 2 and two.state.frame_index == 1
    one.set_insertion(10)
    one.set_imu_rotation(np.eye(3))
    one.reset()
    assert one.state.frame_index == 0 and one.state.processor_state is None
    assert one.state.insert_value_mm is None and one.state.curve_history == ()
    assert two.state.frame_index == 1


def test_source_lifecycle_is_explicit_and_cleans_after_stream_end():
    class Source:
        def __init__(self): self.opens=0; self.closes=0; self.index=0
        def open(self): self.opens+=1
        def read(self):
            self.index += 1
            return np.zeros((2,2)) if self.index == 1 else None
        def close(self): self.closes+=1
    source = Source()
    runtime = RuntimeService(processor, source=source)
    assert source.opens == 0
    runtime.start()
    runtime.thread.join(timeout=2)
    assert source.opens == 1 and source.closes == 1
    assert runtime.poll().frame_index == 0
    runtime.close()
    runtime.close()
    assert source.closes == 1


def test_processing_failure_keeps_prior_session_state_and_does_not_record():
    recorded=[]
    def sometimes(values, **kwargs):
        if values[0,0] < 0: raise ValueError('bad input')
        return processor(values, **kwargs)
    runtime=RuntimeService(sometimes, recorder=recorded.append)
    runtime.process_frame(np.zeros((2,2)))
    with pytest.raises(ValueError): runtime.process_frame(-np.ones((2,2)))
    assert runtime.state.frame_index == 1 and len(recorded) == 1


def test_runtime_restarts_curve_average_when_grid_coordinates_change():
    """同样点数的弧长网格若发生变化，当前帧不混入旧网格坐标。"""
    def changing_grid(signals, **kwargs):
        """返回五个采样位置（m），其弧长受输入控制。"""
        shape, state = processor(signals, **kwargs)
        return replace(shape, distance_m=shape.distance_m + signals[0, 0]), state

    runtime = RuntimeService(changing_grid, moving_window=2)
    runtime.process_frame(np.zeros((2, 2)))
    current = runtime.process_frame(np.ones((2, 2)))
    np.testing.assert_array_equal(current.shape.position_m[:, 1], np.ones(5))


def test_udp_commands_update_runtime_anchor_and_insertion_without_qt():
    from shape_sensing.runtime.transport import UdpTransport
    runtime=RuntimeService(processor)
    transport=UdpTransport(runtime)
    assert transport._handle_udp_datagram(b'{"cmd":"insert_dis","data":12}')['ok']
    assert runtime.state.insert_value_mm == 12
    response=transport._handle_udp_datagram(b'{"cmd":"anchor_configure","data":{"enabled":true,"entrance_offset_m":0.08}}')
    assert response['ok'] and runtime.anchor.enabled and runtime.anchor.entrance_offset_m == .08
    assert not transport._handle_udp_datagram(b'{"cmd":"insert_dis","data":"bad"}')['ok']


def test_final_udp_and_recording_preserve_distinct_curvature_stages(tmp_path, monkeypatch):
    """最终坐标、真实点曲率、积分曲率和时间滤波前曲率分别传入原有输出字段。"""
    import json
    import pickle
    from pathlib import Path
    from shape_sensing.runtime.protocol import DataRecorder
    from shape_sensing.runtime.transport import UdpTransport

    def with_diagnostics(signals, **kwargs):
        """为 [N,3] m 形状附上不同处理阶段的 1/m 曲率和 rad 方向。"""
        shape, state = processor(signals, **kwargs)
        return replace(shape, diagnostics={
            'magnitude': np.array([1., 2.]), 'angle': np.zeros(2),
            'cur_kal': np.full(3, 3.), 'ang_filter': np.zeros(3),
            'cur_interp_before_temporal': np.full(5, 4.), 'ang_interp_before_temporal': np.ones(5),
            'cur_interp': np.full(5, 5.), 'ang_interp': np.full(5, 2.),
        }), state

    monkeypatch.chdir(tmp_path)
    runtime = RuntimeService(with_diagnostics, anchor_enabled=True, anchor_offset_m=.2)
    packets = []
    transport = UdpTransport(runtime)
    transport.socket = SimpleNamespace(sendto=lambda data, address: packets.append(json.loads(data)))
    runtime.publisher = transport.publish
    recorder = DataRecorder()
    try:
        def record(frame):
            """将最终帧位置 m、姿态及各网格诊断交给真实记录器。"""
            recorder.save(frame.timestamp, frame.shape.position_m, frame.end_pose,
                          frame.shape.diagnostics['cur_interp'], frame.raw_signals, frame.shape.diagnostics,
                          r_results_before_anchor=frame.before_anchor_m)
        runtime.recorder = record
        frame = runtime.process_frame(np.zeros((2, 2)), timestamp=1.)
    finally:
        recorder.stop()
    with Path(recorder.filename).open('rb') as source:
        recorded = pickle.load(source)
    np.testing.assert_array_equal(recorded['r_results'], frame.shape.position_m)
    np.testing.assert_array_equal(packets[0]['data'], frame.shape.position_m * 1000)
    np.testing.assert_array_equal(packets[0]['end_pose'], recorded['end_pose'])
    assert packets[0]['intermediates']['magnitude'] == recorded['magnitude'] == [1., 2.]
    assert packets[0]['intermediates']['cur_kal'] == recorded['cur_kal'] == [3.] * 3
    assert packets[0]['cur_interp'] == packets[0]['intermediates']['cur_interp'] == recorded['cur_interp'] == [5.] * 5
    assert recorded['cur_interp_raw'] == [4.] * 5
    assert recorded['ang_interp_raw'] == [1.] * 5


def test_window_callback_only_renders_final_runtime_frame(monkeypatch):
    import ast
    from pathlib import Path
    text=(Path(__file__).parents[1]/'shape_sensing/runtime/window.py').read_text(encoding='utf-8-sig')
    module=ast.parse(text)
    callback=next(node for cls in module.body if isinstance(cls,ast.ClassDef)
                  for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='update_vtk')
    small=ast.Module(body=[callback], type_ignores=[])
    namespace={}
    exec(compile(ast.fix_missing_locations(small),'<callback>','exec'),namespace)
    runtime=RuntimeService(processor)
    runtime.submit(np.zeros((2,2)))
    rendered=[]
    fake=SimpleNamespace(runtime=runtime,_update_frame=lambda curve,pose:rendered.append((curve,pose)))
    namespace['update_vtk'](fake)
    np.testing.assert_array_equal(rendered[0][0],runtime.state.last_frame.shape.position_m)
    np.testing.assert_array_equal(rendered[0][1],runtime.state.last_frame.end_pose)

@pytest.mark.parametrize("blend", [False, True])
def test_production_demo_uses_real_1_3_5_without_csv_or_initializer(monkeypatch, blend):
    import json
    from pathlib import Path
    from shape_sensing.pipelines.session import TaskSession
    from shape_sensing.runtime import demo
    import shape_sensing.runtime.transport as transport_module
    import shape_sensing.runtime.protocol as protocol_module
    root=Path(__file__).parents[1]
    config=json.loads((root/'tests/fixtures/module_split_baseline/reconstruction_config.json').read_text())
    config.update(sensor_id='runtime-test', centre_fibre=False, active_channels=[1,3,4], segmentation=(22,),
        weights={d:np.ones(6) for d in (0,120,240)},
        slopes={d:np.ones((6,1)) for d in (0,120,240)}, global_phases_set=0.)
    session=TaskSession(config,task='demo')
    monkeypatch.setattr(session, 'load_data', lambda:pytest.fail('live path read calibration CSV'))
    import shape_sensing.initialization.service as initialization
    monkeypatch.setattr(initialization.InitializationService,'fit',lambda *a,**k:pytest.fail('implicit module 2'))
    signals=np.load(root/'tests/fixtures/module_split_baseline/real_09B7.npz')['0_circ_2000_input']
    events=[]; rendered=[]; recorded=[]; published=[]; windows=[]
    class Source:
        def __init__(self, config): self.frames=iter([signals, signals*1.1])
        def open(self): events.append('source-open')
        def read(self): return next(self.frames,None)
        def close(self): events.append('source-close')
    class Window:
        def __init__(self,runtime,*a,**k): self.runtime=runtime; windows.append(self)
        def show(self): events.append('window-show')
        def close(self): self.runtime.close()
    class Transport:
        def __init__(self,runtime): self.runtime=runtime
        def open(self): events.append('udp-open')
        def publish(self,frame): published.append(frame)
        def close(self): events.append('udp-close')
    class Recorder:
        def __init__(self,**kwargs): events.append('recorder-open')
        def save(self,**kwargs): recorded.append(kwargs)
        def stop(self): events.append('recorder-close')
    class App:
        def exec_(self):
            runtime=windows[0].runtime
            runtime.thread.join(timeout=2.)
            while not runtime.queue.empty():
                frame=runtime.poll()
                rendered.append(frame)
            return 7
    monkeypatch.setattr(demo,'InterrogatorSource',Source)
    monkeypatch.setitem(sys.modules,'shape_sensing.runtime.window',SimpleNamespace(RuntimeWindow=Window))
    monkeypatch.setattr(transport_module,'UdpTransport',Transport)
    monkeypatch.setattr(protocol_module,'DataRecorder',Recorder)
    options=SimpleNamespace(direction_blend_enabled=blend,direction_reference_direct=0,
        direction_min_curvature=.5,direction_min_consecutive=3,direction_consistency_threshold=.6,
        direction_smooth_alpha=.15,
        demo_kalman_enabled=True,demo_kalman_Rk=1e-4,demo_kalman_Bk=1e-5,imu=False,
        anchor_constraint_enabled=True,anchor_translation_enabled=False,anchor_rotation_enabled=False,
        anchor_offset_m=.06,total_scale=1.,demo_recorder=True)
    assert demo.run_demo(session,options,lambda argv:App()) == 7
    assert len(rendered)==len(recorded)==len(published)==2
    for frame, record, published_frame in zip(rendered,recorded,published):
        assert frame is published_frame
        assert frame.shape.position_m.shape == (120,3)
        np.testing.assert_array_equal(record['r_results'],frame.shape.position_m)
        np.testing.assert_array_equal(record['end_pose'],frame.end_pose)
        np.testing.assert_array_equal(record['cur_interp'],frame.shape.diagnostics['cur_interp'])
    assert events.count('source-close')==events.count('udp-close')==events.count('recorder-close')==1
    assert session.initialization_result is None
    assert windows[0].runtime.state.processor_state.temporal.x.shape == (120,2)

def test_interrogator_constructor_has_no_io_and_read_scales_once(monkeypatch):
    from shape_sensing.runtime.source import InterrogatorSource
    from shape_sensing.contracts import InputConfig
    import shape_sensing.runtime.source as source_module
    import shape_sensing.input_preprocessing._kernel as input_kernel
    calls=[]
    kernel=SimpleNamespace(buffer=bytearray(),
        _configure_zero_reference=lambda *a:calls.append('zero-load'),
        _process_buffer=lambda:calls.append('parse'),
        _prep_frame=lambda:np.full((1,3,2),123.))
    monkeypatch.setattr(input_kernel,'InputKernel',lambda **kw:kernel)
    class Socket:
        def __init__(self,*args): calls.append('socket-open')
        def settimeout(self,*args): pass
        def connect(self,*args): calls.append('connect')
        def recv(self,*args): return b'wire-packet'
        def shutdown(self,*args): pass
        def close(self): calls.append('socket-close')
    monkeypatch.setattr(source_module.socket,'socket',Socket)
    source=InterrogatorSource(InputConfig())
    assert calls == []
    source.open()
    values=source.read()
    np.testing.assert_allclose(values,np.full((3,2),.000123))
    assert kernel.buffer == b'wire-packet'
    source.close()
    source.close()
    assert calls == ['zero-load','socket-open','connect','parse','socket-close']


def test_udp_publish_serializes_same_final_curve_and_curvature():
    import json
    from shape_sensing.runtime.transport import UdpTransport
    def with_diagnostics(values,**kwargs):
        shape,state=processor(values,**kwargs)
        diagnostics={key:np.ones(5) for key in ('magnitude','angle','cur_kal','ang_filter','cur_interp','ang_interp')}
        return replace(shape,diagnostics=diagnostics),state
    runtime=RuntimeService(with_diagnostics)
    frame=runtime.process_frame(np.zeros((3,2)))
    packets=[]
    transport=UdpTransport(runtime)
    transport.socket=SimpleNamespace(sendto=lambda packet,address:packets.append(packet))
    transport.publish(frame)
    packet=json.loads(packets[0])
    np.testing.assert_array_equal(packet['data'],frame.shape.position_m*1000)
    np.testing.assert_array_equal(packet['cur_interp'],frame.shape.diagnostics['cur_interp'])

def test_runtime_passes_cached_imu_tangent_and_closes_all_resources_after_error():
    tangents=[]
    def inspect(values,**kwargs):
        tangents.append(kwargs['tangent_seed'].copy())
        return processor(values,**kwargs)
    runtime=RuntimeService(inspect,imu=True)
    runtime.set_imu_rotation(np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]))
    runtime.process_frame(np.zeros((2,2)))
    runtime.process_frame(np.zeros((2,2)))
    np.testing.assert_array_equal(tangents,[[0.,-1.,0.],[0.,-1.,0.]])
    closed=[]
    def fail():
        closed.append('bad')
        raise RuntimeError('cleanup failed')
    runtime.resources=[SimpleNamespace(close=fail),SimpleNamespace(close=lambda:closed.append('good'))]
    with pytest.raises(RuntimeError,match='cleanup failed'): runtime.close()
    assert closed==['bad','good']
    runtime.close()
