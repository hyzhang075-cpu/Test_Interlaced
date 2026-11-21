"""生产demo装配；所有I/O均发生在显式run_demo调用。"""
import sys
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from .service import RuntimeService, FrameRejected
from .source import InterrogatorSource
from ..reconstruction.temporal import filter_curvature_vectors

@dataclass(frozen=True)
class LiveProcessingState:
    """8持有帧间状态；unwrap方向rad、temporal曲率1/m、direction质量无量纲、last_signals(P,C)。"""
    reconstruction: object = None
    temporal: object = None
    direction: object = None
    last_signals: object = None

class LiveFrameProcessor:
    def __init__(self, session, args):
        """输入TaskSession和CLI配置，输出独立实时处理器；不打开设备或拟合。"""
        self.session = session
        self.args = args
        self.temporal_state = None
        self.selector = None
        if args.direction_blend_enabled:
            from ..curvature.direction import DirectionSelector
            block = session.parameters.parameters_by_region[session.region]
            self.selector = DirectionSelector(block['weights'], block['slopes'], self._base_curvature,
                rotation_deg=session.config.reconstruction.rot_ang, enabled=True,
                reference_direct=args.direction_reference_direct, min_curvature=args.direction_min_curvature,
                min_consecutive=args.direction_min_consecutive, consistency_threshold=args.direction_consistency_threshold,
                smooth_alpha=args.direction_smooth_alpha, multi_region_parameters=block.get('multi_region_parameters'))

    def _base_curvature(self, keys, calib_Data, base_slopes, verbose=False):
        """输入真实signals(P,C)与参考slopes(K,)，输出模块3公开base四元组；无滤波/补点。"""
        from ..contracts import PreparedFrame
        design = self.session.design
        channels = tuple(self.session.config.input.active_channels)
        if self.session.config.input.centre_fibre:
            channels = channels[1:]
        frame = PreparedFrame(design.sampling_distance_m.copy(), design.source_indices.copy(),
                              np.array(calib_Data, copy=True), channels)
        return self.session.curvature.base(frame, design, self.session.parameters,
            direct=self.args.direction_reference_direct, region=self.session.region)

    def _filter(self, prepared):
        """输入PreparedCurvature，输出同积分场；Kalman状态归当前模块8处理器。"""
        prepared, self.temporal_state = filter_curvature_vectors(prepared, self.temporal_state,
            measurement_noise=self.args.demo_kalman_Rk, process_noise=self.args.demo_kalman_Bk)
        return prepared

    def _select(self, frame, parameters):
        """输入PreparedFrame及标定参数，输出当前帧参数副本，不改模块4结果。"""
        from copy import deepcopy
        from ..contracts import CalibrationParameters
        mapping = deepcopy(parameters.parameters_by_region)
        weights, slopes = self.selector.select(frame.signals)
        direct = self.args.direction_reference_direct
        mapping[self.session.region]['weights'][direct] = weights
        mapping[self.session.region]['slopes'][direct] = slopes
        return CalibrationParameters(mapping)

    def __call__(self, signals, state=None, direct=0, tangent_seed=None):
        """输入已采集差分(P,C)及unwrap状态，输出ShapeResult与新状态，距离m。"""
        current = LiveProcessingState() if state is None else state
        values = np.array(signals, dtype=float, copy=True)
        bad_rows = ~np.isfinite(values).all(axis=1)
        if np.any(bad_rows):
            if current.last_signals is None or current.last_signals.shape != values.shape:
                raise FrameRejected('first acquired frame contains nonfinite signals')
            values[bad_rows] = current.last_signals[bad_rows]
        self.temporal_state = deepcopy(current.temporal)
        if self.selector is not None:
            if current.direction is None:
                self.reset()
                self.temporal_state = deepcopy(current.temporal)
            else:
                masses, angle, consistency = current.direction
                self.selector.direction_masses_prev = masses.copy()
                self.selector.current_direction_masses = masses.copy()
                self.selector.current_direction_angle_deg = angle
                self.selector.current_direction_consistency = consistency
        shape, reconstruction = self.session.process_frame(values, state=current.reconstruction, direct=direct,
            tangent_seed=tangent_seed, acquired=True,
            temporal_filter=self._filter if self.args.demo_kalman_enabled else None,
            parameter_selector=self._select if self.selector is not None else None)
        direction = None if self.selector is None else (
            self.selector.direction_masses_prev.copy(), self.selector.current_direction_angle_deg,
            self.selector.current_direction_consistency)
        return shape, LiveProcessingState(reconstruction, deepcopy(self.temporal_state), direction, values.copy())

    def reset(self):
        """输入无，输出None；清空本会话Kalman和方向质量历史。"""
        self.temporal_state = None
        if self.selector is not None:
            direct = self.args.direction_reference_direct
            self.selector.direction_masses_prev = self.selector._one_hot_direction_masses(direct)
            self.selector.current_direction_masses = self.selector.direction_masses_prev.copy()
            self.selector.current_direction_angle_deg = float(direct)
            self.selector.current_direction_consistency = 1.


def create_demo(session, args):
    """输入显式TaskSession和CLI配置，输出(runtime,window)；仅创建窗口不接设备。"""
    from .window import RuntimeWindow
    processor = LiveFrameProcessor(session, args)
    runtime = RuntimeService(processor, reset_processor=processor.reset, source=InterrogatorSource(session.config.input),
        moving_window=1 if args.demo_kalman_enabled else session.config.reconstruction.move_filter,
        queue_size=session.config.input.maxsize, direct=args.direction_reference_direct, imu=args.imu,
        anchor_enabled=args.anchor_constraint_enabled, anchor_translation_enabled=args.anchor_translation_enabled,
        anchor_rotation_enabled=args.anchor_rotation_enabled, anchor_offset_m=args.anchor_offset_m,
        rotation_deg=session.config.reconstruction.rot_ang)
    window = RuntimeWindow(runtime, session.design.sampling_distance_m,
        data_frequency=session.config.input.data_freq, total_scale=args.total_scale,
        title=session.config.view.title)
    from pathlib import Path
    model_path = Path(session.config.view.body_model_path)
    if model_path.is_file():
        window.load_body_model(model_path)
    return runtime, window


def run_demo(session, args, app_factory):
    """输入TaskSession、CLI配置和QApplication工厂，输出Qt退出码int；退出统一清理。"""
    app = app_factory(sys.argv)
    runtime, window = create_demo(session, args)
    try:
        from .transport import UdpTransport
        transport = UdpTransport(runtime)
        runtime.resources.append(transport)
        transport.open()
        runtime.publisher = transport.publish
        if args.demo_recorder:
            from .protocol import DataRecorder
            recorder = DataRecorder(filename_prefix='VTK_ShapeData')
            runtime.resources.append(RecorderAdapter(recorder))
            runtime.recorder = lambda frame: recorder.save(timestamp=frame.timestamp,
                r_results=frame.shape.position_m, r_results_before_anchor=frame.before_anchor_m,
                anchor_point_m=frame.anchor.anchored_point_m, end_pose=frame.end_pose,
                cur_interp=frame.shape.diagnostics['cur_interp'], raw_data=frame.raw_signals,
                intermediates=frame.shape.diagnostics, insert_value=frame.insert_value_mm)
        if args.imu:
            from .imu_source import ImuSource
            imu_source = ImuSource(runtime, args.imu_com)
            runtime.resources.append(imu_source)
            imu_source.open()
        runtime.start()
        window.show()
        return app.exec_()
    finally:
        window.close()
        runtime.close()

class RecorderAdapter:
    def __init__(self, recorder):
        """输入旧异步记录器，输出提供统一close生命周期的适配器。"""
        self.recorder = recorder
    def close(self):
        """输入无，输出None；停止并冲刷记录器。"""
        self.recorder.stop()
