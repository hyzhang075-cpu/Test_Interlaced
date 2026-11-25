"""生产demo装配；所有I/O均发生在显式run_demo调用。"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
import numpy as np
from .service import RuntimeService, FrameRejected
from .source import InterrogatorSource
from ..contracts import ReconstructionState
from ..reconstruction.temporal import TemporalCurvatureState, filter_curvature_vectors


class _DemoStatusPrinter:
    """按单调时钟每秒报告有效帧吞吐量；与采集时间戳无关。"""

    def __init__(self, transport, *, clock=time.monotonic):
        self.transport = transport
        self.clock = clock
        self.last_time = clock()
        self.last_count = 0

    def __call__(self, frame):
        now = self.clock()
        elapsed = now - self.last_time
        if elapsed < 1.0:
            return
        count = frame.frame_index + 1
        fps = (count - self.last_count) / elapsed
        insertion = '--' if frame.insert_value_mm is None else f'{frame.insert_value_mm:.1f} mm'
        print(f'算法 {frame.shape.frame_rule} | 已处理 {count} 帧 | '
              f'处理帧率 {fps:.1f} FPS | 当前推进量 {insertion} | '
              f'发送ip：{self.transport.receiver[0]}', flush=True)
        self.last_time = now
        self.last_count = count


@dataclass(frozen=True)
class LiveProcessingState:
    """生产处理器返回的会话历史；由 RuntimeService 在有效帧后保存。"""
    reconstruction: ReconstructionState | None = None
    temporal: TemporalCurvatureState | None = None
    direction_masses: np.ndarray | None = None
    last_signals: np.ndarray | None = None


class LiveFrameProcessor:
    def __init__(self, session, args):
        """输入TaskSession和CLI配置，输出独立实时处理器；不打开设备或拟合。"""
        self.session = session
        self.args = args
        self.selector = None
        if args.direction_blend_enabled and not session.config.input.core_position_calib:
            from ..curvature.direction import DirectionSelector
            block = session.parameters.parameters_by_region[session.region]
            self.selector = DirectionSelector(block['weights'], block['slopes'], self._base_curvature,
                rotation_deg=session.config.reconstruction.rot_ang, enabled=True,
                reference_direct=args.direction_reference_direct, min_curvature=args.direction_min_curvature,
                min_consecutive=args.direction_min_consecutive, consistency_threshold=args.direction_consistency_threshold,
                smooth_alpha=args.direction_smooth_alpha, multi_region_parameters=block.get('multi_region_parameters'))

    def _base_curvature(self, frame):
        """用本次预处理帧 [P,C] 与参考方向参数计算真实点基础曲率 1/m、方向 rad。"""
        return self.session.curvature.base(frame, self.session.design, self.session.parameters,
            direct=self.args.direction_reference_direct, region=self.session.region)

    def __call__(self, signals, state=None, direct=0, tangent_seed=None):
        """输入已采集差分 [P,C] 与上帧状态，输出形状 [N,3] m 和下一帧状态。

        异常行用上一有效输入修复，缺少可用历史时拒绝帧。方向混合与时间滤波
        的临时状态只在本次调用存在，有效帧由会话统一提交。
        """
        current = LiveProcessingState() if state is None else state
        values = np.array(signals, dtype=float, copy=True)
        bad_rows = ~np.isfinite(values).all(axis=1)
        if np.any(bad_rows):
            if current.last_signals is None or current.last_signals.shape != values.shape:
                raise FrameRejected('first acquired frame contains nonfinite signals')
            values[bad_rows] = current.last_signals[bad_rows]
        temporal = current.temporal
        direction_masses = current.direction_masses

        def filter_frame(prepared):
            """输入积分场 [N]，卡尔曼滤波后返回 [N] 曲率 1/m 和方向 rad。"""
            nonlocal temporal
            prepared, temporal = filter_curvature_vectors(prepared, temporal,
                measurement_noise=self.args.demo_kalman_Rk, process_noise=self.args.demo_kalman_Bk)
            return prepared

        def select_parameters(frame, parameters):
            """用真实点帧 [P,C] 更新三方向质量，返回当前方向的混合标定参数。"""
            nonlocal direction_masses
            weights, slopes, direction_masses = self.selector.select(frame, direction_masses)
            region = self.session.region
            block = parameters.parameters_by_region[region]
            reference = self.args.direction_reference_direct
            blended = {**block, 'weights': {**block['weights'], reference: weights},
                       'slopes': {**block['slopes'], reference: slopes}}
            return replace(parameters, parameters_by_region={**parameters.parameters_by_region, region: blended})

        shape, reconstruction = self.session.process_frame(values, state=current.reconstruction, direct=direct,
            tangent_seed=tangent_seed, acquired=True,
            temporal_filter=filter_frame if self.args.demo_kalman_enabled else None,
            parameter_selector=select_parameters if self.selector is not None else None)
        return shape, LiveProcessingState(reconstruction, temporal, direction_masses, values)


def create_demo(session, args):
    """输入显式TaskSession和CLI配置，输出(runtime,window)；仅创建窗口不接设备。"""
    from .window import RuntimeWindow
    processor = LiveFrameProcessor(session, args)
    runtime = RuntimeService(processor, source=InterrogatorSource(session.config.input),
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
        status = _DemoStatusPrinter(transport)

        def publish_frame(frame):
            transport.publish(frame)
            status(frame)

        runtime.publisher = publish_frame
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
