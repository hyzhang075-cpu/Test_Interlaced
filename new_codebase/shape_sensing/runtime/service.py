"""模块8：无Qt/硬件依赖的实时会话、采集生命周期和同帧下游输出。"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Protocol, Any, Callable
from queue import Queue, Empty, Full
from threading import Event, Thread, RLock
import copy
import time
import numpy as np
from ..contracts import ShapeResult
from ..reconstruction.temporal import average_curve
from ..reconstruction.geometry import rotation_x_matrix, terminal_pose
from .anchor import EntranceAnchorConstraint, AnchorConstraintResult

class FrameRejected(ValueError):
    """尚无有效历史可修复的采集帧；poll/replay跳过，不提交会话状态。"""

class FrameProcessor(Protocol):
    """模块1/3/5组合入口；signals(P,C)，tangent_seed(3,)无量纲，结果位置m。"""
    def __call__(self, signals, state=None, direct=0, tangent_seed=None): ...

class FrameSource(Protocol):
    """显式生命周期采集源；read返回(P,C)原始信号，结束返回None。"""
    def open(self): ...
    def read(self): ...
    def close(self): ...

@dataclass(frozen=True)
class RuntimeFrame:
    """同一最终帧；shape.position_m(N,3)m、end_pose(3,3)、raw_signals(P,C)。"""
    timestamp: float
    frame_index: int
    shape: ShapeResult
    before_anchor_m: np.ndarray
    end_pose: np.ndarray
    raw_signals: np.ndarray
    anchor: AnchorConstraintResult
    insert_value_mm: float | None

@dataclass
class SessionState:
    """模块8独占跨帧状态；processor_state由模块1/5显式返回，曲线历史单位m。"""
    frame_index: int = 0
    processor_state: Any = None
    curve_history: tuple = ()
    imu_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    insert_value_mm: float | None = None
    last_frame: RuntimeFrame | None = None

class RuntimeService:
    """构造不采集/不拟合；live/replay均通过process_frame执行同一处理路径。"""
    def __init__(self, processor: FrameProcessor, *, source: FrameSource | None = None,
                 recorder: Callable | None = None, publisher: Callable | None = None,
                 moving_window=1, queue_size=100, direct=0, imu=False,
                 anchor_enabled=False, anchor_translation_enabled=False,
                 anchor_rotation_enabled=False, anchor_offset_m=0.06, rotation_deg=0., reset_processor=None):
        """输入显式处理器/源/输出回调与运行设置；输出无I/O的新会话实例。

        recorder/publisher接收RuntimeFrame，均使用锚定后的同一个最终shape。
        """
        if moving_window < 1 or queue_size < 1:
            raise ValueError('window and queue size must be positive')
        self.processor = processor
        self.source = source
        self.recorder = recorder
        self.publisher = publisher
        self.moving_window = moving_window
        self.direct = direct
        self.imu = imu
        self.rotation_deg = float(rotation_deg)
        self.reset_processor = reset_processor
        self.state = SessionState()
        self.anchor = EntranceAnchorConstraint(enabled=anchor_enabled,
            translation_enabled=anchor_translation_enabled or anchor_enabled,
            rotation_enabled=anchor_rotation_enabled or anchor_enabled,
            entrance_offset_m=anchor_offset_m)
        self.queue = Queue(maxsize=queue_size)
        self.lock = RLock()
        self.stop_event = Event()
        self.thread = None
        self.error = None
        self._source_open = False
        self.resources = []

    def set_imu_rotation(self, rotation):
        """输入IMU相对姿态(3,3)无量纲；输出None，复制保存本会话最新有效矩阵。"""
        matrix = np.asarray(rotation, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError('IMU rotation must be finite (3,3)')
        with self.lock:
            self.state.imu_rotation = matrix.copy()

    def set_insertion(self, value_mm):
        """输入有限推进量float(mm)，输出None；推进基线由锚定器在有效帧捕获。"""
        if not np.isscalar(value_mm) or not np.isfinite(value_mm):
            raise ValueError('insertion must be finite scalar mm')
        with self.lock:
            self.state.insert_value_mm = float(value_mm)

    def configure_anchor(self, **settings):
        """输入锚定开关/距离m/reset命名参数；输出None，锁内原子更新会话锚点。"""
        with self.lock:
            self.anchor.configure(**settings)

    def process_frame(self, signals, *, timestamp=None):
        """输入signals(P,C)原量纲及可选秒时间戳，输出最终RuntimeFrame。

        顺序为模块1/3/5处理、时间曲线均值、入口锚定，然后向记录/通讯发送同一结果。
        积分弧长网格变化时重新累积均值；失败不提交处理器状态、帧号或历史。
        """
        raw = np.array(signals, dtype=float, copy=True)
        if raw.ndim != 2:
            raise ValueError('signals must have shape (P,C)')
        with self.lock:
            tangent = np.array([1., 0., 0.])
            if self.imu:
                # 原IMU坐标约定：仅改变初始切向，保持原轴映射。
                axis_map = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
                tangent = -self.state.imu_rotation[:, 1] @ axis_map
            shape, processor_state = self.processor(raw.copy(),
                state=copy.deepcopy(self.state.processor_state), direct=self.direct,
                tangent_seed=tangent)
            rotation = rotation_x_matrix(self.rotation_deg)
            curve = np.asarray(shape.position_m, dtype=float) @ rotation.T
            axes = [None if axis is None else np.asarray(axis) @ rotation.T
                    for axis in (shape.tangent, shape.normal, shape.binormal)]
            if curve.ndim != 2 or curve.shape[1] != 3 or not np.isfinite(curve).all():
                raise ValueError('processor must return finite position_m(N,3)')
            previous = self.state.last_frame
            history = self.state.curve_history
            if previous is not None and not np.array_equal(shape.distance_m, previous.shape.distance_m):
                history = ()
            smoothed, history = average_curve(curve, history, self.moving_window)
            anchored = self.anchor.apply(smoothed, self.state.insert_value_mm,
                *axes)
            transformed = [None if axis is None else np.asarray(axis) @ anchored.rotation_matrix.T for axis in axes]
            if all(axis is not None for axis in transformed):
                end_pose = terminal_pose(shape.tangent, shape.normal,
                    shape.binormal, rotation, self.rotation_deg) @ anchored.rotation_matrix.T
            else:
                end_pose = np.eye(3)
            result_shape = replace(shape, position_m=anchored.curve_m,
                tangent=transformed[0], normal=transformed[1], binormal=transformed[2])
            frame = RuntimeFrame(time.time() if timestamp is None else float(timestamp),
                self.state.frame_index, result_shape, smoothed.copy(), end_pose,
                raw.copy(), anchored, self.state.insert_value_mm)
            self.state.processor_state = processor_state
            self.state.curve_history = history
            self.state.frame_index += 1
            self.state.last_frame = frame
        if self.recorder is not None:
            self.recorder(frame)
        if self.publisher is not None:
            self.publisher(frame)
        return frame

    def replay(self, frames):
        """输入可迭代(P,C)帧，逐帧产出RuntimeFrame；不打开源或启动后台线程。"""
        for signals in frames:
            try:
                yield self.process_frame(signals)
            except FrameRejected:
                continue

    def submit(self, signals):
        """输入(P,C)帧，输出None；复制入有限队列，队满时丢弃最旧帧。"""
        item = np.array(signals, copy=True)
        try:
            self.queue.put_nowait(item)
        except Full:
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            self.queue.put_nowait(item)

    def poll(self):
        """输入无，输出下一最终帧或None；Qt和无界面客户端使用同一非阻塞入口。"""
        try:
            signals = self.queue.get_nowait()
        except Empty:
            return None
        try:
            return self.process_frame(signals)
        except FrameRejected:
            return None

    def _collect(self):
        """输入无，输出None；读取源并入队，保存采集错误，最终释放源。"""
        try:
            while not self.stop_event.is_set():
                item = self.source.read()
                if item is None:
                    break
                self.submit(item)
        except Exception as exc:
            self.error = exc
        finally:
            self._close_source()

    def _close_source(self):
        """输入无，输出None；确保已打开的源最多关闭一次。"""
        with self.lock:
            if self._source_open:
                self._source_open = False
                self.source.close()

    def start(self):
        """输入无，输出None；显式打开采集源并创建线程，重复调用不重复启动。"""
        if self.thread is not None and self.thread.is_alive():
            return
        if self.source is None:
            raise ValueError('a source is required for live acquisition')
        self.stop_event.clear()
        self.error = None
        self.source.open()
        self._source_open = True
        self.thread = Thread(target=self._collect, daemon=True, name='shape-acquisition')
        self.thread.start()

    def reset(self):
        """输入无，输出None；清空历史/推进/IMU/队列和锚点，保留显式服务配置。"""
        with self.lock:
            self.state = SessionState()
            if self.reset_processor is not None:
                self.reset_processor()
            self.anchor.configure(reset=True)
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Empty:
                    break

    def close(self):
        """输入无，输出 None；停止采集并关闭资源，保留最后有效帧供回放检查。

        reset 可显式清空帧与历史。若清理失败，释放其他资源后再抛出首个错误。
        """
        self.stop_event.set()
        errors = []
        try:
            self._close_source()
        except Exception as exc:
            errors.append(exc)
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                errors.append(RuntimeError('source did not stop after close'))
        resources, self.resources = self.resources, []
        for resource in resources:
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def run_demo(session, args, app_factory):
    """输入TaskSession、CLI配置与Qt应用工厂，输出退出码int；显式运行才导入Qt。"""
    from .demo import run_demo as execute
    return execute(session, args, app_factory)
