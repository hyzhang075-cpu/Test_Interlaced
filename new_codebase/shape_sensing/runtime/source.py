"""显式启动的解调仪采集源，复用模块1解包/归零/跳变保护。"""
from dataclasses import asdict
import socket
import time
import numpy as np

class InterrogatorSource:
    def __init__(self, config):
        """输入InputConfig，输出无设备I/O的采集源；read输出已裁剪(P,C)历史尺度。"""
        self.config = config
        self.kernel = None
        self.socket = None
        self.closed = True
        self.last_push = 0.

    def open(self):
        """输入无，输出None；显式建立TCP/UDP设备连接并加载可选零点文件。"""
        from ..input_preprocessing._kernel import InputKernel
        values = asdict(self.config)
        self._interlaced_input = None
        if self.config.core_position_calib and self.config.interlaced_grating:
            from ..input_preprocessing import InputService
            # Wire packets carry raw WL IDs, not indices on the merged grid.
            # Keep all raw rows through zeroing and spline interpolation; crop
            # only after merging, exactly as in static preprocessing.
            InputKernel(**values).core_calibration_channel_ids()
            self._interlaced_input = InputService(self.config)
            values.update(interlaced_grating=False, straight_trim=0, reverse=False)
        self.kernel = InputKernel(**values)
        self.kernel._configure_zero_reference(self.config.use_stored_zero, self.config.zero_reference_path)
        protocol = socket.SOCK_STREAM if self.config.interrogator == 'FiberTecc' else socket.SOCK_DGRAM
        if self.config.interrogator not in ('FiberTecc', 'Hippo'):
            raise ValueError('live acquisition supports FiberTecc and Hippo')
        if self.config.core_position_calib and self.config.interrogator == 'Hippo':
            if self.kernel.raw_trim_idx + self.kernel.total_points > 16:
                raise ValueError('Hippo 四芯实时模式最多接收 16 个原始 WL')
            if not all(1 <= channel <= 8 for channel in self.kernel.active_channels):
                raise ValueError('Hippo 四芯实时模式的物理通道必须在 1–8 内')
        self.socket = socket.socket(socket.AF_INET, protocol)
        self.socket.settimeout(3.)
        self.closed = False
        try:
            if protocol == socket.SOCK_STREAM:
                self.socket.connect((self.config.IP, self.config.PORT))
            else:
                self.socket.sendto(b'\x01\x02', (self.config.IP, self.config.PORT))
                self.socket.recv(2048)
                self.socket.sendto(b'\x01\x0A\x55', (self.config.IP, self.config.PORT))
            self.socket.settimeout(.2)
            self.last_push = 0.
        except Exception:
            self.close()
            raise

    def read(self):
        """输入无，输出差分帧(P,C)或结束None；从pm转历史应变尺度1e-6一次。

        FiberTecc沿用包缓存/跳变/零点核，Hippo沿用01/0C解析与首次归零；close可打断阻塞。
        """
        connection = self.socket
        while not self.closed:
            try:
                if self.config.interrogator == 'FiberTecc':
                    chunk = connection.recv(4096)
                    if not chunk:
                        return None
                    self.kernel.buffer.extend(chunk)
                    self.kernel._process_buffer()
                    if self.config.core_position_calib and not np.isfinite(self.kernel.current_snapshot).all():
                        continue
                    now = time.monotonic()
                    if now - self.last_push < 1 / self.config.data_freq:
                        continue
                    self.last_push = now
                    return self._response_frame(self.kernel._prep_frame()[0] * 1e-6)
                wave = self.kernel._recv_010c_once(connection)
                if self.config.core_position_calib:
                    start = self.kernel.raw_trim_idx
                    stop = start + self.kernel.total_points
                    columns = [physical for physical, _ in sorted(
                        self.kernel.active_channels.items(), key=lambda entry: entry[1])]
                    absolute = wave[np.asarray(columns) - 1, start:stop].T
                    if not self.kernel._complete_core_snapshot(absolute):
                        continue
                    zero = np.column_stack([self.kernel.zero_points[physical] for physical in columns])
                    return self._response_frame((absolute - zero) * 1000 * 1e-6)
                values = np.empty((16, self.kernel.num_channels))
                for physical, column in self.kernel.active_channels.items():
                    current = wave[physical - 1]
                    if self.kernel.zero_points[physical][0] is None:
                        self.kernel.zero_points[physical] = current.copy()
                    values[:, column] = (current - self.kernel.zero_points[physical]) * 1000 * 1e-6
                return values[self.kernel.raw_trim_idx:, :].copy()
            except socket.timeout:
                continue
            except OSError:
                if self.closed:
                    return None
                raise
        return None

    def _response_frame(self, values):
        """Return merged four-core responses once, in calibration channel order."""
        if self._interlaced_input is None:
            return values
        return self._interlaced_input.preprocess(
            values, channel_ids=tuple(self.config.active_channels)).signals

    def close(self):
        """输入无，输出None；关闭socket使采集线程退出，重复关闭安全。"""
        self.closed = True
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.socket.close()
            self.socket = None
