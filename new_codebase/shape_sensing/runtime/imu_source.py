"""IMU资源适配；保留原100帧基准与Yesense协议。"""
from threading import Event, Thread
import numpy as np
from scipy.spatial.transform import Rotation
from . import imu

class ImuSource:
    def __init__(self, runtime, port):
        """输入RuntimeService及串口名str，输出无I/O资源实例。"""
        self.runtime = runtime
        self.port = port
        self.connection = None
        self.thread = None
        self.stop_event = Event()
        self.error = None

    def open(self):
        """输入无，输出None；显式打开串口并启动可关闭IMU线程。"""
        self.connection = imu.open_serial(self.port, 460800)
        self.stop_event.clear()
        self.thread = Thread(target=self._receive, daemon=True, name='shape-imu')
        self.thread.start()

    def _receive(self):
        """输入无，输出None；首100帧均值建立基准，再发布相对姿态(3,3)。"""
        output = {'roll': 0., 'pitch': 0., 'yaw': 0.}
        baseline_samples = []
        baseline = None
        try:
            while not self.stop_event.is_set():
                data = self.connection.read_all()
                if not data:
                    self.stop_event.wait(.005)
                    continue
                if not imu.decode_data(data, len(data), output, False):
                    continue
                angles = np.deg2rad([output['roll'], output['pitch'], output['yaw']])
                if len(baseline_samples) < 100:
                    baseline_samples.append(angles)
                    continue
                if baseline is None:
                    baseline = Rotation.from_euler('xyz', np.mean(baseline_samples, axis=0)).as_matrix()
                    continue
                current = Rotation.from_euler('xyz', angles).as_matrix()
                self.runtime.set_imu_rotation(np.linalg.pinv(baseline) @ current)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc

    def close(self):
        """输入无，输出None；关闭串口并等待IMU线程退出。"""
        self.stop_event.set()
        if self.connection is not None:
            self.connection.close()
        if self.thread is not None:
            self.thread.join(timeout=2.)
