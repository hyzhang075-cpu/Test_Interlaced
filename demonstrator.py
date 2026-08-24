import pickle
import queue
from datetime import datetime
import time
import vtk
import socket
import struct
import json
import threading
import numpy as np

from collections import deque
from scipy.spatial.transform import Rotation as R
from numpy.linalg import pinv
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import Qt, QTimer

import IMU
from entrance_anchor import EntranceAnchorConstraint, apply_entrance_anchor

# ==============================================================================
# 电磁联调通讯配置（与 0605_电磁联调 相同的 UDP 协议）
# 1) JSON 明文：每帧末端姿态 + 整根形状 + 曲率，发送给电磁对比程序
# 2) 二进制定长包：发送给本机 C++ 程序用于与电磁传感器同步记录
# ==============================================================================
RECEIVER_IP = '192.168.110.71'
RECEIVER_PORT = 8888
SYNC_IP = '127.0.0.1'
SYNC_PORT = 50000


def _build_sync_packet(timestamp, r_results_postrot, end_pose, cur_interp,
                       raw_data, n_pts, n_raw):
    """Build the legacy fixed-length binary packet used by the C++ recorder."""
    r_flat = np.asarray(r_results_postrot, dtype=float).flatten()
    r_flat = np.pad(r_flat, (0, max(0, 3 * n_pts - len(r_flat))))[:3 * n_pts]
    ep = np.asarray(end_pose if end_pose is not None else [0.0, 0.0, 0.0], dtype=float).flatten()
    ep = np.pad(ep, (0, max(0, 3 - len(ep))))[:3]
    cur_flat = np.asarray(cur_interp if cur_interp is not None else [], dtype=float).flatten()
    cur_flat = np.pad(cur_flat, (0, max(0, n_pts - len(cur_flat))))[:n_pts]
    raw_flat = np.asarray(raw_data if raw_data is not None else [], dtype=float).flatten()
    raw_flat = np.pad(raw_flat, (0, max(0, n_raw - len(raw_flat))))[:n_raw]
    values = np.concatenate((np.asarray([timestamp], dtype=float), r_flat, ep, cur_flat, raw_flat))
    return struct.pack('<' + 'd' * len(values), *values)


def _build_send_obj(end_pose, r_results_postrot, raw_data, intermediates):
    """构造包含重构结果、原始数据及六项中间变量的 JSON 发送对象。"""
    return {
        'end_pose': end_pose.tolist() if end_pose is not None else [0.0, 0.0, 0.0],
        'data': r_results_postrot.tolist(),
        'raw_data': raw_data.tolist() if raw_data is not None else [],
        'intermediates': {
            'magnitude': intermediates['magnitude'].tolist(),
            'angle': intermediates['angle'],
            'cur_kal': intermediates['cur_kal'].tolist(),
            'ang_filter': intermediates['ang_filter'].tolist(),
            'cur_interp': intermediates['cur_interp'].tolist(),
            'ang_interp': intermediates['ang_interp'].tolist(),
        },
        'cur_interp': intermediates['cur_interp'].tolist(),
    }


def _build_end_pose_arrow_matrices(endpoint, end_pose, scale_factor, axis_length):
    """Build three VTK actor matrices from the terminal point and frame axes."""
    endpoint = np.asarray(endpoint, dtype=float)
    end_pose = np.asarray(end_pose, dtype=float)

    if endpoint.shape != (3,):
        raise ValueError("endpoint must have shape (3,)")
    if end_pose.shape != (3, 3):
        raise ValueError("end_pose must have shape (3, 3)")
    if not np.all(np.isfinite(endpoint)) or not np.all(np.isfinite(end_pose)):
        raise ValueError("endpoint and end_pose must contain finite values")
    if not np.isfinite(scale_factor) or not np.isfinite(axis_length) or axis_length <= 0:
        raise ValueError("scale_factor must be finite and axis_length must be positive")

    origin = endpoint * float(scale_factor)
    matrices = []
    for axis in end_pose:
        norm = float(np.linalg.norm(axis))
        if norm <= np.finfo(float).eps:
            raise ValueError("end_pose axes must be non-zero")

        local_x = axis / norm
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(local_x, reference))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        local_z = np.cross(local_x, reference)
        local_z /= np.linalg.norm(local_z)
        local_y = np.cross(local_z, local_x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack((local_x, local_y, local_z)) * float(axis_length)
        matrix[:3, 3] = origin
        matrices.append(matrix)

    return np.asarray(matrices)

# ==============================================================================
# 数据记录器类：用于后台异步保存数据，避免阻塞 VTK 渲染主线程
# ==============================================================================
class DataRecorder:
    def __init__(self, filename_prefix="ShapeData"):
        start_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{filename_prefix}_{start_str}.pkl"
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"--- 数据记录已启动，保存至: {self.filename} ---")

    def save(self, timestamp, r_results, end_pose, cur_interp, raw_data, intermediates, insert_value=None):
        """
        将当前帧的重构结果打包后放入后台写入队列。

        步骤：
        1. 若记录器已停止运行，则直接返回。
        2. 将输入的时间戳、重构结果、末端姿态和曲率插值结果打包为字典。
        3. 对支持 copy() 的对象做浅拷贝，避免后续主线程修改影响已排队的数据。
        4. 以非阻塞方式将数据包放入 self.queue。
        5. 若队列已满，则直接丢弃当前帧，保证渲染实时性。

        输入：
        - timestamp: 当前帧时间戳
        - r_results: 当前帧的重构三维轨迹结果
        - end_pose: 当前帧末端姿态
        - cur_interp: 当前帧插值后的曲率序列

        输出：
        - 无显式返回
        - 副作用：
            - 向后台写入队列 self.queue 中加入一个数据包

        说明：
        - 当前实现使用 queue.Queue 的非阻塞 put，因此写入失败不会阻塞前台渲染线程
        - cur_interp 会被转换为 list；若其为 None，则保存为空列表
        """
        if not self.running:
            return
        try:
            data_packet = {
                'timestamp': timestamp,
                'r_results': r_results.copy() if hasattr(r_results, 'copy') else r_results,
                'end_pose': end_pose.copy() if hasattr(end_pose, 'copy') else end_pose,
                'cur_interp': intermediates['cur_interp'].tolist(),
                'raw_data': list(raw_data) if raw_data is not None else [],
                'magnitude': intermediates['magnitude'].tolist(),
                #'angle': intermediates['angle'].tolist(),
                'angle': intermediates['angle'],
                'cur_kal': intermediates['cur_kal'].tolist(),
                'ang_filter': intermediates['ang_filter'].tolist(),
                'ang_interp': intermediates['ang_interp'].tolist(),
                'insert_value':insert_value.copy() if hasattr(insert_value, 'copy') else insert_value
            }
            # 非阻塞放入，保证渲染实时性
            self.queue.put(data_packet, block=False)
        except queue.Full:
            pass 

    def _worker(self):
        """
        后台循环消费写入队列，并将数据逐包序列化保存到 pkl 文件。

        步骤：
        1. 以写入模式打开当前记录文件。
        2. 在 self.running 为 True 或队列仍非空时持续循环。
        3. 尝试从 self.queue 中取出一个数据包。
        4. 使用 pickle.dump 将数据包顺序写入文件。
        5. 写入成功后调用 task_done 标记当前任务完成。
        6. 当队列暂时为空时继续等待；若写入过程异常，则打印错误信息。

        输入：
        - 无（使用实例中的 self.filename、self.queue 与 self.running）

        输出：
        - 无显式返回
        - 副作用：
            - 持续向 self.filename 对应的 pkl 文件中追加序列化数据包

        说明：
        - 该方法设计为后台守护线程入口函数
        - 文件会在 with 代码块结束后自动关闭
        """
        with open(self.filename, 'wb') as f:
            while self.running or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    pickle.dump(data, f)
                    self.queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"写入错误: {e}")

    def stop(self):
        self.running = False
        self.thread.join()
        print(f"--- 数据记录结束，文件已关闭 ---")

class Demonstrator(QMainWindow):
    DIRECTION_ANCHORS = (0, 120, 240)

    def __init__(self, recon, total_scale, unwrap_threshold, align_threshold, imu=False, imu_com='COM6', imu_maxlen=10, demo_recorder=False,
                 direction_blend_enabled=False, direction_reference_direct=0, direction_min_curvature=0.5, direction_min_consecutive=3,
                 direction_consistency_threshold=0.6, direction_smooth_alpha=0.15,
                 demo_kalman_enabled=False, demo_kalman_Rk=1e-4, demo_kalman_Bk=1e-5,
                 anchor_constraint_enabled=False, title="Viewer", anchor_offset_m=0.07):
        super().__init__()

        self.setWindowTitle(title)
        self.recon = recon

        self.demo_recorder = demo_recorder
        if self.demo_recorder:
            self.recorder = DataRecorder(filename_prefix="VTK_ShapeData")
        else:
            self.recorder = None

        # imu相关参数
        self.imu = imu
        self.imu_com = imu_com
        self.imu_lock = threading.Lock() if imu else None
        self.imu_data_queue = deque(maxlen=imu_maxlen) if imu else None

        # 展示效果控制参数
        self.curve_history = deque(maxlen=self.recon.move_filter)
        self.prev_reordered_data = None
        self.total_scale = total_scale

        self.unwrap_threshold = unwrap_threshold
        self.align_threshold = align_threshold
        self.angle_prev = None
        self.angle_unwrap_prev = None

        # 跨帧卡尔曼滤波（动态展示专用）：开启后替代 curve_history 滑动平均。
        # 状态为曲线各点三维坐标，采用恒定位置模型；kalman_x/kalman_p 在首帧惰性初始化。
        self.demo_kalman_enabled = bool(demo_kalman_enabled)
        self.demo_kalman_Rk = float(demo_kalman_Rk)
        self.demo_kalman_Bk = float(demo_kalman_Bk)
        self.kalman_x = None
        self.kalman_p = None

        # 支气管入口锚点：长度可在启动时配置，也可经进程接口原子更新。
        self.insert_dis = None
        self.entrance_anchor_lock = threading.RLock()
        self.entrance_anchor_constraint = EntranceAnchorConstraint(
            enabled=anchor_constraint_enabled,
            entrance_offset_m=anchor_offset_m,
        )

        # 三方向参数软过渡只服务于动态展示，不写入 Reconstructor / Optimiser 状态。
        # 在 rot_ang 尚未确定时保持 direction_blend_enabled=False，并使用精度最高的
        # direction_reference_direct 单套参数；完成 rot_ang 标定后再打开软过渡。
        self.direction_blend_enabled = bool(direction_blend_enabled)
        self.direction_reference_direct = int(direction_reference_direct)
        self.direction_min_curvature = float(direction_min_curvature)
        self.direction_min_consecutive = max(1, int(direction_min_consecutive))
        self.direction_consistency_threshold = float(direction_consistency_threshold)
        self.direction_smooth_alpha = float(direction_smooth_alpha)
        self._validate_direction_blend_config()

        self.direction_masses_prev = self._one_hot_direction_masses(self.direction_reference_direct)
        self.current_direction_masses = self.direction_masses_prev.copy()
        self.current_direction_angle_deg = float(self.direction_reference_direct)
        self.current_direction_consistency = 1.0

        self.firstFocal = None
        self.firstPosition = None
        self.setWindowTitle("3D Shape Visualization")
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.layout.addWidget(self.vtk_widget, 1)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.renderer.SetBackground(0.2, 0.2, 0.2)

        self.add_ball()
        self.add_axes()
        self.add_end_pose_arrows()

        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("Colors")

        self.line_polydata = vtk.vtkPolyData()
        self.line_polydata.SetPoints(self.points)
        self.line_polydata.SetLines(self.lines)

        self.line_mapper = vtk.vtkPolyDataMapper()
        self.line_mapper.SetInputData(self.line_polydata)
        self.line_actor = vtk.vtkActor()
        self.line_actor.SetMapper(self.line_mapper)
        self.line_actor.GetProperty().SetLineWidth(2)

        self.renderer.AddActor(self.line_actor)
        self.renderer.ResetCamera()

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_vtk)
        self.timer.start(int(1000 / self.recon.data_freq))  

        # 电磁联调 UDP 发送套接字（与 0605_电磁联调 相同）
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SIO_UDP_CONNRESET") and hasattr(self.udp_socket, "ioctl"):
            self.udp_socket.ioctl(socket.SIO_UDP_CONNRESET, False)
        local_address = ('0.0.0.0', 8888)  # 请替换为你实际监听的端口
        self.udp_socket.bind(local_address)

            # 2. 创建一个标志位，用于控制后台线程的启动和停止
        self.udp_running = True

        # 3. 创建并启动后台线程
        # target 指向我们要在线程中运行的函数
        # daemon=True 表示这是一个守护线程，主程序退出时它会自动结束
        self.udp_thread = threading.Thread(target=self._receive_udp_data, daemon=True)
        self.udp_thread.start()


    def _receive_udp_data(self):
        """
        这个函数将在后台线程中运行，专门负责接收UDP数据。
        它不会阻塞主界面。
        """
        print("后台UDP接收线程已启动...")
        while self.udp_running:
            try:
                # 设置一个超时时间，这样循环可以定期检查 self.udp_running 标志
                self.udp_socket.settimeout(1.0) 
                data, addr = self.udp_socket.recvfrom(1024)
                
                json_str = data.decode('utf-8')
                response = json.loads(json_str)
                
                if response.get("cmd") == "insert_dis":
                    insert_dis_value = response["data"]
                    # 使用 print 在控制台输出，或者通过 PyQt 的信号(Signal)更新界面
                    print(f"收到来自 {addr} 的推进量: {insert_dis_value}")
                    self.insert_dis = insert_dis_value
                else:
                    print(f"收到未知指令: {response.get('cmd')}")
                    
            except socket.timeout:
                # settimeout 后会定期抛出这个异常，我们忽略它，继续循环即可
                continue
            except Exception as e:
                if self.udp_running: # 只有在程序正常运行时才打印错误
                    print(f"UDP接收错误: {e}")
                break # 发生其他错误时退出循环

        print("后台UDP接收线程已停止。")

    def configure_anchor_constraint(self, enabled=None, entrance_offset_m=None):
        """Thread-safe protocol-independent hook for future external control adapters."""
        with self.entrance_anchor_lock:
            self.entrance_anchor_constraint.configure(
                enabled=enabled,
                entrance_offset_m=entrance_offset_m,
                reset=True,
            )

    def _apply_anchor_constraint(self, curve_m, insert_value_mm=None):
        with self.entrance_anchor_lock:
            return apply_entrance_anchor(
                self.entrance_anchor_constraint,
                curve_m,
                insert_value_mm,
            )

    # --- 修改关闭事件，确保程序退出时能正确停止线程 ---
    def closeEvent(self, event):
        # 1. 停止UDP接收线程
        self.udp_running = False
        # 等待线程结束，确保资源被正确释放
        if hasattr(self, 'udp_thread'):
            self.udp_thread.join()
        
        # 2. 停止数据记录器线程 (原有的代码)
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.stop()
            
        # 3. 关闭UDP套接字
        if hasattr(self, 'udp_socket'):
            self.udp_socket.close()

        # 4. 调用父类的关闭事件
        super().closeEvent(event)

    def add_ball(self):
        """
        根据采样点位置创建用于显示各采样点的球体标记，并加入渲染器。

        步骤：
        1. 根据 self.recon.ad_samp_dist 生成反向排列的采样位置索引数组 self.indexArr。
        2. 为每个采样位置创建一个 vtkSphereSource。
        3. 为每个球体建立 mapper 与 actor。
        4. 将所有球体 actor 保存到实例成员中，并加入当前 renderer。

        输入：
        - 无（使用 self.recon.ad_samp_dist）

        输出：
        - 无显式返回
        - 更新以下成员：
            - self.indexArr
            - self.sphereArr
            - self.sphereActorArr

        说明：
        - 当前实现中球体初始半径固定为 30.0，初始中心均为原点
        - 球体的实际位置会在后续 _update_frame() 中动态更新
        """
        self.indexArr = self.recon.ad_samp_dist[::-1] * 1000
        self.sphereArr = []
        self.sphereActorArr = []
        for _ in enumerate(self.indexArr):
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(30.0)
            sphere.SetCenter(0.0, 0.0, 0.0)
            sphere.SetThetaResolution(50)
            sphere.SetPhiResolution(50)
            self.sphereArr.append(sphere)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)
            self.sphereActorArr.append(actor)
            self.renderer.AddActor(actor)

    def add_axes(self):
        """
        向 VTK 场景中添加主坐标轴和右下角方向标记部件。

        步骤：
        1. 创建主坐标轴 vtkAxesActor，并设置长度、轴杆类型与标签显示。
        2. 调整主坐标轴标签字体大小。
        3. 将主坐标轴加入当前 renderer。
        4. 创建右下角方向标记坐标轴，并设置标签文字与字体大小。
        5. 使用 vtkOrientationMarkerWidget 将方向标记绑定到当前交互器并启用。

        输入：
        - 无

        输出：
        - 无显式返回
        - 副作用：
            - 向当前渲染窗口中加入主坐标轴与方向标记组件

        说明：
        - 主坐标轴用于场景内参考
        - 右下角方向标记用于交互时显示当前观察方向
        """
        # 添加坐标轴
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1500, 1500, 1500)  # 设置坐标轴长度
        axes.SetShaftType(0)  # 设置轴杆类型为线条
        axes.SetAxisLabels(1)  # 显示轴标签

        # 调整坐标轴标签的字体大小
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)

        self.renderer.AddActor(axes)

        # 添加方向标记部件（右下角的小坐标系）
        orient_axes = vtk.vtkAxesActor()
        orient_axes.SetShaftTypeToCylinder()
        orient_axes.SetXAxisLabelText("X")  # 设置X轴标签
        orient_axes.SetYAxisLabelText("Y")  # 设置Y轴标签
        orient_axes.SetZAxisLabelText("Z")  # 设置Z轴标签

        # 调整方向标记部件的标签字体大小
        orient_axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(4)
        orient_axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(4)
        orient_axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(4)

        orientation_widget = vtk.vtkOrientationMarkerWidget()
        orientation_widget.SetOrientationMarker(orient_axes)
        orientation_widget.SetInteractor(self.interactor)
        orientation_widget.SetViewport(0.8, 0.0, 1.0, 0.2)  # 设置方向标记的位置（右下角）
        orientation_widget.SetEnabled(1)
        orientation_widget.InteractiveOn()

    def add_end_pose_arrows(self):
        """Add independent X/Y/Z arrows for the reconstructed terminal pose."""
        self.end_pose_axis_length = 250.0
        self.end_pose_arrow_sources = []
        self.end_pose_arrow_actors = []
        self.end_pose_arrow_matrices = []

        for color in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            arrow = vtk.vtkArrowSource()
            arrow.SetShaftResolution(24)
            arrow.SetTipResolution(24)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(arrow.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.SetVisibility(False)

            matrix = vtk.vtkMatrix4x4()
            actor.SetUserMatrix(matrix)

            self.end_pose_arrow_sources.append(arrow)
            self.end_pose_arrow_actors.append(actor)
            self.end_pose_arrow_matrices.append(matrix)
            self.renderer.AddActor(actor)

    def _update_end_pose_arrows(self, endpoint, end_pose, scale_factor):
        """Move and rotate the terminal-pose arrows without touching global axes."""
        if end_pose is None:
            for actor in self.end_pose_arrow_actors:
                actor.SetVisibility(False)
            return

        try:
            matrices = _build_end_pose_arrow_matrices(
                endpoint,
                end_pose,
                scale_factor=scale_factor,
                axis_length=self.end_pose_axis_length,
            )
        except ValueError:
            for actor in self.end_pose_arrow_actors:
                actor.SetVisibility(False)
            return

        for actor, vtk_matrix, matrix in zip(
            self.end_pose_arrow_actors,
            self.end_pose_arrow_matrices,
            matrices,
        ):
            for row in range(4):
                for column in range(4):
                    vtk_matrix.SetElement(row, column, matrix[row, column])
            vtk_matrix.Modified()
            actor.SetVisibility(True)

    def receive_IMU(self):
        """
        持续从 IMU 串口读取姿态数据，并将相对初始姿态的旋转矩阵写入缓存队列。

        步骤：
        1. 初始化 IMU 串口连接，并准备姿态解码所需的输出字典。
        2. 持续读取串口数据并调用 IMU.decode_data 进行解析。
        3. 将 roll、pitch、yaw 从角度转换为弧度。
        4. 在前 100 帧中累计姿态数据，用于估计初始参考姿态 R_init。
        5. 在参考姿态建立后，对每一帧新姿态计算当前旋转矩阵 R_0IMU。
        6. 通过 pinv(R_init) @ R_0IMU 计算相对初始姿态的旋转矩阵 R_initIMU。
        7. 在加锁条件下将 R_initIMU 追加到 self.imu_data_queue。

        输入：
        - 无（使用 self.imu_com、self.imu_lock 与 self.imu_data_queue）

        输出：
        - 无显式返回
        - 副作用：
            - 持续向 self.imu_data_queue 写入相对初始姿态的旋转矩阵

        说明：
        - 当前实现假定该方法通常运行在独立线程中
        - 若 IMU 初始化失败，则直接打印错误并返回
        - 当前实现只捕获了部分读取/解析异常
        """
        yis_out = {'tid':1, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, 'q0':1.0, 'q1':1.0, 'q2':0.0, 'q3':0.0, 'sensor_temp':25.0, 'acc_x':0.0, 'acc_y':0.0, 'acc_z':1.0, 'gyro_x':0.0, 'gyro_y':0.0, 'gyro_z':0.0, 'norm_mag_x':0.0, 'norm_mag_y':0.0, 'norm_mag_z':0.0, 'raw_mag_x':0.0, 'raw_mag_y':0.0, 'raw_mag_z':0.0, 'lat':0.0, 'longt':0.0, 'alt':0.0, 'vel_e':0.0, 'vel_n':0.0, 'vel_u':0.0, 'ms':0, 'year': 2022, 'month':8, 'day': 31, 'hour':12, 'minute':0, 'second':0, 'samp_timestamp':0, 'dataready_timestamp':0 ,'status':0}
        roll_list = []
        pitch_list = []
        yaw_list = []
        loop_cnt = 0
        
        try:
            conn=IMU.open_serial(self.imu_com, 460800)
            print("IMU connection established.")
        except Exception as e:
            print(f"IMU initialization error: {e}")
            return

        while True:
            try:
                data = conn.read_all()
                num = len(data)
                if(num > 0):
                    ret = IMU.decode_data(data, num, yis_out, False)
                    if ret is True:
                        roll = np.deg2rad(yis_out['roll'])
                        pitch = np.deg2rad(yis_out['pitch'])
                        yaw = np.deg2rad(yis_out['yaw'])
                        if loop_cnt < 100:
                            roll_list.append(roll); pitch_list.append(pitch); yaw_list.append(yaw)
                            loop_cnt += 1
                        elif loop_cnt == 100:
                            roll_mean = np.mean(roll_list); pitch_mean = np.mean(pitch_list); yaw_mean = np.mean(yaw_list)
                            r_init = R.from_euler('xyz', (roll_mean, pitch_mean, yaw_mean))
                            R_init = r_init.as_matrix()
                            # print("R_init initialized")
                            loop_cnt += 1
                        else:
                            try:
                                r = R.from_euler('xyz', (roll, pitch, yaw))
                                R_0IMU = r.as_matrix()
                                if 'R_init' in locals():
                                    R_initIMU = pinv(R_init) @ R_0IMU
                                    with self.imu_lock:
                                        self.imu_data_queue.append(R_initIMU)
                            except Exception: pass
            except IndexError: 
                pass

    def closeEvent(self, event):
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.stop()
        super().closeEvent(event)

    def _validate_direction_blend_config(self):
        """校验仅供动态展示使用的三方向参数混合配置。"""
        if self.direction_reference_direct not in self.DIRECTION_ANCHORS:
            raise ValueError(
                f"direction_reference_direct 必须属于 {self.DIRECTION_ANCHORS}，"
                f"当前为 {self.direction_reference_direct}"
            )

        if not 0.0 < self.direction_smooth_alpha <= 1.0:
            raise ValueError("direction_smooth_alpha 必须位于 (0, 1]")

        if not 0.0 <= self.direction_consistency_threshold <= 1.0:
            raise ValueError("direction_consistency_threshold 必须位于 [0, 1]")

        required = self.DIRECTION_ANCHORS if self.direction_blend_enabled else (self.direction_reference_direct,)
        missing_weights = [d for d in required if d not in self.recon.weights]
        missing_slopes = [d for d in required if d not in self.recon.slopes]
        if missing_weights or missing_slopes:
            raise ValueError(
                "动态方向参数不完整: "
                f"weights 缺少 {missing_weights}, slopes 缺少 {missing_slopes}"
            )

        if self.direction_blend_enabled:
            weight_shapes = {d: np.asarray(self.recon.weights[d]).shape for d in self.DIRECTION_ANCHORS}
            slope_shapes = {d: np.asarray(self.recon.slopes[d]).shape for d in self.DIRECTION_ANCHORS}
            if len(set(weight_shapes.values())) != 1:
                raise ValueError(f"三方向 weights 形状不一致，无法插值: {weight_shapes}")
            if len(set(slope_shapes.values())) != 1:
                raise ValueError(f"三方向 slopes 形状不一致，无法插值: {slope_shapes}")

    def _one_hot_direction_masses(self, direct):
        masses = np.zeros(len(self.DIRECTION_ANCHORS), dtype=float)
        masses[self.DIRECTION_ANCHORS.index(int(direct))] = 1.0
        return masses

    def _half_cosine_direction_masses(self, angle_deg):
        """
        在圆周上只混合当前角度两侧的两套参数。

        例如 angle_deg 位于 [0, 120] 时，使用半余弦从 0 度参数平滑过渡到
        120 度参数；240 到 360 度区间会自动过渡回 0 度参数。
        """
        angle_deg = float(angle_deg) % 360.0
        sector = int(np.floor(angle_deg / 120.0)) % len(self.DIRECTION_ANCHORS)
        left_direct = self.DIRECTION_ANCHORS[sector]
        right_direct = self.DIRECTION_ANCHORS[(sector + 1) % len(self.DIRECTION_ANCHORS)]

        t = np.clip((angle_deg - left_direct) / 120.0, 0.0, 1.0)
        right_mass = 0.5 - 0.5 * np.cos(np.pi * t)
        left_mass = 1.0 - right_mass

        masses = np.zeros(len(self.DIRECTION_ANCHORS), dtype=float)
        masses[sector] = left_mass
        masses[(sector + 1) % len(self.DIRECTION_ANCHORS)] = right_mass
        return masses

    def _find_first_stable_bending_region(self, magnitude):
        """返回近端起第一个满足曲率阈值和最少连续点数的完整连续弯曲区。"""
        magnitude = np.asarray(magnitude, dtype=float)
        valid = np.isfinite(magnitude) & (magnitude >= self.direction_min_curvature)

        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            elif not is_valid and start is not None:
                if idx - start >= self.direction_min_consecutive:
                    return np.arange(start, idx, dtype=int)
                start = None

        if start is not None and len(valid) - start >= self.direction_min_consecutive:
            # print(f"bend starts at {start}, length={valid}")
            return np.arange(start, len(valid), dtype=int)
        return None

    def _estimate_direction_masses(self, calib_Data):
        """
        用指定的最佳方向参数做第一遍粗计算，并估计当前帧的方向质量。

        只使用近端第一个稳定弯曲区。区域内采用曲率加权圆周平均；低曲率、
        无稳定弯曲区或方向一致性不足时冻结上一帧质量。有效的新质量再进行
        指数平滑，以减少跨帧边界抖动。
        """
        reference_slope = np.asarray(
            self.recon.slopes[self.direction_reference_direct][0],
            dtype=float,
        )
        _, magnitude, angle, _ = self.recon._get_base(
            keys=None,
            calib_Data=calib_Data,
            base_slopes=reference_slope,
            verbose=False,
        )

        region = self._find_first_stable_bending_region(magnitude)
        if region is None:
            return self.direction_masses_prev.copy()

        region_magnitude = np.asarray(magnitude, dtype=float)[region]
        region_angle = np.asarray(angle, dtype=float)[region]
        finite = np.isfinite(region_magnitude) & np.isfinite(region_angle)
        if not np.any(finite):
            return self.direction_masses_prev.copy()

        region_magnitude = region_magnitude[finite]
        # 与现有注释中的方向判断保持同一角度约定：方向角加 rot_ang 后再映射到
        # 0/120/240 度参数坐标。软过渡应在 rot_ang 完成标定后才启用。
        region_angle = region_angle[finite] + np.deg2rad(self.recon.rot_ang)

        magnitude_sum = float(np.sum(region_magnitude))
        if magnitude_sum <= 0.0:
            return self.direction_masses_prev.copy()

        resultant = np.sum(region_magnitude * np.exp(1j * region_angle))
        consistency = float(np.abs(resultant) / magnitude_sum)
        self.current_direction_consistency = consistency
        if consistency < self.direction_consistency_threshold:
            return self.direction_masses_prev.copy()

        angle_deg = float(np.rad2deg(np.angle(resultant)) % 360.0)
        print(f"detected bending angle is {angle_deg}")
        raw_masses = self._half_cosine_direction_masses(angle_deg)

        alpha = self.direction_smooth_alpha
        smoothed_masses = (1.0 - alpha) * self.direction_masses_prev + alpha * raw_masses
        smoothed_masses /= np.sum(smoothed_masses)

        self.direction_masses_prev = smoothed_masses
        self.current_direction_masses = smoothed_masses.copy()
        self.current_direction_angle_deg = angle_deg
        return smoothed_masses

    def _blend_direction_parameters(self, direction_masses):
        """按当前帧方向质量混合三套完整 weights / slopes 参数。"""
        direction_masses = np.asarray(direction_masses, dtype=float)
        if direction_masses.shape != (len(self.DIRECTION_ANCHORS),):
            raise ValueError(f"direction_masses 形状错误: {direction_masses.shape}")

        reference = self.direction_reference_direct
        blended_weights = np.zeros_like(np.asarray(self.recon.weights[reference], dtype=float))
        blended_slopes = np.zeros_like(np.asarray(self.recon.slopes[reference], dtype=float))

        for mass, direct in zip(direction_masses, self.DIRECTION_ANCHORS):
            blended_weights += mass * np.asarray(self.recon.weights[direct], dtype=float)
            blended_slopes += mass * np.asarray(self.recon.slopes[direct], dtype=float)

        return blended_weights, blended_slopes

    def _get_dynamic_parameters(self, calib_Data):
        """返回当前帧参数；关闭软过渡时始终使用指定的最佳方向参数。"""
        if not self.direction_blend_enabled:
            direct = self.direction_reference_direct
            masses = self._one_hot_direction_masses(direct)
            self.current_direction_masses = masses
            self.current_direction_angle_deg = float(direct)
            return self.recon.weights[direct], self.recon.slopes[direct]

        masses = self._estimate_direction_masses(calib_Data=calib_Data)
        return self._blend_direction_parameters(direction_masses=masses)

    def _apply_curve_kalman(self, r_frame):
        """
        对当前帧重构曲线做跨帧卡尔曼滤波（动态展示专用，逐点逐坐标独立）。

        模型：恒定位置模型（状态即各点三维坐标，假设短时间内位置不变）。
        - Rk 为观测噪声参数，越大越平滑、响应越慢；
        - Bk 为状态转移误差参数，越大对新观测响应越快。
        首帧或曲线点数变化时用观测值直接初始化状态。
        """
        z = np.asarray(r_frame, dtype=float)
        if self.kalman_x is None or self.kalman_x.shape != z.shape:
            self.kalman_x = z.copy()
            self.kalman_p = np.full_like(z, self.demo_kalman_Bk)
            return self.kalman_x.copy()

        # 预测：恒定位置模型，状态不变，协方差累加过程噪声
        p_pred = self.kalman_p + self.demo_kalman_Bk
        # 更新
        K = p_pred / (p_pred + self.demo_kalman_Rk)
        self.kalman_x = self.kalman_x + K * (z - self.kalman_x)
        self.kalman_p = (1.0 - K) * p_pred
        return self.kalman_x.copy()

    def update_vtk(self):
        """
        从实时数据队列中读取一帧数据，完成预处理、重构、可选 IMU 姿态融合、平滑与 VTK 显示更新。

        步骤：
        1. 记录当前处理帧的时间戳。
        2. 以非阻塞方式从 self.recon.data_queue 中读取一帧数据；若队列为空则直接返回。
        3. 将读入数据按当前单位约定转换为 calib_Data。
        4. 若启用了中心光纤补偿，则调用 self.recon._process_centreData() 去除中心通道影响。
        5. 检查当前帧是否存在非有限值；若存在，则优先使用上一帧同位置数据进行替换，否则直接跳过本帧。
        6. 若启用方向软过渡，用指定参考方向完成第一遍方向估计，经过冻结和指数平滑后混合三方向参数；否则使用参考方向单套参数。
        7. 若启用 IMU，则从 IMU 队列中读取最新姿态矩阵，并据此构造初始切向量 init_T；否则使用默认 init_T。
        8. 使用当前帧混合参数调用 _reconstruct_frame() 完成最终重构，并保持 direct=0 的统一显示坐标系。
        9. 提取末端姿态 End_pose。
        10. 将最近若干帧重构结果存入 self.curve_history，并在达到窗口长度后做均值平滑。
        11. 若启用 recorder，则将当前帧结果写入后台记录器。
        12. 调用 _update_frame() 将最终曲线写入 VTK 显示。

        输入：
        - 无（使用实例中的实时队列、IMU 队列与重构器参数）

        输出：
        - 无显式返回
        - 副作用：
            - 更新 self.prev_reordered_data
            - 更新 self.curve_history
            - 可能向 recorder 写入一帧记录
            - 触发一次 VTK 场景刷新

        说明：
        - 三方向参数过渡仅在本动态路径中生效，不修改静态优化和静态重构参数
        - 参数方向软分配与最终显示坐标系相互独立，显示旋转始终使用 direct=0
        - 当前实现对含 NaN / inf 的坏行会优先尝试用上一帧对应行替换
        - 当前实现中 IMU 姿态只用于更新初始切向量 init_T
        """
        # 记录当前处理帧的时间戳
        frame_timestamp = time.time()

        # 定义初始状态
        last_valid_R_0IMU = np.eye(3)
        R_imu_rot = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

        # 修改点 1：消除竞态条件，使用非阻塞 get() 并捕获异常
        try:
            Data = self.recon.data_queue.get_nowait()
        except queue.Empty:
            return  # 队列为空，直接返回等待下一次定时器触发

        # ---------- 处理数据 ----------
        calib_Data = Data[0] * 1e-6
        raw_data = calib_Data.copy()

        # 处理中心光纤的数据
        if self.recon.centre_fibre:
            _, _, calib_Data, _ = self.recon._process_centreData(calib_Data)

        # 确保超量程不会造成展示中断
        row_bad = ~np.isfinite(calib_Data).all(axis=1)   # shape (16,)
        if np.any(row_bad):
            print(f"超量程")
            if self.prev_reordered_data is not None and self.prev_reordered_data.shape == calib_Data.shape:
                # 用上一帧同一行替换
                calib_Data = calib_Data.copy()
                calib_Data[row_bad, :] = self.prev_reordered_data[row_bad, :]
            else:
                # 第一帧就坏了：没法用上一帧补，只能跳过这一帧（或把坏行置0）
                return
        self.prev_reordered_data = calib_Data.copy()

        # ---------- 动态三方向参数软分配 ----------
        # 方向软过渡关闭时使用 direction_reference_direct 对应的最佳单套参数；
        # 打开时先估计当前弯曲方向，再将三方向完整参数混合。曲率等级的第二维
        # 软分配仍由 _Strain_2_Cur() 在最终重构中完成。
        weight, slope = self._get_dynamic_parameters(calib_Data=calib_Data)

        # ---------- 预处理 ----------
        if self.imu:
            if self.imu_data_queue:
                with self.imu_lock:
                    R_0IMU = self.imu_data_queue[-1]  # 非阻塞获取
                    last_valid_R_0IMU = R_0IMU  # 更新最新的有效数据
            else:
                if last_valid_R_0IMU is not None:
                    # print("update_vtk is using the cached R_0IMU.")
                    R_0IMU = last_valid_R_0IMU  # 使用缓存数据
                else:
                    print("update_vtk: No valid R_0IMU available, skipping update.")
            init_T = -R_0IMU[:3, 1] @ R_imu_rot
        else:
            init_T = [1.0, 0.0, 0.0]

        # ---------- 重构 ----------
        self.angle_prev, self.angle_unwrap_prev, cur_interp, _, r, T, Y, Z, intermediates = self.recon._reconstruct_frame(keys=None, calib_shape='circ', calib_Data=calib_Data, weight=weight, slope=slope, magnitude_ratio=1.0, init_T=init_T, verbose=False,
                                                                                            unwrap_threshold=self.unwrap_threshold, align_threshold=self.align_threshold, angle_prev=self.angle_prev, angle_unwrap_prev=self.angle_unwrap_prev)
        rot_matrix = self.recon._get_rot_x(direct=0)
        r_results_postrot = r @ rot_matrix.T

        # 提取末端姿态 (End_pose)
        # 根据参考代码逻辑，End_pose 通常是末端节点的切向向量，即 T 的最后一行
        End_pose = self.recon._build_end_pose(T, Y, Z)
        
        # --------------  平滑滤波 --------------
        if self.demo_kalman_enabled:
            # 跨帧卡尔曼滤波（替代滑动平均）
            r_results_postrot = self._apply_curve_kalman(r_results_postrot)
        else:
            self.curve_history.append(r_results_postrot.copy())
            if len(self.curve_history) == self.recon.move_filter:
                r_results_postrot = np.mean(np.array(self.curve_history), axis=0)

        # ---------- 支气管入口锚点约束 ----------
        # 必须位于跨帧滤波之后、所有保存/显示/通讯之前，保证下游坐标一致。
        r_results_postrot = self._apply_anchor_constraint(
            r_results_postrot,
            self.insert_dis,
        )

        #----------------- 保存pickle数据 -----------------
        if hasattr(self, 'recorder') and self.recorder:
            insert_value = self.insert_dis.copy() if hasattr(self.insert_dis, 'copy') else self.insert_dis
            self.recorder.save(timestamp=frame_timestamp, r_results=r_results_postrot, end_pose=End_pose, cur_interp=cur_interp, raw_data=raw_data, intermediates=intermediates, insert_value=insert_value)

        self._update_frame(r_results_postrot, End_pose)

        # ---------------- 电磁联调通讯（与 0605_电磁联调 相同协议） ----------------
        # 坐标单位换算为 mm 后发送
        r_results_postrot = r_results_postrot * 1000

        # 1) 通过 UDP 以 JSTON 明文推送给软件端
        try:
            sendObj = _build_send_obj(End_pose, r_results_postrot, raw_data, intermediates)
            print("send=======")
            self.udp_socket.sendto(json.dumps(sendObj).encode(), (RECEIVER_IP, RECEIVER_PORT))
        except Exception as e:
            print(f"JSON发送错误: {e}")

        # 2) 通过 UDP 以定长二进制包发送给本机 C++ 程序用于同步记录
        # 数据格式: 时间戳(1) + 每点位置(3*interp_num) + 末端姿态(3)
        # + 曲率(interp_num) + 原始数据(total_points*num_channels) 个 double。
        # 当前配置下: 1 + 180 + 3 + 60 + 112 = 356 个 double。
        try:
            data_save = Data[0] * 1e-6
            n_pts = int(self.recon.interp_num)
            n_raw = int(self.recon.total_points * self.recon.num_channels)
            pack_data = _build_sync_packet(
                timestamp=frame_timestamp,
                r_results_postrot=r_results_postrot,
                end_pose=End_pose,
                cur_interp=cur_interp,
                raw_data=data_save,
                n_pts=n_pts,
                n_raw=n_raw,
            )
            # self.udp_socket.sendto(pack_data, (SYNC_IP, SYNC_PORT))
        except Exception:
            pass
        # -------------------------------------------------------------------------

    def _update_frame(self, data, end_pose=None):
        """
        将当前三维重构曲线写入 VTK 点集与连线结构，并刷新显示。

        步骤：
        1. 根据 self.total_scale 计算显示缩放因子。
        2. 获取当前曲线点数 num_points。
        3. 若当前 VTK 点数与 num_points 不一致，则重新初始化点集与线段拓扑结构。
        4. 遍历所有点，将三维坐标按缩放因子写入 vtkPoints。
        5. 根据 self.indexArr 计算各标记球体对应的曲线索引，并更新球体中心位置。
        6. 用曲线末端点与 End_pose 更新三根末端姿态箭头。
        7. 标记 vtkPoints 与 polydata 已修改，并触发渲染窗口刷新。

        输入：
        - data: 当前三维重构曲线，形状通常为 (num_points, 3)
        - end_pose: _build_end_pose(T, Y, Z) 返回的未经过 rot_matrix 旋转的末端局部 X/Y/Z 方向

        输出：
        - 无显式返回
        - 副作用：
            - 更新 VTK 点集、线段与球体位置
            - 触发当前窗口重新渲染

        说明：
        - 当前实现只在首次运行或点数变化时重建线段拓扑，以减少每帧开销
        - 显示坐标会乘以固定比例因子 5000 * self.total_scale
        """
        scale_factor = 5000 * self.total_scale
        num_points = len(data)

        # 避免每帧 Reset()。仅在首次运行或点数变化时初始化拓扑结构
        if self.points.GetNumberOfPoints() != num_points:
            self.points.Reset()
            self.lines.Reset()
            self.points.SetNumberOfPoints(num_points)
            
            # 一次性构建线段连接关系
            self.lines.Allocate(num_points - 1)
            for i in range(num_points - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                self.lines.InsertNextCell(line)

        # 就地更新点坐标
        for i, point in enumerate(data):
            self.points.SetPoint(i, point[0] * scale_factor, point[1] * scale_factor, point[2] * scale_factor)

        # 更新标记球体位置
        for i, len_num in enumerate(self.indexArr):
            idx = round((len_num / self.indexArr[0]) * num_points)
            idx = max(0, min(num_points - 1, idx))
            self.sphereArr[i].SetCenter(data[idx][0] * scale_factor, data[idx][1] * scale_factor, data[idx][2] * scale_factor)

        if num_points > 0:
            self._update_end_pose_arrows(data[-1], end_pose, scale_factor)

        # 标记数据已修改，触发 VTK 渲染管线更新
        self.points.Modified()
        self.line_polydata.Modified()
        self.vtk_widget.GetRenderWindow().Render()

