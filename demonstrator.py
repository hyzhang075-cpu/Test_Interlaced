import pickle
import queue
import datetime
import time
import vtk
import threading
import numpy as np

from collections import deque
from scipy.spatial.transform import Rotation as R
from numpy.linalg import pinv
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import Qt, QTimer

import IMU

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

    def save(self, timestamp, r_results, end_pose, cur_interp):
        if not self.running:
            return
        try:
            data_packet = {
                'timestamp': timestamp,
                'r_results': r_results.copy() if hasattr(r_results, 'copy') else r_results,
                'end_pose': end_pose.copy() if hasattr(end_pose, 'copy') else end_pose,
                'cur_interp': list(cur_interp) if cur_interp is not None else []
            }
            # 非阻塞放入，保证渲染实时性
            self.queue.put(data_packet, block=False)
        except queue.Full:
            pass 

    def _worker(self):
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
    def __init__(self, recon, total_scale, imu=False, imu_com='COM6', imu_maxlen=10, demo_recorder=False, title="Viewer"):
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

    def add_ball(self):
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
        orient_axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(8)
        orient_axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(8)
        orient_axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(8)

        orientation_widget = vtk.vtkOrientationMarkerWidget()
        orientation_widget.SetOrientationMarker(orient_axes)
        orientation_widget.SetInteractor(self.interactor)
        orientation_widget.SetViewport(0.8, 0.0, 1.0, 0.2)  # 设置方向标记的位置（右下角）
        orientation_widget.SetEnabled(1)
        orientation_widget.InteractiveOn()

    def receive_IMU(self):
        # ... (保持原有的IMU代码不变) ...
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

    def update_vtk(self):
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

        # ---------- 判定当前方向 ----------
        # _, _, angle = self.recon._get_base(radius=None, calib_Data=calib_Data, base_slopes=self.recon.slopes[0][0], verbose=False) # 这个地方要再看，初始方向判定固定使用得是0方向的base_slope
        # init_ang = (angle[0] + np.deg2rad(self.recon.rot_ang)) % (2 * np.pi)
        # if -np.pi / 3 <= init_ang < np.pi / 3:
        #     direct = 0
        # elif np.pi / 3 <= init_ang < np.pi:
        #     direct = 120
        # else:
        #     direct = 240
        direct = 0  # 保持原代码逻辑

        # ---------- 预处理 ----------
        cur_interp, torsion, interp_ss, init_N, _ = self.recon._preprocess(keys=None, calib_shape='circ', calib_Data=calib_Data, weight=self.recon.weights[direct], slope=self.recon.slopes[direct], verbose=False)

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
        r, T, N, B = self.recon._F_S_frame(kappa=cur_interp, tau=torsion, ss=interp_ss, init_T=init_T, init_N=init_N)
        rot_matrix = self.recon._get_rot_x(direct=0)
        r_results_postrot = r @ rot_matrix.T

        # 提取末端姿态 (End_pose)
        # 根据参考代码逻辑，End_pose 通常是末端节点的切向向量，即 T 的最后一行
        End_pose = T[-1] if len(T) > 0 else None
        
        # --------------  平滑滤波 --------------
        self.curve_history.append(r_results_postrot.copy())
        if len(self.curve_history) == self.recon.move_filter:
            r_results_postrot = np.mean(np.array(self.curve_history), axis=0)

        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.save(timestamp=frame_timestamp, r_results=r_results_postrot, end_pose=End_pose, cur_interp=cur_interp)

        self._update_frame(r_results_postrot)

    def _update_frame(self, data):
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

        # 标记数据已修改，触发 VTK 渲染管线更新
        self.points.Modified()
        self.line_polydata.Modified()
        self.vtk_widget.GetRenderWindow().Render()

