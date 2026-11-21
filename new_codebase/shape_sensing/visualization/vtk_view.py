"""Qt/VTK 实时视图方法；不包含通信或帧调度。"""
import numpy as np
import vtk
from scipy.spatial.transform import Rotation as R



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

def _build_anchor_marker_segments(anchor_m, scale_factor, half_size):
    """Build four body-diagonal line segments for a view-independent 3D star marker."""
    anchor = np.asarray(anchor_m, dtype=float)
    scale_factor = float(scale_factor)
    half_size = float(half_size)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor_m must be a finite 3D point")
    if not np.isfinite(scale_factor):
        raise ValueError("scale_factor must be finite")
    if not np.isfinite(half_size) or half_size <= 0.0:
        raise ValueError("half_size must be finite and positive")

    center = anchor * scale_factor
    diagonals = np.array(
        [
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            [[-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
            [[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]],
            [[-1.0, 1.0, 1.0], [1.0, -1.0, -1.0]],
        ]
    )
    return center + diagonals * half_size

class VTKViewMixin:
    """Demonstrator 的纯场景对象与帧绘制操作。"""

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
        distance = self.sampling_distance_m if hasattr(self, 'sampling_distance_m') else self.recon.ad_samp_dist
        self.indexArr = distance[::-1] * 1000
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

    def add_anchor_marker(self):
        """Add a hidden white 3D star that marks the fixed bronchial entrance."""
        self.anchor_marker_half_size = 45.0
        self.anchor_marker_points = vtk.vtkPoints()
        self.anchor_marker_points.SetNumberOfPoints(8)
        for point_index in range(8):
            self.anchor_marker_points.SetPoint(point_index, 0.0, 0.0, 0.0)

        self.anchor_marker_lines = vtk.vtkCellArray()
        for segment_index in range(4):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 2 * segment_index)
            line.GetPointIds().SetId(1, 2 * segment_index + 1)
            self.anchor_marker_lines.InsertNextCell(line)

        self.anchor_marker_polydata = vtk.vtkPolyData()
        self.anchor_marker_polydata.SetPoints(self.anchor_marker_points)
        self.anchor_marker_polydata.SetLines(self.anchor_marker_lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.anchor_marker_polydata)
        self.anchor_marker_actor = vtk.vtkActor()
        self.anchor_marker_actor.SetMapper(mapper)
        self.anchor_marker_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.anchor_marker_actor.GetProperty().SetLineWidth(3.0)
        self.anchor_marker_actor.SetVisibility(False)
        self.renderer.AddActor(self.anchor_marker_actor)

    def _update_anchor_marker(self, scale_factor):
        """Show the fixed anchor on the next GUI frame, or hide it when unavailable."""
        with self.entrance_anchor_lock:
            enabled = self.entrance_anchor_constraint.enabled
            anchor = self.entrance_anchor_constraint.anchor_m
            anchor = None if anchor is None else np.asarray(anchor, dtype=float).copy()

        if not enabled or anchor is None:
            self.anchor_marker_actor.SetVisibility(False)
            return

        try:
            segments = _build_anchor_marker_segments(
                anchor,
                scale_factor=scale_factor,
                half_size=self.anchor_marker_half_size,
            )
        except ValueError:
            self.anchor_marker_actor.SetVisibility(False)
            return

        for point_index, point in enumerate(segments.reshape(-1, 3)):
            self.anchor_marker_points.SetPoint(point_index, *point)
        self.anchor_marker_points.Modified()
        self.anchor_marker_polydata.Modified()
        self.anchor_marker_actor.SetVisibility(True)

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
        - end_pose: 末端标架先乘 rot_matrix，再将 Y/Z 绕旋转后的 T 反向旋转 rot_ang

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
        self._update_anchor_marker(scale_factor)

        # 标记数据已修改，触发 VTK 渲染管线更新
        self.points.Modified()
        self.line_polydata.Modified()
        self.vtk_widget.GetRenderWindow().Render()
