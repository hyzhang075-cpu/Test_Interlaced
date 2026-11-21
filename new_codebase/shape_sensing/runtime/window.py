"""Qt/VTK薄窗口：定时器只消费RuntimeService最终帧。"""
import numpy as np
import vtk
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from ..visualization.vtk_view import VTKViewMixin

class RuntimeWindow(VTKViewMixin, QMainWindow):
    def __init__(self, runtime, sampling_distance_m, *, data_frequency=100, total_scale=1., title='Viewer'):
        """输入RuntimeService、真实采样距离(P,)m与显示配置，输出显式创建的窗口。"""
        super().__init__()
        self.runtime = runtime
        self.sampling_distance_m = np.array(sampling_distance_m, copy=True)
        self.total_scale = total_scale
        self.entrance_anchor_lock = runtime.lock
        self.entrance_anchor_constraint = runtime.anchor
        self.firstFocal = None
        self.firstPosition = None
        self.setWindowTitle(title)
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
        self.add_anchor_marker()

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
        self.timer.start(int(1000 / data_frequency))


    def update_vtk(self):
        """输入无，输出None；仅渲染模块8最终同一帧，不预处理/拟合/滤波/积分。"""
        frame = self.runtime.poll()
        if frame is not None:
            self._update_frame(frame.shape.position_m, frame.end_pose)

    def closeEvent(self, event):
        """输入Qt关闭事件，输出None；停定时器并统一释放模块8拥有的资源。"""
        self.timer.stop()
        self.runtime.close()
        super().closeEvent(event)

    def load_body_model(self, model_path):
        """输入STL路径str，输出vtkActor；仅展示明确文件，不参与数值计算。"""
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(model_path))
        reader.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(.25)
        self.renderer.AddActor(actor)
        self.body_model_actor = actor
        return actor
