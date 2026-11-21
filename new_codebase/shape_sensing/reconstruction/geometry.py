"""模块5公开显示坐标几何；复用原标架/旋转公式，不持有跨帧状态。"""
from .integration import ShapeIntegrator, ShapeIntegrationMixin


def rotation_x_matrix(rotation_deg, direct=0., rot_dir=-1.):
    """输入安装旋转/实验方向float(deg)及方向符号，输出旋转矩阵(3,3)无量纲。

    沿用原_get_rot_x公式，显示坐标旋转与参数方向选择彼此独立。
    """
    return ShapeIntegrator(rot_ang=float(rotation_deg))._get_rot_x(direct=direct, rot_dir=rot_dir)


def terminal_pose(tangent, normal, binormal, rotation, rotation_deg):
    """输入三轴(N,3)、旋转(3,3)无量纲及安装角deg，输出末端三轴(3,3)或空帧None。

    沿用原末端标架公式：先全局旋转，再绕旋转后T反向补偿安装角。
    """
    return ShapeIntegrationMixin._build_end_pose(tangent, normal, binormal, rotation, rotation_deg)
