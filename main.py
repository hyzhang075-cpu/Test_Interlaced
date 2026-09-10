import argparse
from email import parser
import time
import sys
import threading
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from optimiser import Optimiser
from visualisor import Visualisor
# from proto_demonstrator import Demonstrator 完整样机展示代码
from demonstrator import Demonstrator



def compute_error_ang(optim, vis, init):
    errs = {}
    for direct in optim.exp_directs:
        optim.direct = direct
        rs = {}
        for calib_shape in optim.calib_shapes:
            r_results_postrot, interp_dist = optim.reconstruct_shapes(calib_shape=calib_shape, direct=direct, verbose=False)
            rs[calib_shape] = r_results_postrot
        optim._compute_error(interp_dist=interp_dist, rs=rs)
        errs[direct] = optim.avg_point_err['calib'] + optim.k_end * optim.avg_end_err['calib'] + optim.k_pitch * optim.avg_pitch_err

        if init:
            print(f"---------- {direct}方向初始化完成 ----------")
            vis.init_visualise(direct=direct)
            vis.show_reconstruction(direct=direct, rs=rs, interp_dist=interp_dist, showanime=True)
            optim._print_errors()
    return errs
        
def optim_ang(rot_ang, ang_step, ang_range, optim, vis):
    # 计算初始误差并绘制初始图像
    prev_errs = compute_error_ang(optim=optim, vis=vis, init=False)

    print(f"---------- 旋转角度优化开始 ----------")
    start = time.time()
    best_rot_ang = rot_ang
    for ang in np.arange(rot_ang - ang_range, rot_ang + ang_range, ang_step):
        optim.rot_ang = ang
        ang_errs = compute_error_ang(optim=optim, vis=None, init=False)
        if sum([ang_errs[direct] for direct in ang_errs]) < sum([prev_errs[direct] for direct in prev_errs]):
            best_rot_ang = ang
            prev_errs = ang_errs
    end = time.time()
    print(f"---------- 旋转角度优化完成 ----------")
    print(f"优化耗时: {end - start}")
    print(f"最优旋转角度: {best_rot_ang}")
    print(f"最小误差: {prev_errs}")

    return prev_errs

def _str2bool(v):
    """argparse 布尔取值：显式支持 True/False，避免 type=bool 时 'False' 被当成 True 的陷阱。"""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes')


def _finite_nonnegative_float(value):
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative value")
    return number


def add_anchor_arguments(parser):
    """Register startup settings for the protocol-independent entrance anchor."""
    class _EnableFullAnchor(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, True)
            setattr(namespace, 'anchor_translation_enabled', True)
            setattr(namespace, 'anchor_rotation_enabled', True)

    parser.add_argument(
        '--anchor_translation_enabled',
        action='store_true',
        help='启用入口位置平移锚定，将当前入口点平移到首帧固定入口坐标',
    )
    parser.add_argument(
        '--anchor_rotation_enabled',
        action='store_true',
        help='启用入口姿态旋转锚定，使当前入口T/Y/Z与首帧入口标架对齐',
    )
    parser.add_argument(
        '--anchor_constraint_enabled',
        action=_EnableFullAnchor,
        nargs=0,
        help='兼容旧参数：同时启用入口平移锚定和旋转锚定',
    )
    parser.add_argument(
        '--anchor_offset_m',
        type=_finite_nonnegative_float,
        default=0.06,
        help='末端到主隆突/支气管入口的初始距离，单位为m；运行中可通过进程接口修改',
    )
    return parser


def main():
# ---------- 初始化 ----------
    parser = argparse.ArgumentParser()
    # 硬件参数
    parser.add_argument('--sensor_id', type=str, default="06B6_NeTi_v2_tube", help='传感器编号')
    parser.add_argument('--start_value', type=float, default=7.0, help='传感器第一个传感点到固定件的距离, 单位为mm')
    parser.add_argument('--end_value', type=float, default=11.0, help='传感器最后一个传感点到末端的距离, 单位为mm')
    # parser.add_argument('--R', type=float, default=0.235e-3, help='传感器半径, 单位为m') #肠镜
    parser.add_argument('--R', type=float, default=0.1e-3, help='传感器半径, 单位为m') # 支气管镜
    parser.add_argument('--ideal_len', type=float, default=0.878, help='传感器长度, 单位为m') # 支气管镜
    # parser.add_argument('--ideal_len', type=float, default=1.35, help='传感器长度, 单位为m') #肠镜
    # parser.add_argument('--samp_dist_preset', type=list, default=[510, 540, 570, 630, 710, 810, 910, 1010, 1110, 1210, 1310, 1410, 1510, 1610, 1710, 1815], help='FBG传感点位置(A组及中心), 单位为mm') #肠镜
    # parser.add_argument('--samp_dist_preset', type=list, default=[510, 520, 530, 540, 550, 580, 610, 640, 670, 720, 770, 850, 930, 1010, 1110, 1230], help='FBG传感点位置, 单位为mm')# 16点支气管镜
    parser.add_argument('--samp_dist_preset', type=list, default=[510, 521, 532, 543, 554, 588, 622, 656, 707, 758, 809, 860, 911, 962, 1013, 1064, 1115, 1166, 1217, 1268, 1319, 1370], help='FBG传感点位置, 单位为mm')# 22点支气管镜
    # parser.add_argument('--samp_dist_preset', type=list, default=[510, 521, 532, 543, 554, 579, 604, 629, 654, 696, 738, 780, 822, 864, 906, 948, 990, 1032, 1074, 1116, 1158, 1200], help='FBG传感点位置, 单位为mm')# 22点支气管镜，69cm
    parser.add_argument('--samp_dist_preset_sub', type=list, default=[120, 160, 200, 295, 445, 595, 745, 895], help='FBG传感点位置(B组), 单位为mm')
    parser.add_argument('--centre_fibre', type=bool, default=True, help='是否为1+x构型')
    parser.add_argument('--interlaced_grating', type=bool, default=False, help='栅区是否交错排列')
    parser.add_argument('--interlaced_offset', type=float, default=np.pi/3, help='交错排列时的角度偏移，单位为度')
    # parser.add_argument('--global_phases_set', type=float, default=-0.20541970186130923, help='全局相位，初始值为None') #全段相位
    parser.add_argument('--global_phases_set', type=float, default=None, help='全局相位，初始值为None')
    # 标定实验参数
    parser.add_argument('--interrogator', type=str, default='FiberTecc', choices=['FiberTecc', 'Hippo', 'LUNA'], help='解调仪器')
    parser.add_argument('--active_channels', type=list, default=[1,2,3,4], help='传感器连接解调仪的通道以及其顺序')
    parser.add_argument('--channel_change', type=bool, default=False, help='通道排布是否变化')
    parser.add_argument('--channel_layout', type=list, default=[(5, [2, 1, 4, 3, 7, 5, 6]), (7, [5, 1, 4, 3, 2, 7, 6]), (4, [7, 1, 4, 3, 2, 5, 6])], help='当通道排布出现错序，每一段的长度及其排布顺序(列表嵌套元组格式)')
    parser.add_argument('--active_angles', type=list, default=None, help='外围通道角度排布(度数)，若为None则默认均匀排布')
    parser.add_argument('--folder_path', type=str, default="./0909-06B6末端固定前端滑轨定标重测/0度", help='数据存储路径')
    # parser.add_argument('--folder_path', type=str, default="./0909-06B6末端固定前端滑轨定标重测/15cm", help='数据存储路径')
    # parser.add_argument('--folder_path', type=str, default="./0909-06B6末端固定前端滑轨定标重测/5cm", help='数据存储路径')
    parser.add_argument('--rot_dir', type=list, default={0:-1,120:-1,240:-1}, help='各个标定实验按逆(1)还是顺(-1)顺时针调整的方向')
    parser.add_argument('--exp_directs', type=list, default=[0], help='实验使用的方向')
    parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300]}, help='标定圆弧形状半径')
    # parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 150, 100, 50]}, help='标定圆弧形状半径')
    # parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 150, 100, 50, 30, 20, 15]}, help='标定圆弧形状半径')
    parser.add_argument('--end_len', type=float, default=0.05, help='end形状中进入末端大曲率弧圈的包含长度')
    # 优化使用的超参数
    parser.add_argument('--direct', type=int, default=0, help='优化曲率权重与扭转系数的标定方向')
    parser.add_argument('--freeze', type=list, default=None, help='冻结部分分区的参数，使得它们不参与优化。如果不需要冻结则调整为None。分区index从0开始。')
    parser.add_argument('--weight_bounds', type=tuple, default=(0.1, 5.0), help='曲率权重范围')
    parser.add_argument('--slope_upper_bound', type=tuple, default=50.0, help='扭转系数大小上限')
    parser.add_argument('--thresholds', type=list, default=[0.8, 1.2, 2.0, 3.0, 4.0], help='各曲率分级的阈值')
    parser.add_argument('--max_iter', type=int, default=10, help='优化的最大轮次')
    parser.add_argument('--k_point', type=float, default=1.0, help='优化目标中末端误差的权重')
    parser.add_argument('--k_end', type=float, default=0.5, help='优化目标中末端误差的权重')
    parser.add_argument('--k_pitch', type=float, default=1.0, help='优化目标中螺距误差的权重')
    parser.add_argument('--k_radius', type=float, default=1.0, help='优化目标中半径误差的权重')
    parser.add_argument('--method', type=str, default="Powell", help='优化算法, 用于输入scipy.optimize.minimimze')
    parser.add_argument('--hyper_params', type=dict, default=[20000, 1e-6, 1e-6], help='优化算法使用的超参数')
    parser.add_argument('--min_loss_diff', type=float, default=1e-4, help='收敛判定')
    parser.add_argument('--ss', type=float, default=2.6e-3, help='LUNA传感点间隔, 单位为m')
    # 动态展示参数
    parser.add_argument('--IP', type=str, default='192.168.1.75', choices=['192.168.1.75', '10.0.0.122'], help='FiberTecc使用192.168.1.75, Hippo使用10.0.0.122')
    parser.add_argument('--PORT', type=int, default=26546, choices=[26546, 4010], help='FiberTecc使用26546, Hippo使用4010')
    parser.add_argument('--maxsize', type=int, default=10, choices=[10, 2], help='FiberTecc使用10, Hippo使用2')
    parser.add_argument('--data_freq', type=int, default=100, help='解调仪频率')
    parser.add_argument('--use_stored_zero', type=_str2bool, default=True, help='demo 是否使用文件中的绝对波长零状态；False 时使用首帧归零')
    parser.add_argument('--zero_reference_path', type=str, default=str(Path(__file__).resolve().parent / 'params' / '06B6_NeTi_v2_tube_zero_reference.json'), help='统一零状态 JSON 文件路径，文件中的 SensorID、通道及栅点数必须与当前配置一致')
    parser.add_argument('--total_scale', type=float, default=1.0, help='动态显示大小')
    parser.add_argument('--imu', type=bool, default=False, help='是否启用IMU')
    parser.add_argument('--imu_com', type=str, default='COM6', help='IMU接口')
    parser.add_argument('--imu_maxlen', type=int, default=10)
    parser.add_argument('--demo_recorder', type=bool, default=True, help='是否在动态展示时记录数据')
    parser.add_argument('--unwrap_threshold', type=float, default=0.65, help='若两帧同一点角度变化小于该阈值，则统一unwrap方向')
    parser.add_argument('--align_threshold', type=float, default=0.20, help='若某帧第一点角度变化小于该阈值，则将第一点角度与上一帧对齐')
    parser.add_argument('--direction_blend_enabled', action='store_true', help='动态展示中启用0/120/240度参数半余弦软过渡；应在rot_ang确定后再打开')
    parser.add_argument('--direction_reference_direct', type=int, default=0, choices=[0, 120, 240], help='动态第一遍方向估计及关闭软过渡时使用的最高精度参数方向')
    parser.add_argument('--direction_min_curvature', type=float, default=0.5, help='动态方向估计的有效弯曲点最小曲率，低于该值时冻结上一帧方向质量')
    parser.add_argument('--direction_min_consecutive', type=int, default=3, help='动态方向估计中近端稳定弯曲区要求的最少连续点数')
    parser.add_argument('--direction_consistency_threshold', type=float, default=0.6, help='曲率加权圆周平均的最小方向一致性，低于该值时冻结上一帧方向质量')
    parser.add_argument('--direction_smooth_alpha', type=float, default=0.15, help='动态方向质量跨帧指数平滑系数，范围(0,1]')
    parser.add_argument('--demo_kalman_enabled', type=_str2bool, default=True, help='动态展示中对重构曲线启用跨帧卡尔曼滤波（True开启，开启后替代原有的curve_history滑动平均；False关闭）')
    parser.add_argument('--demo_kalman_Rk', type=float, default=1e-4, help='跨帧卡尔曼滤波的观测噪声参数，越大越平滑、响应越慢，量纲为m^2')
    parser.add_argument('--demo_kalman_Bk', type=float, default=1e-5, help='跨帧卡尔曼滤波的状态转移误差参数，越大对新观测响应越快，量纲为m^2')
    add_anchor_arguments(parser)
    # 动态样机展示相关参数
    # parser.add_argument('--body_model_path', type=str, default="./model.stl", help='人体STL模型路径')
    # parser.add_argument('--entry_length_cm', type=float, default=0.0, help='初始手动入体长度，单位cm')
    # parser.add_argument('--max_curvature', type=float, default=2.3, help='曲率入口识别阈值')
    # parser.add_argument('--back_num', type=int, default=8, help='曲率入口识别后向前回退的点数')
    # parser.add_argument('--entry_exclude', type=int, default=20, help='曲率入口识别时忽略前若干点')
    # parser.add_argument('--enable_t_align', type=bool, default=True, help='是否启用入口T向量对齐')
    # parser.add_argument('--t2_smooth_window', type=int, default=3, help='入口T向量滤波窗口大小')
    # parser.add_argument('--clip_pct', type=float, default=0.38, help='显示人体模型时的帧间位移截断比例')
    # parser.add_argument('--clip_k', type=float, default=3.0, help='显示人体模型时的帧间位移截断倍率')
    # parser.add_argument('--still_rms_th', type=float, default=0.05, help='静止冻结的RMS位移阈值')
    # parser.add_argument('--still_hold_frames', type=int, default=10, help='触发静止冻结所需连续帧数')
    # 重构使用的超参数
    parser.add_argument('--soft', type=bool, default=True, help='是否使用软化判定')
    parser.add_argument('--reverse', type=bool, default=False, help='是否反向重构')
    parser.add_argument('--discont', type=float, default=np.pi, help='np.unwrap跳变判断阈值')
    parser.add_argument('--frame_rule', type=str, default='TNN', choices=['TNB', 'TNN'], help="选择使用FS标架还是平行传递标架")
    parser.add_argument('--straight_threshold', type=float, default=2.0, help='零螺距形状中判断直线段的曲率阈值')
    parser.add_argument('--rot_threshold', type=float, default=4.0, help='零螺距形状中判断螺旋段的曲率阈值')
    parser.add_argument('--min_consecutive', type=int, default=3, help='曲率分级判定前平滑滤波窗口大小')
    parser.add_argument('--move_filter', type=int, default=3, help='动/静态重构前平滑滤波窗口大小')
    parser.add_argument('--move_filter_for_vector', type=_str2bool, default=False, help='向量曲率插值前是否启用空间方向窗口滤波；默认False，直接插值原始kx/ky',)
    parser.add_argument('--kalman_params', type=dict, default={'angle': {'Rk': 0.001, 'Bk': 0.001}, 'magnitude': {'Rk': 0.0001, 'Bk': 0.01}}, help='卡尔曼滤波参数')
    parser.add_argument('--smoothing_params', type=dict, default={'window_length': 21,'polyorder': 3}, help='LUNA数据平滑参数')
    parser.add_argument('--interp_num', type=int, default=120, help='插值目标点数')
    parser.add_argument(
        '--curvature_interp',
        type=str,
        default='vector_pchip',
        choices=['hermite', 'pchip', 'scalar_pchip', 'vector_linear', 'vector_pchip'],
        help=(
            '曲率空间插值方法：hermite为原始基线；pchip是scalar_pchip的兼容别名；'
            'vector_linear/vector_pchip分别对曲率向量分量做线性/PCHIP插值'
        ),
    )
    parser.add_argument('--nonnegative_curvature', action='store_true', help='对插值后的曲率施加非负约束')
    parser.add_argument('--sharpness', type=float, default=4.8, help='sigmoid软评分硬度')
    # parser.add_argument('--end_weights', type=bool, default=False, help='是否对末端另算权重') #肉眼看穿过标准圆圈的点数（前后段weight不一致），然后在直线段看见的点数来判断在距离多少（旁边有格尺），然后回到init图中看看离末端有几个fbg（例如9个），那就在下面'--end_start', type=int, default=-6, help='末端起点')这里改成-9
    parser.add_argument('--end_weights', type=bool, default=False, help='是否对末端另算权重') #考虑末端15cm
    parser.add_argument('--end_start', type=int, default=-7, help='末端起点')
    parser.add_argument('--TNB_threshold', type=float, default=0.0, help='TNB重构直线判定阈值')
    # 初始化时必须手动调整的超参数
    # parser.add_argument('--straight_trim', type=float, default=-0.155, help='默认不读取的部分，若只知道剩余长度，则调整为负数。')
    parser.add_argument('--straight_trim', type=float, default=0.0, help='默认不读取的部分，若只知道剩余长度，则调整为负数。')
    # parser.add_argument('--segmentation', type=list, default=[2, 3], help='对扭转角度-传感点位置曲线的点数')#末端15cm
    # parser.add_argument('--segmentation', type=list, default=[2, 2, 3], help='对扭转角度-传感点位置曲线的点数')#末端15cm
    parser.add_argument('--segmentation', type=list, default=[2, 3, 5, 2, 3, 2, 2, 3], help='对扭转角度-传感点位置曲线的点数')#优先调分段
    parser.add_argument('--connect', type=bool, default=False, help='拟合初始分段时是否令各分段首尾相连')
    parser.add_argument('--turn_threshold', type=float, default=2.0, help='判断形状弯曲方向是否改变的阈值')
    parser.add_argument('--sensing_ranges', type=dict, default={1: (583, 1048), 3: (678, 1146), 4: (749, 1215)}, help='LUNA各通道使用的传感点范围')
    # 角度优化初始值
    # parser.add_argument('--rot_ang', type=float, default=-145, help='绕x轴旋转角度, 初始为0.0')  #动态上机角度
    parser.add_argument('--rot_ang', type=float, default=0.0, help='绕x轴旋转角度, 初始为0.0')
    # # 曲率权重与扭转系数相关初始值, 曲率向量插值, 全段参数, 不带窗, 末端调weight，只有0度有效
    # parser.add_argument('--weights', default={0: np.array([1.070912693028867, 0.8789969457923975, 1.0503537574283366, 0.7834015424915021, 1.345119068369388, 0.36765192846044026]), 
    #                                                       120: np.array([0.8834070763118534, 0.85251461024247, 0.986924957649386, 0.9452724668609392, 0.9725634153903076, 0.9499982676616073, 0.2974572176118146, 2.364679449655484, 0.10000000152172318, 1.0604064407338645, 0.44551347690428705, 0.8243089591676417]), 
    #                                                       240: np.array([0.7867436848812638, 0.9969041223121436, 0.8298518879670551, 1.1115326395320613, 0.5702994386370039, 1.5798283235946788, 3.400371328236248, 0.10000062173304661, 2.231475970279864, 0.4183739322611567, 1.8996450745922804, 0.953123821823926])}, help='曲率权重')
    # parser.add_argument('--slopes', default={0: np.array([[-1.1314115331505255, -10.895705895616416, 5.184639247338778, -9.062341002198842, 3.1311331660872215, -19.312546781869692, -37.68516204685518, -2.8027593977144454], [-16.62888950024221, -10.99159700763849, 6.153222595202812, -15.064088108925855, 12.752295189039241, 3.36076952651291, 49.99996678126025, -49.99996678126025], [-3.595872620652059, -10.894656776408395, 5.118246772791866, -6.34947135356131, -1.0014221141947595, -9.918893251389335, 49.99996678126025, -49.99996678126025], [-1.199478913268285, -11.19518950375212, 3.910190737850387, -16.469119680818597, 18.373623687664008, -9.474280273685833, 49.9999997720543, -49.99996678126025], [-3.2873924933032366, -12.293823607927768, 8.217497497142528, -2.2634929588447843, -19.674373342979894, 2.402487705656266, -0.21167473184107022, 49.99996678126025], [8.04317588690287, -13.258297245401543, -2.0187497319889145, -16.149088895501464, 49.9999999551903, -49.99996678126025, 49.99996678126025, 3.3587469635050264]]), 
    #                                                     120: np.array([[-12.828408552287456, -9.660210787817029, 7.630721934225222, -8.316378424133006, 12.236550095184619, -2.11845516753808, 4.292688546499804, -49.99996678126025], [-19.017298802057717, -8.283278169308197, 6.84716400284566, -7.810189078511994, 10.897955534508188, 4.31129970066748, 18.252710691946287, -49.99996678126025], [-10.379083471278523, -10.499107406535098, 8.140460016747564, -10.060627404919888, 12.115878315219163, -0.32677593994438586, 18.381835450609323, 37.27395612992368], [-17.262023839969302, -9.271997381518112, 7.191153873790344, -8.464842710567064, 11.45512337608043, 2.6066960851053635, -5.461988110978801, -49.99996678126025], [-11.394338619612483, -10.205125424779862, 8.139901885790902, -10.248509615140962, 12.169985962386189, 1.2800595891063282, 16.32310983367443, 0.700059917491817], [-18.831696568554584, -8.874357643498374, 7.399052192216785, -8.757061443787114, 11.474403534989582, 2.7496901433051244, 10.509198644192884, -49.99996678126025]]),
    #                                                                   240: np.array([[-15.753437734255483, -7.492630557400586, 6.82487052400456, -6.8520500684282375, 8.83024792338105, 1.212951049305152, -41.57405355972927, 22.964181519257913], [-20.66799438844964, -6.776705896728184, 5.63496372120728, -7.163143253184144, 12.5093612759363, 2.1380952602362466, 65.76230411056459, 65.76229958173887], [-15.712141723954192, -7.706921023867793, 7.398354249496547, -8.36463954191213, 10.36247070157027, 1.700805506504947, -19.981830727317664, 65.76229958173887], [-15.345600369190322, -8.181994005709742, 4.255236139128114, -5.782931675570792, 10.901804326393206, 3.8229293012945336, 10.828218342063177, -65.76229958173887], [-15.577545930269098, -7.171362441835742, 14.180824405901827, -11.472368985281506, 10.711070932314414, 2.816920692241342, -4.005981333883456, -65.76229958173887], [-12.76053019437769, -9.686897178718416, -22.267853719660234, 5.292893501952441, 8.507735902044905, 5.538954429081346, 6.1033067320378915, 65.76229958173887]])}, help='扭转系数')
    # # 曲率权重与扭转系数相关初始值, 曲率向量插值, 全段参数, 不带窗, 15cm
    # parser.add_argument('--weights', default={0: np.array([1.1005804874175051, 0.10000059096651391, 0.6713507268457743, 0.8974203737024119, 0.9477927046223203, 0.9674498280281112]), 
    #                                                           120: np.array([0.8834070763118534, 0.85251461024247, 0.986924957649386, 0.9452724668609392, 0.9725634153903076, 0.9499982676616073, 0.2974572176118146, 2.364679449655484, 0.10000000152172318, 1.0604064407338645, 0.44551347690428705, 0.8243089591676417]), 
    #                                                           240: np.array([0.7867436848812638, 0.9969041223121436, 0.8298518879670551, 1.1115326395320613, 0.5702994386370039, 1.5798283235946788, 3.400371328236248, 0.10000062173304661, 2.231475970279864, 0.4183739322611567, 1.8996450745922804, 0.953123821823926])}, help='曲率权重')
    # parser.add_argument('--slopes', default={0: np.array([[-32.73088631926295, -24.12060628588178, -18.86074744179465], 
    # [19.798476975160384, 49.99996678154388, -49.99996667386808], 
    # [5.020640500670412, 49.99996678126025, -49.99996678126025], 
    # [-1.1917961165890218, 3.9029429861528415, -49.99996678126025], 
    # [-3.301274427472478, 49.99996678126025, 49.99996678126025], 
    # [-4.309474391268686, -8.67431407089824, 34.201195544309726]]), 
    #                                                         120: np.array([[-12.828408552287456, -9.660210787817029, 7.630721934225222, -8.316378424133006, 12.236550095184619, -2.11845516753808, 4.292688546499804, -49.99996678126025], [-19.017298802057717, -8.283278169308197, 6.84716400284566, -7.810189078511994, 10.897955534508188, 4.31129970066748, 18.252710691946287, -49.99996678126025], [-10.379083471278523, -10.499107406535098, 8.140460016747564, -10.060627404919888, 12.115878315219163, -0.32677593994438586, 18.381835450609323, 37.27395612992368], [-17.262023839969302, -9.271997381518112, 7.191153873790344, -8.464842710567064, 11.45512337608043, 2.6066960851053635, -5.461988110978801, -49.99996678126025], [-11.394338619612483, -10.205125424779862, 8.139901885790902, -10.248509615140962, 12.169985962386189, 1.2800595891063282, 16.32310983367443, 0.700059917491817], [-18.831696568554584, -8.874357643498374, 7.399052192216785, -8.757061443787114, 11.474403534989582, 2.7496901433051244, 10.509198644192884, -49.99996678126025]]),
    #                                                                       240: np.array([[-15.753437734255483, -7.492630557400586, 6.82487052400456, -6.8520500684282375, 8.83024792338105, 1.212951049305152, -41.57405355972927, 22.964181519257913], [-20.66799438844964, -6.776705896728184, 5.63496372120728, -7.163143253184144, 12.5093612759363, 2.1380952602362466, 65.76230411056459, 65.76229958173887], [-15.712141723954192, -7.706921023867793, 7.398354249496547, -8.36463954191213, 10.36247070157027, 1.700805506504947, -19.981830727317664, 65.76229958173887], [-15.345600369190322, -8.181994005709742, 4.255236139128114, -5.782931675570792, 10.901804326393206, 3.8229293012945336, 10.828218342063177, -65.76229958173887], [-15.577545930269098, -7.171362441835742, 14.180824405901827, -11.472368985281506, 10.711070932314414, 2.816920692241342, -4.005981333883456, -65.76229958173887], [-12.76053019437769, -9.686897178718416, -22.267853719660234, 5.292893501952441, 8.507735902044905, 5.538954429081346, 6.1033067320378915, 65.76229958173887]])}, help='扭转系数')
    # 曲率权重与扭转系数相关初始值, 曲率向量插值, 全段参数, 不带窗, combined
    parser.add_argument('--weights', default={0: np.array([1.070912693028867, 0.8789969457923975, 1.0503537574283366, 0.7834015424915021, 1.345119068369388, 0.36765192846044026, 1.1005804874175051, 0.10000059096651391, 0.6713507268457743, 0.8974203737024119, 0.9477927046223203, 0.9674498280281112]), 
                                                              120: np.array([0.8834070763118534, 0.85251461024247, 0.986924957649386, 0.9452724668609392, 0.9725634153903076, 0.9499982676616073, 0.2974572176118146, 2.364679449655484, 0.10000000152172318, 1.0604064407338645, 0.44551347690428705, 0.8243089591676417]), 
                                                              240: np.array([0.7867436848812638, 0.9969041223121436, 0.8298518879670551, 1.1115326395320613, 0.5702994386370039, 1.5798283235946788, 3.400371328236248, 0.10000062173304661, 2.231475970279864, 0.4183739322611567, 1.8996450745922804, 0.953123821823926])}, help='曲率权重')
    parser.add_argument('--slopes', default={0: np.array([[-1.1314115331505255, -10.895705895616416, 5.184639247338778, -9.062341002198842, 3.1311331660872215, -19.312546781869692, -24.12060628588178, -18.86074744179465], 
                                                          [-16.62888950024221, -10.99159700763849, 6.153222595202812, -15.064088108925855, 12.752295189039241, 3.36076952651291,  49.99996678154388, -49.99996667386808], 
                                                          [-3.595872620652059, -10.894656776408395, 5.118246772791866, -6.34947135356131, -1.0014221141947595, -9.918893251389335, 49.99996678126025, -49.99996678126025], 
                                                          [-1.199478913268285, -11.19518950375212, 3.910190737850387, -16.469119680818597, 18.373623687664008, -9.474280273685833, 3.9029429861528415, -49.99996678126025], 
                                                          [-3.2873924933032366, -12.293823607927768, 8.217497497142528, -2.2634929588447843, -19.674373342979894, 2.402487705656266, 49.99996678126025, 49.99996678126025], 
                                                          [8.04317588690287, -13.258297245401543, -2.0187497319889145, -16.149088895501464, 49.9999999551903, -49.99996678126025, -8.67431407089824, 34.201195544309726]]), 
                                                            120: np.array([[-12.828408552287456, -9.660210787817029, 7.630721934225222, -8.316378424133006, 12.236550095184619, -2.11845516753808, 4.292688546499804, -49.99996678126025], [-19.017298802057717, -8.283278169308197, 6.84716400284566, -7.810189078511994, 10.897955534508188, 4.31129970066748, 18.252710691946287, -49.99996678126025], [-10.379083471278523, -10.499107406535098, 8.140460016747564, -10.060627404919888, 12.115878315219163, -0.32677593994438586, 18.381835450609323, 37.27395612992368], [-17.262023839969302, -9.271997381518112, 7.191153873790344, -8.464842710567064, 11.45512337608043, 2.6066960851053635, -5.461988110978801, -49.99996678126025], [-11.394338619612483, -10.205125424779862, 8.139901885790902, -10.248509615140962, 12.169985962386189, 1.2800595891063282, 16.32310983367443, 0.700059917491817], [-18.831696568554584, -8.874357643498374, 7.399052192216785, -8.757061443787114, 11.474403534989582, 2.7496901433051244, 10.509198644192884, -49.99996678126025]]),
                                                                          240: np.array([[-15.753437734255483, -7.492630557400586, 6.82487052400456, -6.8520500684282375, 8.83024792338105, 1.212951049305152, -41.57405355972927, 22.964181519257913], [-20.66799438844964, -6.776705896728184, 5.63496372120728, -7.163143253184144, 12.5093612759363, 2.1380952602362466, 65.76230411056459, 65.76229958173887], [-15.712141723954192, -7.706921023867793, 7.398354249496547, -8.36463954191213, 10.36247070157027, 1.700805506504947, -19.981830727317664, 65.76229958173887], [-15.345600369190322, -8.181994005709742, 4.255236139128114, -5.782931675570792, 10.901804326393206, 3.8229293012945336, 10.828218342063177, -65.76229958173887], [-15.577545930269098, -7.171362441835742, 14.180824405901827, -11.472368985281506, 10.711070932314414, 2.816920692241342, -4.005981333883456, -65.76229958173887], [-12.76053019437769, -9.686897178718416, -22.267853719660234, 5.292893501952441, 8.507735902044905, 5.538954429081346, 6.1033067320378915, 65.76229958173887]])}, help='扭转系数')
    # 不传入Optimiser的参数
    parser.add_argument('--task', type=str, required=True, choices=['init', 'ang', 'optim', 'recon', 'demo'], help='选择运行目的为初始化、旋转角度优化、曲率权重与扭转系数优化、重构还是动态展示')
    parser.add_argument('--ang_step', type=float, default=0.1, help='旋转角度搜索步长, 初始为0.1')
    parser.add_argument('--ang_range', type=float, default=1.0, help='旋转角度搜索距离, 初始为180.0')
    args = parser.parse_args()

    # 存储零状态只作用于动态展示；其他任务保持原有静态数据流程。
    if args.task != 'demo':
        args.use_stored_zero = False

    # 避免Terminal指令读取错误
    if args.weights == "None":
        args.weights = None
    if args.slopes == "None":
        args.slopes = None
    
    # 只传给 Demonstrator 的参数，不传给 Optimiser
    # demo_param_keys = ['total_scale', 'imu', 'imu_com', 'imu_maxlen', 'demo_recorder', 'unwrap_threshold', 'align_threshold',
    #                    'body_model_path', 'entry_length_cm', 'max_curvature', 'back_num', 'entry_exclude', 'enable_t_align', 't2_smooth_window', 'clip_pct', 'clip_k', 'still_rms_th', 'still_hold_frames'] # 用于完整样机展示
    demo_param_keys = [
        'total_scale', 'imu', 'imu_com', 'imu_maxlen', 'demo_recorder', 'unwrap_threshold', 'align_threshold',
        'direction_blend_enabled', 'direction_reference_direct', 'direction_min_curvature', 'direction_min_consecutive',
        'direction_consistency_threshold', 'direction_smooth_alpha',
        'demo_kalman_enabled', 'demo_kalman_Rk', 'demo_kalman_Bk',
        'anchor_constraint_enabled', 'anchor_translation_enabled',
        'anchor_rotation_enabled', 'anchor_offset_m',
    ]

    optim_exclude_keys = ['task', 'ang_step', 'ang_range'] + demo_param_keys
    params = {k: getattr(args, k) for k in vars(args) if k not in optim_exclude_keys}

    demo_params = {k: getattr(args, k) for k in demo_param_keys}

    # 初始化优化器与可视化工具
    optim = Optimiser(**params)
    print(f"curvature interpolation mode: {optim.curvature_interp}")


    if args.task == 'demo':
        app = QApplication(sys.argv)
        demo = Demonstrator(recon=optim, **demo_params)
        demo.show()
        
        if demo.imu: 
            IMU_thread = threading.Thread(target=demo.receive_IMU)
            IMU_thread.daemon = True
            IMU_thread.start()
            print("---------- 开始接收IMU数据 ----------")       
        else:
            print("---------- 未启用IMU ----------")

        if optim.interrogator == 'FiberTecc':
            data_receiver_thread = threading.Thread(target=optim.receive_data_FiberTecc, daemon=True)
        elif optim.interrogator == 'Hippo':
            data_receiver_thread = threading.Thread(target=optim.receive_data_Hippo, daemon=True)
        data_receiver_thread.start()
        sys.exit(app.exec_())

    else:
        vis = Visualisor(recon=optim)
        if args.task == 'init':
            compute_error_ang(optim=optim, vis=vis, init=True)
        elif args.task == 'ang':
            optim_ang(rot_ang=args.rot_ang, ang_step=args.ang_step, ang_range=args.ang_range, optim=optim, vis=vis)
        elif args.task in ['optim']:
            optim.optim_coeff(verbose=True)            
        elif args.task == 'recon':
            rs = {}
            for calib_shape in optim.calib_shapes:
                r_results_postrot, interp_dist = optim.reconstruct_shapes(calib_shape=calib_shape, direct=args.direct, verbose=False)
                rs[calib_shape] = r_results_postrot
            optim._compute_error(interp_dist=interp_dist, rs=rs)
            vis.show_reconstruction(direct=args.direct, rs=rs, interp_dist=interp_dist, showanime=True)
            optim._print_errors()
    

if __name__ == '__main__':
    main()
