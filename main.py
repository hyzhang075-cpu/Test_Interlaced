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

def main():
# ---------- 初始化 ----------
    parser = argparse.ArgumentParser()
    # 硬件参数
    parser.add_argument('--sensor_id', type=str, default="06B6_NeTi", help='传感器编号')
    parser.add_argument('--start_value', type=float, default=10.0, help='传感器第一个传感点到固定件的距离, 单位为mm')
    parser.add_argument('--end_value', type=float, default=5.0, help='传感器最后一个传感点到末端的距离, 单位为mm')
    # parser.add_argument('--R', type=float, default=0.235e-3, help='传感器半径, 单位为m') #肠镜
    parser.add_argument('--R', type=float, default=0.1e-3, help='传感器半径, 单位为m') # 支气管镜
    parser.add_argument('--ideal_len', type=float, default=0.875, help='传感器长度, 单位为m') # 支气管镜
    # parser.add_argument('--ideal_len', type=float, default=1.35, help='传感器长度, 单位为m') #肠镜
    # parser.add_argument('--samp_dist_preset', type=list, default=[510, 540, 570, 630, 710, 810, 910, 1010, 1110, 1210, 1310, 1410, 1510, 1610, 1710, 1815], help='FBG传感点位置(A组及中心), 单位为mm') #肠镜
    # parser.add_argument('--samp_dist_preset', type=list, default=[510, 520, 530, 540, 550, 580, 610, 640, 670, 720, 770, 850, 930, 1010, 1110, 1230], help='FBG传感点位置, 单位为mm')# 16点支气管镜
    parser.add_argument('--samp_dist_preset', type=list, default=[510, 521, 532, 543, 554, 588, 622, 656, 707, 758, 809, 860, 911, 962, 1013, 1064, 1115, 1166, 1217, 1268, 1319, 1370], help='FBG传感点位置, 单位为mm')# 22点支气管镜
    # parser.add_argument('--samp_dist_preset', type=list, default=[510, 521, 532, 543, 554, 579, 604, 629, 654, 696, 738, 780, 822, 864, 906, 948, 990, 1032, 1074, 1116, 1158, 1200], help='FBG传感点位置, 单位为mm')# 22点支气管镜，69cm
    parser.add_argument('--samp_dist_preset_sub', type=list, default=[120, 160, 200, 295, 445, 595, 745, 895], help='FBG传感点位置(B组), 单位为mm')
    parser.add_argument('--centre_fibre', type=bool, default=True, help='是否为1+x构型')
    parser.add_argument('--interlaced_grating', type=bool, default=False, help='栅区是否交错排列')
    parser.add_argument('--interlaced_offset', type=float, default=np.pi/3, help='交错排列时的角度偏移，单位为度')
    parser.add_argument('--global_phases_set', type=float, default=None, help='全局相位，初始值为None')
    # 标定实验参数
    parser.add_argument('--interrogator', type=str, default='FiberTecc', choices=['FiberTecc', 'Hippo', 'LUNA'], help='解调仪器')
    parser.add_argument('--active_channels', type=list, default=[1,2,3,4], help='传感器连接解调仪的通道以及其顺序')
    parser.add_argument('--channel_change', type=bool, default=False, help='通道排布是否变化')
    parser.add_argument('--channel_layout', type=list, default=[(5, [2, 1, 4, 3, 7, 5, 6]), (7, [5, 1, 4, 3, 2, 7, 6]), (4, [7, 1, 4, 3, 2, 5, 6])], help='当通道排布出现错序，每一段的长度及其排布顺序(列表嵌套元组格式)')
    parser.add_argument('--active_angles', type=list, default=None, help='外围通道角度排布(度数)，若为None则默认均匀排布')
    parser.add_argument('--folder_path', type=str, default="./0730-06B6-点胶固定镍钛管+末端测试/全段测试", help='数据存储路径')
    parser.add_argument('--rot_dir', type=list, default={0:-1,120:-1,240:-1}, help='各个标定实验按逆(1)还是顺(-1)顺时针调整的方向')
    parser.add_argument('--exp_directs', type=list, default=[0, 120, 240], help='实验使用的方向')
    parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200]}, help='标定圆弧形状半径')
    # parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200, 100, 75, 50, 30]}, help='标定圆弧形状半径')
    # parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200, 100, 75, 50, 30, 20, 15]}, help='标定圆弧形状半径')
    parser.add_argument('--end_len', type=float, default=0.05, help='end形状中进入末端大曲率弧圈的包含长度')
    # 优化使用的超参数
    parser.add_argument('--direct', type=int, default=0, help='优化曲率权重与扭转系数的标定方向')
    parser.add_argument('--freeze', type=list, default=None, help='冻结部分分区的参数，使得它们不参与优化。如果不需要冻结则调整为None。分区index从0开始。')
    parser.add_argument('--weight_bounds', type=tuple, default=(0.1, 5.0), help='曲率权重范围')
    parser.add_argument('--slope_upper_bound', type=tuple, default=30.0, help='扭转系数大小上限')
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
    parser.add_argument('--straight_trim', type=float, default=0.0, help='默认不读取的部分，若只知道剩余长度，则调整为负数。')
    parser.add_argument('--discont', type=float, default=np.pi, help='np.unwrap跳变判断阈值')
    parser.add_argument('--frame_rule', type=str, default='TNN', choices=['TNB', 'TNN'], help="选择使用FS标架还是平行传递标架")
    parser.add_argument('--straight_threshold', type=float, default=2.0, help='零螺距形状中判断直线段的曲率阈值')
    parser.add_argument('--rot_threshold', type=float, default=4.0, help='零螺距形状中判断螺旋段的曲率阈值')
    parser.add_argument('--min_consecutive', type=int, default=3, help='曲率分级判定前平滑滤波窗口大小')
    parser.add_argument('--move_filter', type=int, default=3, help='动/静态重构前平滑滤波窗口大小')
    parser.add_argument('--kalman_params', type=dict, default={'angle': {'Rk': 0.001, 'Bk': 0.001}, 'magnitude': {'Rk': 0.0001, 'Bk': 0.01}}, help='卡尔曼滤波参数')
    parser.add_argument('--smoothing_params', type=dict, default={'window_length': 21,'polyorder': 3}, help='LUNA数据平滑参数')
    parser.add_argument('--interp_num', type=int, default=60, help='插值目标点数')
    parser.add_argument('--sharpness', type=float, default=4.8, help='sigmoid软评分硬度')
    # parser.add_argument('--end_weights', type=bool, default=False, help='是否对末端另算权重') #肉眼看穿过标准圆圈的点数（前后段weight不一致），然后在直线段看见的点数来判断在距离多少（旁边有格尺），然后回到init图中看看离末端有几个fbg（例如9个），那就在下面'--end_start', type=int, default=-6, help='末端起点')这里改成-9
    parser.add_argument('--end_weights', type=bool, default=False, help='是否对末端另算权重') #考虑末端15cm
    parser.add_argument('--end_start', type=int, default=-7, help='末端起点')
    parser.add_argument('--TNB_threshold', type=float, default=0.0, help='TNB重构直线判定阈值')
    # 初始化时必须手动调整的超参数
    # parser.add_argument('--segmentation', type=list, default=[2, 6], help='对扭转角度-传感点位置曲线的点数')#优先调分段
    parser.add_argument('--segmentation', type=list, default=[2, 3, 5, 2, 3, 7], help='对扭转角度-传感点位置曲线的点数')#优先调分段
    # parser.add_argument('--segmentation', type=list, default=[2, 3, 5, 2, 3, 2, 5], help='对扭转角度-传感点位置曲线的点数')#优先调分段
    parser.add_argument('--connect', type=bool, default=False, help='拟合初始分段时是否令各分段首尾相连')
    parser.add_argument('--turn_threshold', type=float, default=2.0, help='判断形状弯曲方向是否改变的阈值')
    parser.add_argument('--sensing_ranges', type=dict, default={1: (583, 1048), 3: (678, 1146), 4: (749, 1215)}, help='LUNA各通道使用的传感点范围')
    # 角度优化初始值
    parser.add_argument('--rot_ang', type=float, default=-60, help='绕x轴旋转角度, 初始为0.0')
    # 曲率权重与扭转系数相关初始值, 原始全段参数
    parser.add_argument('--weights', default={0: np.array([0.8731661733675625, 0.9115250480738559, 0.8821677801033272, 0.8907000715922333, 0.8855793956382506, 0.9042403166550511]), 
                                              120: np.array([0.9286483420227251, 0.9115785905665769, 0.9303130902491028, 0.9043369969923305, 0.8968953709433376, 0.8876631304186952]), 
                                              240: np.array([0.8647348985128309, 0.9186126647466022, 0.8970296434749025, 0.8972916729732984, 0.8956171969426803, 0.8976175854902237])}, help='曲率权重')
    parser.add_argument('--slopes', default={0: np.array([[-17.212745717272483, -10.609666705616046, 7.218852454562629, 2.7736646625811145, 0.6571864173273458, 9.853050586331596], [-29.999995718376887, 0.5187629643523597, 0.07629418407301028, 1.3798365771069705, 11.957140606747098, 6.034514589040578], [-15.942056297810076, -6.384274074351701, 6.052266177718926, -1.7911717625800896, 7.815994449577535, 0.47100534563037705], [-8.739523410257213, -10.461876592659129, 4.925706934907043, 1.3109567372721473, 7.06970189281955, 6.767169600078585], [-23.895921114244985, -2.7868732691846057, 3.9815939467837635, -5.092460167456333, 12.09586213502729, 0.9355258163602358], [-6.235379249579586, -4.595660064389552, 4.3892727563839316, -4.991442527149374, 11.451338238538337, 3.4674959950991555]]), 
                                            120: np.array([[-24.193761509079174, -4.533170587196295, 5.509710978710244, -5.662815639338861, 11.63706382990372, -8.370256263782231], [-21.042751749633517, -6.160708891783905, 2.962769677670296, -3.6985061971334967, 15.409956371130168, -0.3848159303916009], [-29.56915622181359, -4.4687200395981534, 5.087900589732432, -1.6642222864147032, 6.054626779689361, 6.019354758383769], [-29.279771592172725, -5.336937079526172, 4.119955741073221, -2.4364597200784077, 11.469893725966783, -1.2399179004848229], [-29.77218705786623, -6.670806972093258, 5.827306611756221, -2.610270723134177, 8.303236948621148, 5.444213319766701], [-29.999980128703026, -9.9244469758181, 7.168486137575096, -4.715363282433343, 10.149873831157992, 1.8775097368415656]]),
                                                          240: np.array([[-12.239950409879247, -12.312554097580607, 8.809943653659316, -8.33466797131219, 8.795235928007518, 0.4867765337482919], [-13.066663705955818, -11.006250581162991, 8.760008793021798, -15.730809557875263, 16.307068220293893, 8.627063755550935], [-5.969843981933276, -13.573067245251725, 7.720617423261948, -3.871512958913836, 4.217754907489323, 3.812432445879975], [-17.94229822611044, -8.298031562966381, 3.069179382475759, 3.5547075546795863, 7.383649140175335, 6.535023683007421], [-24.329616678709097, -0.8224950414052646, 3.46917575447114, -1.6880772343299522, 9.887736240604855, 3.924146896153953], [-18.08133303653446, -5.167039514612586, 4.297369029050207, -1.6422440126782303, 9.633848922528667, 4.71457459404181]])}, help='扭转系数')
    # 曲率权重与扭转系数相关初始值, 15cm参数
    # parser.add_argument('--weights', default={0: np.array([0.8731661733675625, 0.9115250480738559, 0.8821677801033272, 0.8907000715922333, 0.8855793956382506, 0.9042403166550511, 1.1007341264087735, 1.08570359326273, 0.9501242182176861, 0.9668183757316279, 1.0085346431630267, 0.8940028756477347]), 
    #                                               120: np.array([0.9286483420227251, 0.9115785905665769, 0.9303130902491028, 0.9043369969923305, 0.8968953709433376, 0.8876631304186952, 1.0191220431025068, 0.8702328343376972, 0.9710811880480233, 0.9008418105472392, 0.924400661967903, 0.8731638055768233]), 
    #                                               240: np.array([0.8647348985128309, 0.9186126647466022, 0.8970296434749025, 0.8972916729732984, 0.8956171969426803, 0.8976175854902237, 0.9617572458606467, 0.8538175464567751, 0.891919352750199, 0.8763396991959259, 0.9848820118222292, 0.9484031601409824])}, help='曲率权重')
    # parser.add_argument('--slopes', default={0: np.array([
    #     [-17.212745717272483, -10.609666705616046, 7.218852454562629, 2.7736646625811145, 0.6571864173273458, 27.099344199717663, -21.36424903188906], 
    #     [-29.999995718376887, 0.5187629643523597, 0.07629418407301028, 1.3798365771069705, 11.957140606747098, 17.49773011366227, -20.348867847876242], 
    #     [-15.942056297810076, -6.384274074351701, 6.052266177718926, -1.7911717625800896, 7.815994449577535, 13.98269367548682, -3.810355914172986], 
    #     [-8.739523410257213, -10.461876592659129, 4.925706934907043, 1.3109567372721473, 7.06970189281955, 16.174626145114416, -10.655354588370425], 
    #     [-23.895921114244985, -2.7868732691846057, 3.9815939467837635, -5.092460167456333, 12.09586213502729, -6.093565385807544, 18.198135885335304], 
    #     [-6.235379249579586, -4.595660064389552, 4.3892727563839316, -4.991442527149374, 11.451338238538337, 5.585372155747211, 2.0591240441270067]]), 
    #                                             120: np.array([
    #                                                 [-24.193761509079174, -4.533170587196295, 5.509710978710244, -5.662815639338861, 11.63706382990372, -14.453584533083578, 7.9183743396895885], 
    #                                                 [-21.042751749633517, -6.160708891783905, 2.962769677670296, -3.6985061971334967, 15.409956371130168, -16.702274309235918, 29.999980068756148], 
    #                                                 [-29.56915622181359, -4.4687200395981534, 5.087900589732432, -1.6642222864147032, 6.054626779689361, -4.741440432741673, -0.312188685550775], 
    #                                                 [-29.279771592172725, -5.336937079526172, 4.119955741073221, -2.4364597200784077, 11.469893725966783, -6.351438320108748, 14.450634363518503], 
    #                                                 [-29.77218705786623, -6.670806972093258, 5.827306611756221, -2.610270723134177, 8.303236948621148, 9.69305549119484, 4.04021207372789], 
    #                                                 [-29.999980128703026, -9.9244469758181, 7.168486137575096, -4.715363282433343, 10.149873831157992, -16.31382852875096, 7.648778072085138]]),
    #                                                           240: np.array([
    #                                                               [-12.239950409879247, -12.312554097580607, 8.809943653659316, -8.33466797131219, 8.795235928007518, 29.99998006801156, -16.24112176141038], 
    #                                                               [-13.066663705955818, -11.006250581162991, 8.760008793021798, -15.730809557875263, 16.307068220293893, 29.999980067863394, 4.882501267303498], 
    #                                                               [-5.969843981933276, -13.573067245251725, 7.720617423261948, -3.871512958913836, 4.217754907489323, 29.999980144265653, -26.106784435176113], 
    #                                                               [-17.94229822611044, -8.298031562966381, 3.069179382475759, 3.5547075546795863, 7.383649140175335, 29.999980068598752, -18.661818941363446], 
    #                                                               [-24.329616678709097, -0.8224950414052646, 3.46917575447114, -1.6880772343299522, 9.887736240604855, 29.99998003704963, -21.62043169856916], 
    #                                                               [-18.08133303653446, -5.167039514612586, 4.297369029050207, -1.6422440126782303, 9.633848922528667, 29.999980037214446, -1.595734614765548]])}, help='扭转系数')
    # 不传入Optimiser的参数
    parser.add_argument('--task', type=str, required=True, choices=['init', 'ang', 'optim', 'recon', 'demo'], help='选择运行目的为初始化、旋转角度优化、曲率权重与扭转系数优化、重构还是动态展示')
    parser.add_argument('--ang_step', type=float, default=0.1, help='旋转角度搜索步长, 初始为0.1')
    parser.add_argument('--ang_range', type=float, default=1.0, help='旋转角度搜索距离, 初始为180.0')
    args = parser.parse_args()

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
    ]

    optim_exclude_keys = ['task', 'ang_step', 'ang_range'] + demo_param_keys
    params = {k: getattr(args, k) for k in vars(args) if k not in optim_exclude_keys}

    demo_params = {k: getattr(args, k) for k in demo_param_keys}

    # 初始化优化器与可视化工具
    optim = Optimiser(**params)


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
