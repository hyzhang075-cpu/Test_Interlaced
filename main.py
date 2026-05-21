import argparse
import time
import sys
import threading
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from optimiser import Optimiser
from visualisor import Visualisor
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

def main():
    # ---------- 初始化 ----------
    parser = argparse.ArgumentParser()
    # 硬件参数
    parser.add_argument('--sensor_id', type=str, default="04C4", help='传感器编号')
    parser.add_argument('--start_value', type=float, default=93, help='传感器第一个传感点到固定件的距离, 单位为mm')
    parser.add_argument('--R', type=float, default=0.076e-3, help='传感器半径, 单位为m')
    parser.add_argument('--ideal_len', type=float, default=0.9, help='传感器长度, 单位为m')
    parser.add_argument('--samp_dist_preset', type=list, default=[100, 140, 180, 220, 370, 520, 670, 820], help='FBG传感点位置(A组及中心), 单位为mm')
    parser.add_argument('--samp_dist_preset_sub', type=list, default=[120, 160, 200, 295, 445, 595, 745, 895], help='FBG传感点位置(B组), 单位为mm')
    parser.add_argument('--centre_fibre', type=bool, default=True, help='是否为1+x构型')
    parser.add_argument('--interlaced_grating', type=bool, default=True, help='栅区是否交错排列')
    parser.add_argument('--interlaced_offset', type=float, default=np.pi/3, help='交错排列时的角度偏移，单位为度')
    parser.add_argument('--global_phases_set', type=float, default=2.171893469999961, help='全局相位，初始值为None')
    # 标定实验参数
    parser.add_argument('--interrogator', type=str, default='Hippo', choices=['FiberTecc', 'Hippo', 'LUNA'], help='解调仪器')
    parser.add_argument('--active_channels', type=list, default=[1, 2, 3, 4, 5, 6, 7], help='传感器连接解调仪的通道以及其顺序')
    parser.add_argument('--channel_change', type=bool, default=False, help='通道排布是否变化')
    parser.add_argument('--channel_layout', type=list, default=[(6, [2, 1, 4, 3, 7, 5, 6]), (6, [5, 1, 4, 3, 2, 7, 6]), (4, [7, 1, 4, 3, 2, 5, 6])], help='当通道排布出现错序，每一段的长度及其排布顺序(列表嵌套元组格式)')
    parser.add_argument('--active_angles', type=list, default=None, help='外围通道角度排布(度数)，若为None则默认均匀排布')
    parser.add_argument('--folder_path', type=str, default=str((Path(__file__).resolve()).parents[2] / "形状感知公共文件/支气管镜/传感器测试数据/0512-04C4-峰澜(1+6)-FBG"), help='数据存储路径')
    parser.add_argument('--rot_dir', type=list, default=[-1], help='各个标定实验按逆(1)还是顺(-1)时针调整的方向')
    parser.add_argument('--exp_directs', type=list, default=[0, 120, 240], help='实验使用的方向')
    parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300]}, help='标定圆弧形状半径')
    parser.add_argument('--end_len', type=float, default=0.08, help='end形状中进入末端大曲率弧圈的包含长度')
    # 优化使用的超参数
    parser.add_argument('--direct', type=int, default=0, help='优化曲率权重与扭转系数的标定方向')
    parser.add_argument('--freeze', type=list, default=None, help='冻结部分分区的参数，使得它们不参与优化。如果不需要冻结则调整为None')
    parser.add_argument('--weight_bounds', type=tuple, default=(0.0, 2.0), help='曲率权重范围')
    parser.add_argument('--slope_upper_bound', type=tuple, default=20.0, help='扭转系数大小上限')
    parser.add_argument('--thresholds', type=list, default=[0.8, 1.2, 2.0, 3.0, 4.0], help='各曲率分级的阈值')
    # parser.add_argument('--thresholds', type=list, default=[1.2, 2.0, 4.5], help='各曲率分级的阈值')
    parser.add_argument('--max_iter', type=int, default=5, help='优化的最大轮次')
    parser.add_argument('--k_point', type=float, default=1.0, help='优化目标中末端误差的权重')
    parser.add_argument('--k_end', type=float, default=0.3, help='优化目标中末端误差的权重')
    parser.add_argument('--k_pitch', type=float, default=1.0, help='优化目标中螺距误差的权重')
    parser.add_argument('--k_radius', type=float, default=1.0, help='优化目标中半径误差的权重')
    parser.add_argument('--method', type=str, default="Powell", help='优化算法, 用于输入scipy.optimize.minimimze')
    parser.add_argument('--hyper_params', type=dict, default=[10000, 1e-4, 1e-4], help='优化算法使用的超参数')
    parser.add_argument('--min_loss_diff', type=float, default=1e-4, help='收敛判定')
    parser.add_argument('--ss', type=float, default=2.6e-3, help='LUNA传感点间隔, 单位为m')
    # 动态展示参数
    parser.add_argument('--IP', type=str, default='10.0.0.122', choices=['192.168.1.75', '10.0.0.122'], help='FiberTecc使用192.168.1.75, Hippo使用10.0.0.122')
    parser.add_argument('--PORT', type=int, default=4010, choices=[26546, 4010], help='FiberTecc使用26546, Hippo使用4010')
    parser.add_argument('--maxsize', type=int, default=2, choices=[10, 2], help='FiberTecc使用10, Hippo使用2')
    parser.add_argument('--data_freq', type=int, default=100, help='解调仪频率')
    parser.add_argument('--total_scale', type=float, default=1.0, help='动态显示大小')
    parser.add_argument('--imu', type=bool, default=False, help='是否启用IMU')
    parser.add_argument('--imu_com', type=str, default='COM6', help='IMU接口')
    parser.add_argument('--imu_maxlen', type=int, default=10)
    parser.add_argument('--demo_recorder', type=bool, default=False, help='是否在动态展示时记录数据')
    parser.add_argument('--unwrap_threshold', type=float, default=0.65, help='若两帧同一点角度变化小于该阈值，则统一unwrap方向')
    parser.add_argument('--align_threshold', type=float, default=0.20, help='若某帧第一点角度变化小于该阈值，则将第一点角度与上一帧对齐')
    # 重构使用的超参数
    parser.add_argument('--soft', type=bool, default=False, help='是否使用软化判定')
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
    parser.add_argument('--end_weights', type=bool, default=False, help='是否对末端另算权重')
    parser.add_argument('--end_start', type=int, default=-7, help='末端起点')
    parser.add_argument('--TNB_threshold', type=float, default=0.0, help='TNB重构直线判定阈值')
    # 初始化时必须手动调整的超参数
    parser.add_argument('--segmentation', type=list, default=[2, 3, 6, 5], help='对扭转角度-传感点位置曲线的点数')
    parser.add_argument('--connect', type=bool, default=True, help='拟合初始分段时是否令各分段首尾相连')
    parser.add_argument('--turn_threshold', type=float, default=2.0, help='判断形状弯曲方向是否改变的阈值')
    parser.add_argument('--sensing_ranges', type=dict, default={1: (583, 1048), 3: (678, 1146), 4: (749, 1215)}, help='LUNA各通道使用的传感点范围')
    # 角度优化初始值
    parser.add_argument('--rot_ang', type=float, default=0.0, help='绕x轴旋转角度, 初始为0.0')
    # 曲率权重与扭转系数相关初始值
    parser.add_argument('--weights', default={0: np.array([1.9451975511432067, 1.5675431895656657, 1.5544730205643247, 2.81809147463091]),
                                            120: np.array([2.4418347408360432, 1.7568396404504978, 1.577966295128975, 1.1956906956321312]),
                                            240: np.array([1.5421765879893197, 2.000500848840206, 1.7604901282284617, 2.999996502794499])}, help='应变权重')
    parser.add_argument('--slopes', default={0: np.array([[-10.624202770906944, -12.559725635848974, -25.10550827675271, -25.10550827675271], [3.8695322222926145, -17.448721267338467, -12.502916437157307, -25.10550827675271], [-11.648771833436427, -15.956576669971179, -24.52286630454854, -25.10550827675271], [-25.10550827675271, -25.10550827675271, -25.10550827675271, -25.10550827675271]]),
                                            120: np.array([[5.661037751186545, -11.742677520277228, 4.8725380525877, -3.2873066766532117], [22.565100205344173, -15.099331408292464, -1.9245264830082809, 22.565100205344173], [22.565100205344173, -22.565098034226104, 6.965594802463428, -22.565100205344173], [-22.565100205344173, -22.565100205344173, -22.565100205344173, -22.565100205344173]]),
                                            240: np.array([[-1.2333211775042203, -2.9837571701833507, -11.772470176050568, -19.999919306079978], [6.94170820492411, -2.228582429012452, -19.999919306079978, -19.999919306079978], [3.7336660163498907, -4.51241223823026, -19.999919306079978, -19.999919306079978], [19.999919306079978, 19.999919306079978, -19.999919306079978, -19.999919306079978]])}, help='扭转系数')
    # 不传入Optimiser的参数
    parser.add_argument('--task', type=str, required=True, choices=['init', 'ang', 'weights', 'slopes', 'recon', 'demo'], help='选择运行目的为初始化、旋转角度优化、曲率权重与扭转系数优化、重构还是动态展示')
    parser.add_argument('--ang_step', type=float, default=0.1, help='旋转角度搜索步长, 初始为0.1')
    parser.add_argument('--ang_range', type=float, default=180, help='旋转角度搜索距离, 初始为180.0')
    args = parser.parse_args()

    # 避免Terminal指令读取错误
    if args.weights == "None":
        args.weights = None
    if args.slopes == "None":
        args.slopes = None
    
    params = {k: getattr(args, k) for k in vars(args) if k not in ['task', 'ang_step', 'ang_range', 'total_scale', 'imu', 'imu_com', 'imu_maxlen', 'demo_recorder', 'unwrap_threshold', 'align_threshold']}
    
    # 初始化优化器与可视化工具
    optim = Optimiser(**params)
    

    if args.task == 'demo':
        app = QApplication(sys.argv)
        demo = Demonstrator(recon=optim, total_scale=args.total_scale, unwrap_threshold=args.unwrap_threshold, align_threshold=args.align_threshold, imu=args.imu, imu_com=args.imu_com, demo_recorder=args.demo_recorder)
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
        elif args.task in ['weights', 'slopes']:
            optim.optim_coeff(coeff=args.task, verbose=True)            
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