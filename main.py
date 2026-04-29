import argparse
import time
import sys
import threading
import numpy as np

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
    parser.add_argument('--sensor_id', type=str, default="04B1_cath", help='传感器编号')
    parser.add_argument('--start_value', type=float, default=69.0, help='传感器第一个传感点到固定件的距离, 单位为mm')
    parser.add_argument('--R', type=float, default=0.1e-3, help='传感器半径, 单位为m')
    parser.add_argument('--ideal_len', type=float, default=0.96, help='传感器长度, 单位为m')
    parser.add_argument('--samp_dist_preset', type=list, default=[510, 520, 530, 540, 550, 580, 610, 640, 670, 720, 770, 850, 930, 1010, 1110, 1230], help='FBG传感点位置, 单位为mm')
    parser.add_argument('--centre_fibre', type=bool, default=True, help='是否为1+x构型')
    parser.add_argument('--global_phases_set', type=float, default=None, help='全局相位，初始值为None')
    # 标定实验参数
    parser.add_argument('--interrogator', type=str, default='Hippo', choices=['FiberTecc', 'Hippo', 'LUNA'], help='解调仪器')
    parser.add_argument('--active_channels', type=list, default=[2, 4, 5, 6, 3, 1, 7], help='传感器连接解调仪的通道以及其顺序')
    parser.add_argument('--active_angles', type=list, default=None, help='外围通道角度排布(度数)，若为None则默认均匀排布')
    parser.add_argument('--folder_path', type=str, default="./FBG/0427-04B1-自研（1+6）FBG+支气管镜导管", help='数据存储路径')
    parser.add_argument('--rot_dir', type=int, default=-1, help='各个标定实验按逆(1)还是顺(-1)时针调整的方向')
    parser.add_argument('--exp_directs', type=list, default=[0, 120, 240], help='实验使用的方向')
    parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200]}, help='标定圆弧形状半径')
    parser.add_argument('--end_len', type=float, default=0.08, help='end形状中进入末端大曲率弧圈的包含长度')
    # 优化使用的超参数
    parser.add_argument('--direct', type=int, default=0, help='优化曲率权重与扭转系数的标定方向')
    parser.add_argument('--weight_bounds', type=tuple, default=(0.0, 3.0), help='曲率权重范围')
    parser.add_argument('--slope_upper_bound', type=tuple, default=20.0, help='扭转系数大小上限')
    parser.add_argument('--thresholds', type=list, default=[0.8, 1.2, 2.0, 3.0, 4.0], help='各曲率分级的阈值')
    parser.add_argument('--max_iter', type=int, default=5, help='优化的最大轮次')
    parser.add_argument('--k_end', type=float, default=0.0, help='优化目标中末端误差的权重')
    parser.add_argument('--k_pitch', type=float, default=0.0, help='优化目标中螺距误差的权重')
    parser.add_argument('--method', type=str, default="Powell", help='优化算法, 用于输入scipy.optimize.minimimze')
    parser.add_argument('--hyper_params', type=dict, default=[10000, 1e-5, 1e-5], help='优化算法使用的超参数')
    parser.add_argument('--min_loss_diff', type=float, default=1e-5, help='收敛判定')
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
    parser.add_argument('--demo_recorder', type=bool, default=True, help='是否在动态展示时记录数据')
    # 重构使用的超参数
    parser.add_argument('--min_consecutive', type=int, default=3, help='曲率分级判定前平滑滤波窗口大小')
    parser.add_argument('--move_filter', type=int, default=3, help='动/静态重构前平滑滤波窗口大小')
    parser.add_argument('--kaiman_params', type=dict, default={'angle': {'Rk': 0.001, 'Bk': 0.001}, 'magnitude': {'Rk': 0.0001, 'Bk': 0.01}}, help='卡尔曼滤波参数')
    parser.add_argument('--smoothing_params', type=dict, default={'window_length': 21,'polyorder': 3}, help='LUNA数据平滑参数')
    parser.add_argument('--interp_num', type=int, default=60, help='插值目标点数')
    parser.add_argument('--sharpness', type=float, default=4.8, help='sigmoid软评分硬度')
    parser.add_argument('--end_weights', type=bool, default=True, help='是否对末端另算权重')
    parser.add_argument('--end_start', type=int, default=-7, help='末端起点')
    parser.add_argument('--TNB_threshold', type=float, default=0.0, help='TNB重构直线判定阈值')
    # 初始化时必须手动调整的超参数
    parser.add_argument('--segmentation', type=list, default=[4, 2, 3, 7], help='对扭转角度-传感点位置曲线的点数')
    parser.add_argument('--connect', type=bool, default=False, help='拟合初始分段时是否令各分段首尾相连')
    parser.add_argument('--turn_threshold', type=float, default=2.0, help='判断形状弯曲方向是否改变的阈值')
    parser.add_argument('--sensing_ranges', type=dict, default={1: (583, 1048), 3: (678, 1146), 4: (749, 1215)}, help='LUNA各通道使用的传感点范围')
    # 角度优化初始值
    parser.add_argument('--rot_ang', type=float, default=0.0, help='绕x轴旋转角度, 初始为0.0')
    # 曲率权重与扭转系数相关初始值
    parser.add_argument('--weights', default={0: np.array([1.2299416483686587, 1.276499313933142, 1.1812079449097777, 1.189648051040153, 1.2598584648298368, 1.0457138971301634, 1.2299416483686587, 1.276499313933142, 1.1812079449097777, 1.189648051040153, 1.2598584648298368, 1.0457138971301634]),
                                            120: np.array([1.2708228202188316, 1.2802453337543294, 1.3944370362639333, 1.1100681704398814, 1.3344454238821395, 1.0718245718968196, 1.2708228202188316, 1.2802453337543294, 1.3944370362639333, 1.1100681704398814, 1.3344454238821395, 1.0718245718968196]),
                                            240: np.array([1.2444081597852923, 1.2285330061778428, 1.5079415698997187, 1.2410011479780314, 1.4714730660364241, 1.0711381008818552, 1.2444081597852923, 1.2285330061778428, 1.5079415698997187, 1.2410011479780314, 1.4714730660364241, 1.0711381008818552])}, help='应变权重')
    parser.add_argument('--slopes', default={0: np.array([[-18.191365208420084, -13.022951978318934, -16.01179367280889, -12.664072631591484], [-14.071067866290932, -21.278239414573626, 0.9405927498457501, -25.982281713011407], [-17.623958457293707, -22.524124967301912, -1.575468781323087, -26.082193757414224], [-11.056323521482438, 9.71549422264249, 11.59323990118162, 26.082193757414224], [-21.35141964662741, -26.082196654264106, -14.592319478625033, -26.082193757414224], [-5.208154134846858, -23.122264261099808, -18.74727945401354, -8.960358901337779]]),
                                            120: np.array([[21.488579151309832, 1.5466779437491316, -22.292080491455035, -15.512971644662352], [16.103393577239395, 8.606202728641763, -41.20058910620496, 0.7329529975080714], [18.87322423069608, 8.330503910971913, -21.650003418902802, -7.535158934526144], [12.918510693033394, 16.064448805402193, -41.185652403285566, -0.15080246524708152], [20.547990652360827, 10.90661714888755, -27.361542479342162, 2.4445458160491214], [7.02409273670132, 19.538920064619592, -41.20032559111566, 5.559996595512033]]),
                                            240: np.array([[-19.524672148512963, 8.430995256188108, 23.65070396959523, -31.748816319016573], [-13.312544578667726, -0.9223685001605338, -4.515452636354289, -18.98498621801104], [-21.86459432667205, 12.449820207297016, 23.19009305815221, 27.965717240814804], [-14.194055587038617, -28.470311548041902, 12.644468649934216, -35.66058993548225], [-23.990137130392, -8.562768133881013, -22.3063470957393, -35.66058993548225], [-13.783784735331428, 9.003241073083167, 35.66058993548226, 4.912863353435324]])}, help='扭转系数')
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
    
    params = {k: getattr(args, k) for k in vars(args) if k not in ['task', 'ang_step', 'ang_range', 'total_scale', 'imu', 'imu_com', 'imu_maxlen', 'demo_recorder']}
    
    # 初始化优化器与可视化工具
    optim = Optimiser(**params)
    

    if args.task == 'demo':
        app = QApplication(sys.argv)
        demo = Demonstrator(recon=optim, total_scale=args.total_scale, imu=args.imu, imu_com=args.imu_com, demo_recorder=args.demo_recorder)
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