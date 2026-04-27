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
    print("-------------------- 初始化开始 --------------------")
    parser = argparse.ArgumentParser()
    # 硬件参数
    parser.add_argument('--sensor_id', type=str, default="04B1", help='传感器编号')
    parser.add_argument('--start_value', type=float, default=163.0, help='传感器第一个传感点到固定件的距离, 单位为mm')
    parser.add_argument('--R', type=float, default=0.1e-3, help='传感器半径, 单位为m')
    parser.add_argument('--ideal_len', type=float, default=0.96, help='传感器长度, 单位为m')
    parser.add_argument('--samp_dist_preset', type=list, default=[510, 520, 530, 540, 550, 580, 610, 640, 670, 720, 770, 850, 930, 1010, 1110, 1230], help='FBG传感点位置, 单位为mm')
    parser.add_argument('--centre_fibre', type=bool, default=True, help='是否为1+x构型')
    parser.add_argument('--global_phases_set', type=float, default=-2.6051262303184806, help='全局相位，初始值为None')
    # 标定实验参数
    parser.add_argument('--interrogator', type=str, default='Hippo', choices=['FiberTecc', 'Hippo', 'LUNA'], help='解调仪器')
    parser.add_argument('--active_channels', type=list, default=[2, 4, 5, 6, 3, 1, 7], help='传感器连接解调仪的通道以及其顺序')
    parser.add_argument('--active_angles', type=list, default=None, help='外围通道角度排布(度数)，若为None则默认均匀排布')
    parser.add_argument('--folder_path', type=str, default="./FBG/0413-04B1-自研（1+6）FBG", help='数据存储路径')
    parser.add_argument('--rot_dir', type=int, default=-1, help='各个标定实验按逆(1)还是顺(-1)时针调整的方向')
    parser.add_argument('--exp_directs', type=list, default=[0, 120, 240], help='实验使用的方向')
    parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200], 'end': {2000: [30, 20, 15], 1500: [150, 100, 50]}}, help='标定圆弧形状半径')
    parser.add_argument('--end_len', type=float, default=0.08, help='end形状中进入末端大曲率弧圈的包含长度')
    # 优化使用的超参数
    parser.add_argument('--weight_bounds', type=tuple, default=(0.0, 2.0), help='曲率权重范围')
    parser.add_argument('--slope_upper_bound', type=tuple, default=45.0, help='扭转系数大小上限')
    parser.add_argument('--thresholds', type=list, default=[0.8, 1.2, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0], help='各曲率分级的阈值')
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
    parser.add_argument('--TNB_threshold', type=float, default=0.15, help='TNB重构直线判定阈值')
    # 初始化时必须手动调整的超参数
    parser.add_argument('--segmentation', type=list, default=[4, 3, 2, 7], help='对扭转角度-传感点位置曲线的点数')
    parser.add_argument('--turn_threshold', type=float, default=2.0, help='判断形状弯曲方向是否改变的阈值')
    parser.add_argument('--sensing_ranges', type=dict, default={1: (583, 1048), 3: (678, 1146), 4: (749, 1215)}, help='LUNA各通道使用的传感点范围')
    # 角度优化初始值
    parser.add_argument('--rot_ang', type=float, default=8.50, help='绕x轴旋转角度, 初始为0.0')
    # 曲率权重与扭转系数相关初始值
    parser.add_argument('--weights', default={0: np.array([1.387401010324464, 1.3155911190855032, 1.3342541993891601, 1.2131594595148374, 1.3058430658058238, 1.290918256177195, 1.660101964779637, 4.034695958928312e-06, 4.034696001026539e-06, 1.999995965303999, 1.999995965303999, 1.999995965303999, 1.999995965303999, 1.999995965303999, 1.999995965303999, 0.29632691089638546, 0.46332179965361914, 0.16379500980953052]),
                                            120: np.array([1.1328699146746692, 1.6672640093222235, 1.357122655766316, 1.4461944045959096, 1.9748289581553002, 1.2635621017462142, 4.0346960010265374e-06, 0.23652433770782827, 1.999995965303999, 1.999995965303999, 1.9999959653040589, 0.3944592058325099, 4.034695943015791e-06, 0.8835945437196792, 0.7933287739929115, 0.010499990178392963, 4.0346960010265374e-06, 1.094216044320855]),
                                            240: np.array([1.2765739663259739, 1.304304961869965, 1.3250365537913709, 1.4504041728715542, 1.3269321929106683, 1.739801758063423, 4.034562234878428e-06, 0.9277090904178914, 4.0346960010265374e-06, 1.999995965303999, 1.999995965303999, 4.0346960010266094e-06, 0.20391177892906873, 4.0346960010265374e-06, 1.9539083643115978, 4.030665275468824e-06, 0.442874807908722, 0.012486618808595624])}, help='应变权重')
    parser.add_argument('--slopes', default={0: np.array([[-18.24873694710094, 12.14740038721649, 4.952758455905366, 4.101225581650594], [-16.589238504567028, 5.102057782943631, 22.226263733467487, 8.224581801759282], [-17.126697130669633, 8.849589375166445, 0.48612379017988677, 3.531999104670078], [-15.687669489580113, 10.77041898353375, 22.990898591104934, 35.16145269118918], [-18.537410356184438, 7.252237806010346, -10.693787264409933, -25.329884742244122], [-12.926874032950762, -16.864801936343753, 6.404831617333144, -44.89294259450884], [38.65121248426324, -44.99974648477765, 22.707685089989077, 3.1177276184176446], [44.97423015115044, 44.999921940884796, 25.377547923414152, -1.633743904414629], [44.99981843867995, 44.99981843867995, 44.99977797970412, -19.23446262410442]]),
                                            120: np.array([[-11.632666229421577, -23.348948762018672, -9.303212886209705, 44.999818438682645], [-4.082736656125723, 44.968255187620294, 15.530420855198587, 44.99981843867995], [-0.6452714266158313, 29.647175619012902, -15.383382439648456, -44.99981843867995], [-3.2303409441228204, 36.82617739710456, -36.243598288651725, 29.998828565159215], [3.4618205800622466, 17.96419088035305, 16.92815477817895, 44.99981843867995], [-7.700971507292049, 34.030696275372264, 32.05235421466306, 44.99981843867995], [44.99981843867995, 44.99981843867995, -16.524165026339126, 44.99981843867995], [44.99448439288386, 44.99981843867995, 29.402626278679914, -34.42880502264437], [44.99981843867995, 44.99981843867995, 44.99981843867995, -1.5348291538969825]]),
                                            240: np.array([[16.507076359915814, 5.144851731368412, 24.544935401755218, 44.99999999998301], [10.971653760418551, 25.93702312743927, -20.677886805675023, -3.735403985064146], [6.239229374027361, 41.830307862720424, -5.39027937739337, -41.00644345395389], [6.28206371000055, 41.85225568234658, -30.735322150779034, -44.99984330083201], [5.043056748990524, 44.99981843867995, -44.99981843867995, -44.99981843867995], [4.161551259152149, -1.66434870666405, -44.99981843867995, -44.99981843867995], [-44.99981843867995, -44.99981758378016, -44.99981843867995, -44.99981843867995], [-44.96685869584651, -44.99981843867995, -15.774605447190375, 44.99981843867995], [44.99981843867995, 44.99981843867995, 44.99981843907449, 44.99999977027874]])}, help='扭转系数')
    # 不传入Optimiser的参数
    parser.add_argument('--task', type=str, required=True, choices=['init', 'ang', 'weights', 'slopes', 'recon', 'demo'], help='选择运行目的为初始化、旋转角度优化、曲率权重与扭转系数优化、重构还是动态展示')
    parser.add_argument('--init_loss', type=float, default=np.inf, help='当前优化使用的初始误差')
    parser.add_argument('--ang_step', type=float, default=0.1, help='旋转角度搜索步长, 初始为0.1')
    parser.add_argument('--ang_range', type=float, default=180, help='旋转角度搜索距离, 初始为180.0')
    parser.add_argument('--direct', type=int, default=0, help='优化曲率权重与扭转系数的标定方向')
    args = parser.parse_args()

    # 避免Terminal指令读取错误
    if args.weights == "None":
        args.weights = None
    if args.slopes == "None":
        args.slopes = None
    
    params = {k: getattr(args, k) for k in vars(args) if k not in ['task', 'init_loss', 'ang_step', 'ang_range', 'direct', 'total_scale', 'imu', 'imu_com', 'imu_maxlen', 'demo_recorder']}
    
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
            optim.direct = args.direct
            optim.optim_coeff(init_loss=args.init_loss, coeff=args.task, verbose=True)            
        elif args.task == 'recon':
            optim.direct = args.direct
            rs = {}
            for calib_shape in optim.calib_shapes:
                r_results_postrot, interp_dist = optim.reconstruct_shapes(calib_shape=calib_shape, direct=args.direct, verbose=False)
                rs[calib_shape] = r_results_postrot
            optim._compute_error(interp_dist=interp_dist, rs=rs)
            vis.show_reconstruction(direct=args.direct, rs=rs, interp_dist=interp_dist, showanime=True)
            optim._print_errors()
    

if __name__ == '__main__':
    main()