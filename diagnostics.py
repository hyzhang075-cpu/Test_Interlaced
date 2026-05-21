import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
import argparse
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq

from optimiser import Optimiser
from visualisor import Visualisor


# =========================
# 参数读取与 Optimiser 初始化
# =========================
parser = argparse.ArgumentParser()
# 硬件参数
parser.add_argument('--sensor_id', type=str, default="04B1_cath0509", help='传感器编号')
parser.add_argument('--start_value', type=float, default=80, help='传感器第一个传感点到固定件的距离, 单位为mm')
parser.add_argument('--R', type=float, default=0.1e-3, help='传感器半径, 单位为m')
parser.add_argument('--ideal_len', type=float, default=0.9, help='传感器长度, 单位为m')
parser.add_argument('--samp_dist_preset', type=list, default=[510, 520, 530, 540, 550, 580, 610, 640, 670, 720, 770, 850, 930, 1010, 1110, 1230], help='FBG传感点位置, 单位为mm')
parser.add_argument('--centre_fibre', type=bool, default=True, help='是否为1+x构型')
parser.add_argument('--global_phases_set', type=float, default=2.218625282483061, help='全局相位，初始值为None')
# 标定实验参数
parser.add_argument('--interrogator', type=str, default='Hippo', choices=['FiberTecc', 'Hippo', 'LUNA'], help='解调仪器')
parser.add_argument('--active_channels', type=list, default=[1, 5, 3, 4, 2, 6], help='传感器连接解调仪的通道以及其顺序')
parser.add_argument('--active_angles', type=list, default=[0, 60, 120, 180, 240], help='外围通道角度排布(度数)，若为None则默认均匀排布')
parser.add_argument('--folder_path', type=str, default="./FBG/0429-04B1-自研（1+6）FBG+支气管镜导管中心", help='数据存储路径')
parser.add_argument('--rot_dir', type=int, default=1, help='各个标定实验按逆(1)还是顺(-1)时针调整的方向')
parser.add_argument('--exp_directs', type=list, default=[0, 120, 240], help='实验使用的方向')
parser.add_argument('--radii', type=list, default={'circ': [0, 2000, 1500, 1000, 800, 600, 400, 300, 200]}, help='标定圆弧形状半径')
parser.add_argument('--end_len', type=float, default=0.08, help='end形状中进入末端大曲率弧圈的包含长度')
# 优化使用的超参数
parser.add_argument('--direct', type=int, default=0, help='优化曲率权重与扭转系数的标定方向')
parser.add_argument('--freeze', type=list, default=None, help='冻结部分分区的参数，使得它们不参与优化。如果不需要冻结则调整为None')
parser.add_argument('--weight_bounds', type=tuple, default=(0.0, 2.0), help='曲率权重范围')
parser.add_argument('--slope_upper_bound', type=tuple, default=20.0, help='扭转系数大小上限')
parser.add_argument('--max_iter', type=int, default=5, help='优化的最大轮次')
parser.add_argument('--k_point', type=float, default=1.0, help='优化目标中末端误差的权重')
parser.add_argument('--k_end', type=float, default=0.3, help='优化目标中末端误差的权重')
parser.add_argument('--k_pitch', type=float, default=1.0, help='优化目标中螺距误差的权重')
parser.add_argument('--k_radius', type=float, default=1.0, help='优化目标中半径误差的权重')
parser.add_argument('--method', type=str, default="Powell", help='优化算法, 用于输入scipy.optimize.minimimze')
parser.add_argument('--hyper_params', type=dict, default=[10000, 1e-5, 1e-5], help='优化算法使用的超参数')
parser.add_argument('--min_loss_diff', type=float, default=1e-5, help='收敛判定')
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
parser.add_argument('--unwrap_threshold', type=float, default=1.0, help='若两帧同一点角度变化小于该阈值，则统一unwrap方向')
parser.add_argument('--align_threshold', type=float, default=0.2, help='若某帧第一点角度变化小于该阈值，则将第一点角度与上一帧对齐')
# 重构使用的超参数
parser.add_argument('--discont', type=float, default=np.pi, help='np.unwrap跳变判断阈值')
parser.add_argument('--frame_rule', type=str, default='TNN', choices=['TNB', 'TNN'], help="选择使用FS标架还是平行传递标架")
parser.add_argument('--thresholds', type=list, default=[0.8, 1.2, 2.0], help='各曲率分级的阈值')
parser.add_argument('--straight_threshold', type=float, default=2.0, help='零螺距形状中判断直线段的曲率阈值')
parser.add_argument('--rot_threshold', type=float, default=4.0, help='零螺距形状中判断螺旋段的曲率阈值')
parser.add_argument('--min_consecutive', type=int, default=3, help='曲率分级判定前平滑滤波窗口大小')
parser.add_argument('--move_filter', type=int, default=3, help='动/静态重构前平滑滤波窗口大小')
parser.add_argument('--kaiman_params', type=dict, default={'angle': {'Rk': 0.001, 'Bk': 0.001}, 'magnitude': {'Rk': 0.0001, 'Bk': 0.01}}, help='卡尔曼滤波参数')
parser.add_argument('--smoothing_params', type=dict, default={'window_length': 21,'polyorder': 3}, help='LUNA数据平滑参数')
parser.add_argument('--interp_num', type=int, default=60, help='插值目标点数')
parser.add_argument('--sharpness', type=float, default=4.8, help='sigmoid软评分硬度')
parser.add_argument('--end_weights', type=bool, default=False, help='是否对末端另算权重')
parser.add_argument('--end_start', type=int, default=-7, help='末端起点')
parser.add_argument('--TNB_threshold', type=float, default=0.0, help='TNB重构直线判定阈值')
# 初始化时必须手动调整的超参数
parser.add_argument('--segmentation', type=list, default=[2, 2, 6, 6], help='对扭转角度-传感点位置曲线的点数')
parser.add_argument('--connect', type=bool, default=False, help='拟合初始分段时是否令各分段首尾相连')
parser.add_argument('--turn_threshold', type=float, default=2.0, help='判断形状弯曲方向是否改变的阈值')
parser.add_argument('--sensing_ranges', type=dict, default={1: (583, 1048), 3: (678, 1146), 4: (749, 1215)}, help='LUNA各通道使用的传感点范围')
# 角度优化初始值
# parser.add_argument('--rot_ang', type=float, default=-40.0, help='绕x轴旋转角度, 初始为0.0')
parser.add_argument('--rot_ang', type=float, default=0.0, help='绕x轴旋转角度, 初始为0.0')
# 曲率权重与扭转系数相关初始值
# parser.add_argument('--weights', default={0: np.array([1.3968169831498476, 1.0667532002480278, 1.1375383427393826, 1.0694821821171347, 1.0313653701868237, 1.2620742266982392]),
#                                         120: np.array([0.8564815682411855, 0.9399042282350683, 0.9752028346404654, 1.0518474361869776, 1.1617849856823705, 1.0070899544165117]),
#                                         240: np.array([1.0000059608609866, 1.0000059608609866, 1.000005900084534, 1.0000000000355318, 1.2109376880518632, 1.0000059608609866])}, help='应变权重')
# parser.add_argument('--slopes', default={0: np.array([[-12.670259683175164, -10.303256531742232, -23.91382970972733, -24.99987174166408], [-16.488393406694072, -5.600158131621509, 13.471535759004597, -0.505724879585917], [-7.234049891260806, -14.957042843568413, 5.523691941301739, 4.90490327572986], [-10.296368467626953, -12.859617256422288, 7.7569795030276545, -24.99989913259996], [-10.184653555248115, -12.275412302308663, 5.469393293782229, -24.999899132599968], [12.88113623630428, -24.999899132599975, 24.179808505054506, -11.998245888787693]]),
#                                         120: np.array([[-24.957031336929976, 11.679329036559754, -23.61974653150268, -24.999899132599975], [-20.063002677584198, -10.733698941682697, -3.778205390616809, 24.999899132599975], [-15.09302709159872, -11.275970544049112, 5.908033848008622, 24.819561028981294], [-16.300914799546135, -10.914521006385227, 6.029247867020968, -12.35148638804945], [-15.799198048305648, -8.969110839189412, 1.2188157047425456, -6.504225843398681], [-10.225022532810945, -16.180097584685253, 13.424054750985048, 10.208686760510664]]),
#                                         240: np.array([[-5.512663320168729, -19.90576205843398, -19.707023957962285, -19.999919306079978], [-15.229838195095844, -15.993861646478964, 14.701434055838673, -19.999919306079978], [-16.211299868737367, -10.116083494473319, -4.661251176781066, -19.987144190681395], [-14.379244007134835, -13.60479990168532, 5.410221576980541, 19.999919306079978], [-15.729363378166902, -8.563002120960734, 9.248239254450265, -7.3424760335371495], [-15.017571401985299, -16.989504671728085, -7.216047713894436, -19.999919306079978]])}, help='扭转系数')
parser.add_argument('--weights', default={0: np.array([1.4399114951505265, 1.1374347240112066, 1.2664003165163586, 1.1040923131807858]),
                                            120: np.array([1.4399114951505265, 1.1374347240112066, 1.2664003165163586, 1.1040923131807858]),
                                            240: np.array([1.4399114951505265, 1.1374347240112066, 1.2664003165163586, 1.1040923131807858])}, help='应变权重')
parser.add_argument('--slopes', default={0: np.array([[12.679395211465765, 4.183885877957795, -15.10103781307189, -20.344393081570818], [14.433359191228446, 1.6004557928581038, -11.566342908176729, -24.950686895760395], [14.423753383332844, 2.4256670095956068, 7.890614324064342, 24.999899132599975], [20.17883387214279, -3.508811958754833, 6.624940446239565, -24.99989913228523]]),
                                            120: np.array([[12.679395211465765, 4.183885877957795, -15.10103781307189, -20.344393081570818], [14.433359191228446, 1.6004557928581038, -11.566342908176729, -24.950686895760395], [14.423753383332844, 2.4256670095956068, 7.890614324064342, 24.999899132599975], [20.17883387214279, -3.508811958754833, 6.624940446239565, -24.99989913228523]]),
                                            240: np.array([[12.679395211465765, 4.183885877957795, -15.10103781307189, -20.344393081570818], [14.433359191228446, 1.6004557928581038, -11.566342908176729, -24.950686895760395], [14.423753383332844, 2.4256670095956068, 7.890614324064342, 24.999899132599975], [20.17883387214279, -3.508811958754833, 6.624940446239565, -24.99989913228523]])}, help='扭转系数')
# 不传入Optimiser的参数
# parser.add_argument('--task', type=str, required=True, choices=['init', 'ang', 'weights', 'slopes', 'recon', 'demo'], help='选择运行目的为初始化、旋转角度优化、曲率权重与扭转系数优化、重构还是动态展示')
parser.add_argument('--ang_step', type=float, default=0.1, help='旋转角度搜索步长, 初始为0.1')
parser.add_argument('--ang_range', type=float, default=180, help='旋转角度搜索距离, 初始为180.0')
args = parser.parse_args()

if args.weights == "None":
    args.weights = None
if args.slopes == "None":
    args.slopes = None

params = {k: getattr(args, k) for k in vars(args) if k not in ['task', 'ang_step', 'ang_range', 'total_scale', 'imu', 'imu_com', 'imu_maxlen', 'demo_recorder', 'unwrap_threshold', 'align_threshold']}

optim = Optimiser(**params)
vis = Visualisor(recon=optim)


# =========================
# 基本配置
# =========================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

filename = "VTK_ShapeData_20260511_182953.pkl"

SEGMENT_RULE = [2, 2, 6, 6]
TOP_K_ENDPOINT_DISTANCE = 10
TOP_K_CUR_INTERP_CHANGE = 3
TOP_K_CURVATURE_ANGLE_CHANGE = 3

# 是否绘制 endpoint-distance 最大的帧对的 TNB 标架
PLOT_MAX_ENDPOINT_TNB = True
TNB_PLOT_ARROW_LENGTH = 0.03
TNB_PLOT_SCALE = 1.0

# 如果上一帧和当前帧几乎重合，可以给两条曲线加一点纯显示用的 Y 方向错位。
# 设为 0.0 就是在真实坐标中叠加绘制；设为 0.01 表示上一帧 -0.01m、当前帧 +0.01m。
TNB_PAIR_VISUAL_Y_OFFSET = 0.01

# timestamp / time 字段检查
PRINT_PACKET_KEYS_ON_READ = False
PRINT_TIMESTAMP_REPORT = True
PRINT_TIMESTAMP_IN_TOP_PAIRS = False
TIMESTAMP_KEYWORDS = ('timestamp', 'time', 'ts')

# 若想指定一对相邻帧而不是自动使用 endpoint-distance 最大的帧对，
# 在这里填写二元组，例如：SPECIFIED_ADJACENT_FRAME_PAIR = (2012, 2013)。
# 设为 None 时保持原逻辑：自动寻找 endpoint-distance 最大的帧对。
# SPECIFIED_ADJACENT_FRAME_PAIR = None
SPECIFIED_ADJACENT_FRAME_PAIR = None

# 指定要分析的 timestamp 区间。
# None 表示不限制这一端。
ANALYSIS_TIMESTAMP_RANGE = (1778495577, 1778495724)
# ANALYSIS_TIMESTAMP_RANGE = (None, None)
TIMESTAMP_KEY = 'timestamp'

# 如果 timestamp 不随帧递增，或者按 timestamp 裁剪会删光数据，则退回按帧率裁剪。
TRIM_FALLBACK_USE_FRAME_COUNT = True
TRIM_FALLBACK_FRAME_RATE = args.data_freq

# 精简版只保留 calib_data 显式重构得到的 r/T/N/B 或 r/T/N1/N2。
# 以下两个旧开关保留为兼容配置，但当前主流程不再使用多来源 r/TNB 图。
PLOT_BOTH_R_TNB_SOURCES = False
PLOT_DYNAMIC_RECON_SMOOTH_R_TNB = False
DYNAMIC_RECON_SMOOTH_WINDOW = args.move_filter

# 基于 calib_data 重构 + 动态展示式滑动平均末端轨迹的整体运动诊断图。
PLOT_CALIB_RECON_CYCLOID_FITTING_AND_DEVIATION = False
PLOT_CALIB_RECON_TRAJECTORY_JITTER = False

# 是否把每帧 calib_data 显式重构得到的完整 r 坐标序列导出为 3D 视频。
# 这里不使用 pkl 中记录的 packet['r_results']，而是使用 diagnostics 中重新从 calib_data / calib_shape='dynamic'
# 计算出来的 r，因此可用于检查“重新重构结果”本身的动态效果。
GENERATE_CALIB_RECON_VIDEO = False
CALIB_RECON_VIDEO_OUTPUT = "calib_shape_reconstructed_shape_demo.mp4"
CALIB_RECON_VIDEO_FPS = 30
CALIB_RECON_VIDEO_WRITER = "ffmpeg"
CALIB_RECON_VIDEO_DPI = 150
CALIB_RECON_VIDEO_STRIDE = 1
CALIB_RECON_VIDEO_MAX_FRAMES = None
CALIB_RECON_VIDEO_USE_DYNAMIC_SMOOTHING = True
CALIB_RECON_VIDEO_SMOOTH_WINDOW = DYNAMIC_RECON_SMOOTH_WINDOW
CALIB_RECON_VIDEO_ELEV = 20
CALIB_RECON_VIDEO_AZIM = -60

CYCLOID_RADIUS = 0.4
CYCLOID_MARKER_STEP = 100
# 在 PLOT_CALIB_RECON_CYCLOID_FITTING_AND_DEVIATION 图中额外标注这些 timestamp 对应的帧。
# 如果对应帧被 trajectory_arr[::10] 跳过，会自动选择下采样后 timestamp 最近的点。
CYCLOID_ANNOTATE_TIMESTAMPS = [1778495466, 1778495577, 1778495724, 1778495784]
# None 表示不限制最大允许时间差；若设为数值，则超过该秒数时仍会标注，但会打印 warning。
CYCLOID_TIMESTAMP_MATCH_MAX_DIFF = None
# 与旧 recorded-data 分析脚本保持一致：前 20s 去程被删后，剩余轨迹按回程处理，
# 用逆序轨迹的起点/最远点解算理论摆线，但偏差仍对原始顺序 trajectory_arr 逐帧计算。
CYCLOID_FIT_USE_REVERSE_TRAJECTORY = True
JITTER_FRAME_DT = 1.0

# diagnostics 中从 calib_data 重构时使用的标架规则。
# 设为 args.frame_rule 时，会跟随命令行 / 参数区的 --frame_rule；
# 因此 --frame_rule TNB 会画 T/N/B，--frame_rule TNN 会画 T/N1/N2。
DIAG_RECON_FRAME_RULE = args.frame_rule

# SPECIFIED_FRAME_PAIR_MODE:
# - 'original': 使用原始 pkl 中的帧号，即输出里显示的 Original Frame；
# - 'trimmed': 使用去除前 TRIM_INITIAL_SECONDS 后的 data_list 下标，即输出里显示的 Trimmed Frame。
SPECIFIED_FRAME_PAIR_MODE = 'trimmed'

# 保留旧开关名作为默认值；现在只在需要单独调用 reconstruct_packet_tnb_for_plot 时使用。
TNB_PAIR_USE_PACKET_R_RESULTS = False

# 是否对目标帧对额外画诊断图。
# 若 SPECIFIED_ADJACENT_FRAME_PAIR 不为 None，则目标帧对就是指定帧对；
# 否则目标帧对就是 endpoint-distance 最大的 Rank 1 帧对。
PLOT_MAX_PAIR_CUR_INTERP_AND_ANGLES = True

# 是否额外画由 calib_data 重新计算得到的原始 curvature magnitude 与原始 angle。
# 注意：cur_interp / ang_interp 是经过滤波与插值后的重构输入；
# magnitude / angle 是 _Strain_2_Cur 后、插值前的曲率大小与方向角。
PLOT_TARGET_PAIR_RAW_CURVATURE_AND_ANGLE = True

# 是否额外画 np.unwrap 之后、逐帧方向限制之前的 angle。
# 这样可以区分变化来自 np.unwrap，还是来自 framewise unwrap limit。
PLOT_TARGET_PAIR_ANGLE_UNWRAP_RAW = True

# 是否打印目标帧对的 magnitude / angle / angle_unwrap_raw / angle_unwrap 及其帧间差值。
# 这里的 magnitude 是 _Strain_2_Cur 计算后，并经过 zero_init 或 angle[0]=angle[1] 分支处理后的曲率。
PRINT_TARGET_PAIR_ANGLE_VALUES = True

# 是否额外画 angle_unwrap 经过 moving average 之后、Hermite 插值之前的 ang_filter。
# 这张图用于区分异常走势来自滑动平均，还是来自后续 Hermite 插值。
PLOT_TARGET_PAIR_ANGLE_FILTER = True

# 是否输出由 calib_data 重新计算得到的 curvature / angle 变化排行。
PRINT_TARGET_PAIR_RECON_CURVATURE_AND_ANGLE = False

# 是否在 diagnostics 中启用逐帧 angle_unwrap 方向限制。
# 注意：这只影响本 diagnostics 脚本显式展开的 calib_data 重构流程，不修改 reconstructor.py。
APPLY_FRAMEWISE_UNWRAP_LIMIT = True
PRINT_FRAMEWISE_UNWRAP_DEBUG = False

# 如果 slopes[direct] 是二维矩阵，则取第几行作为角度计算用的分段扭转系数
ANGLE_SLOPE_LEVEL = 0

# 当首点曲率接近直线段阈值时，用上一帧未经 zero_init / angle[0]=angle[1] 处理的 raw angle[0]
# 替换当前帧的 raw angle[0]，以降低起点法向量在近零曲率处的跳变。
APPLY_LOW_CURVATURE_START_ANGLE0_HOLD = True
PRINT_LOW_CURVATURE_START_ANGLE0_DEBUG = False

print(f"Diagnostic calib_data reconstruction frame_rule: {DIAG_RECON_FRAME_RULE}")

data_list = []
end_effector_trajectory = []


# =========================
# 工具函数
# =========================

# def _get_unwrap_direct(angle, angle_unwrap):
#     """
#     返回每个相邻点 i-1 -> i 在 unwrap 后实际走的方向。

#     +1 表示 angle_unwrap[i] >= angle_unwrap[i-1]
#     -1 表示 angle_unwrap[i] <  angle_unwrap[i-1]

#     这里故意只看 angle_unwrap 的差分符号，不再比较
#     angle_unwrap_diff == angle_diff，避免浮点数精确相等导致方向误判。
#     """
#     angle_unwrap = np.asarray(angle_unwrap, dtype=np.float64)
#     return [1 if diff >= 0 else -1 for diff in np.diff(angle_unwrap)]

def select_timestamp_range(data_list, timestamp_range=(None, None), keep_missing_timestamp=False):
    """
    按具体 timestamp 区间筛选 data_list。

    输入：
    - timestamp_range: (start_timestamp, end_timestamp)
        - start_timestamp 为 None 时不限制起点；
        - end_timestamp 为 None 时不限制终点；
        - 区间为闭区间 [start_timestamp, end_timestamp]。
    - keep_missing_timestamp:
        - False: 没有 timestamp 的帧不保留；
        - True: 没有 timestamp 的帧保留。

    输出：
    - selected: 只包含指定 timestamp 区间内的 packet。
    """
    start_timestamp, end_timestamp = timestamp_range

    if start_timestamp is None and end_timestamp is None:
        print("\n========== timestamp range selection skipped ==========")
        print("ANALYSIS_TIMESTAMP_RANGE is (None, None); all frames are kept.")
        return data_list

    selected = []
    missing_timestamp_count = 0
    before_count = 0
    after_count = 0

    first_kept_original_idx = None
    last_kept_original_idx = None
    first_kept_timestamp = None
    last_kept_timestamp = None

    print("\n========== timestamp range selection report ==========")
    print(f"Requested timestamp range: [{start_timestamp}, {end_timestamp}]")

    for frame_idx, packet in enumerate(data_list):
        timestamp = get_packet_timestamp(packet)

        if timestamp is None:
            missing_timestamp_count += 1

            if keep_missing_timestamp:
                selected.append(packet)

            continue

        if start_timestamp is not None and timestamp < start_timestamp:
            before_count += 1
            continue

        if end_timestamp is not None and timestamp > end_timestamp:
            after_count += 1
            continue

        selected.append(packet)

        original_idx = packet.get('_diag_original_frame_idx', frame_idx) if isinstance(packet, dict) else frame_idx

        if first_kept_original_idx is None:
            first_kept_original_idx = original_idx
            first_kept_timestamp = timestamp

        last_kept_original_idx = original_idx
        last_kept_timestamp = timestamp

    print(f"Original frames: {len(data_list)}")
    print(f"Selected frames: {len(selected)}")
    print(f"Frames before range: {before_count}")
    print(f"Frames after range: {after_count}")
    print(f"Frames without numeric timestamp: {missing_timestamp_count}")

    if len(selected) > 0:
        print(
            f"Selected original frame range: {first_kept_original_idx} -> {last_kept_original_idx}"
        )
        print(
            f"Selected timestamp range: "
            f"{first_kept_timestamp} ({format_unix_timestamp(first_kept_timestamp)}) -> "
            f"{last_kept_timestamp} ({format_unix_timestamp(last_kept_timestamp)})"
        )
    else:
        print("Warning: selected timestamp range contains no usable frames.")

    return selected

def _get_unwrap_direct(angle, angle_unwrap):
    """
    修复：使用 np.isclose 替代 == 进行浮点数比较
    """
    unwrap_directs = []
    for i in range(1, len(angle)):
        angle_diff = angle[i] - angle[i-1]
        angle_unwrap_diff = angle_unwrap[i] - angle_unwrap[i-1]

        # 使用 np.isclose 容忍浮点误差
        if angle_diff >= 0 and np.isclose(angle_unwrap_diff, angle_diff, atol=1e-7):
            unwrap_direct = 1
        elif angle_diff >= 0 and angle_unwrap_diff < angle_diff:
            unwrap_direct = -1
        elif angle_diff < 0 and np.isclose(angle_unwrap_diff, angle_diff, atol=1e-7): 
            unwrap_direct = -1
        else:
            unwrap_direct = 1
        unwrap_directs.append(unwrap_direct)
    return unwrap_directs

def get_segment_slices(segment_rule, total_rows):
    slices = []
    start = 0

    for seg_idx, seg_len in enumerate(segment_rule):
        end = start + seg_len

        if start >= total_rows:
            break

        end = min(end, total_rows)
        slices.append((seg_idx, start, end))
        start = end

    return slices


def get_direct_value(value, direct):
    if isinstance(value, dict):
        if direct in value:
            return value[direct]
        if str(direct) in value:
            return value[str(direct)]
        return next(iter(value.values()))

    return value


def get_seg_slopes_for_angle(optim, direct, num_segments, slope_level=0):
    if not hasattr(optim, "slopes") or optim.slopes is None:
        return None

    slopes_for_direct = get_direct_value(optim.slopes, direct)
    slopes_arr = np.asarray(slopes_for_direct, dtype=np.float64)

    if slopes_arr.ndim == 1:
        if len(slopes_arr) < num_segments:
            return None
        return slopes_arr[:num_segments]

    if slopes_arr.ndim == 2:
        level = min(max(int(slope_level), 0), slopes_arr.shape[0] - 1)
        if slopes_arr.shape[1] < num_segments:
            return None
        return slopes_arr[level, :num_segments]

    return None

def circular_abs_angle_diff(curr_angle, prev_angle):
    """
    计算圆周角度差的绝对值，避免 0 和 2π 附近产生虚假大跳变。
    """
    return np.abs((curr_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi)

def make_framewise_unwrap_processor(unwrap_threshold, debug=False):
    """
    构造一个带状态的逐帧 angle_unwrap 修正器。

    规则：
    - 逐帧比较同一个传感点上的原始 angle；
    - 如果当前帧和上一帧该点的圆周角差不超过 unwrap_threshold，
      就强制当前帧在相邻点 i-1 -> i 的 unwrap 方向沿用上一帧；
    - 如果某个点被强制修正，则把该修正带来的相位偏移 offset 累加，
      并同步平移后续点，避免后续点仍停留在旧相位分支；
    - 该修正发生在 angle_unwrap 进入 moving average / Hermite 插值 / TNB或TNN 重构之前。

    返回的 processor 带有 reset() 方法，遇到重构失败或非连续帧时可以清空状态。
    """
    state = {
        "prev_angle": None,
        "prev_angle_unwrap": None,
        "prev_unwrap_directs": None,
        "prev_context": None,
        "last_prev_unwrap_directs_used": None,
        "last_frame_unwrap_directs": None,
        "last_changed_indices": [],
        "last_context": None,
    }

    def reset():
        state["prev_angle"] = None
        state["prev_angle_unwrap"] = None
        state["prev_unwrap_directs"] = None
        state["prev_context"] = None
        state["last_prev_unwrap_directs_used"] = None
        state["last_frame_unwrap_directs"] = None
        state["last_changed_indices"] = []
        state["last_context"] = None

    def processor(angle, angle_unwrap, angle_unwrap_raw=None, frame_context=None):
        angle = np.asarray(angle, dtype=np.float64)
        angle_unwrap = np.asarray(angle_unwrap, dtype=np.float64).copy()

        prev_angle = state["prev_angle"]
        prev_unwrap_directs = state["prev_unwrap_directs"]
        changed_indices = []

        if prev_angle is not None and prev_unwrap_directs is not None:
            common_len = min(len(angle), len(prev_angle), len(angle_unwrap))

            if common_len > 1:
                offset = 0.0

                for i in range(1, common_len):
                    # 先把之前修正造成的累计相位偏移应用到当前点。
                    # 这样当前点与 angle_unwrap[i - 1] 处在同一个被修正后的相位链上。
                    angle_unwrap[i] += offset

                    curr_raw_step = angle[i] - angle[i - 1]
                    prev_raw_step = prev_angle[i] - prev_angle[i - 1]

                    # 比较两帧中同一相邻段 i-1 -> i 的原始角度差是否稳定。
                    # 即判断 (a_i^t - a_{i-1}^t) 与 (a_i^{t-1} - a_{i-1}^{t-1}) 是否接近。
                    # 使用圆周差避免 0 / 2π 附近产生虚假大跳变。
                    segment_step_diff = np.abs(
                        (curr_raw_step - prev_raw_step + np.pi) % (2 * np.pi) - np.pi
                    )

                    if np.isclose(segment_step_diff, 0.0, atol=unwrap_threshold, rtol=0.0):
                        old_unwrap_val = angle_unwrap[i]

                        prev_unwrap_step = state["prev_angle_unwrap"][i] - state["prev_angle_unwrap"][i - 1]

                        candidate_steps = np.array([
                            curr_raw_step - 2 * np.pi,
                            curr_raw_step,
                            curr_raw_step + 2 * np.pi,
                        ], dtype=np.float64)

                        step = candidate_steps[np.argmin(np.abs(candidate_steps - prev_unwrap_step))]

                        new_unwrap_val = angle_unwrap[i - 1] + step

                        if not np.isclose(new_unwrap_val, old_unwrap_val, atol=1e-12, rtol=0.0):
                            angle_unwrap[i] = new_unwrap_val

                            # 当前点被切换相位分支后，后续点也必须整体跟着平移。
                            step_offset = angle_unwrap[i] - old_unwrap_val
                            offset += step_offset

                            changed_indices.append(i)

                # 如果当前帧比上一帧长，common_len 之后的点虽然没有上一帧方向可比较，
                # 但仍应继承前面修正产生的累计相位平移。
                if offset != 0.0 and common_len < len(angle_unwrap):
                    angle_unwrap[common_len:] += offset

                if debug and changed_indices:
                    print(
                        f"  framewise unwrap limited at indices={changed_indices}; "
                        f"frame_context={frame_context}, prev_context={state['prev_context']}"
                    )

        frame_unwrap_directs = _get_unwrap_direct(angle, angle_unwrap)

        state["last_prev_unwrap_directs_used"] = (
            None if prev_unwrap_directs is None else list(prev_unwrap_directs)
        )
        state["last_frame_unwrap_directs"] = list(frame_unwrap_directs)
        state["last_changed_indices"] = list(changed_indices)
        state["last_context"] = frame_context

        state["prev_angle"] = angle.copy()
        state["prev_angle_unwrap"] = angle_unwrap.copy()
        state["prev_unwrap_directs"] = frame_unwrap_directs
        state["prev_context"] = frame_context

        return angle_unwrap

    processor.reset = reset
    processor.state = state
    return processor


def make_diag_unwrap_processor_if_enabled():
    """
    根据 diagnostics 全局开关创建逐帧 unwrap processor。
    """
    if not APPLY_FRAMEWISE_UNWRAP_LIMIT:
        return None

    return make_framewise_unwrap_processor(
        unwrap_threshold=float(args.unwrap_threshold),
        debug=PRINT_FRAMEWISE_UNWRAP_DEBUG
    )


def make_low_curvature_angle0_processor(debug=False):
    """
    构造一个带状态的首点 raw angle 修正器。

    规则：
    - 在 _Strain_2_Cur 得到 raw magnitude / raw angle 之后，
      且在 zero_init 插点或 angle[0] = angle[1] 之前调用；
    - 如果当前帧 magnitude[0] 小于或近似等于 args.TNB_threshold，
      且上一帧存在未经任何处理的 raw angle[0]，则用上一帧的 raw angle[0]
      替换当前帧的 raw angle[0]；
    - 保存给下一帧使用的 prev_raw_angle0 始终是当前帧刚从 _Strain_2_Cur
      得到的原始 angle[0]，不是替换后的值。
    """
    state = {
        "prev_raw_angle0": None,
        "prev_context": None,
        "last_original_angle0": None,
        "last_replaced_angle0": None,
        "last_used_prev_raw_angle0": None,
        "last_magnitude0": None,
        "last_context": None,
    }

    def reset():
        state["prev_raw_angle0"] = None
        state["prev_context"] = None
        state["last_original_angle0"] = None
        state["last_replaced_angle0"] = None
        state["last_used_prev_raw_angle0"] = None
        state["last_magnitude0"] = None
        state["last_context"] = None

    def processor(magnitude, angle, frame_context=None):
        magnitude = np.asarray(magnitude, dtype=np.float64)
        angle = np.asarray(angle, dtype=np.float64).copy()

        original_angle0 = None
        replaced_angle0 = False
        used_prev_raw_angle0 = None
        magnitude0 = None

        if len(angle) > 0:
            original_angle0 = float(angle[0])

        if len(magnitude) > 0:
            magnitude0 = float(magnitude[0])

        prev_raw_angle0 = state["prev_raw_angle0"]

        if original_angle0 is not None and magnitude0 is not None and prev_raw_angle0 is not None:
            threshold = float(getattr(args, "TNB_threshold", 0.0))
            below_or_close_threshold = (
                magnitude0 < threshold
                or np.isclose(magnitude0, threshold, atol=1e-12, rtol=0.0)
            )

            if below_or_close_threshold:
                angle[0] = prev_raw_angle0
                replaced_angle0 = True
                used_prev_raw_angle0 = float(prev_raw_angle0)

                if debug:
                    print(
                        f"  low-curvature raw angle[0] held: original={original_angle0:.6f}, "
                        f"new={used_prev_raw_angle0:.6f}, magnitude[0]={magnitude0:.6g}, "
                        f"threshold={threshold:.6g}, frame_context={frame_context}, "
                        f"prev_context={state['prev_context']}"
                    )

        # state["prev_raw_angle0"] = original_angle0
        state["prev_raw_angle0"] = float(angle[0]) if len(angle) > 0 else None
        state["prev_context"] = frame_context
        state["last_original_angle0"] = original_angle0
        state["last_replaced_angle0"] = replaced_angle0
        state["last_used_prev_raw_angle0"] = used_prev_raw_angle0
        state["last_magnitude0"] = magnitude0
        state["last_context"] = frame_context

        return angle

    processor.reset = reset
    processor.state = state
    return processor


def make_diag_angle0_processor_if_enabled():
    """
    根据 diagnostics 全局开关创建首点 raw angle 修正器。
    """
    if not APPLY_LOW_CURVATURE_START_ANGLE0_HOLD:
        return None

    return make_low_curvature_angle0_processor(
        debug=PRINT_LOW_CURVATURE_START_ANGLE0_DEBUG
    )


def format_short_value(value, max_len=120):
    """
    把 timestamp / time 字段的值压缩成适合打印的一行。
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            text = repr(value.item())
        else:
            first_value = value.reshape(-1)[0] if value.size else None
            text = f"ndarray(shape={value.shape}, dtype={value.dtype}, first={first_value})"
    else:
        text = repr(value)

    if len(text) > max_len:
        text = text[:max_len - 3] + '...'

    return text


def get_timestamp_items(packet, keywords=TIMESTAMP_KEYWORDS):
    """
    返回 packet 中 key 名包含 timestamp / time / ts 的字段。
    """
    if not isinstance(packet, dict):
        return []

    timestamp_items = []

    for key, value in packet.items():
        key_text = str(key).lower()

        if any(keyword in key_text for keyword in keywords):
            timestamp_items.append((key, value))

    return timestamp_items


def print_packet_keys(frame_idx, packet):
    """
    打印单帧 packet 的所有一级 key。
    """
    if not isinstance(packet, dict):
        print(f"Frame {frame_idx}: packet type={type(packet).__name__}, not a dict")
        return

    print(f"Frame {frame_idx} packet keys: {list(packet.keys())}")


def print_timestamp_items_for_frame(frame_idx, packet, keywords=TIMESTAMP_KEYWORDS):
    """
    打印单帧中疑似 timestamp / time 的字段。
    """
    timestamp_items = get_timestamp_items(packet, keywords=keywords)

    if not timestamp_items:
        print(f"Frame {frame_idx}: no timestamp-like keys matched {keywords}")
        return

    for key, value in timestamp_items:
        print(f"Frame {frame_idx}: {key} = {format_short_value(value)}")


def print_timestamp_report(data_list, keywords=TIMESTAMP_KEYWORDS, max_examples=5):
    """
    汇总所有 packet 中疑似 timestamp / time 的字段，并打印少量样例。
    """
    print("\n========== timestamp / time key report ==========")

    if not data_list:
        print("No packets loaded.")
        return

    all_keys = []

    for packet in data_list:
        if isinstance(packet, dict):
            all_keys.extend(packet.keys())

    unique_keys = sorted({str(key) for key in all_keys})
    matched_keys = [key for key in unique_keys if any(keyword in key.lower() for keyword in keywords)]

    first_packet_keys = list(data_list[0].keys()) if isinstance(data_list[0], dict) else type(data_list[0]).__name__
    print(f"First packet keys: {first_packet_keys}")
    print(f"Timestamp-like keys matched by {keywords}: {matched_keys}")

    if not matched_keys:
        print("No timestamp-like keys found. If timestamp is nested, inspect the full first packet manually.")
        return

    examples_printed = 0

    for frame_idx, packet in enumerate(data_list):
        timestamp_items = get_timestamp_items(packet, keywords=keywords)

        if not timestamp_items:
            continue

        for key, value in timestamp_items:
            print(f"Example {examples_printed + 1}: Frame {frame_idx}, {key} = {format_short_value(value)}")
            examples_printed += 1

            if examples_printed >= max_examples:
                return


def numeric_scalar_or_none(value):
    """
    尝试把 timestamp 字段转成 float 标量。
    """
    try:
        arr = np.asarray(value)

        if arr.ndim == 0:
            return float(arr.item())

        if arr.size == 1:
            return float(arr.reshape(-1)[0])

    except (TypeError, ValueError):
        return None

    return None


def get_packet_timestamp(packet, timestamp_key=TIMESTAMP_KEY, keywords=TIMESTAMP_KEYWORDS):
    """
    从 packet 中取出可用于排序/裁剪的 timestamp。

    优先使用 TIMESTAMP_KEY='timestamp'；若不存在，再从 timestamp/time/ts
    这类字段中找第一个可转成 float 的值。
    """
    if not isinstance(packet, dict):
        return None

    if timestamp_key in packet:
        timestamp = numeric_scalar_or_none(packet[timestamp_key])

        if timestamp is not None:
            return timestamp

    for key, value in get_timestamp_items(packet, keywords=keywords):
        timestamp = numeric_scalar_or_none(value)

        if timestamp is not None:
            return timestamp

    return None


def format_unix_timestamp(timestamp):
    """
    把 Unix timestamp 秒数转成便于人工检查的本地时间字符串。
    """
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S.%f')
    except (OSError, OverflowError, ValueError):
        return "unavailable"


def trim_initial_frames_by_count(data_list, seconds=20.0, frame_rate=100):
    """
    在 timestamp 无法可靠裁剪时，按采样频率去除前 seconds 秒。
    """
    if seconds is None or seconds <= 0:
        return data_list

    if frame_rate is None or frame_rate <= 0:
        print("Frame-count trim skipped because frame_rate is unavailable or non-positive.")
        return data_list

    remove_count = int(round(float(seconds) * float(frame_rate)))
    remove_count = min(remove_count, len(data_list))
    trimmed = data_list[remove_count:]

    print("\n========== frame-count trim report ==========")
    print(f"Timestamp trim was not usable; removed first {remove_count} frames using frame_rate={frame_rate} Hz.")
    print(f"Frame-count trim result: {len(trimmed)} frames kept from {len(data_list)} original frames.")

    return trimmed


def trim_initial_seconds_by_timestamp(data_list, seconds=20.0, fallback_frame_rate=None):
    """
    在正式处理前去除起始 seconds 秒数据。

    优先按每帧 timestamp 裁剪；如果 timestamp 不随帧递增、范围不足，
    或按 timestamp 会删光数据，则退回按 frame_rate 裁剪。
    """
    if seconds is None or seconds <= 0:
        return data_list

    timestamp_records = []

    for frame_idx, packet in enumerate(data_list):
        timestamp = get_packet_timestamp(packet)

        if timestamp is not None:
            timestamp_records.append((frame_idx, timestamp))

    print("\n========== timestamp trim report ==========")

    if not timestamp_records:
        print("No numeric timestamp found.")
        if TRIM_FALLBACK_USE_FRAME_COUNT:
            return trim_initial_frames_by_count(data_list, seconds=seconds, frame_rate=fallback_frame_rate)
        print("No initial-second trimming applied.")
        return data_list

    first_timestamp_idx, first_timestamp = timestamp_records[0]
    last_timestamp_idx, last_timestamp = timestamp_records[-1]
    timestamp_span = last_timestamp - first_timestamp
    threshold = first_timestamp + float(seconds)

    print(f"First timestamp frame: {first_timestamp_idx}, timestamp={first_timestamp} ({format_unix_timestamp(first_timestamp)})")
    print(f"Last timestamp frame: {last_timestamp_idx}, timestamp={last_timestamp} ({format_unix_timestamp(last_timestamp)})")
    print(f"Timestamp span: {timestamp_span:.6f} seconds")
    print(f"Trim threshold: timestamp >= {threshold} ({format_unix_timestamp(threshold)})")

    timestamp_unusable = (not np.isfinite(timestamp_span)) or timestamp_span <= 1e-9

    if timestamp_unusable:
        print("Timestamp does not appear to increase across frames; it may be a file/session timestamp rather than per-frame time.")
        if TRIM_FALLBACK_USE_FRAME_COUNT:
            return trim_initial_frames_by_count(data_list, seconds=seconds, frame_rate=fallback_frame_rate)
        print("No initial-second trimming applied.")
        return data_list

    trimmed = []
    removed = 0
    missing_timestamp_kept = 0

    for packet in data_list:
        timestamp = get_packet_timestamp(packet)

        if timestamp is None:
            trimmed.append(packet)
            missing_timestamp_kept += 1
            continue

        if timestamp >= threshold:
            trimmed.append(packet)
        else:
            removed += 1

    print(f"Timestamp trim result: {removed} frames removed, {len(trimmed)} frames kept.")

    if missing_timestamp_kept > 0:
        print(f"Warning: {missing_timestamp_kept} frames without numeric timestamp were kept.")

    if len(trimmed) == 0 and TRIM_FALLBACK_USE_FRAME_COUNT:
        print("Timestamp trim removed all frames; falling back to frame-count trimming.")
        return trim_initial_frames_by_count(data_list, seconds=seconds, frame_rate=fallback_frame_rate)

    return trimmed


def build_dynamic_style_calib_endpoint_trajectory(
    optim, data_list, direct=0, smooth_window=3, apply_direct_rotation=True
):
    """
    从已经裁剪过的 data_list 中，使用每帧记录的 calib_data 重新计算 r，
    然后按动态展示中的 trailing window 规则对整条 r 曲线做滑动平均，
    最后提取平滑后曲线的末端点作为轨迹点。

    注意：
    - 这里不使用 packet['r_results']；
    - smooth_window=3 时，对第 0、1 帧不做完整三帧平均，保持启动阶段行为；
    - 输入的 data_list 应该已经经过 TRIM_INITIAL_SECONDS 裁剪。
    """
    smooth_window = int(smooth_window) if smooth_window is not None else 3
    smooth_window = max(smooth_window, 1)

    trajectory = []
    raw_trajectory = []
    original_frame_indices = []
    trajectory_timestamps = []
    # 与 data_list 一一对齐，保存每一帧在全序列顺序重构时得到的完整 r/T/N/B 或 r/T/N1/N2 结果。
    # 这样后续 Rank 1 或手动指定 pair 的绘图不会重新初始化跨帧状态。
    raw_frame_results = []
    curve_history = []
    unwrap_processor = make_diag_unwrap_processor_if_enabled()
    angle0_processor = make_diag_angle0_processor_if_enabled()

    for frame_idx, packet in enumerate(data_list):
        original_idx = packet.get('_diag_original_frame_idx', frame_idx) if isinstance(packet, dict) else frame_idx

        result = reconstruct_packet_tnb_for_plot(
            optim=optim, packet=packet, direct=direct, apply_direct_rotation=apply_direct_rotation,
            use_packet_r_results=False, allow_shape_only=False,
            source_label=f"trajectory_dynamic_style_calib_reconstruction_{get_frame_vector_title(DIAG_RECON_FRAME_RULE)}",
            unwrap_processor=unwrap_processor, angle0_processor=angle0_processor,
            frame_context={"trimmed_frame_idx": frame_idx, "original_frame_idx": original_idx, "source": "endpoint_trajectory"}
        )

        if result is None:
            raw_frame_results.append(None)
            logging.warning(f"第 {original_idx} 帧 calib_data 重构失败，跳过该帧的 calib 重构末端轨迹点。")
            curve_history.clear()
            if unwrap_processor is not None and hasattr(unwrap_processor, "reset"):
                unwrap_processor.reset()
            if angle0_processor is not None and hasattr(angle0_processor, "reset"):
                angle0_processor.reset()
            continue
        
        raw_r = np.asarray(result["r"], dtype=np.float64)

        if raw_r.ndim != 2 or raw_r.shape[1] != 3:
            raw_frame_results.append(None)
            logging.warning(f"第 {original_idx} 帧 calib_data 重构 r 维度异常: {raw_r.shape}")
            curve_history.clear()
            if unwrap_processor is not None and hasattr(unwrap_processor, "reset"):
                unwrap_processor.reset()
            if angle0_processor is not None and hasattr(angle0_processor, "reset"):
                angle0_processor.reset()
            continue

        raw_frame_results.append(result)
        raw_trajectory.append(raw_r[-1, :].copy())
        curve_history.append(raw_r.copy())

        if len(curve_history) > smooth_window:
            curve_history.pop(0)

        if len(curve_history) == smooth_window:
            r_stack = np.asarray(curve_history, dtype=np.float64)

            if all(curve.shape == raw_r.shape for curve in curve_history):
                displayed_r = np.mean(r_stack, axis=0)
            else:
                logging.warning(f"第 {original_idx} 帧滑动平均窗口内 r 维度不一致，使用当前 raw_r。")
                displayed_r = raw_r
        else:
            displayed_r = raw_r

        trajectory.append(displayed_r[-1, :])
        original_frame_indices.append(original_idx)
        trajectory_timestamps.append(get_packet_timestamp(packet))

    return trajectory, raw_trajectory, original_frame_indices, trajectory_timestamps, raw_frame_results



def build_curve_frames_from_reconstruction_results(
    reconstruction_results, use_dynamic_smoothing=True, smooth_window=3, stride=1, max_frames=None
):
    """
    从 build_dynamic_style_calib_endpoint_trajectory 返回的 full-sequence 重构结果中提取完整 r 曲线，
    并可选地按动态展示中的 trailing window 规则对整条 r 曲线做滑动平均。

    输入：
    - reconstruction_results: 每帧的重构结果列表；有效元素应包含 result["r"]；
    - use_dynamic_smoothing: True 时输出滑动平均后的显示曲线；False 时输出 raw r；
    - smooth_window: 滑动平均窗口大小；
    - stride: 每隔多少帧取一帧，用于快速生成预览视频；
    - max_frames: 最多输出多少帧，None 表示不限制。

    输出：
    - video_frames: list[np.ndarray]，每个元素形状为 (num_points, 3)；
    - kept_indices: video_frames 对应的原 reconstruction_results 下标。
    """
    stride = max(int(stride), 1)
    smooth_window = max(int(smooth_window), 1)

    video_frames = []
    kept_indices = []
    curve_history = []

    for frame_idx, result in enumerate(reconstruction_results):
        if result is None or "r" not in result:
            curve_history.clear()
            continue

        raw_r = np.asarray(result["r"], dtype=np.float64)

        if raw_r.ndim != 2 or raw_r.shape[1] != 3 or not np.all(np.isfinite(raw_r)):
            curve_history.clear()
            continue

        curve_history.append(raw_r.copy())

        if len(curve_history) > smooth_window:
            curve_history.pop(0)

        if use_dynamic_smoothing and len(curve_history) == smooth_window and all(curve.shape == raw_r.shape for curve in curve_history):
            frame_r = np.mean(np.asarray(curve_history, dtype=np.float64), axis=0)
        else:
            frame_r = raw_r

        if frame_idx % stride != 0:
            continue

        video_frames.append(frame_r.copy())
        kept_indices.append(frame_idx)

        if max_frames is not None and len(video_frames) >= int(max_frames):
            break

    return video_frames, kept_indices


def generate_3d_shape_video_from_frames(
    frames_data, output_filename="calib_shape_reconstructed_shape_demo.mp4", fps=30,
    writer="ffmpeg", dpi=150, title="Calib-shape reconstructed 3D Shape Evolution", elev=20, azim=-60
):
    """
    将已经重构出的三维曲线序列导出为视频。

    与 pickle_3d_video.py 的区别：
    - 不再从 pkl 读取 packet['r_results']；
    - 直接使用 diagnostics 当前流程从 calib_data / calib_shape='dynamic' 重构出的 r 坐标；
    - 坐标轴范围仍按所有帧的全局范围固定，避免视频播放时视角跳动。
    """
    valid_frames = []

    for frame_idx, pts in enumerate(frames_data):
        pts = np.asarray(pts, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 3 or not np.all(np.isfinite(pts)):
            print(f"Video frame {frame_idx} skipped because its shape/value is invalid: shape={pts.shape}")
            continue

        valid_frames.append(pts)

    num_frames = len(valid_frames)
    print(f"准备使用 calib_data / calib_shape 重构坐标生成视频：有效帧数 = {num_frames}")

    if num_frames == 0:
        print("未找到可用于生成视频的有效重构坐标，视频生成跳过。")
        return False

    all_points = np.vstack(valid_frames)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    def expand_axis_limits(v_min, v_max, margin_ratio=0.05):
        if np.isclose(v_min, v_max):
            margin = 0.01 if np.isclose(v_min, 0.0) else abs(v_min) * margin_ratio
        else:
            margin = (v_max - v_min) * margin_ratio
        return v_min - margin, v_max + margin

    x_min, x_max = expand_axis_limits(x_min, x_max)
    y_min, y_max = expand_axis_limits(y_min, y_max)
    z_min, z_max = expand_axis_limits(z_min, z_max)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(title)

    line, = ax.plot([], [], [], lw=2, color='b')
    endpoint = ax.scatter([], [], [], color='red', s=20)
    frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame_idx):
        pts = valid_frames[frame_idx]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        endpoint._offsets3d = ([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]])
        frame_text.set_text(f"Frame {frame_idx + 1}/{num_frames}")
        return line, endpoint, frame_text

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=False)

    try:
        print(f"正在编码视频并保存至: {output_filename}")
        ani.save(output_filename, writer=writer, fps=fps, dpi=dpi)
        print("calib_data / calib_shape 重构坐标视频生成完成！")
        return True
    except Exception as exc:
        print(f"视频生成失败：{exc}")
        print("如果使用 writer='ffmpeg'，请确认系统中已经安装 ffmpeg，或改用可用 writer。")
        return False
    finally:
        plt.close(fig)

def plot_endpoint_trajectory_3d(trajectory_arr, title, path_label='Trajectory Path'):
    """
    画 3D 末端轨迹图。
    """
    trajectory_arr = np.asarray(trajectory_arr, dtype=np.float64)

    if trajectory_arr.ndim != 2 or trajectory_arr.shape[1] != 3 or len(trajectory_arr) == 0:
        raise ValueError(f"Invalid trajectory_arr shape: {trajectory_arr.shape}")

    x_coords = trajectory_arr[:, 0]
    y_coords = trajectory_arr[:, 1]
    z_coords = trajectory_arr[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_coords, y_coords, z_coords, label=path_label, color='blue', linewidth=1.5, alpha=0.7)
    ax.scatter(x_coords, y_coords, z_coords, color='red', s=10, label='End Points')
    ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=50, label='Start Point', marker='^')
    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='black', s=50, label='End Point', marker='s')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.show()


def generate_3d_cycloid(theta_arr, start_pt, r_vec, s_vec, radius):
    """
    生成空间平面摆线点。

    这里沿用旧 DynamicAna 脚本中的定义，但调用方传入的是
    calib_data 显式重构并经过动态展示式滑动平均后的末端轨迹。
    """
    theta_arr = np.asarray(theta_arr, dtype=np.float64)
    start_pt = np.asarray(start_pt, dtype=np.float64)
    r_vec = np.asarray(r_vec, dtype=np.float64)
    s_vec = np.asarray(s_vec, dtype=np.float64)

    points = np.zeros((len(theta_arr), 3), dtype=np.float64)

    for i, theta in enumerate(theta_arr):
        roll_disp = radius * (theta - np.sin(theta))
        swing_disp = radius * (1 - np.cos(theta))
        points[i] = start_pt + roll_disp * r_vec + swing_disp * s_vec

    return points


def plot_calib_reconstructed_cycloid_and_deviation(
    trajectory_arr, original_frame_indices=None, trajectory_timestamps=None, annotate_timestamps=None,
    timestamp_match_max_diff=None, radius=0.4, marker_step=100, reverse_for_return=True
):
    """
    用 calib_data 重构出来的末端轨迹进行 3D 空间摆线拟合和逐帧偏差分析。

    输入 trajectory_arr 应为 build_dynamic_style_calib_endpoint_trajectory 返回的轨迹，
    即 calib_data -> r 重构后，再按动态展示式 trailing window 做滑动平均得到的末端轨迹。
    本函数不读取也不使用 packet['r_results']。

    reverse_for_return=True 时，与旧 recorded-data 分析脚本保持一致：
    假设 trim 掉前 20s 后剩余轨迹主要是回程，因此用逆序轨迹的起点和最远点
    来解算理论摆线；但 frame_deviations 仍按原始 trajectory_arr 顺序逐帧计算。
    """
    trajectory_arr_full = np.asarray(trajectory_arr, dtype=np.float64)

    if trajectory_arr_full.ndim != 2 or trajectory_arr_full.shape[1] != 3 or len(trajectory_arr_full) < 2:
        raise ValueError(f"Invalid trajectory_arr shape for cycloid fitting: {trajectory_arr_full.shape}")

    # 保留原有每 10 帧取 1 帧的绘图/拟合逻辑，但同步下采样所有与轨迹点一一对应的元数据。
    subsample_indices = np.arange(0, len(trajectory_arr_full), 10, dtype=int)
    trajectory_arr = trajectory_arr_full[subsample_indices]

    def downsample_metadata(metadata, name):
        if metadata is None:
            return None

        metadata_arr = np.asarray(metadata, dtype=object)

        if len(metadata_arr) != len(trajectory_arr_full):
            logging.warning(
                f"{name} length mismatch: expected {len(trajectory_arr_full)}, got {len(metadata_arr)}; "
                f"related cycloid labels will fall back when needed."
            )
            return None

        return metadata_arr[subsample_indices]

    original_frame_indices = downsample_metadata(original_frame_indices, "original_frame_indices")
    trajectory_timestamps = downsample_metadata(trajectory_timestamps, "trajectory_timestamps")

    if trajectory_arr.ndim != 2 or trajectory_arr.shape[1] != 3 or len(trajectory_arr) < 2:
        raise ValueError(f"Invalid trajectory_arr shape for cycloid fitting: {trajectory_arr.shape}")

    if not np.all(np.isfinite(trajectory_arr)):
        raise ValueError("trajectory_arr contains NaN or inf; cycloid fitting skipped.")

    x_coords = trajectory_arr[:, 0]
    min_x_idx = int(np.argmin(x_coords))

    if reverse_for_return:
        # 与旧 recorded-data 脚本一致：前 20s 去程被删后，剩余数据视为回程。
        # 逆序只用于确定理论摆线的起点与最远点；偏差和绘图仍保留原始帧顺序。
        fit_trajectory = trajectory_arr[::-1]
        fit_min_x_idx = int(np.argmin(fit_trajectory[:, 0]))
        P_start = fit_trajectory[0]
        P_far = fit_trajectory[fit_min_x_idx]
        fit_mode_text = "reverse-return fitting"
    else:
        fit_trajectory = trajectory_arr
        fit_min_x_idx = min_x_idx
        P_start = fit_trajectory[0]
        P_far = fit_trajectory[fit_min_x_idx]
        fit_mode_text = "forward fitting"

    print("最远端的 calib_data 重构末端坐标为：", P_far)
    logging.info(f"开始进行 calib_data 重构轨迹的动态空间平面摆线验证 ({fit_mode_text})...")

    u_roll = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    v_sf = P_far - P_start
    v_yz = np.array([0.0, v_sf[1], v_sf[2]], dtype=np.float64)
    norm_yz = np.linalg.norm(v_yz)

    if norm_yz < 1e-6:
        logging.warning("calib_data 重构轨迹起点与最远点的 YZ 偏差极小，退化为理想 XY 平面")
        u_swing = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        u_swing = v_yz / norm_yz

    logging.info(f"calib_data 重构轨迹解算起点坐标: {P_start}")
    logging.info(f"calib_data 重构轨迹解算摆动方向向量 (YZ平面): {u_swing}")

    max_roll_dist = abs(P_far[0] - P_start[0])
    theta_max_guess = (max_roll_dist / float(radius)) + 1.0
    theta_dense = np.linspace(0, theta_max_guess, 3000)

    theoretical_curve = generate_3d_cycloid(theta_dense, P_start, u_roll, u_swing, float(radius))

    distances = cdist(trajectory_arr, theoretical_curve)
    frame_deviations = np.min(distances, axis=1)

    max_dev = float(np.max(frame_deviations))
    mean_dev = float(np.mean(frame_deviations))
    max_dev_idx = int(np.argmax(frame_deviations))

    if original_frame_indices is not None and len(original_frame_indices) == len(trajectory_arr):
        max_dev_original_idx = original_frame_indices[max_dev_idx]
        min_x_original_idx = original_frame_indices[min_x_idx]
        logging.info(
            f"calib_data 重构轨迹最大空间偏差: {max_dev:.4f} m "
            f"(trajectory_idx={max_dev_idx}, original_frame={max_dev_original_idx})"
        )
        logging.info(
            f"calib_data 重构轨迹平均空间偏差: {mean_dev:.4f} m; "
            f"min_x trajectory_idx={min_x_idx}, original_frame={min_x_original_idx}"
        )
    else:
        logging.info(f"calib_data 重构轨迹最大空间偏差: {max_dev:.4f} m (trajectory_idx={max_dev_idx})")
        logging.info(f"calib_data 重构轨迹平均空间偏差: {mean_dev:.4f} m")

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')

    scatter = ax1.scatter(
        trajectory_arr[:, 0], trajectory_arr[:, 1], trajectory_arr[:, 2],
        c=np.arange(len(trajectory_arr)), cmap='viridis', s=15,
        label='Calib-data Reconstructed Trajectory'
    )

    ax1.plot(
        theoretical_curve[:, 0], theoretical_curve[:, 1], theoretical_curve[:, 2],
        color='black', linewidth=3, label='Theoretical 3D Cycloid'
    )

    marker_step = max(int(marker_step), 1)
    marker_indices = np.arange(0, len(trajectory_arr), marker_step)

    if len(marker_indices) == 0:
        marker_indices = np.array([0], dtype=int)

    marker_points = trajectory_arr[marker_indices]

    ax1.scatter(
        marker_points[:, 0], marker_points[:, 1], marker_points[:, 2],
        color='red', marker='^', s=60, edgecolor='black', zorder=10,
        label=f'{marker_step}-Trajectory-Index Markers'
    )

    for idx, point in zip(marker_indices, marker_points):
        if original_frame_indices is not None and len(original_frame_indices) == len(trajectory_arr):
            frame_text = f"F{original_frame_indices[int(idx)]}"
        else:
            frame_text = f"F{idx}"
        ax1.text(point[0], point[1], point[2], f' {frame_text}', color='darkred', fontsize=9, zorder=11)

    timestamp_annotation_records = []
    annotate_timestamps = [] if annotate_timestamps is None else list(annotate_timestamps)

    if annotate_timestamps:
        if trajectory_timestamps is None or len(trajectory_timestamps) != len(trajectory_arr):
            logging.warning(
                "CYCLOID_ANNOTATE_TIMESTAMPS is set, but trajectory_timestamps is unavailable "
                "or not aligned with the downsampled trajectory; timestamp annotation skipped."
            )
        else:
            timestamp_values = np.array([numeric_scalar_or_none(ts) for ts in trajectory_timestamps], dtype=np.float64)
            valid_timestamp_mask = np.isfinite(timestamp_values)

            if not np.any(valid_timestamp_mask):
                logging.warning(
                    "CYCLOID_ANNOTATE_TIMESTAMPS is set, but no finite trajectory timestamps are available; "
                    "timestamp annotation skipped."
                )
            else:
                for target_timestamp in annotate_timestamps:
                    target_timestamp_value = numeric_scalar_or_none(target_timestamp)

                    if target_timestamp_value is None or not np.isfinite(target_timestamp_value):
                        logging.warning(f"Invalid target timestamp skipped: {target_timestamp!r}")
                        continue

                    timestamp_diffs = np.abs(timestamp_values - target_timestamp_value)
                    timestamp_diffs[~valid_timestamp_mask] = np.inf
                    nearest_idx = int(np.argmin(timestamp_diffs))
                    nearest_diff = float(timestamp_diffs[nearest_idx])

                    if timestamp_match_max_diff is not None and nearest_diff > float(timestamp_match_max_diff):
                        logging.warning(
                            f"Target timestamp {target_timestamp_value} matched to nearest plotted point with "
                            f"diff={nearest_diff:.6f}s, exceeding CYCLOID_TIMESTAMP_MATCH_MAX_DIFF={timestamp_match_max_diff}."
                        )

                    point = trajectory_arr[nearest_idx]
                    original_frame = (
                        original_frame_indices[nearest_idx]
                        if original_frame_indices is not None and len(original_frame_indices) == len(trajectory_arr)
                        else subsample_indices[nearest_idx]
                    )
                    actual_timestamp = timestamp_values[nearest_idx]

                    timestamp_annotation_records.append({
                        "target_timestamp": float(target_timestamp_value),
                        "matched_timestamp": float(actual_timestamp),
                        "timestamp_diff": nearest_diff,
                        "trajectory_idx_after_subsample": nearest_idx,
                        "trajectory_idx_before_subsample": int(subsample_indices[nearest_idx]),
                        "original_frame": original_frame,
                    })

                    label = (
                        f"TS→F{original_frame}\n"
                        f"Δt={nearest_diff:.3f}s"
                    )
                    ax1.scatter(
                        point[0], point[1], point[2], color='cyan', marker='*', s=160,
                        edgecolor='black', zorder=20, label='_nolegend_'
                    )
                    ax1.text(
                        point[0], point[1], point[2], f' {label}',
                        color='darkcyan', fontsize=9, zorder=21
                    )

                if timestamp_annotation_records:
                    ax1.scatter([], [], [], color='cyan', marker='*', s=160, edgecolor='black', label='Specified Timestamp Markers')
                    print("\n========== cycloid specified timestamp annotation report ==========")

                    for record in timestamp_annotation_records:
                        print(
                            f"target_timestamp={record['target_timestamp']} ({format_unix_timestamp(record['target_timestamp'])}) -> "
                            f"matched_timestamp={record['matched_timestamp']} ({format_unix_timestamp(record['matched_timestamp'])}), "
                            f"diff={record['timestamp_diff']:.6f}s, "
                            f"trajectory_idx_before_subsample={record['trajectory_idx_before_subsample']}, "
                            f"trajectory_idx_after_subsample={record['trajectory_idx_after_subsample']}, "
                            f"original_frame={record['original_frame']}"
                        )

    ax1.scatter(*P_start, color='green', s=50, label='Start Point', zorder=5)
    ax1.scatter(*P_far, color='purple', s=50, label='Furthest Point', zorder=5)

    ax1.set_title('3D Spatial Cycloid Fitting: calib_data reconstructed trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    fig.colorbar(scatter, ax=ax1, label='Calib trajectory index', shrink=0.5)

    ax2 = fig.add_subplot(122)
    frames = np.arange(len(frame_deviations))

    ax2.plot(frames, frame_deviations, color='red', linewidth=1.5, label='Spatial Deviation')
    ax2.axvline(x=min_x_idx, color='gray', linestyle='--', label='Turnaround Index (min_x)')
    ax2.axhline(y=mean_dev, color='blue', linestyle='-.', label=f'Mean Dev: {mean_dev:.4f}m')

    ax2.set_title('Frame-by-Frame Spatial Deviation to Theoretical Curve: calib_data reconstructed')
    ax2.set_xlabel('Calib trajectory index after trimming/skipped invalid frames')
    ax2.set_ylabel('Deviation Distance (m)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return {
        "theoretical_curve": theoretical_curve,
        "frame_deviations": frame_deviations,
        "min_x_idx": min_x_idx,
        "max_dev_idx": max_dev_idx,
        "mean_dev": mean_dev,
        "max_dev": max_dev,
        "subsample_indices": subsample_indices,
        "timestamp_annotation_records": timestamp_annotation_records,
    }


def plot_calib_reconstructed_trajectory_jitter(trajectory_arr, frame_dt=1.0):
    """
    对 calib_data 重构出来的末端轨迹绘制 jitter / jerk / FFT 分析。

    spatial_jitter 定义为：每一帧 calib_data 重构末端点与 Savitzky-Golay 平滑轨迹
    之间的三维欧氏距离。本函数不读取也不使用 packet['r_results']。
    """
    trajectory_arr = np.asarray(trajectory_arr, dtype=np.float64)

    if trajectory_arr.ndim != 2 or trajectory_arr.shape[1] != 3:
        raise ValueError(f"Invalid trajectory_arr shape for jitter analysis: {trajectory_arr.shape}")

    n_frames = len(trajectory_arr)

    if n_frames < 5:
        raise ValueError("calib_data 重构轨迹帧数过少，无法稳定计算 jitter。")

    if not np.all(np.isfinite(trajectory_arr)):
        raise ValueError("trajectory_arr contains NaN or inf; jitter analysis skipped.")

    window_length = min(15, n_frames if n_frames % 2 == 1 else n_frames - 1)

    if window_length < 5:
        window_length = 5 if n_frames >= 5 else n_frames

    # Savitzky-Golay 的 window_length 必须是奇数，且 polyorder < window_length。
    if window_length % 2 == 0:
        window_length -= 1

    polyorder = min(3, window_length - 1)
    smoothed_traj = savgol_filter(trajectory_arr, window_length=window_length, polyorder=polyorder, axis=0)
    spatial_jitter = np.linalg.norm(trajectory_arr - smoothed_traj, axis=1)

    print(
        f"calib_data 重构轨迹平均空间抖动: {np.mean(spatial_jitter):.4f} m, "
        f"最大空间抖动: {np.max(spatial_jitter):.4f} m"
    )

    velocity = np.gradient(trajectory_arr, frame_dt, axis=0)
    acceleration = np.gradient(velocity, frame_dt, axis=0)
    jerk = np.gradient(acceleration, frame_dt, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    N = len(spatial_jitter)
    yf = fft(spatial_jitter)
    xf = fftfreq(N, frame_dt)[:N // 2]
    power_spectrum = 2.0 / N * np.abs(yf[0:N // 2])

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Trajectory Jitter & Vibration Analysis: calib_data reconstructed trajectory', fontsize=16, fontweight='bold')

    frames = np.arange(len(trajectory_arr))

    ax1 = fig.add_subplot(311)
    ax1.plot(frames, spatial_jitter * 1000, color='coral', linewidth=1.5, label='Spatial Jitter')
    ax1.axhline(
        y=np.mean(spatial_jitter) * 1000, color='red', linestyle='--',
        label=f'Mean Jitter: {np.mean(spatial_jitter) * 1000:.2f} mm'
    )
    ax1.set_ylabel('Spatial Jitter (mm)')
    ax1.set_title('Deviation from Smoothed Calib-Data-Reconstructed End-Effector Trajectory')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(312)
    ax2.plot(frames, jerk_magnitude, color='teal', linewidth=1.5, label='Jerk Magnitude')
    jerk_threshold = np.mean(jerk_magnitude) + 2 * np.std(jerk_magnitude)
    ax2.axhline(y=jerk_threshold, color='orange', linestyle='-.', label='+2 Std Dev Threshold')
    ax2.set_ylabel('Jerk (m/frame³)')
    ax2.set_title('Kinematic Domain: Jerk')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    ax3 = fig.add_subplot(313)
    ax3.plot(xf[1:], power_spectrum[1:], color='purple', linewidth=1.5, label='Jitter Power Spectrum')
    ax3.set_xlabel('Frequency (cycles/frame)' if frame_dt == 1.0 else 'Frequency (Hz)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Frequency Domain: FFT of Spatial Jitter')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return spatial_jitter

def resolve_specified_adjacent_frame_pair(data_list, specified_pair, pair_mode='original'):
    """
    将用户指定的相邻帧对转换为当前 data_list 中的下标。

    输入：
    - specified_pair: None 或 (prev_frame, curr_frame)
    - pair_mode:
        - 'original': specified_pair 使用原始 pkl 帧号；
        - 'trimmed': specified_pair 使用裁剪后的 data_list 下标。

    输出：
    - None：未指定或无法解析；
    - (prev_idx, curr_idx)：当前 data_list 中的下标。
    """
    if specified_pair is None:
        return None

    if not isinstance(specified_pair, (tuple, list)) or len(specified_pair) != 2:
        print(f"Invalid SPECIFIED_ADJACENT_FRAME_PAIR={specified_pair}; expected None or a pair like (100, 101).")
        return None

    try:
        first = int(specified_pair[0])
        second = int(specified_pair[1])
    except (TypeError, ValueError):
        print(f"Invalid SPECIFIED_ADJACENT_FRAME_PAIR={specified_pair}; frame numbers must be integers.")
        return None

    if second != first + 1:
        print(
            f"Invalid SPECIFIED_ADJACENT_FRAME_PAIR={specified_pair}; "
            "the two frames must be adjacent and ordered as (prev_frame, curr_frame)."
        )
        return None

    pair_mode = str(pair_mode).lower()

    if pair_mode == 'trimmed':
        prev_idx, curr_idx = first, second

        if prev_idx < 0 or curr_idx >= len(data_list):
            print(
                f"Specified trimmed frame pair {specified_pair} is out of range after trimming. "
                f"Valid trimmed frame range is 0 to {len(data_list) - 1}."
            )
            return None

        return prev_idx, curr_idx

    if pair_mode == 'original':
        original_to_trimmed = {}

        for trimmed_idx, packet in enumerate(data_list):
            if isinstance(packet, dict):
                original_idx = packet.get('_diag_original_frame_idx', trimmed_idx)
            else:
                original_idx = trimmed_idx

            original_to_trimmed[int(original_idx)] = trimmed_idx

        if first not in original_to_trimmed or second not in original_to_trimmed:
            print(
                f"Specified original frame pair {specified_pair} is not available after trimming. "
                "If these frames were removed by TRIM_INITIAL_SECONDS, lower TRIM_INITIAL_SECONDS or use a later pair."
            )
            return None

        return original_to_trimmed[first], original_to_trimmed[second]

    print(f"Unknown SPECIFIED_FRAME_PAIR_MODE={pair_mode!r}; expected 'original' or 'trimmed'.")
    return None



def normalise_diag_frame_rule(frame_rule):
    """
    将诊断重构标架规则规范化为 'TNB' 或 'TNN'。
    """
    if frame_rule is None:
        frame_rule = DIAG_RECON_FRAME_RULE

    frame_rule = str(frame_rule).upper()

    if frame_rule not in {'TNB', 'TNN'}:
        print(f"Unknown diagnostic frame_rule={frame_rule!r}; fallback to 'TNB'.")
        return 'TNB'

    return frame_rule


def get_frame_vector_labels(frame_rule=None):
    """
    返回当前标架下三组向量的显示标签。
    - TNB: T / N / B
    - TNN: T / N1 / N2
    """
    frame_rule = normalise_diag_frame_rule(frame_rule)
    return ('T', 'N', 'B') if frame_rule == 'TNB' else ('T', 'N1', 'N2')


def get_frame_vector_title(frame_rule=None):
    """
    返回适合用于图标题 / 日志的标架向量名称。
    """
    return '/'.join(get_frame_vector_labels(frame_rule))






def reconstruct_packet_tnb_for_plot(
    optim, packet, direct=0, apply_direct_rotation=True, verbose=False,
    use_packet_r_results=False, allow_shape_only=False, source_label="calib_data", frame_rule=None,
    unwrap_processor=None, angle0_processor=None, frame_context=None
):
    """
    diagnostics 专用：显式展开 calib_data -> angle/angle_unwrap -> r/T/N/B 的重构流程。

    与 Reconstructor._reconstruct_frame 的主要差别：
    - 不调用 _reconstruct_frame；
    - angle、angle_unwrap_raw、angle_unwrap 等都保存在返回字典的 intermediates 中；
    - 可通过 angle0_processor 在 zero_init / angle[0]=angle[1] 前修正首点 raw angle；
    - 可通过 unwrap_processor 在 angle_unwrap 进入滤波、插值、重构前做逐帧限制。

    参数：
    - frame_rule='TNB'：返回并绘制 T/N/B；
    - frame_rule='TNN'：返回并绘制 T/N1/N2；
    - use_packet_r_results=False：r 与标架全部来自 calib_data 重构；
    - use_packet_r_results / allow_shape_only 参数保留为兼容旧调用；当前精简版不再使用 packet['r_results']。
    - angle0_processor=None：不做首点 raw angle 跨帧修正；否则在 zero_init / angle[0]=angle[1] 前调用。
    - unwrap_processor=None：不限制 angle_unwrap；否则在 np.unwrap 后、moving average 前调用。
    """
    recon_frame_rule = normalise_diag_frame_rule(frame_rule)
    vector_title = get_frame_vector_title(recon_frame_rule)
    def shape_only_or_none(message):
        print(f"  {message}")
        return None

    if not isinstance(packet, dict):
        return shape_only_or_none(f"packet is not a dict; {vector_title} reconstruction skipped.")

    if 'calib_data' not in packet or packet['calib_data'] is None:
        return shape_only_or_none(f"frame missing calib_data; {vector_title} reconstruction skipped.")

    calib_arr = np.asarray(packet['calib_data'], dtype=np.float64)

    if calib_arr.ndim != 2:
        return shape_only_or_none(f"calib_data is not 2D: shape={calib_arr.shape}; {vector_title} reconstruction skipped.")

    if not np.all(np.isfinite(calib_arr)):
        return shape_only_or_none(f"calib_data contains NaN or inf; {vector_title} reconstruction skipped.")

    required_methods = [
        '_Strain_2_Cur', '_moving_average_scipy', '_Kalman_filter',
        '_KT_Hermit_interp_tang', '_F_S_frame', '_parallel_frame'
    ]

    for method_name in required_methods:
        if not hasattr(optim, method_name):
            return shape_only_or_none(f"optim has no {method_name} method; {vector_title} reconstruction skipped.")

    weight = get_direct_value(optim.weights, direct)
    slope = get_direct_value(optim.slopes, direct)
    original_frame_rule = getattr(optim, "frame_rule", None)

    try:
        if original_frame_rule is not None:
            optim.frame_rule = recon_frame_rule

        # 1. 等价于 _reconstruct_frame 的第一步：从 calib_data 得到 curvature magnitude 与 angle。
        magnitude, angle = optim._Strain_2_Cur(
            keys=None, calib_shape='dynamic', calib_Data=calib_arr,
            weight=weight, slope=slope, verbose=verbose
        )

        magnitude = np.asarray(magnitude, dtype=np.float64)
        angle = np.asarray(angle, dtype=np.float64)
        # for i in range(len(magnitude)):
        #     if optim.samp_dist[i] < 0.5:
        #         magnitude[i] = 0.0
        #         angle_ori = angle[i].copy()
        #         angle[i] = 0.0
        #         if optim.samp_dist[i+1] >= 0.5:
        #             for j in range(i+1, len(magnitude)):
        #                 angle[j] -= angle_ori


        # 2. 在插点 / angle[0]=angle[1] 之前，对近零首点曲率的 raw angle[0] 做跨帧保持。
        if angle0_processor is not None:
            angle = angle0_processor(
                magnitude=magnitude, angle=angle, frame_context=frame_context
            )

        # 3. 保持与 _reconstruct_frame 一致的起点处理。
        if getattr(optim, 'zero_init', False):
            magnitude = np.insert(magnitude, 0, magnitude[0])
            angle = np.insert(angle, 0, angle[0])
        elif len(angle) > 1:
            angle[0] = angle[1]

        # 4. 显式生成 angle_unwrap。逐帧限制必须放在这里，才能影响后续重构。
        discont = getattr(optim, "discont", getattr(args, "discont", np.pi))
        angle_unwrap_raw = np.unwrap(angle, discont=discont)
        angle_unwrap = angle_unwrap_raw.copy()

        if unwrap_processor is not None:
            angle_unwrap = unwrap_processor(
                angle=angle,
                angle_unwrap=angle_unwrap,
                angle_unwrap_raw=angle_unwrap_raw,
                frame_context=frame_context
            )

        angle_unwrap = np.asarray(angle_unwrap, dtype=np.float64)

        # 5. 与 _reconstruct_frame 保持一致：angle_unwrap -> filter/interp；magnitude -> Kalman/interp。
        ang_filter = optim._moving_average_scipy(data=angle_unwrap, window_size=optim.move_filter, mode='reflect')
        cur_kal = optim._Kalman_filter(
            data=magnitude,
            Rk=optim.kalman_params['magnitude']['Rk'],
            Bk=optim.kalman_params['magnitude']['Bk']
        )

        context_str = f"calib_shape: dynamic, keys(radius/end_radius): None"

        if not np.all(np.isfinite(ang_filter)):
            raise ValueError(f"[{context_str}] ang_filter contains NaN or inf. Check explicit angle_unwrap path.")
        if not np.all(np.isfinite(cur_kal)):
            raise ValueError(f"[{context_str}] cur_kal contains NaN or inf. Check _Strain_2_Cur output.")

        ang_interp, interp_dist = optim._KT_Hermit_interp_tang(data=ang_filter)
        # ang_interp, interp_dist = optim._KT_Hermit_interp_tang(data=angle_unwrap)
        cur_interp, interp_dist = optim._KT_Hermit_interp_tang(data=cur_kal)
        interp_ss = np.diff(interp_dist)

        init_T = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        init_N = np.array([0.0, np.cos(ang_interp[0]), np.sin(ang_interp[0])], dtype=np.float64)

        # 6. 与 _reconstruct_frame 后半段一致：根据 frame_rule 做 TNB 或 TNN 重构。
        if recon_frame_rule == 'TNB':
            ang_diff = np.diff(ang_interp)
            torsion = (ang_diff - 2 * np.pi * np.floor((ang_diff + np.pi) / (2 * np.pi))) / interp_ss
            r, T, N_like, B_like = optim._F_S_frame(kappa=cur_interp, tau=torsion, ss=interp_ss, init_T=init_T, init_N=init_N)
        elif recon_frame_rule == 'TNN':
            r, T, N_like, B_like = optim._parallel_frame(kappa=cur_interp, angles=ang_interp, ss=interp_ss, init_T=init_T, init_N=init_N)
        else:
            raise ValueError(f"Unknown recon_frame_rule={recon_frame_rule}")

    except Exception as e:
        return shape_only_or_none(f"failed to explicitly reconstruct {vector_title} for this frame: {e}")
    finally:
        if original_frame_rule is not None:
            optim.frame_rule = original_frame_rule

    r = np.asarray(r, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    N_like = np.asarray(N_like, dtype=np.float64)
    B_like = np.asarray(B_like, dtype=np.float64)

    if apply_direct_rotation and hasattr(optim, "_get_rot_x"):
        rot_matrix = np.asarray(optim._get_rot_x(direct=direct), dtype=np.float64)
        r = r @ rot_matrix.T
        T = T @ rot_matrix.T
        N_like = N_like @ rot_matrix.T
        B_like = B_like @ rot_matrix.T

    source = f"explicit_angle_calib_data_reconstructed_r_{vector_title}"

    intermediates = {
        "magnitude": magnitude.copy(),
        "angle": angle.copy(),
        "angle_unwrap_raw": angle_unwrap_raw.copy(),
        "angle_unwrap": angle_unwrap.copy(),
        "ang_filter": np.asarray(ang_filter, dtype=np.float64).copy(),
        "cur_kal": np.asarray(cur_kal, dtype=np.float64).copy(),
        "ang_interp": np.asarray(ang_interp, dtype=np.float64).copy(),
        "cur_interp": np.asarray(cur_interp, dtype=np.float64).copy(),
        "interp_dist": np.asarray(interp_dist, dtype=np.float64).copy(),
        "frame_context": frame_context,
        "angle0_held": False,
        "raw_angle0_original": None,
        "raw_angle0_replacement": None,
        "raw_magnitude0": None,
        "unwrap_limited": unwrap_processor is not None,
        "prev_unwrap_directs_used": None,
        "frame_unwrap_directs": None,
        "unwrap_changed_indices": None,
    }

    if angle0_processor is not None and hasattr(angle0_processor, "state"):
        angle0_state = angle0_processor.state
        intermediates["angle0_held"] = bool(angle0_state.get("last_replaced_angle0", False))
        intermediates["raw_angle0_original"] = angle0_state.get("last_original_angle0")
        intermediates["raw_angle0_replacement"] = angle0_state.get("last_used_prev_raw_angle0")
        intermediates["raw_magnitude0"] = angle0_state.get("last_magnitude0")

    if unwrap_processor is not None and hasattr(unwrap_processor, "state"):
        unwrap_state = unwrap_processor.state
        prev_unwrap_directs_used = unwrap_state.get("last_prev_unwrap_directs_used")
        frame_unwrap_directs = unwrap_state.get("last_frame_unwrap_directs")
        unwrap_changed_indices = unwrap_state.get("last_changed_indices")

        if prev_unwrap_directs_used is not None:
            intermediates["prev_unwrap_directs_used"] = np.asarray(prev_unwrap_directs_used, dtype=np.int64).copy()
        if frame_unwrap_directs is not None:
            intermediates["frame_unwrap_directs"] = np.asarray(frame_unwrap_directs, dtype=np.int64).copy()
        if unwrap_changed_indices is not None:
            intermediates["unwrap_changed_indices"] = np.asarray(unwrap_changed_indices, dtype=np.int64).copy()

    return {
        "cur_interp": cur_interp, "interp_dist": interp_dist, "r": r,
        "T": T, "N": N_like, "B": B_like,
        "frame_rule": recon_frame_rule, "vector_labels": get_frame_vector_labels(recon_frame_rule),
        "source": source_label or source,
        "intermediates": intermediates
    }

def plot_tnb_frame_pair(
    prev_result, curr_result, prev_frame_idx, curr_frame_idx, endpoint_distance=None,
    scale=1.0, arrow_length=0.03, visual_y_offset=TNB_PAIR_VISUAL_Y_OFFSET,
    source_label="TNB frame pair", endpoint_label="endpoint_distance"
):
    """
    在同一张图中绘制上一帧和当前帧的曲线及 T/N/B 标架。

    若 visual_y_offset > 0，会对两帧做纯显示用错位：
    - 上一帧 y -= visual_y_offset
    - 当前帧 y += visual_y_offset
    这样即使两帧真实坐标非常接近，也能在图上明确看到两条形状。
    """
    prev_r = np.asarray(prev_result["r"], dtype=np.float64)
    curr_r = np.asarray(curr_result["r"], dtype=np.float64)

    prev_display_r = prev_r.copy()
    curr_display_r = curr_r.copy()

    if visual_y_offset and visual_y_offset != 0:
        prev_display_r[:, 1] -= visual_y_offset
        curr_display_r[:, 1] += visual_y_offset

    plotted_end_distance = np.linalg.norm(curr_r[-1] - prev_r[-1])
    print(f"  {source_label} endpoint distance between plotted r curves: {plotted_end_distance:.6f} m")
    print(f"  visual_y_offset used only for plotting: {visual_y_offset:.6f} m")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    prev_label = f"Previous Curve F{prev_frame_idx}"
    curr_label = f"Current Curve F{curr_frame_idx}"

    if visual_y_offset and visual_y_offset != 0:
        prev_label += f" (y-{visual_y_offset:g}m)"
        curr_label += f" (y+{visual_y_offset:g}m)"

    ax.plot(prev_display_r[:, 0], prev_display_r[:, 1], prev_display_r[:, 2], label=prev_label, color='gray', linestyle='--', linewidth=2.0, alpha=0.95)
    ax.plot(curr_display_r[:, 0], curr_display_r[:, 1], curr_display_r[:, 2], label=curr_label, color='blue', linestyle='-', linewidth=2.2, alpha=0.95)

    ax.scatter(prev_display_r[0, 0], prev_display_r[0, 1], prev_display_r[0, 2], color='black', marker='^', s=45, label=f"F{prev_frame_idx} start")
    ax.scatter(prev_display_r[-1, 0], prev_display_r[-1, 1], prev_display_r[-1, 2], color='black', marker='s', s=45, label=f"F{prev_frame_idx} end")
    ax.scatter(curr_display_r[0, 0], curr_display_r[0, 1], curr_display_r[0, 2], color='red', marker='^', s=55, label=f"F{curr_frame_idx} start")
    ax.scatter(curr_display_r[-1, 0], curr_display_r[-1, 1], curr_display_r[-1, 2], color='red', marker='s', s=55, label=f"F{curr_frame_idx} end")

    ax.plot(
        [prev_display_r[-1, 0], curr_display_r[-1, 0]],
        [prev_display_r[-1, 1], curr_display_r[-1, 1]],
        [prev_display_r[-1, 2], curr_display_r[-1, 2]],
        color='black', linestyle=':', linewidth=1.5, label='Displayed endpoint gap'
    )

    def draw_frame_vectors(result, display_r, label_prefix, colors, alpha=1.0):
        vector_labels = result.get("vector_labels", get_frame_vector_labels(result.get("frame_rule", DIAG_RECON_FRAME_RULE)))
        vector_specs = [
            (vector_labels[0], result.get("T"), colors[0], arrow_length),
            (vector_labels[1], result.get("N"), colors[1], arrow_length * scale),
            (vector_labels[2], result.get("B"), colors[2], arrow_length * scale)
        ]
        vectors = []

        for vec_name, vec_value, color, length in vector_specs:
            if vec_value is None:
                continue

            vec_arr = np.asarray(vec_value, dtype=np.float64)

            if vec_arr.shape != display_r.shape:
                print(f"  skip {label_prefix} {vec_name}: vector shape {vec_arr.shape} does not match curve shape {display_r.shape}")
                continue

            vectors.append((vec_name, vec_arr, color, length))

        if not vectors:
            print(f"  no frame vectors available for {label_prefix}; drawing curve only.")
            return

        for i in range(len(display_r)):
            p = display_r[i]

            for vec_name, vec_arr, color, length in vectors:
                ax.quiver(
                    p[0], p[1], p[2], vec_arr[i, 0], vec_arr[i, 1], vec_arr[i, 2],
                    color=color, length=length, normalize=True, alpha=alpha,
                    label=f"{label_prefix} {vec_name}" if i == 0 else ""
                )

    draw_frame_vectors(prev_result, prev_display_r, f"F{prev_frame_idx}", colors=('orange', 'limegreen', 'violet'), alpha=0.55)
    draw_frame_vectors(curr_result, curr_display_r, f"F{curr_frame_idx}", colors=('red', 'green', 'purple'), alpha=0.95)

    all_points = np.vstack((prev_display_r, curr_display_r))
    mins = np.nanmin(all_points, axis=0)
    maxs = np.nanmax(all_points, axis=0)
    center = (mins + maxs) / 2
    half_range = max(np.max(maxs - mins) / 2, 0.05) * 1.25

    ax.set(
        xlim=(center[0] - half_range, center[0] + half_range),
        ylim=(center[1] - half_range, center[1] + half_range),
        zlim=(center[2] - half_range, center[2] + half_range)
    )

    ax.legend(loc='upper left')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = f"{source_label}: F{prev_frame_idx} -> F{curr_frame_idx}"

    if endpoint_distance is not None:
        title += f", {endpoint_label}={endpoint_distance:.6f} m"

    ax.set_title(title)
    ax.view_init(elev=30, azim=60)
    plt.show()




def plot_packet_tnb_for_diagnostic(
    optim, visualisor, packet, frame_idx, direct=0, endpoint_distance=None,
    scale=1.0, arrow_length=0.03, prev_packet=None, prev_frame_idx=None,
    data_list_for_dynamic=None, prev_trimmed_idx=None, curr_trimmed_idx=None,
    plot_dynamic_smooth=False, dynamic_smooth_window=3,
    precomputed_prev_result=None, precomputed_curr_result=None,
    require_precomputed_pair=False
):
    """
    对目标帧对只画 calib_data 显式重构得到的 r/T/N/B 或 r/T/N1/N2。

    已删除：
    - 使用记录曲线 + 重新计算 TNB 的混合图；
    - recorded/calib/dynamic 多来源距离对比打印；
    - dynamic-style 第三套 r/TNB 图。
    """
    print(f"\n---------- calib_data reconstructed r/{get_frame_vector_title(DIAG_RECON_FRAME_RULE)} plot for target frame pair ----------")

    if prev_packet is None or prev_frame_idx is None:
        unwrap_processor = make_diag_unwrap_processor_if_enabled()
        angle0_processor = make_diag_angle0_processor_if_enabled()
        curr_result = reconstruct_packet_tnb_for_plot(
            optim=optim, packet=packet, direct=direct, use_packet_r_results=False,
            allow_shape_only=False, source_label=f"calib_data_reconstructed_r_{get_frame_vector_title(DIAG_RECON_FRAME_RULE)}",
            unwrap_processor=unwrap_processor, angle0_processor=angle0_processor,
            frame_context={"original_frame_idx": frame_idx, "source": "calib_single_frame"}
        )

        if curr_result is None:
            return None

        if visualisor is None or not hasattr(visualisor, "plot_curve_with_tnb"):
            print("  visualisor.plot_curve_with_tnb is unavailable; TNB plot skipped.")
            return None

        dist_label = f"calib_data reconstructed frame {frame_idx}"
        if endpoint_distance is not None:
            dist_label += f", endpoint_distance={endpoint_distance:.6f} m"
        visualisor.plot_curve_with_tnb(dist=dist_label, r=curr_result["r"], T=curr_result["T"], N=curr_result["N"], B=curr_result["B"], scale=scale, arrow_length=arrow_length)
        return {"calib_data_reconstructed": {"curr": curr_result}}

    if precomputed_prev_result is not None and precomputed_curr_result is not None:
        calib_prev = precomputed_prev_result
        calib_curr = precomputed_curr_result
        print("  using full-sequence cached reconstruction results for this frame pair.")
    else:
        if require_precomputed_pair:
            print("  pair plot skipped because full-sequence cached reconstruction is unavailable for at least one frame.")
            return None

        # 按时间顺序 prev -> curr 调用；否则逐帧 unwrap 限制和首点 raw angle 保持会拿错上一帧。
        calib_unwrap_processor = make_diag_unwrap_processor_if_enabled()
        calib_angle0_processor = make_diag_angle0_processor_if_enabled()
        calib_prev = reconstruct_packet_tnb_for_plot(
            optim=optim, packet=prev_packet, direct=direct, use_packet_r_results=False,
            allow_shape_only=False, source_label=f"calib_data_reconstructed_r_{get_frame_vector_title(DIAG_RECON_FRAME_RULE)}",
            unwrap_processor=calib_unwrap_processor, angle0_processor=calib_angle0_processor,
            frame_context={"original_frame_idx": prev_frame_idx, "source": "calib_pair_prev"}
        )
        calib_curr = reconstruct_packet_tnb_for_plot(
            optim=optim, packet=packet, direct=direct, use_packet_r_results=False,
            allow_shape_only=False, source_label=f"calib_data_reconstructed_r_{get_frame_vector_title(DIAG_RECON_FRAME_RULE)}",
            unwrap_processor=calib_unwrap_processor, angle0_processor=calib_angle0_processor,
            frame_context={"original_frame_idx": frame_idx, "source": "calib_pair_curr"}
        )

        if calib_prev is None or calib_curr is None:
            print("  calib_data reconstructed r/TNB pair plot skipped because at least one frame failed reconstruction.")
            return None

    calib_endpoint_distance = np.linalg.norm(np.asarray(calib_curr["r"])[-1] - np.asarray(calib_prev["r"])[-1])
    plot_tnb_frame_pair(
        calib_prev, calib_curr, prev_frame_idx, frame_idx, endpoint_distance=calib_endpoint_distance,
        scale=scale, arrow_length=arrow_length, source_label=f"Calib_data reconstructed r/{get_frame_vector_title(DIAG_RECON_FRAME_RULE)}",
        endpoint_label="calib reconstructed endpoint_distance"
    )

    return {"calib_data_reconstructed": {"prev": calib_prev, "curr": calib_curr}}

def extract_reconstruction_diagnostics_from_result(optim, result):
    """
    从 reconstruct_packet_tnb_for_plot 已经得到的结果中提取曲率/角度诊断量。
    可用于全序列缓存结果，避免目标 pair 重新初始化跨帧状态后再重构。
    """
    if result is None:
        print("  reconstruction diagnostics skipped because explicit reconstruction failed.")
        return None

    intermediates = result.get("intermediates")

    if intermediates is None:
        print("  reconstruction diagnostics skipped because intermediates are unavailable.")
        return None

    diag = {
        "magnitude": np.asarray(intermediates["magnitude"], dtype=np.float64),
        "angle": np.asarray(intermediates["angle"], dtype=np.float64),
        "angle_unwrap_raw": np.asarray(intermediates["angle_unwrap_raw"], dtype=np.float64),
        "angle_unwrap": np.asarray(intermediates["angle_unwrap"], dtype=np.float64),
        "ang_filter": np.asarray(intermediates["ang_filter"], dtype=np.float64),
        "cur_kal": np.asarray(intermediates["cur_kal"], dtype=np.float64),
        "cur_interp": np.asarray(intermediates["cur_interp"], dtype=np.float64),
        "ang_interp": np.asarray(intermediates["ang_interp"], dtype=np.float64),
        "interp_dist": np.asarray(intermediates["interp_dist"], dtype=np.float64),
        "r": np.asarray(result["r"], dtype=np.float64),
        "T": np.asarray(result["T"], dtype=np.float64),
        "N": np.asarray(result["N"], dtype=np.float64),
        "B": np.asarray(result["B"], dtype=np.float64),
    }

    for direction_key in ["prev_unwrap_directs_used", "frame_unwrap_directs", "unwrap_changed_indices"]:
        if intermediates.get(direction_key) is not None:
            diag[direction_key] = np.asarray(intermediates[direction_key], dtype=np.int64)
        else:
            diag[direction_key] = None

    for scalar_key in ["angle0_held", "raw_angle0_original", "raw_angle0_replacement", "raw_magnitude0"]:
        diag[scalar_key] = intermediates.get(scalar_key)

    raw_dist_source = getattr(optim, 'ad_samp_dist', getattr(optim, 'samp_dist', None))

    if raw_dist_source is None:
        raw_dist = np.arange(len(diag["magnitude"]), dtype=np.float64)
    else:
        raw_dist = np.asarray(raw_dist_source, dtype=np.float64)

    if len(raw_dist) != len(diag["magnitude"]):
        common_len = min(len(raw_dist), len(diag["magnitude"]))
        raw_dist = raw_dist[:common_len]

        for key in ["magnitude", "angle", "angle_unwrap_raw", "angle_unwrap", "ang_filter", "cur_kal"]:
            diag[key] = diag[key][:common_len]

    diag["raw_dist"] = raw_dist
    return diag


def calculate_reconstruction_diagnostics_for_packet(
    optim, packet, direct=0, verbose=False, unwrap_processor=None,
    angle0_processor=None, frame_context=None
):
    """
    直接调用 diagnostics 的显式 calib_data 重构函数，并从返回结果的 intermediates
    中提取曲率/角度诊断量，避免再维护一套重复的曲率角度计算流程。
    """
    result = reconstruct_packet_tnb_for_plot(
        optim=optim, packet=packet, direct=direct, apply_direct_rotation=False,
        verbose=verbose, use_packet_r_results=False, allow_shape_only=False,
        source_label="diagnostic_intermediates_from_explicit_reconstruction",
        unwrap_processor=unwrap_processor, angle0_processor=angle0_processor,
        frame_context=frame_context
    )

    return extract_reconstruction_diagnostics_from_result(optim, result)


def plot_single_series_subplot(ax, x_values, y_values, frame_idx, title, y_label):
    """
    在一个子图里画单帧的一维序列；若 y_values 是二维，则逐列画出。
    """
    y_values = np.asarray(y_values, dtype=np.float64)
    x_values = np.asarray(x_values, dtype=np.float64)

    if y_values.ndim == 1:
        ax.plot(x_values, y_values, linewidth=1.6, label=f"Frame {frame_idx}")
    elif y_values.ndim == 2:
        if y_values.shape[0] == len(x_values):
            for col_idx in range(y_values.shape[1]):
                ax.plot(x_values, y_values[:, col_idx], linewidth=1.2, label=f"Frame {frame_idx}, col {col_idx}")
        elif y_values.shape[1] == len(x_values):
            for row_idx in range(y_values.shape[0]):
                ax.plot(x_values, y_values[row_idx, :], linewidth=1.2, label=f"Frame {frame_idx}, row {row_idx}")
        else:
            flat_y = y_values.reshape(-1)
            ax.plot(np.arange(len(flat_y)), flat_y, linewidth=1.2, label=f"Frame {frame_idx}, flattened")
            ax.set_xlabel("Flattened index")
    else:
        flat_y = y_values.reshape(-1)
        ax.plot(np.arange(len(flat_y)), flat_y, linewidth=1.2, label=f"Frame {frame_idx}, flattened")
        ax.set_xlabel("Flattened index")

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')


def plot_frame_pair_series(prev_diag, curr_diag, prev_frame_idx, curr_frame_idx, key, figure_title, y_label, x_key='interp_dist'):
    """
    画一张包含两个子图的图：上图对应上一帧，下图对应当前帧。
    """
    if prev_diag is None or curr_diag is None:
        print(f"  {figure_title} skipped because one frame has no diagnostic data.")
        return

    if key not in prev_diag or key not in curr_diag:
        print(f"  {figure_title} skipped because key={key} is unavailable.")
        return

    prev_y = np.asarray(prev_diag[key], dtype=np.float64)
    curr_y = np.asarray(curr_diag[key], dtype=np.float64)
    prev_x = np.asarray(prev_diag.get(x_key, np.arange(len(prev_y))), dtype=np.float64)
    curr_x = np.asarray(curr_diag.get(x_key, np.arange(len(curr_y))), dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle(figure_title, fontsize=14, fontweight='bold')

    plot_single_series_subplot(
        axes[0], prev_x, prev_y, prev_frame_idx,
        title=f"Previous frame {prev_frame_idx}", y_label=y_label
    )
    plot_single_series_subplot(
        axes[1], curr_x, curr_y, curr_frame_idx,
        title=f"Current/max-jump frame {curr_frame_idx}", y_label=y_label
    )

    x_label = "Interpolated distance" if x_key == 'interp_dist' else "Raw sensing distance"
    axes[0].set_xlabel(x_label)
    axes[1].set_xlabel(x_label)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def print_target_pair_angle_values(prev_diag, curr_diag, prev_frame_idx, curr_frame_idx):
    """
    打印目标帧对的曲率 magnitude、原始 angle、np.unwrap 后的 angle_unwrap_raw、
    以及经过 framewise unwrap limit 后真正进入重构的 angle_unwrap。

    注意：这里的 magnitude 已经经过 reconstruct_packet_tnb_for_plot 中的
    zero_init 插点，或 non-zero-init 情况下与 angle[0]=angle[1] 同阶段的处理。
    """
    keys = ["magnitude", "angle", "angle_unwrap_raw", "angle_unwrap"]

    if prev_diag is None or curr_diag is None:
        print("  target pair curvature/angle values skipped because one frame has no diagnostic data.")
        return

    missing_keys = [key for key in keys if key not in prev_diag or key not in curr_diag]

    if missing_keys:
        print(f"  target pair curvature/angle values skipped because keys are unavailable: {missing_keys}")
        return

    def common_diff(curr_arr, prev_arr):
        curr_arr = np.asarray(curr_arr, dtype=np.float64)
        prev_arr = np.asarray(prev_arr, dtype=np.float64)
        common_len = min(len(curr_arr), len(prev_arr))
        return curr_arr[:common_len] - prev_arr[:common_len]

    print("\n---------- target pair curvature / angle / unwrap values ----------")
    print(f"Previous frame {prev_frame_idx}:")

    with np.printoptions(precision=6, suppress=True):
        for key in keys:
            print(f"  {key} = {np.asarray(prev_diag[key], dtype=np.float64)}")

        print(f"Current frame {curr_frame_idx}:")
        for key in keys:
            print(f"  {key} = {np.asarray(curr_diag[key], dtype=np.float64)}")

        print("Difference current - previous:")
        for key in keys:
            print(f"  {key} diff = {common_diff(curr_diag[key], prev_diag[key])}")

        print("Framewise limit modification relative to np.unwrap:")
        print(f"  previous angle_unwrap - angle_unwrap_raw = {common_diff(prev_diag['angle_unwrap'], prev_diag['angle_unwrap_raw'])}")
        print(f"  current angle_unwrap - angle_unwrap_raw = {common_diff(curr_diag['angle_unwrap'], curr_diag['angle_unwrap_raw'])}")

        print("Framewise unwrap direction diagnostics:")
        direction_keys = [
            ("prev_unwrap_directs_used", "state['prev_unwrap_directs'] used when processing this frame"),
            ("frame_unwrap_directs", "directions computed from this frame's final angle_unwrap"),
            ("unwrap_changed_indices", "indices changed by framewise limit"),
        ]

        for diag_label, diag_dict in [(f"Previous frame {prev_frame_idx}", prev_diag), (f"Current frame {curr_frame_idx}", curr_diag)]:
            print(f"  {diag_label}:")
            for key, description in direction_keys:
                value = diag_dict.get(key)
                if value is None:
                    print(f"    {key} ({description}) = None")
                else:
                    print(f"    {key} ({description}) = {np.asarray(value, dtype=np.int64)}")

        prev_frame_directs = prev_diag.get("frame_unwrap_directs")
        curr_prev_directs_used = curr_diag.get("prev_unwrap_directs_used")

        if prev_frame_directs is not None and curr_prev_directs_used is not None:
            same_directs = np.array_equal(
                np.asarray(prev_frame_directs, dtype=np.int64),
                np.asarray(curr_prev_directs_used, dtype=np.int64)
            )
            print(f"  current prev_unwrap_directs_used == previous frame_unwrap_directs: {same_directs}")


def plot_curvature_angle_and_interp_for_frame_pair(
    optim, prev_packet, curr_packet, prev_frame_idx, curr_frame_idx, direct=0,
    pair_label="target frame pair", plot_raw_curvature_angle=True,
    precomputed_prev_result=None, precomputed_curr_result=None,
    require_precomputed_pair=False
):
    """
    对目标相邻帧对画诊断图。

    目标帧对的定义：
    - 若 SPECIFIED_ADJACENT_FRAME_PAIR 不为 None，则使用指定帧对；
    - 否则使用 endpoint-distance 最大的 Rank 1 帧对。

    绘图内容：
    - cur_interp：滤波 + Hermite 插值后的曲率大小；
    - ang_interp：unwrap + moving average + Hermite 插值后的角度；
    - magnitude：_Strain_2_Cur 后、插值前的曲率大小；
    - angle：_Strain_2_Cur 后、unwrap 前的原始角度；
    - angle_unwrap_raw：仅经过 np.unwrap、尚未经过逐帧方向限制的角度；
    - angle_unwrap：经过逐帧方向限制后，真正进入滤波/插值/重构的角度；
    - ang_filter：angle_unwrap 经过 moving average 后、Hermite 插值前的角度。
    """
    print(f"\n---------- curvature / angle plots for {pair_label} ----------")

    if precomputed_prev_result is not None and precomputed_curr_result is not None:
        print("  using full-sequence cached reconstruction intermediates for this frame pair.")
        prev_diag = extract_reconstruction_diagnostics_from_result(optim, precomputed_prev_result)
        curr_diag = extract_reconstruction_diagnostics_from_result(optim, precomputed_curr_result)
    else:
        if require_precomputed_pair:
            print("  curvature/angle pair plots skipped because full-sequence cached reconstruction is unavailable for at least one frame.")
            return None

        unwrap_processor = make_diag_unwrap_processor_if_enabled()
        angle0_processor = make_diag_angle0_processor_if_enabled()
        prev_diag = calculate_reconstruction_diagnostics_for_packet(
            optim=optim, packet=prev_packet, direct=direct, unwrap_processor=unwrap_processor,
            angle0_processor=angle0_processor,
            frame_context={"original_frame_idx": prev_frame_idx, "source": "diagnostic_pair_prev"}
        )
        curr_diag = calculate_reconstruction_diagnostics_for_packet(
            optim=optim, packet=curr_packet, direct=direct, unwrap_processor=unwrap_processor,
            angle0_processor=angle0_processor,
            frame_context={"original_frame_idx": curr_frame_idx, "source": "diagnostic_pair_curr"}
        )

    if prev_diag is None or curr_diag is None:
        print("  curvature/angle pair plots skipped because at least one frame failed diagnostics calculation.")
        return None

    if PRINT_TARGET_PAIR_ANGLE_VALUES:
        print_target_pair_angle_values(prev_diag, curr_diag, prev_frame_idx, curr_frame_idx)

    plot_frame_pair_series(
        prev_diag, curr_diag, prev_frame_idx, curr_frame_idx,
        key='cur_interp', figure_title=f"cur_interp for {pair_label}: F{prev_frame_idx} -> F{curr_frame_idx}",
        y_label="cur_interp / curvature magnitude", x_key='interp_dist'
    )
    plot_frame_pair_series(
        prev_diag, curr_diag, prev_frame_idx, curr_frame_idx,
        key='ang_interp', figure_title=f"ang_interp for {pair_label}: F{prev_frame_idx} -> F{curr_frame_idx}",
        y_label="angle after unwrap/filter/interp (rad)", x_key='interp_dist'
    )

    if plot_raw_curvature_angle:
        plot_frame_pair_series(
            prev_diag, curr_diag, prev_frame_idx, curr_frame_idx,
            key='magnitude', figure_title=f"raw curvature magnitude for {pair_label}: F{prev_frame_idx} -> F{curr_frame_idx}",
            y_label="raw curvature magnitude", x_key='raw_dist'
        )
        plot_frame_pair_series(
            prev_diag, curr_diag, prev_frame_idx, curr_frame_idx,
            key='angle', figure_title=f"raw angle before unwrap for {pair_label}: F{prev_frame_idx} -> F{curr_frame_idx}",
            y_label="raw curvature angle before unwrap (rad)", x_key='raw_dist'
        )

        if PLOT_TARGET_PAIR_ANGLE_UNWRAP_RAW:
            plot_frame_pair_series(
                prev_diag, curr_diag, prev_frame_idx, curr_frame_idx,
                key='angle_unwrap_raw', figure_title=f"raw np.unwrap angle before framewise limit for {pair_label}: F{prev_frame_idx} -> F{curr_frame_idx}",
                y_label="angle after np.unwrap, before framewise limit (rad)", x_key='raw_dist'
            )

        plot_frame_pair_series(
            prev_diag, curr_diag, prev_frame_idx, curr_frame_idx,
            key='angle_unwrap', figure_title=f"final raw angle unwrap after framewise limit for {pair_label}: F{prev_frame_idx} -> F{curr_frame_idx}",
            y_label="final raw curvature angle after unwrap/limit (rad)", x_key='raw_dist'
        )

        if PLOT_TARGET_PAIR_ANGLE_FILTER:
            plot_frame_pair_series(
                prev_diag, curr_diag, prev_frame_idx, curr_frame_idx,
                key='ang_filter', figure_title=f"ang_filter after moving average for {pair_label}: F{prev_frame_idx} -> F{curr_frame_idx}",
                y_label="angle after unwrap/limit + moving average (rad)", x_key='raw_dist'
            )

    return {"prev": prev_diag, "curr": curr_diag}


def plot_cur_interp_and_angles_for_frame_pair(optim, prev_packet, curr_packet, prev_frame_idx, curr_frame_idx, direct=0):
    """
    兼容旧调用名；默认对目标帧对同时画 cur_interp / ang_interp / raw curvature / raw angle。
    """
    return plot_curvature_angle_and_interp_for_frame_pair(
        optim=optim, prev_packet=prev_packet, curr_packet=curr_packet,
        prev_frame_idx=prev_frame_idx, curr_frame_idx=curr_frame_idx,
        direct=direct, pair_label="target frame pair",
        plot_raw_curvature_angle=PLOT_TARGET_PAIR_RAW_CURVATURE_AND_ANGLE
    )






def print_top_endpoint_distance_change_diagnostics(
    optim, data_list, trajectory_arr, segment_rule=None, top_k=3,
    top_k_cur_interp=3, top_k_angle=3, angle_direct=0, angle_slope_level=0,
    visualisor=None, plot_max_tnb=True, tnb_scale=1.0, tnb_arrow_length=0.03,
    print_timestamps=False, plot_max_cur_interp_angles=True,
    specified_adjacent_pair=None, specified_pair_mode='original',
    sequence_reconstruction_results=None
):
    """
    基于 calib_data 重构轨迹寻找相邻帧末端位移最大的帧对。

    已删除详细数值诊断打印：
    - calib_data 分段最大变化；
    - cur_interp top-k 数值变化；
    - raw / reconstructed curvature-angle top-k 数值变化；
    - 每帧 endpoint 坐标与 timestamp 明细。

    当前只负责：
    - 选择目标帧对；
    - 对 Rank 1 / 指定帧对画曲率角度诊断图；
    - 对 Rank 1 / 指定帧对画 calib_data 重构 r/T/N/B 或 r/T/N1/N2。
    """
    if sequence_reconstruction_results is not None:
        if len(sequence_reconstruction_results) != len(data_list):
            print(
                "Warning: sequence_reconstruction_results length does not match data_list; "
                "falling back to trajectory_arr for endpoint-distance ranking."
            )
            sequence_reconstruction_results = None

    if sequence_reconstruction_results is not None:
        endpoint_distance = np.full(len(data_list), np.nan)

        for curr_idx in range(1, len(data_list)):
            prev_result = sequence_reconstruction_results[curr_idx - 1]
            curr_result = sequence_reconstruction_results[curr_idx]

            if prev_result is None or curr_result is None:
                continue

            prev_r = np.asarray(prev_result["r"], dtype=np.float64)
            curr_r = np.asarray(curr_result["r"], dtype=np.float64)

            if prev_r.ndim == 2 and curr_r.ndim == 2 and prev_r.shape[1] == 3 and curr_r.shape[1] == 3:
                endpoint_distance[curr_idx] = np.linalg.norm(curr_r[-1] - prev_r[-1])
    else:
        if len(trajectory_arr) < 2:
            print("trajectory_arr has fewer than 2 frames; cannot compute endpoint distance.")
            return None

        endpoint_distance = np.full(len(trajectory_arr), np.nan)
        endpoint_distance[1:] = np.linalg.norm(trajectory_arr[1:] - trajectory_arr[:-1], axis=1)

    specified_indices = resolve_specified_adjacent_frame_pair(
        data_list=data_list, specified_pair=specified_adjacent_pair, pair_mode=specified_pair_mode
    )

    if specified_indices is None and specified_adjacent_pair is not None:
        raise ValueError(
            "SPECIFIED_ADJACENT_FRAME_PAIR was set, but it could not be resolved. "
            "The script will not fall back to the max endpoint-distance/jitter pair. "
            f"specified_pair={specified_adjacent_pair}, "
            f"specified_pair_mode={specified_pair_mode!r}, "
            f"trimmed_data_len={len(data_list)}. "
            "For SPECIFIED_FRAME_PAIR_MODE='trimmed', use indices in the timestamp-selected data_list; "
            "for SPECIFIED_FRAME_PAIR_MODE='original', use original pkl frame indices that remain after timestamp selection."
        )

    if specified_indices is not None:
        top_indices = np.array([specified_indices[1]], dtype=int)
        target_pair_label = "specified adjacent frame pair"
        print(f"\n========== Specified adjacent frame pair: {specified_adjacent_pair} ({specified_pair_mode} frame indices) ==========")
    else:
        valid_indices = np.where(np.isfinite(endpoint_distance))[0]
        valid_indices = valid_indices[valid_indices > 0]

        if len(valid_indices) == 0:
            print("No valid adjacent endpoint distances were found.")
            return endpoint_distance

        top_k = min(top_k, len(valid_indices))
        top_indices = valid_indices[np.argsort(endpoint_distance[valid_indices])[-top_k:][::-1]]
        target_pair_label = "max calib-reconstructed endpoint-distance Rank 1 frame pair"
        print(f"\n========== Top {top_k} raw calib-reconstructed endpoint-distance frame pairs before moving average ==========")

    for rank, curr_idx in enumerate(top_indices, start=1):
        prev_idx = curr_idx - 1
        prev_packet = data_list[prev_idx]
        curr_packet = data_list[curr_idx]
        prev_original_idx = prev_packet.get('_diag_original_frame_idx', prev_idx) if isinstance(prev_packet, dict) else prev_idx
        curr_original_idx = curr_packet.get('_diag_original_frame_idx', curr_idx) if isinstance(curr_packet, dict) else curr_idx

        print(
            f"Rank {rank}: Trimmed Frame {prev_idx} -> {curr_idx} "
            f"(Original Frame {prev_original_idx} -> {curr_original_idx}), "
            f"raw_calib_reconstructed_endpoint_distance_before_moving_average={endpoint_distance[curr_idx]:.6f} m"
        )

        precomputed_prev_result = None
        precomputed_curr_result = None
        use_required_cached_pair = sequence_reconstruction_results is not None

        if sequence_reconstruction_results is not None:
            precomputed_prev_result = sequence_reconstruction_results[prev_idx]
            precomputed_curr_result = sequence_reconstruction_results[curr_idx]

        if rank == 1 and plot_max_cur_interp_angles:
            plot_curvature_angle_and_interp_for_frame_pair(
                optim=optim, prev_packet=prev_packet, curr_packet=curr_packet,
                prev_frame_idx=prev_original_idx, curr_frame_idx=curr_original_idx, direct=angle_direct,
                pair_label=target_pair_label,
                plot_raw_curvature_angle=PLOT_TARGET_PAIR_RAW_CURVATURE_AND_ANGLE,
                precomputed_prev_result=precomputed_prev_result,
                precomputed_curr_result=precomputed_curr_result,
                require_precomputed_pair=use_required_cached_pair
            )

        if rank == 1 and plot_max_tnb:
            plot_packet_tnb_for_diagnostic(
                optim=optim, visualisor=visualisor, packet=curr_packet, frame_idx=curr_original_idx,
                direct=angle_direct, endpoint_distance=endpoint_distance[curr_idx],
                scale=tnb_scale, arrow_length=tnb_arrow_length,
                prev_packet=prev_packet, prev_frame_idx=prev_original_idx,
                data_list_for_dynamic=None, prev_trimmed_idx=None, curr_trimmed_idx=None,
                plot_dynamic_smooth=False,
                dynamic_smooth_window=DYNAMIC_RECON_SMOOTH_WINDOW,
                precomputed_prev_result=precomputed_prev_result,
                precomputed_curr_result=precomputed_curr_result,
                require_precomputed_pair=use_required_cached_pair
            )

    return endpoint_distance

def print_nan_report(data_list):
    """
    汇总打印所有含有 NaN 的帧数，以及 NaN 的行列位置。
    如果没有 NaN，则不打印任何 NaN report。
    """
    nan_records = []

    for frame_idx, packet in enumerate(data_list):
        if 'calib_data' not in packet or packet['calib_data'] is None:
            continue

        calib_arr = np.asarray(packet['calib_data'])
        nan_positions = np.argwhere(np.isnan(calib_arr))

        if nan_positions.size == 0:
            continue

        if calib_arr.ndim == 2:
            for r, c in nan_positions:
                nan_records.append((frame_idx, int(r), int(c)))
        else:
            for pos in nan_positions:
                nan_records.append((frame_idx, tuple(pos)))

    if len(nan_records) == 0:
        return

    print("\n========== NaN report in calib_data ==========")

    nan_frame_indices = sorted(set(record[0] for record in nan_records))
    print(f"Frames containing NaN: {nan_frame_indices}")

    for record in nan_records:
        if len(record) == 3:
            frame_idx, r, c = record
            print(f"Frame {frame_idx}: row={r}, col={c}, value=nan")
        else:
            frame_idx, idx = record
            print(f"Frame {frame_idx}: index={idx}, value=nan")




# =========================
# 读取数据
# =========================

try:
    with open(filename, 'rb') as f:
        frame_count = 0

        while True:
            try:
                packet = pickle.load(f)

                if isinstance(packet, dict):
                    packet['_diag_original_frame_idx'] = frame_count

                data_list.append(packet)

                if PRINT_PACKET_KEYS_ON_READ:
                    print_packet_keys(frame_count, packet)

                frame_count += 1

            except EOFError:
                break

except FileNotFoundError:
    logging.error(f"文件未找到: {filename}")
except Exception as e:
    logging.error(f"读取数据时发生未知错误: {e}")


logging.info(f"共读取到 {len(data_list)} 帧原始数据")

if PRINT_TIMESTAMP_REPORT:
    print_timestamp_report(data_list)

data_list = select_timestamp_range(
    data_list=data_list,
    timestamp_range=ANALYSIS_TIMESTAMP_RANGE,
    keep_missing_timestamp=False
)
# 下面只使用 trim 后的 calib_data 显式重构轨迹；不再使用记录文件中的 r_results 轨迹。
calib_dynamic_endpoint_trajectory, calib_raw_endpoint_trajectory, calib_dynamic_original_frame_indices, calib_dynamic_timestamps, calib_full_sequence_reconstruction_results = build_dynamic_style_calib_endpoint_trajectory(
    optim=optim, data_list=data_list, direct=args.direct, smooth_window=3
)

logging.info(f"按 ANALYSIS_TIMESTAMP_RANGE={ANALYSIS_TIMESTAMP_RANGE} 筛选后，剩余 {len(data_list)} 帧数据")
logging.info(f"成功提取到 {len(calib_dynamic_endpoint_trajectory)} 个 calib_data 重构 + window=3 滑动平均末端点坐标")

# =========================
# calib_data 重构轨迹可视化与目标帧对诊断
# =========================

if not calib_dynamic_endpoint_trajectory:
    raise RuntimeError("没有可用于分析的 calib_data 重构末端点轨迹数据。")

calib_dynamic_trajectory_arr = np.asarray(calib_dynamic_endpoint_trajectory, dtype=np.float64)
calib_raw_trajectory_arr = np.asarray(calib_raw_endpoint_trajectory, dtype=np.float64)

if GENERATE_CALIB_RECON_VIDEO:
    calib_video_frames, calib_video_frame_indices = build_curve_frames_from_reconstruction_results(
        reconstruction_results=calib_full_sequence_reconstruction_results,
        use_dynamic_smoothing=CALIB_RECON_VIDEO_USE_DYNAMIC_SMOOTHING,
        smooth_window=CALIB_RECON_VIDEO_SMOOTH_WINDOW, stride=CALIB_RECON_VIDEO_STRIDE,
        max_frames=CALIB_RECON_VIDEO_MAX_FRAMES
    )
    print(
        f"calib_data / calib_shape 重构视频帧准备完成："
        f"输出帧数={len(calib_video_frames)}, stride={CALIB_RECON_VIDEO_STRIDE}, "
        f"use_dynamic_smoothing={CALIB_RECON_VIDEO_USE_DYNAMIC_SMOOTHING}"
    )
    generate_3d_shape_video_from_frames(
        frames_data=calib_video_frames, output_filename=CALIB_RECON_VIDEO_OUTPUT,
        fps=CALIB_RECON_VIDEO_FPS, writer=CALIB_RECON_VIDEO_WRITER, dpi=CALIB_RECON_VIDEO_DPI,
        title=f"{filename} - calib_data / calib_shape reconstructed 3D Shape Evolution",
        elev=CALIB_RECON_VIDEO_ELEV, azim=CALIB_RECON_VIDEO_AZIM
    )

plot_endpoint_trajectory_3d(
    trajectory_arr=calib_dynamic_trajectory_arr,
    title='End-Effector 3D Trajectory Analysis: calib_data reconstructed + moving-average r, window=3',
    path_label='Calib-data Reconstructed + Smoothed Trajectory Path'
)

# 下面两组整体运动诊断全部使用 calib_data 重构 + 动态展示式滑动平均得到的末端轨迹，
# 不读取也不使用 packet['r_results']。
if PLOT_CALIB_RECON_CYCLOID_FITTING_AND_DEVIATION:
    cycloid_diagnostics = plot_calib_reconstructed_cycloid_and_deviation(
        trajectory_arr=calib_dynamic_trajectory_arr,
        original_frame_indices=calib_dynamic_original_frame_indices,
        trajectory_timestamps=calib_dynamic_timestamps,
        annotate_timestamps=CYCLOID_ANNOTATE_TIMESTAMPS,
        timestamp_match_max_diff=CYCLOID_TIMESTAMP_MATCH_MAX_DIFF,
        radius=CYCLOID_RADIUS, marker_step=CYCLOID_MARKER_STEP,
        reverse_for_return=CYCLOID_FIT_USE_REVERSE_TRAJECTORY
    )
else:
    cycloid_diagnostics = None

if PLOT_CALIB_RECON_TRAJECTORY_JITTER:
    calib_reconstructed_spatial_jitter = plot_calib_reconstructed_trajectory_jitter(
        trajectory_arr=calib_dynamic_trajectory_arr,
        frame_dt=JITTER_FRAME_DT
    )
else:
    calib_reconstructed_spatial_jitter = None

# 基于 calib_data 重构轨迹的相邻帧末端距离选择目标帧对。
# 详细数值诊断打印已删除；曲率角度诊断量直接来自显式重构结果的 intermediates。
endpoint_distance = print_top_endpoint_distance_change_diagnostics(
    optim=optim, data_list=data_list, trajectory_arr=calib_raw_trajectory_arr, segment_rule=None,
    top_k=TOP_K_ENDPOINT_DISTANCE, top_k_cur_interp=TOP_K_CUR_INTERP_CHANGE,
    top_k_angle=TOP_K_CURVATURE_ANGLE_CHANGE, angle_direct=args.direct,
    angle_slope_level=ANGLE_SLOPE_LEVEL, visualisor=vis, plot_max_tnb=PLOT_MAX_ENDPOINT_TNB,
    tnb_scale=TNB_PLOT_SCALE, tnb_arrow_length=TNB_PLOT_ARROW_LENGTH,
    print_timestamps=False,
    plot_max_cur_interp_angles=PLOT_MAX_PAIR_CUR_INTERP_AND_ANGLES,
    specified_adjacent_pair=SPECIFIED_ADJACENT_FRAME_PAIR,
    specified_pair_mode=SPECIFIED_FRAME_PAIR_MODE,
    sequence_reconstruction_results=calib_full_sequence_reconstruction_results
)

# 汇总所有含 NaN 的 calib_data 帧。
print_nan_report(data_list)
