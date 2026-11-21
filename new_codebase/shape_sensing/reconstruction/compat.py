# 历史类兼容入口集中于本模块；新生产入口请使用 shape_sensing.reconstruction.service.ReconstructionService。
from copy import deepcopy

import numpy as np
from ..common.utils import dict_recur_extract_path, dict_recur_paths, dict_recur_unflatten
from ..configuration.regions import validate_weight_group_ids
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from ..initialization.compat import Initialiser
from ..curvature.model import CurvatureModelMixin
from ..reconstruction.integration import ShapeIntegrationMixin
from .preparation import CurvaturePreparationKernel
from ..evaluation.ideal_curvature import IdealCurvatureMixin




class Reconstructor(CurvatureModelMixin, Initialiser, ShapeIntegrationMixin, IdealCurvatureMixin, CurvaturePreparationKernel):
    """
    形状重构模块
    """
    def __init__(self, soft=True, frame_rule='TNB', rot_ang=0.0, turn_threshold=2.0, init_weight=1.0, weights=None, slopes=None, min_consecutive=3, sharpness=4.8, end_start=-6, end_weights=False, thresholds=[0.8, 1.5, 2.0, 3.0, 4.0],
                 TNB_threshold=0.15, interp_num=120, move_filter=3, kalman_params={'angle': {'Rk': 0.001, 'Bk': 0.001}, 'magnitude': {'Rk': 0.0001, 'Bk': 0.01}},
                 curvature_interp='hermite', nonnegative_curvature=False,
                 *args, move_filter_for_vector=False, multi_region_parameters=None, **kwargs):
        weight_group_ids = kwargs.pop('weight_group_ids', None)
        end5cm_independent_weight_score = kwargs.pop('end5cm_independent_weight_score', None)
        legacy_groupwise = kwargs.pop('end5cm_groupwise_weight_score', False)
        if end5cm_independent_weight_score is None:
            end5cm_independent_weight_score = legacy_groupwise
        super().__init__(*args, **kwargs)

        self.rot_seg_fit(verbose=True, update_phase=False) # 打印初始信息，但不更新全局相位
        self.magnitude_ratio = dict_recur_extract_path(self.magnitude_results, lambda arr, path: 1.0 if path[-1] == 0 else float((1000 / path[-1]) / np.mean(arr)))
        # 重构所需参数
        self.soft = soft
        self.frame_rule = frame_rule
        self.turn_threshold = turn_threshold
        self.thresholds = thresholds # 曲率分级阈值
        self.min_consecutive = min_consecutive
        self.sharpness = sharpness
        self.end5cm_independent_weight_score = bool(end5cm_independent_weight_score)
        # Runtime compatibility for older callers; production code reads the canonical name.
        self.end5cm_groupwise_weight_score = self.end5cm_independent_weight_score

        self.num_levels = len(self.thresholds) + 1
        self.base_segments = self._calculate_segments(total_points=self.total_points)
        self.end_mask = np.arange(self.total_points) >= self.total_points + end_start
        self.end_start = end_start
        self.TNB_threshold = TNB_threshold # FS重构时直接复制上一点的曲率大小阈值

        # 需要优化的应变权重、扭转系数以及旋转角度
        self.end_weights = end_weights # 是否优化末端权重
        self.weight_group_ids = validate_weight_group_ids(weight_group_ids, self.total_points)
        self.init_weight = init_weight
        if not weights:
            if self.weight_group_ids is not None:
                self.weights = {direct: init_weight*np.ones((max(self.weight_group_ids)+1)*self.num_levels)
                                for direct in self.exp_directs}
            else:
                self.weights = {direct: np.ones(2 * self.num_levels) for direct in self.exp_directs} if self.end_weights else {direct: init_weight * np.ones(self.num_levels) for direct in self.exp_directs}
        else:
            self.weights = weights

        if not slopes:
            # 若涉及S形弯优化，则准备两组扭转系数，否则一组
            self.slopes = {direct: np.array([np.concatenate((self.slopes_init[direct], self.slopes_init[direct])) for _ in range(self.num_levels)]) for direct in self.exp_directs} if 'S' in self.calib_shapes else {direct: np.array([self.slopes_init[direct] for _ in range(self.num_levels)]) for direct in self.exp_directs}
        else:
            self.slopes = slopes
        self.rot_ang = rot_ang

        # 滤波、插值所需参数
        self.move_filter = move_filter
        self.move_filter_for_vector = bool(move_filter_for_vector)
        self.kalman_params = kalman_params
        self.interp_num = interp_num
        self.curvature_interp = self._normalise_curvature_interp(curvature_interp)
        self.nonnegative_curvature = bool(nonnegative_curvature)
        self.multi_region_parameters = (
            self._normalise_multiregion_parameters(multi_region_parameters)
            if multi_region_parameters is not None else None
        )
        self.last_parameter_assembly = None

    def reconstruct_shapes(self, calib_shape, direct, verbose=False, force_ideal_magnitude=False):
        """
        对指定标定形状和方向下的所有数据执行三维形状重构。

        参数：
        - calib_shape: 标定形状类型，例如 'circ'、'S'、'end'、'end3'、'rot0'、'up'、'down'。
        - direct: 标定方向角，对应 self.exp_directs 中的一个方向。
        - verbose: 是否打印每组重构的中间信息。

        处理流程：
        1. 从 self.processed_Data[calib_shape][direct] 中取出当前方向下的数据。
        2. 读取当前 direct 对应的 weight 和 slope。
        3. 根据 calib_shape 构造待重构数据的路径树：
        - circ/S: 单层路径，以 radius 为 key；
        - end/end3/rot0/up/down: 双层路径，以 radius 和末级参数为 key。
        4. 根据 direct 和当前形状的 rot_dir 生成绕 x 轴旋转矩阵。
        5. 遍历路径树中的所有叶节点，逐组调用 _reconstruct_frame()：
        - 将应变转换为曲率大小和方向角；
        - 滤波、插值；
        - 根据 frame_rule 重构三维曲线。
        6. 对每条重构曲线施加方向旋转 r @ rot_matrix.T。
        7. 将 flat_results 还原为与输入数据一致的嵌套结构。

        输出：
        - r_results_postrot: 与标定数据结构一致的重构结果字典。
        每个叶节点为形状 (interp_num, 3) 的三维曲线坐标数组。
        - interp_dist: 插值后的空间采样距离数组。

        说明：
        - 当前实现中 magnitude_ratio 被强制设为 1.0。
        - rot_matrix 只在重构曲线完成后应用，不影响曲率和角度估计过程。
        - 若 verbose=True，会打印 keys 和旋转前的 r。
        """
        # ---------- 初始准备 ----------
        calib_Data = self.processed_Data[calib_shape][direct]
        weight = self.weights[direct]
        slope = self.slopes[direct]

        interp_dist = None
        # 字典按实验方向角取值；旧列表仍按 calib_shapes 顺序取值。
        if isinstance(self.rot_dir, dict):
            rot_dir = self.rot_dir.get(direct, self.rot_dir.get(str(direct)))
            if rot_dir is None:
                raise ValueError(f"rot_dir 缺少方向 {direct} 的旋转符号")
        elif np.isscalar(self.rot_dir):
            rot_dir = self.rot_dir
        else:
            rot_dir = self.rot_dir[self.calib_shapes.index(calib_shape)]
        rot_matrix = self._get_rot_x(direct=direct, rot_dir=rot_dir)

        # ---------- 构造路径树 ----------
        if calib_shape in ['circ', 'S']:
            path_tree = {radius: None for radius in self.radii[direct][calib_shape]}
        else:
            path_tree = {radius: {idx: None for idx in self.radii[direct][calib_shape][radius]} for radius in self.radii[direct][calib_shape]}

        # ---------- 遍历所有叶节点路径并重构 ----------
        flat_results = {}

        for path in dict_recur_paths(path_tree):
            keys = list(path)

            radius = keys[0]
            # magnitude_ratio = self.magnitude_ratio[calib_shape][direct][radius]
            magnitude_ratio = 1.0 # 如果不使用magnitude_ratio, 就把上一行注释掉，用这一行

            if isinstance(magnitude_ratio, dict):
                magnitude_ratio = magnitude_ratio[keys[1]]

            _, _, _, interp_dist, r, _, _, _, _ = self._reconstruct_frame(keys=keys, calib_shape=calib_shape, calib_Data=calib_Data, weight=weight, slope=slope, magnitude_ratio=magnitude_ratio, init_T=[1.0, 0.0, 0.0], verbose=verbose, force_ideal_magnitude=force_ideal_magnitude, direct=direct)

            r_postrot = r @ rot_matrix.T
            flat_results[path] = r_postrot

            if verbose:
                print(f"keys: {keys}")
                print(f"r: {r}")
                print()

        # ---------- 还原为原本嵌套结构 ----------
        r_results_postrot = dict_recur_unflatten(flat_results)

        return r_results_postrot, interp_dist

    def _reconstruct_frame(self, keys, calib_shape, calib_Data, weight, slope, magnitude_ratio, init_T, verbose=False, unwrap_threshold=0.65, align_threshold=0.15, angle_prev=None, angle_unwrap_prev=None, force_ideal_magnitude=False, direct=None):
        """
        对单组应变数据执行完整的单帧三维重构流程。

        参数：
        - keys: 当前数据在 calib_Data 中的路径。
        * None: 直接使用 calib_Data；
        * [radius]&#58; 使用 calib_Data[radius]；
        * [radius, idx]&#58; 使用 calib_Data[radius][idx]。
        - calib_shape: 当前标定形状类型。
        - calib_Data: 当前 direct 下的标定数据字典。
        - weight: 当前 direct 下的曲率等级应变权重。
        - slope: 当前 direct 下的曲率等级分段扭转系数。
        - magnitude_ratio: 曲率大小比例修正系数，用于曲率分级前的 score_magnitude。
        - init_T: 初始切向量，通常为 [1.0, 0.0, 0.0]。
        - verbose: 是否打印插值距离、角度、曲率和初始法向量等中间信息。
        - unwrap_threshold: 帧间 unwrap 分支选择阈值。
        - align_threshold: 若首点曲率低于该阈值且存在上一帧角度，则用上一帧首点角度对齐当前首点。
        - angle_prev: 上一帧 raw angle，用于动态重构时保持帧间相位连续。
        - angle_unwrap_prev: 上一帧 unwrap angle，用于动态重构时选择最接近上一帧的 2π 分支。

        处理流程：
        1. 调用 _Strain_2_Cur() 得到最终用于重构的曲率大小 magnitude 与方向角 angle。
        2. 若提供上一帧角度且当前首点曲率很小，则用上一帧首点角度对齐当前首点。
        3. 根据 zero_init 和 end_point 处理虚拟点：
        - zero_init=True: 在开头复制第一个真实点的 magnitude 和 angle；
        - zero_init=False: 将 angle[0] 设置为 angle[1]；
        - end_point=True: 在末尾复制最后一个真实点的 magnitude 和 angle。
        4. 调用 _framewise_unwrap_limit() 得到连续角度 angle_unwrap。
        5. 对 angle_unwrap 做空间移动平均，得到 ang_filter；仅标量插值路径使用它，
           向量路径仍计算、校验并保存它，但插值使用 angle_unwrap。
        6. 对 magnitude 做单帧空间 Kalman 滤波，得到 cur_kal；标量路径使用它，
           向量路径仅校验/保存该中间量，不将其用于插值。
        7. 检查滤波后的角度和曲率是否包含 NaN / inf。
        8. 按 self.ad_samp_dist 插值：标量路径对 ang_filter 做 Hermite、对 cur_kal
           按所选标量方法插值；向量路径将补齐虚拟点的 magnitude 与 angle_unwrap 转成两分量，
           根据 move_filter_for_vector 选择用分量空间平均修正方向或保留原始分量，
           恢复输入模长后再做线性/PCHIP 插值，
           最后还原曲率大小和方向。
        9. 根据插值后的首点角度构造初始法向量 init_N。
        10. 根据 self.frame_rule 选择重构方式：
            - 'TNB': 由角度差分计算 torsion，并调用 _F_S_frame()；
            - 'TNN': 调用 _parallel_frame()。

        输出：
        - 若 frame_rule == 'TNB':
        返回 angle, angle_unwrap, cur_interp, interp_dist, r, T, N, B。
        - 若 frame_rule == 'TNN':
        返回 angle, angle_unwrap, cur_interp, interp_dist, r, T, N1, N2。

        异常：
        - 若滤波后的角度或曲率包含非有限值，抛出 ValueError 并附带 calib_shape 和 keys。
        - 若插值过程中数据长度与 self.ad_samp_dist 不一致或包含非法值，抛出带上下文的 ValueError。

        说明：
        - 本方法是单条曲线的完整重构入口。
        - 动态展示时可通过 angle_prev / angle_unwrap_prev 保持相邻帧 angle unwrap 的连续性。
        - self.cur_interp 会被更新为当前帧插值后的曲率数组，主要用于调试或外部诊断。
        """
        # ---------- 计算重构使用的曲率大小与方向 ----------
        magnitude, angle = self._Strain_2_Cur(keys=keys, calib_shape=calib_shape, calib_Data=calib_Data, weight=weight, slope=slope, magnitude_ratio=magnitude_ratio, verbose=verbose, direct=direct)
        # 模块5只接收模块3已算好的真实点曲率；旧入口也共享同一数值核。
        self.unwrap_threshold = unwrap_threshold
        self.align_threshold = align_threshold
        ideal_curvature = None
        if force_ideal_magnitude:
            ideal_curvature = lambda distance: self._construct_ideal_magnitude_for_recon(
                keys=keys, calib_shape=calib_shape, sample_dist=distance)
        cur_interp, ang_interp, interp_dist, intermediates = self.prepare_arrays(
            magnitude, angle, angle_prev, angle_unwrap_prev,
            ideal_curvature=ideal_curvature, calib_shape=calib_shape, keys=keys)
        angle = intermediates['angle_prepared']
        angle_unwrap = intermediates['angle_unwrap']
        cur_kal = intermediates['cur_kal']
        ang_filter = intermediates['ang_filter']
        magnitude_raw = intermediates['magnitude']
        angle_raw = intermediates['angle']
        interp_ss = np.diff(interp_dist)

        # Frenet-Serret 路径仍以首点曲率方向作为初始主法向。
        init_N = np.array([0, np.cos(ang_interp[0]), np.sin(ang_interp[0])])

        if verbose:
            print(f"interp_dist: {interp_dist}")
            print(f"ang_interp: {ang_interp}")
            print(f"cur_interp: {cur_interp}")
            print(f"init_N: {init_N}")

        intermediates = {
            'magnitude': magnitude_raw.copy(),
            'angle': angle_raw.copy(),
            'cur_kal': cur_kal.copy(),
            'ang_filter': ang_filter.copy(),
            'cur_interp': cur_interp.copy(),
            'ang_interp': ang_interp.copy(),
        }

        if self.frame_rule == 'TNB':
            # Frenet-Serret重构
            ang_diff = np.diff(ang_interp)
            torsion = (ang_diff - 2 * np.pi * np.floor((ang_diff + np.pi) / (2 * np.pi))) / interp_ss
            if verbose:
                print(f"torsion: {torsion}")
            r, T, N, B = self._F_S_frame(kappa=cur_interp, tau=torsion, ss=interp_ss, init_T=init_T, init_N=init_N)
            return angle, angle_unwrap, cur_interp, interp_dist, r, T, N, B, intermediates
        elif self.frame_rule == 'TNN':
            # 平行标架重构
            r, T, Y, Z, _, _ = self._parallel_frame(
                kappa=cur_interp,
                angles=ang_interp,
                ss=interp_ss,
                init_T=init_T,
                init_N=init_N,
                return_principal_frame=True,
            )
            return angle, angle_unwrap, cur_interp, interp_dist, r, T, Y, Z, intermediates
