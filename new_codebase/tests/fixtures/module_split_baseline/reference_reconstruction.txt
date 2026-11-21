from copy import deepcopy

import numpy as np
from ..common.utils import dict_recur_extract_path, dict_recur_paths, dict_recur_unflatten
from ..configuration.regions import validate_weight_group_ids
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from ..initialization.fitting import Initialiser
from ..curvature.model import CurvatureModelMixin
from ..reconstruction.integration import ShapeIntegrationMixin
from ..evaluation.ideal_curvature import IdealCurvatureMixin




class Reconstructor(CurvatureModelMixin, Initialiser, ShapeIntegrationMixin, IdealCurvatureMixin):
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

    @staticmethod
    def _normalise_curvature_interp(method):
        """Return the canonical production curvature interpolation mode."""
        aliases = {'pchip': 'scalar_pchip'}
        canonical = aliases.get(method, method)
        supported = {'hermite', 'scalar_pchip', 'vector_linear', 'vector_pchip'}
        if canonical not in supported:
            raise ValueError(
                "curvature_interp must be one of: hermite, pchip, "
                "scalar_pchip, vector_linear, vector_pchip"
            )
        return canonical

    @staticmethod
    def _vector_curvature_method(mode):
        """Map a production vector mode to its component interpolation method."""
        methods = {
            'vector_linear': 'linear',
            'vector_pchip': 'pchip',
        }
        try:
            return methods[mode]
        except KeyError as exc:
            raise ValueError(f"unsupported vector curvature mode: {mode}") from exc

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
        magnitude_raw = magnitude.copy()
        angle_raw = angle.copy()
        if force_ideal_magnitude:
            magnitude = self._construct_ideal_magnitude_for_recon(keys=keys, calib_shape=calib_shape, sample_dist=self.samp_dist)
        if angle_prev is not None and magnitude[0] < align_threshold:
            angle[0] = angle_prev[0]

        if self.zero_init:
            magnitude = np.insert(magnitude, 0, magnitude[0])  # 起点复制第一个传感点
            angle = np.insert(angle, 0, angle[0])
        else:
            angle[0] = angle[1]

        if getattr(self, "end_point", False):
            magnitude = np.append(magnitude, magnitude[-1])  # 末端复制最后一个传感点
            angle = np.append(angle, angle[-1])
        # print(magnitude)
        # ---------- 重构前预处理 ----------
        angle_unwrap = self._framewise_unwrap_limit(angle=angle, angle_prev=angle_prev, angle_unwrap_prev=angle_unwrap_prev, unwrap_threshold=unwrap_threshold)
        # 此处沿单帧的空间采样点平均，不是跨帧滤波。
        # ang_filter 仅供下方标量路径插值；向量路径只将它作为中间量校验/保存。
        ang_filter = self._moving_average_scipy(data=angle_unwrap, window_size=self.move_filter, mode='reflect')
        # 标量路径使用此空间 Kalman 输出；向量路径保留它用于中间量校验/记录，
        # 实际插值直接使用 magnitude，避免再叠加一层曲率空间滤波。
        cur_kal = self._Kalman_filter(data=magnitude, Rk=self.kalman_params['magnitude']['Rk'], Bk=self.kalman_params['magnitude']['Bk'])

        # 增加上下文信息字符串，方便报错时追踪
        context_str = f"calib_shape: {calib_shape}, keys(radius/end_radius): {keys}"

        # 检查预处理后的角度与曲率序列是否包含 NaN / inf
        if not np.all(np.isfinite(ang_filter)):
            raise ValueError(f"[{context_str}] ang_filter 包含 NaN 或 inf。请检查 _Strain_2_Cur 的输出。")
        if not np.all(np.isfinite(cur_kal)):
            raise ValueError(f"[{context_str}] cur_kal 包含 NaN 或 inf。请检查 _Strain_2_Cur 的输出。")

        # 使用 try-except 捕获插值过程中的报错并追加上下文信息
        try:
            if self.curvature_interp in {'vector_linear', 'vector_pchip'}:
                if force_ideal_magnitude:
                    curvature_samples = self._construct_ideal_magnitude_for_recon(
                        keys=keys,
                        calib_shape=calib_shape,
                        sample_dist=self.ad_samp_dist,
                    )
                else:
                    curvature_samples = magnitude
                # 使用未做角度窗口平均的 angle_unwrap；是否执行独立的分量方向窗口
                # 由 move_filter_for_vector 控制。force_ideal_magnitude 时输入理想曲率，
                # 绕过 cur_kal 输出。
                cur_interp, ang_interp, interp_dist = self._interpolate_curvature_vector(
                    magnitude=curvature_samples,
                    angle=angle_unwrap,
                    method=self._vector_curvature_method(self.curvature_interp),
                )
            else:
                # Preserve the historical scalar path: angle Hermite plus an
                # independently interpolated scalar curvature magnitude.
                ang_interp, interp_dist = self._KT_Hermit_interp_tang(data=ang_filter)

                if force_ideal_magnitude:
                    cur_interp = self._construct_ideal_magnitude_for_recon(keys=keys, calib_shape=calib_shape, sample_dist=interp_dist)
                else:
                    if not np.all(np.isfinite(cur_kal)):
                        raise ValueError(f"[{context_str}] cur_kal 包含 NaN 或 inf。请检查 _Strain_2_Cur 的输出。")

                    cur_interp, interp_dist = self._interpolate_curvature(
                        data=cur_kal,
                        method=self.curvature_interp,
                        nonnegative=self.nonnegative_curvature,
                    )

            if not np.all(np.isfinite(cur_interp)):
                raise ValueError(f"[{context_str}] cur_interp 包含 NaN 或 inf。")

            self.cur_interp = cur_interp

        except ValueError as e:
            raise ValueError(f"数据追踪信息 -> {context_str} | 原始报错: {e}")
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



    def _framewise_unwrap_limit(self, angle, angle_prev=None, angle_unwrap_prev=None, unwrap_threshold=0.65):
        """
        在逐帧动态重构中限制 angle unwrap 的相位分支跳变。

        该方法与 diagnostics 中 make_framewise_unwrap_processor 的核心逻辑等价，
        用于避免相邻帧之间由于 np.unwrap 自动选择不同 2π 分支而导致角度序列突然跳变。

        参数：
        - angle: 当前帧原始曲率方向角序列，单位 rad，通常范围在 [-π, π]。
        - angle_prev: 上一帧原始曲率方向角序列。
        - angle_unwrap_prev: 上一帧 unwrap 后的曲率方向角序列。
        - unwrap_threshold: 判断当前帧与上一帧相邻段 raw step 是否足够接近的阈值。

        处理流程：
        1. 先对当前帧 angle 调用 np.unwrap(angle, discont=self.discont) 得到 angle_unwrap。
        2. 若没有上一帧信息，直接返回 angle_unwrap。
        3. 从第 1 个点开始逐点检查相邻段 step = angle[i] - angle[i-1]。
        4. 若当前帧和上一帧对应相邻段的 raw step 差异小于 unwrap_threshold，
        则不直接接受 np.unwrap 的结果，而是在 step-2π、step、step+2π 中选择
        最接近上一帧 unwrap step 的分支。
        5. 若当前点发生分支修正，将 offset 累积并传递给后续点。
        6. 若当前序列长于上一帧序列，则对剩余点继续叠加最终 offset。

        输出：
        - angle_unwrap: 修正后的当前帧连续角度序列。

        说明：
        - 该方法主要用于动态序列，不是静态初始化所必需。
        - 它同时考虑当前帧内部连续性和相邻帧之间的连续性。
        """
        angle_unwrap = np.unwrap(angle, discont=self.discont)
        if angle_prev is None or angle_unwrap_prev is None:
            return angle_unwrap

        offset = 0.0

        for i in range(1, len(angle_prev)):
            angle_unwrap[i] += offset

            step = angle[i] - angle[i - 1]
            step_prev = angle_prev[i] - angle_prev[i - 1]

            segment_step_diff = np.abs((step - step_prev + np.pi) % (2 * np.pi) - np.pi)

            if np.isclose(segment_step_diff, 0.0, atol=unwrap_threshold, rtol=0.0):
                unwrap_val = angle_unwrap[i]
                unwrap_step_prev = angle_unwrap_prev[i] - angle_unwrap_prev[i - 1]

                candidate_steps = np.array([step - 2 * np.pi, step, step + 2 * np.pi], dtype=np.float64)
                chosen_step = candidate_steps[np.argmin(np.abs(candidate_steps - unwrap_step_prev))]

                angle_unwrap[i] = angle_unwrap[i - 1] + chosen_step

                step_offset = angle_unwrap[i] - unwrap_val
                offset += step_offset

        if offset != 0.0 and len(angle_prev) < len(angle_unwrap):
            angle_unwrap[len(angle_prev):] += offset

        return angle_unwrap










    def _moving_average_scipy(self, data, window_size, mode):
        """
        使用 scipy.ndimage.uniform_filter1d 实现一维移动平均滤波。

        在重构调用中，数组下标表示空间采样点，因此这是按点数等权的空间平均，
        不考虑相邻点的实际物理间距，也不保存跨帧状态。window_size=1 不做平滑。
        size=3、reflect 时，内部点取左右邻点和自身的均值；末点取
        (倒数第二点 + 2 * 末点) / 3，首点同理。

        输入：
        - data: 输入的一维数组
        - window_size: 移动平均窗口大小
        - mode: 边界处理方式，如 'reflect'、'constant'、'nearest'、'mirror'、'wrap'

        输出：
        - output: 滤波后的数组
        """
        if window_size < 1:
            raise ValueError("Window size must be a positive integer.")
        if window_size > len(data):
            raise ValueError("Window size cannot be larger than the data length.")

        output = uniform_filter1d(data, size=window_size, mode=mode, cval=0.0, origin=0)
        return output

    def _KT_Hermit_interp_tang(self, data):
        """
        根据 self.ad_samp_dist 使用 CubicHermiteSpline对一维数据进行 Hermite 插值。

        输入：
        - data: 原始一维数据数组，其长度必须与 self.ad_samp_dist 一致

        输出：
        - interp_data: 插值后的数据数组
        - interp_samp_dist: 插值后的均匀采样距离数组
        """
        # 确保 samp_dist 是 NumPy 数组
        samp_dist = np.array(self.ad_samp_dist, dtype=float)

        # 检查数据有效性
        if not np.all(np.isfinite(samp_dist)):
            raise ValueError("samp_dist 包含非有限值（NaN 或 inf）")
        if not np.all(np.diff(samp_dist) > 0):
            raise ValueError("samp_dist 必须是递增的数组")

        # 创建插值后的样本点（根据更新后的 samp_dist 范围生成）
        interp_samp_dist = np.linspace(samp_dist.min(), samp_dist.max(), num=self.interp_num)

        data = np.array(data)
        # 检查数据和 samp_dist 的形状是否一致
        if data.shape[0] != len(samp_dist):
            raise ValueError(f"数据的长度与 samp_dist 的长度不一致")

        # 检查数据是否包含非有限值
        if not np.all(np.isfinite(data)):
            raise ValueError(f"数据包含非有限值（NaN 或 inf）")

        # 计算导数（数值梯度）
        try:
            dy = np.gradient(data, samp_dist, axis=0)
        except ValueError as e:
            raise ValueError(f"梯度计算失败：{e}")

        # 检查梯度是否包含非有限值
        if not np.all(np.isfinite(dy)):
            raise ValueError(f"梯度计算结果包含非有限值（NaN 或 inf）")

        # # 强制设置 x=0 处的导数为 0（确保在原点处与 x 轴相切）
        # dy[0] = 0

        # 创建 Hermite 插值对象并插值
        interp_data = []
        hermite_spline = CubicHermiteSpline(samp_dist, data, dy)
        # 通过均匀的 interp_samp_dist 进行重采样
        interp_data = hermite_spline(interp_samp_dist)

        return interp_data, interp_samp_dist

    def _interpolate_curvature(self, data, method=None, nonnegative=None):
        """按选定策略对曲率进行空间插值。

        ``hermite`` 保留历史实现；``pchip`` 使用单调保形插值，避免在
        相邻采样点之间因三次 Hermite 斜率而产生额外过冲。非负约束只在
        曲率插值结果上执行，不改变方向角，因此可单独比较插值与约束的影响。
        """
        method = self.curvature_interp if method is None else method
        nonnegative = self.nonnegative_curvature if nonnegative is None else bool(nonnegative)
        if method == 'hermite':
            interp_data, interp_samp_dist = self._KT_Hermit_interp_tang(data)
        elif method in {'pchip', 'scalar_pchip'}:
            samp_dist = np.asarray(self.ad_samp_dist, dtype=float)
            data = np.asarray(data, dtype=float)
            if samp_dist.ndim != 1 or data.ndim != 1 or data.shape[0] != samp_dist.size:
                raise ValueError("曲率数据与采样距离必须是一维且长度一致")
            if samp_dist.size < 2 or not np.all(np.isfinite(samp_dist)) or not np.all(np.diff(samp_dist) > 0):
                raise ValueError("samp_dist 必须包含至少两个严格递增的有限值")
            if not np.all(np.isfinite(data)):
                raise ValueError("曲率数据包含 NaN 或 inf")
            interp_samp_dist = np.linspace(samp_dist[0], samp_dist[-1], num=self.interp_num)
            interp_data = PchipInterpolator(samp_dist, data)(interp_samp_dist)
        else:
            raise ValueError("scalar curvature interpolation method must be 'hermite' or 'scalar_pchip'")

        if nonnegative:
            interp_data = np.maximum(np.asarray(interp_data, dtype=float), 0.0)
        return np.asarray(interp_data, dtype=float), np.asarray(interp_samp_dist, dtype=float)

    def _smooth_curvature_direction(self, magnitude, angle,
                                    coherence_low=0.25, coherence_high=0.75,
                                    antipodal_start=np.deg2rad(150.0)):
        """用当前帧的邻域向量修正方向，保留每个点的原始曲率模长。

        不读取或更新上一帧状态，静态标定/优化与动态重构使用同一规则。
        magnitude、angle 为已经过调用方校验的一维等长数组，模长须非负。
        窗口内的曲率大小自然成为方向平均的权重，低曲率角度噪声影响较小。

        rho = |平均向量| / 平均模长衡量方向一致性。rho 低时保留当前点方向，
        高时采用平均方向，中间用 smoothstep 连续过渡，避免优化时硬切换。
        修正角接近 ±pi 时逐渐退回原方向，避免反向分支选择不稳定。
        默认 rho 阈值 0.25/0.75 和反向保护起点150度是可调初值，非标定结果；
        连续权重并不意味着整条 PCHIP/参数优化链处处可微。

        如果领域方向一致性很低，保持原方向不变，避免在零曲率/数值抵消处制造不可靠的方向。
        如果领域方向接近当前方向的反向，逐渐退回原方向，避免在 ±π 处不稳定的分支选择。
        """
        if not 0 <= coherence_low < coherence_high <= 1:
            raise ValueError('coherence thresholds must satisfy 0 <= low < high <= 1')
        if not 0 < antipodal_start < np.pi:
            raise ValueError('antipodal_start must be between 0 and pi')
        # 曲率向量的两分量表示，便于窗口平均和插值。cos/sin 对 2π 周期不变，可避免直接插值角度时的 ±π 分支跳变。
        components = magnitude[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
        averaged = np.column_stack([
            self._moving_average_scipy(components[:, j], self.move_filter, 'reflect')
            for j in range(2)
        ])
        # 平均模长用于计算方向一致性，避免零曲率点的噪声影响。平均向量模长用于计算最终的方向修正。
        mean_magnitude = self._moving_average_scipy(magnitude, self.move_filter, 'reflect')
        # 计算平均向量的模长，用于衡量方向一致性。
        resultant = np.hypot(averaged[:, 0], averaged[:, 1])
        # 零曲率/数值抵消处不计算有效方向，不用 epsilon 归一化来制造方向。
        tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, mean_magnitude)
        # 仅在平均模长和平均向量模长都大于阈值时才计算方向一致性，否则保持原方向。如果曲率接近零，方向本身没有可靠物理意义，因此不进行方向修正。
        valid = (mean_magnitude > tolerance) & (resultant > tolerance)
        # 方向一致性计算
        coherence = np.zeros_like(magnitude)
        np.divide(resultant, mean_magnitude, out=coherence, where=valid)
        # 根据阈值归一化到 [0, 1]，并使用 smoothstep 函数平滑过渡。
        fraction = np.clip((coherence - coherence_low) / (coherence_high - coherence_low), 0., 1.)
        weight = fraction * fraction * (3. - 2. * fraction) #smoothstep function: 3f^2 - 2f^3
        # 计算当前点与平均方向的角度差，并用arctan2将其限制在 [-π, π] 范围内，避免 ±π 分支跳变。
        delta = np.zeros_like(angle)
        difference = np.arctan2(averaged[valid, 1], averaged[valid, 0]) - angle[valid]
        delta[valid] = np.arctan2(np.sin(difference), np.cos(difference))
        # 当 delta 接近 ±π 时，逐渐退回原方向，避免反向分支选择不稳定。
        fraction = np.clip((np.pi - np.abs(delta)) / (np.pi - antipodal_start), 0., 1.)
        weight *= fraction * fraction * (3. - 2. * fraction)
        corrected_angle = angle + weight * delta
        # 只借用平均后的方向；零模长仍为零，不把邻点的大曲率摊到当前点。
        return magnitude[:, None] * np.column_stack((np.cos(corrected_angle), np.sin(corrected_angle)))

    def _interpolate_curvature_vector(self, magnitude, angle, method):
        """对局部弯曲平面内的曲率向量进行空间滤波和插值。

        输入 magnitude 是补齐虚拟点、未经 Kalman 滤波的曲率；理想曲率模式传入理论值。
        angle 是 angle_unwrap，不是预先做过角度窗口平均的 ang_filter。
        两者均与 ad_samp_dist 一一对应，含调用方添加的虚拟点。

        流程：极坐标转两分量 → 可选窗口估计方向并保护抵消/反向情况 → 恢复原始模长
        → 各分量线性/PCHIP 插值
        → 还原模长与方向。分量不是最终三维曲线的全局 X、Y 坐标。
        cos/sin 对 2π 周期不变，可避免直接插值角度时的 ±π 分支跳变。

        注意：move_filter_for_vector=True 时，本函数显式执行窗口平均；这不是 PCHIP
        内部操作。窗口仅修正方向，输入节点的模长保持不变；PCHIP点间向量仍可能抵消，
        因此本方法不保证点间模长或最终形状半径完全无偏。不需要上一帧数据。
        返回 cur_interp（曲率模长）、ang_interp（弧度）、interp_dist（米）。
        """
        samp_dist = np.asarray(self.ad_samp_dist, dtype=float)
        magnitude = np.asarray(magnitude, dtype=float)
        angle = np.asarray(angle, dtype=float)
        if magnitude.ndim != 1 or angle.ndim != 1 or magnitude.shape != angle.shape:
            raise ValueError("magnitude and angle must be one-dimensional arrays with equal length")
        if magnitude.size != samp_dist.size:
            raise ValueError("curvature-vector samples must match ad_samp_dist")
        if (
            samp_dist.size < 2
            or not np.all(np.isfinite(samp_dist))
            or not np.all(np.diff(samp_dist) > 0.0)
            or not np.all(np.isfinite(magnitude))
            or not np.all(np.isfinite(angle))
        ):
            raise ValueError("curvature-vector interpolation inputs must be finite and ordered")
        if method not in {'linear', 'pchip'}:
            raise ValueError("curvature-vector interpolation method must be 'linear' or 'pchip'")
        if np.any(magnitude < 0):
            raise ValueError('curvature magnitude must be nonnegative')
        # 可选三点窗口只提供方向候选；关闭时直接插值原始曲率向量分量。
        # 两种路径都保留调用方输入的 magnitude，且静态/动态没有历史状态依赖。
        if self.move_filter_for_vector:
            components = self._smooth_curvature_direction(magnitude, angle)
        else:
            components = magnitude[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
        interp_dist = np.linspace(samp_dist[0], samp_dist[-1], num=self.interp_num)
        if method == 'linear':
            interpolated = np.column_stack(
                [np.interp(interp_dist, samp_dist, components[:, component]) for component in range(2)]
            )
        else:
            # axis=0 沿空间点插值，两列分量独立处理。PCHIP 本身不做窗口平均，
            # 输入节点由 move_filter_for_vector 决定；点间向量模长仍可能因方向变化而下降。
            # 等距输出网格未必包含原采样节点，输出峰值可能低于输入节点峰值。
            interpolated = PchipInterpolator(samp_dist, components, axis=0)(interp_dist)

        # 将插值后的局部曲率向量还原成后续标架重构需要的曲率大小和方向。
        cur_interp = np.hypot(interpolated[:, 0], interpolated[:, 1])
        ang_interp = np.arctan2(interpolated[:, 1], interpolated[:, 0])
        return cur_interp, ang_interp, interp_dist

    def _Kalman_filter(self, data, Rk, Bk):
        """
        对一维数据执行简化 Kalman 滤波。

        _reconstruct_frame 传入单帧 magnitude，因此循环沿光纤空间点推进，
        每次调用重新初始化，不是对同一传感点进行跨帧 Kalman 滤波。
        输出为当前观测与前一空间点滤波结果的加权组合，仅向后传递信息。
        默认 Rk=0.0001、Bk=0.01 时稳态增益约 0.9902，空间平滑较弱。
        当前重构入口只调用 magnitude 的 Kalman，未调用 angle 的 Kalman。

        输入：
        - data: 输入的一维数组
        - Rk: 观测噪声参数，越小则输出越贴近原始观测值。
        - Bk: 状态转移误差参数，越大对新观测的响应通常更快。

        输出：
        - output: 滤波后的数组
        """
        data = np.asarray(data)
        kalman_num = data.shape[0]
        data_last = data[0]
        pk_last = Bk
        output = np.zeros_like(data)
        for kal in range(kalman_num):
            xk_bar = data_last
            pk_bar = pk_last + Bk
            Kgain = pk_bar/(pk_bar+Rk)

            output[kal] = xk_bar+Kgain*(data[kal]-xk_bar)
            pk_last = pk_bar - Kgain*pk_bar

            data_last = output[kal]
        return output
