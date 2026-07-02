import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt

from initialiser import Initialiser

from utils import dict_recur_paths, dict_recur_unflatten, dict_recur_extract_path

class Reconstructor(Initialiser):
    """
    形状重构模块
    """
    def __init__(self, soft=True, frame_rule='TNB', rot_ang=0.0, turn_threshold=2.0, init_weight=1.0, weights=None, slopes=None, min_consecutive=3, sharpness=4.8, end_start=-6, end_weights=False, thresholds=[0.8, 1.5, 2.0, 3.0, 4.0], 
                 TNB_threshold=0.15, interp_num=60, move_filter=3, kalman_params={'angle': {'Rk': 0.001, 'Bk': 0.001}, 'magnitude': {'Rk': 0.0001, 'Bk': 0.01}},  
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rot_seg_fit(verbose=True, update_phase=False) # 打印初始信息，但不更新全局相位
        self.magnitude_ratio = dict_recur_extract_path(self.magnitude_results, lambda arr, path: 1.0 if path[-1] == 0 else float((1000 / path[-1]) / np.mean(arr)))
        # self.magnitude_ratio = dict_recur_extract_path(self.magnitude_results, lambda arr, path: 1.0)
        print(f" magnitude_ratio: {self.magnitude_ratio}")
        # 重构所需参数
        self.soft = soft
        self.frame_rule = frame_rule
        self.turn_threshold = turn_threshold
        self.thresholds = thresholds # 曲率分级阈值
        self.min_consecutive = min_consecutive
        self.sharpness = sharpness

        self.num_levels = len(self.thresholds) + 1 
        self.base_segments = self._calculate_segments(total_points=self.total_points)
        self.end_mask = np.arange(self.total_points) >= self.total_points + end_start
        self.TNB_threshold = TNB_threshold # FS重构时直接复制上一点的曲率大小阈值

        # 需要优化的应变权重、扭转系数以及旋转角度
        self.end_weights = end_weights # 是否优化末端权重
        self.init_weight = init_weight
        if not weights:
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
        self.kalman_params = kalman_params
        self.interp_num = interp_num

    def reconstruct_shapes(self, calib_shape, direct, verbose=False):
        """
        对指定标定形状与方向下的所有数据进行三维形状重构。
        """

        # ---------- 初始准备 ----------
        calib_Data = self.processed_Data[calib_shape][direct]
        weight = self.weights[direct]
        slope = self.slopes[direct]

        interp_dist = None
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
            magnitude_ratio = self.magnitude_ratio[calib_shape][direct][radius]

            if isinstance(magnitude_ratio, dict):
                magnitude_ratio = magnitude_ratio[keys[1]]

            _, _, _, interp_dist, r, _, _, _ = self._reconstruct_frame(keys=keys, calib_shape=calib_shape, calib_Data=calib_Data, weight=weight, slope=slope, magnitude_ratio=magnitude_ratio, init_T=[1.0, 0.0, 0.0], verbose=verbose)

            r_postrot = r @ rot_matrix.T
            flat_results[path] = r_postrot

            if verbose:
                print(f"keys: {keys}")
                print(f"r: {r}")
                print()

        # ---------- 还原为原本嵌套结构 ----------
        r_results_postrot = dict_recur_unflatten(flat_results)

        return r_results_postrot, interp_dist

    def _reconstruct_frame(self, keys, calib_shape, calib_Data, weight, slope, magnitude_ratio, init_T, verbose=False, unwrap_threshold=0.65, align_threshold=0.15, angle_prev=None, angle_unwrap_prev=None):
        # ---------- 计算重构使用的曲率大小与方向 ----------
        magnitude, angle = self._Strain_2_Cur(keys=keys, calib_shape=calib_shape, calib_Data=calib_Data, weight=weight, slope=slope, magnitude_ratio=magnitude_ratio, verbose=verbose)

        if angle_prev is not None and magnitude[0] < align_threshold:
            angle[0] = angle_prev[0]
        
        if self.zero_init:
            magnitude = np.insert(magnitude, 0, magnitude[0])  #在第一个位置复制第一个值
            angle = np.insert(angle, 0, angle[0])   #把原点的弯曲角补齐
        else:
            angle[0] = angle[1]    

        # ---------- 重构前预处理 ----------
        angle_unwrap = self._framewise_unwrap_limit(angle=angle, angle_prev=angle_prev, angle_unwrap_prev=angle_unwrap_prev, unwrap_threshold=unwrap_threshold)
        ang_filter = self._moving_average_scipy(data=angle_unwrap, window_size=self.move_filter, mode='reflect')
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
            ang_interp, interp_dist = self._KT_Hermit_interp_tang(data=ang_filter)
            cur_interp, interp_dist = self._KT_Hermit_interp_tang(data=cur_kal)
            self.cur_interp = cur_interp
        except ValueError as e:
            # 将上下文信息拼接到原有的报错信息中
            raise ValueError(f"数据追踪信息 -> {context_str} | 原始报错: {e}")
        interp_ss = np.diff(interp_dist)

        # 计算初始N向量
        init_N = np.array([0, np.cos(ang_interp[0]), np.sin(ang_interp[0])])

        if verbose:
            print(f"interp_dist: {interp_dist}")
            print(f"ang_interp: {ang_interp}")
            print(f"cur_interp: {cur_interp}")
            print(f"init_N: {init_N}")

        if self.frame_rule == 'TNB':
            # Frenet-Serret重构
            ang_diff = np.diff(ang_interp)
            torsion = (ang_diff - 2 * np.pi * np.floor((ang_diff + np.pi) / (2 * np.pi))) / interp_ss
            if verbose:
                print(f"torsion: {torsion}")
            r, T, N, B = self._F_S_frame(kappa=cur_interp, tau=torsion, ss=interp_ss, init_T=init_T, init_N=init_N)  
            return angle, angle_unwrap, cur_interp, interp_dist, r, T, N, B
        elif self.frame_rule == 'TNN':
            # 平行标架重构
            r, T, N1, N2 = self._parallel_frame(kappa=cur_interp, angles=ang_interp, ss=interp_ss, init_T=init_T, init_N=init_N)   
            return angle, angle_unwrap, cur_interp, interp_dist, r, T, N1, N2
        
    def _framewise_unwrap_limit(self, angle, angle_prev=None, angle_unwrap_prev=None, unwrap_threshold=0.65):
        """
        与 diagnostics 中 make_framewise_unwrap_processor 的核心逻辑等价。

        规则：
        - 先由 np.unwrap 得到 angle_unwrap；
        - 如果当前帧和上一帧在同一个相邻段 i-1 -> i 上的 raw step 变化小于 unwrap_threshold，
        则当前段不单纯接受 np.unwrap 的分支，而是选择最接近上一帧 unwrapped step 的 2π 分支；
        - 若当前点发生相位分支切换，则把 offset 累加并传递给后续点。
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

    def _Strain_2_Cur(self, keys, calib_shape, calib_Data, weight, slope, magnitude_ratio=1.0, verbose=False):
        """
        将一组应变数据转换为最终用于重构的曲率大小与方向角。

        步骤：
        1. 取 slope[0] 作为基础分段扭转系数 base_slopes，先计算基础曲率大小与方向角。
        2. 根据基础曲率大小进行曲率分级：
        - soft=True 时，使用 sigmoid 分数与 _alloc_mass() 得到软分配 masses；
        - soft=False 时，根据 thresholds 与 min_consecutive 找到硬覆盖区域，并生成 one-hot masses。
        3. 使用 masses 在不同曲率等级之间选择/混合对应的 weight 与 slope。
        4. 按曲率等级权重对原始应变进行加权，得到 weighted_strain。
        5. 根据 _get_base() 返回的 turn_segments，将各等级在基础分段上的 slope 组合成逐点扭转系数映射 slope_map。
        6. 根据 slope_map 生成动态分段结构与对应分段扭转系数。
        7. 基于新的动态角度分布与加权后的应变，重新计算最终曲率大小和方向角。

        输入：
        - keys:
            - None：直接使用 calib_Data.T
            - [key1]：使用单层数据，如 circ / S 下的单组数据
            - [key1, key2]：使用双层嵌套数据，如 end / rot0 / up / down 下的单组数据
        - calib_shape: 当前方向下的标定形状类型
        - calib_Data: 当前方向下的标定数据字典
        - weight: 当前方向下的曲率等级权重
        - slope: 当前方向下的分级分段扭转系数
        - verbose: 是否打印详细中间结果

        输出：
        - magnitude: 最终曲率大小数组
        - angle: 最终曲率方向角数组

        说明：
        - 当 calib_shape == 'S' 时，当前实现会区分普通段与转折段，并从 slope 中取两组分段扭转系数进行组合
        - 其他形状下仅使用一组基础分段扭转系数
        """
        # ---------- 初始准备 ----------
        base_slopes = slope[0]

        # ---------- 使用基础扭转系数计算曲率 ----------
        strain_data, magnitude, angle, turn_segments = self._get_base(keys=keys, calib_Data=calib_Data, base_slopes=base_slopes, verbose=False)

        # # 作图画出magnitude和angle的变化
        # plt.figure()
        # plt.plot(magnitude)
        # # plt.plot(angle)
        # plt.legend()
        # plt.show()

        # ---------- 进行曲率分级和分级参数计算 ----------
        score_magnitude = magnitude * magnitude_ratio
        # print(score_magnitude)
        if not self.soft:
            scores = None
            hard_level_idx = np.zeros(self.total_points, dtype=int)
            hard_regions = [None] * self.num_levels
            min_consecutive = max(1, int(self.min_consecutive))

            for level, threshold in enumerate(self.thresholds, start=1):
                consecutive_count = 0
                region_start = None
                region_indices = None

                for i, mag in enumerate(score_magnitude):
                    if mag > threshold:
                        if consecutive_count == 0:
                            region_start = i
                        consecutive_count += 1
                    else:
                        if consecutive_count >= min_consecutive:
                            region_indices = np.arange(region_start, i)
                            break
                        consecutive_count = 0
                        region_start = None

                if region_indices is None and consecutive_count >= min_consecutive:
                    region_indices = np.arange(region_start, self.total_points)

                if region_indices is not None and len(region_indices) > 0:
                    hard_level_idx[region_indices] = level
                    hard_regions[level] = region_indices

            masses = np.zeros((self.num_levels, self.total_points))
            masses[hard_level_idx, np.arange(self.total_points)] = 1.0
        else:
            scores = self._get_scores(magnitude=score_magnitude)
            masses = self._alloc_mass(scores=scores)
        if verbose:
            if scores is not None:
                print(f"scores: {scores}")
            else:
                print(f"hard level index: {hard_level_idx}")
                print(f"hard regions: {hard_regions}")
            for i in range(self.num_levels-1):
                print(f"Probability mass for curvature class {i+1}: {masses[i]}")

        # ---------- 计算加权后的应变 ----------
        weight_vecs = np.array([np.where(self.end_mask, weight[i+self.num_levels] if self.end_weights else weight[i], weight[i]) for i in range(self.num_levels)]) 
        final_weight = np.sum(masses * weight_vecs, axis=0)
        weighted_strain = strain_data * final_weight
        if verbose:
            print(f"final weight: {final_weight}")
            print(f"weighted strain: {weighted_strain}")

        # ---------- 重新分段并分配扭转系数 ----------
        slope_vecs = np.zeros((self.num_levels, self.total_points))
        if calib_shape == 'S':
            turn_masks = np.zeros(self.total_points, dtype=bool)
            for k, (t_start, t_end) in enumerate(turn_segments):
                if k % 2 != 0:
                    turn_masks[t_start:t_end] = True
            for i in range(self.num_levels):
                for j, (start, end) in enumerate(self.base_segments):
                    turn_mask = turn_masks[start:end]
                    slope_vecs[i, start:end] = np.where(turn_mask, slope[i, j + self.num_segments], slope[i, j])
        else:
            for i in range(self.num_levels):
                for j, (start, end) in enumerate(self.base_segments):
                        slope_vecs[i, start:end] = slope[i, j]
                    
        slope_map = np.sum(masses * slope_vecs, axis=0)
        final_segments, final_slope = self._generate_dynamic_segments(slope_map=slope_map)
        if verbose:
            print(f"final segments: {final_segments}")
            print(f"final slopes: {final_slope}")
            
        # ---------- 使用处理过的新应变、分段与扭转系数计算最终结果 ----------
        final_theta_dynamic = self._calculate_dynamic_angles(mode='recon', segments=final_segments, seg_slopes=final_slope)
        magnitude, angle = self._calculate_curvature_and_angle(epsilon=weighted_strain, alpha_dynamic=final_theta_dynamic)
        if verbose:
            print(f"final_theta_dynamic: {final_theta_dynamic}")
            print(f"magnitude: {magnitude}")
            print(f"angle: {angle}")
            print()

        return magnitude, angle
    
    def _get_base(self, keys, calib_Data, base_slopes, verbose=False):
        """
        使用基础分段扭转系数计算一组数据的基础曲率大小、方向角及转折分段。

        步骤：
        1. 根据基础分段扭转系数 base_slopes 与 self.base_segments 生成动态角度 theta_dynamic。
        2. 按 keys 指定的数据路径提取当前组应变数据。
        3. 基于原始应变与动态角度计算基础曲率大小和方向角。
        4. 对方向角做 unwrap。
        5. 根据相邻点方向角变化是否超过 self.turn_threshold，识别转折点并构造 turn_segments。

        输入：
        - keys:
            - None：直接使用 calib_Data.T
            - [key1]：使用单层数据，如 circ / S 下的单组数据
            - [key1, key2]：使用双层嵌套数据，如 end / rot0 / up / down 下的单组数据
        - calib_Data: 当前方向下的标定数据字典
        - base_slopes: 基础分段扭转系数列表
        - verbose: 是否打印详细中间结果

        输出：
        - strain_data: 原始应变矩阵
        - magnitude: 基础曲率大小数组
        - angle: 基础曲率方向角数组
        - turn_segments: 根据方向角转折检测得到的分段区间列表

        说明：
        - 本方法给出的 magnitude 与 angle 仅用于后续曲率分级、分段与权重计算
        - turn_segments 由方向角跳变阈值规则自动检测得到
        """
        theta_dynamic = self._calculate_dynamic_angles(mode='recon', segments=self.base_segments, seg_slopes=base_slopes)
        if keys is None:
            strain_data = calib_Data.T
        elif len(keys) == 1:
            radius = keys[0]
            strain_data = calib_Data[radius].T
        elif len(keys) == 2:
            radius, idx = keys
            strain_data = calib_Data[radius][idx].T
        magnitude, angle = self._calculate_curvature_and_angle(epsilon=strain_data, alpha_dynamic=theta_dynamic)
        angle_unwrap = np.unwrap(angle, discont=self.discont)
        turn_idx = [0]
        for i in range(1, len(angle)):
            if np.abs(angle_unwrap[i] - angle_unwrap[i-1]) > self.turn_threshold:
                turn_idx.append(i)
        turn_segments = [(turn_idx[i], turn_idx[i+1]) for i in range(len(turn_idx) - 1)]
        turn_segments.append((turn_idx[-1], self.total_points))
        if verbose:
            print(f"---------- radius: {'dynamic' if not keys else keys} ----------")
            print(f"strain_data: {strain_data}")
            print(f"theta_dynamic: {theta_dynamic}")
            print(f"magnitude(base): {magnitude}")
            print(f"angle(base): {angle}")
        return strain_data, magnitude, angle, turn_segments

    def _alloc_mass(self, scores):
        """
        根据各曲率等级分数分配概率质量。

        说明：
        - 从高等级到低等级逐层扣减，每一层从剩余质量 remained 中分配 score 比例的质量，最终最低等级使用剩余的全部质量。
        - 返回结果各层质量之和在每个点上应接近 1，并最终裁剪到 [0, 1]。

        输入：
        - scores: 各曲率等级的分数字段，形状通常为 (num_levels - 1, total_points)

        输出：
        - final_masses: 各等级的概率质量分布，形状为 (num_levels, total_points)
        """
        masses = []
        remained = np.ones(self.total_points)
        for score in scores[::-1]:
            mass = remained * score
            masses.append(mass)
            remained -= mass
        final_masses = np.vstack((remained, np.asarray(masses[::-1])))
        return np.clip(final_masses, 0.0, 1.0)

    def _get_scores(self, magnitude):
        """
        根据曲率大小计算各等级的软分级分数。

        说明：
        - 对每个阈值使用 sigmoid 函数计算超过该阈值的程度。
        - sharpness 控制 sigmoid 的陡峭程度。
        - 若 min_consecutive > 1，则再对每个等级分数做一次移动平均，以增强连续性。

        输入：
        - magnitude: 曲率大小数组

        输出：
        - scores: 各等级对应的分数数组，形状通常为 (num_levels - 1, total_points)
        """
        scores = []
        for i in range(self.num_levels-1):
            level_score = 1.0 / (1.0 + np.exp(-self.sharpness * (np.asarray(magnitude) - self.thresholds[i])))
            if self.min_consecutive > 1:
                level_score = self._moving_average_scipy(data=level_score, window_size=self.min_consecutive, mode='constant')
            scores.append(level_score)
        return np.asarray(scores)
    
    def _generate_dynamic_segments(self, slope_map):
        """
        根据逐点扭转系数映射生成动态分段结构。

        输入：
        - slope_map: 每个采样点处的扭转系数数组

        输出：
        - final_segments: 动态分段区间列表，格式为 [(start, end), ...]
        - final_slope: 每个分段对应的扭转系数列表
        """
        final_segments = []
        final_slope = []
        current_slope = slope_map[0]
        segment_start = 0
        for i in range(1, self.total_points):
            if slope_map[i] != current_slope:
                final_segments.append((segment_start, i))
                final_slope.append(current_slope)
                current_slope = slope_map[i]
                segment_start = i
        final_segments.append((segment_start, self.total_points))
        final_slope.append(current_slope)
        return final_segments, final_slope
    
    def _moving_average_scipy(self, data, window_size, mode):
        """
        使用 scipy.ndimage.uniform_filter1d 实现一维移动平均滤波。

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
    
    def _Kalman_filter(self, data, Rk, Bk):
        """
        对一维数据执行简化 Kalman 滤波。

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
    
    def _F_S_frame(self, kappa, tau, ss, init_T, init_N):
        """
        使用 Frenet-Serret 标架公式递推计算三维曲线及 T/N/B 标架。

        说明：
        - 当某点曲率小于 self.TNB_threshold 时，认为局部形状变化过小，直接沿用上一点的 T/N/B。
        - 当前实现的返回顺序为 (r, N, T, B)。

        输入：
        - kappa: 曲率数组，形状为 (num_points,)
        - tau: 扭率数组，形状通常为 (num_points - 1,) 或插入后与步长对齐
        - ss: 相邻点之间的弧长间隔数组
        - init_T: 初始切向量
        - init_N: 初始法向量

        输出：
        - r: 三维曲线位置数组，形状为 (num_points, 3)
        - N: 法向量数组，形状为 (num_points, 3)
        - T: 切向量数组，形状为 (num_points, 3)
        - B: 副法向量数组，形状为 (num_points, 3)
        """
            # Define normlize func
        cal_norm = lambda v: v / np.linalg.norm(v)
        
        num_points = len(ss) + 1  #range函数会不包括这个值
        
        # 初始化 T, N, B, r
        r = np.zeros((num_points, 3))  # 三维位置向量
        T = np.zeros((num_points, 3))  # 切向量
        N = np.zeros((num_points, 3))  # 法向量
        B = np.zeros((num_points, 3))  # 副法向量
        
        # 起点的初始值
        T[0] = init_T
        N[0] = init_N
        B[0] = np.cross(init_T, init_N)

        # 在开头插入 0，保证各数组的长度相同
        ss = np.insert(ss, 0, 0)
        tau = np.insert(tau, 0, 0)
        
        # 迭代计算
        for i in range(1, num_points):
            if kappa[i] < self.TNB_threshold:
                N[i] = N[i-1]
                B[i] = B[i-1]
                T[i] = T[i-1]
            else:
                # 计算 N(i)
                numerator = ss[i] * tau[i] * B[i-1] - ss[i] * kappa[i] * T[i-1] + N[i-1]
                denominator = 1 + ss[i]**2 * tau[i]**2 + ss[i]**2 * kappa[i]**2
                N[i] = numerator / denominator
                N[i] = cal_norm(np.array(N[i]))
                
                # 计算 T(i)
                T[i] = ss[i] * kappa[i] * N[i] + T[i-1]
                T[i] = cal_norm(np.array(T[i]))

                # 计算 B(i)
                B[i] = -ss[i] * tau[i] * N[i] + B[i-1]
                B[i] = cal_norm(np.array(B[i]))

            # 计算 r(i)
            r[i] = ss[i] * T[i] + r[i-1]
        
        return r, T, N, B
    
    def _parallel_frame(self, kappa, angles, ss, init_T, init_N):
        """
        使用平行标架 / TNN 形式递推计算三维曲线及 T/N1/N2 标架。

        说明：
        - 根据 angles 相对于首点角度的变化，将曲率分解为 kappa1 与 kappa2。
        - N1 由 init_N 初始化，N2 由 np.cross(init_T, init_N) 初始化。
        - 当前实现不使用扭率 tau，也不使用 self.TNB_threshold。
        - 当前实现的返回顺序为 (r, T, N1, N2)。

        输入：
        - kappa: 曲率大小数组，形状为 (num_points,)
        - angles: 曲率方向角数组，形状为 (num_points,)
        - ss: 相邻插值点之间的弧长间隔数组，形状为 (num_points - 1,)
        - init_T: 初始切向量
        - init_N: 初始 N1 方向

        输出：
        - r: 三维曲线位置数组，形状为 (num_points, 3)
        - T: 切向量数组，形状为 (num_points, 3)
        - N1: 第一法向基数组，形状为 (num_points, 3)
        - N2: 第二法向基数组，形状为 (num_points, 3)
        """
        # Define normlize func
        cal_norm = lambda v: v / np.linalg.norm(v)
        
        num_points = len(ss) + 1  #range函数会不包括这个值
        
        # 初始化 T, N, B, r
        r = np.zeros((num_points, 3))  # 三维位置向量
        T = np.zeros((num_points, 3))  # 切向量
        N1 = np.zeros((num_points, 3))  # 法向量
        N2 = np.zeros((num_points, 3))  # 副法向量
        
        # 起点的初始值
        T[0] = init_T
        N1[0] = init_N
        N2[0] = np.cross(init_T, init_N) # 也可以反过来init_N\times init_T

        angles_rel = angles - angles[0]
        kappa1 = kappa * np.cos(angles_rel)
        kappa2 = kappa * np.sin(angles_rel)

        # 在开头插入 0，保证各数组的长度相同
        ss = np.insert(ss, 0, 0)

        # 迭代计算
        for i in range(1, num_points):
            numerator = T[i - 1] + ss[i] * (kappa1[i] * N1[i - 1] + kappa2[i] * N2[i - 1])
            denominator = 1.0 + ss[i]**2 * (kappa1[i]**2 + kappa2[i]**2)
            T[i] = numerator / denominator
            T[i] = cal_norm(np.array(T[i]))

            N1[i] = N1[i - 1] - ss[i] * kappa1[i] * T[i]
            N1[i] = cal_norm(N1[i])

            N2[i] = N2[i - 1] - ss[i] * kappa2[i] * T[i]
            N2[i] = cal_norm(N2[i])

            r[i] = ss[i] * T[i] + r[i-1]

        return r, T, N1, N2

    def _get_rot_x(self, direct, rot_dir=-1):
        """
        生成绕 x 轴旋转的旋转矩阵。

        输入：
        - direct: 当前方向角（度）

        输出：
        - rot_matrix: 3×3 的绕 x 轴旋转矩阵
        """
        sigma = np.deg2rad((direct) * rot_dir + self.rot_ang)
        rot_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(sigma), -np.sin(sigma)],
            [0.0, np.sin(sigma), np.cos(sigma)]
        ])
        return rot_matrix