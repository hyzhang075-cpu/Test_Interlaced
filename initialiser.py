import numpy as np
from scipy import stats

from sensor import Sensor

from utils import dict_recur_extract_path, dict_recur_flatten, dict_recur_unflatten

class Initialiser(Sensor):
    """
    形状重构的初始化模块
    """
    def __init__(self, segmentation, connect=True, global_phases_set=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.segmentation = segmentation # 对应原本的segments
        self.num_segments = len(segmentation)
        self.connect = connect
        self.global_phases_set = global_phases_set
        if not self.global_phases_set:
            self.global_phases_set = 0.0
            self.rot_seg_fit(verbose=False, update_phase=True) # 初始化正确的全局相位与全局扭转系数

    def rot_seg_fit(self, verbose, update_phase):
        """
        初始化全局相位，并基于 circ / S / rot0 标定数据联合估计初始分段扭转系数。

        步骤：
        1. 在初始化模式下生成动态角度基准 theta_dynamic。
        2. 对所有存在的标定形状递归计算 angle raw / unwrap / init。
        3. 将 circ / S / rot0 中的初始角度递归汇总，用于全局相位估计。
        4. 将 circ / S / rot0 中的 unwrap 角度曲线递归拉平后联合做：
        - 全局分段拟合
        - 个体分段拟合
        5. 将全局拟合得到的各段斜率取负，作为初始扭转系数 self.slopes_init[direct]。
        """

        fit_shapes = {'circ', 'S', 'rot0'}

        theta_dynamic = self._calculate_dynamic_angles(mode='init', segments=None, seg_slopes=None)

        self.slopes_init = {}
        self.angle_results_raw = {}
        self.angle_results_unwrap = {}
        self.global_fit_direct = {}
        self.individual_fits_direct = {}

        for calib_shape in ['circ', 'S', 'rot0', 'end']:
            if calib_shape in self.calib_shapes:
                self.angle_results_raw[calib_shape] = {}
                self.angle_results_unwrap[calib_shape] = {}

        for direct in self.exp_directs:
            phase_pool = []
            fit_angle_data_flat = {}

            if verbose:
                print(f"-------------------- {direct}方向初始化开始 --------------------")

            for calib_shape in ['circ', 'S', 'rot0', 'end']:
                if calib_shape not in self.calib_shapes:
                    continue
                if direct not in self.processed_Data[calib_shape]:
                    continue

                calib_data = self.processed_Data[calib_shape][direct]
                if calib_shape in ['circ', 'S']:
                    target_data = {radius: calib_data[radius] for radius in self.radii[direct][calib_shape] if radius in calib_data}
                else:
                    target_data = calib_data

                bundled_tree = dict_recur_extract_path(target_data,
                    lambda leaf, path: (
                        lambda strain_data: (
                            lambda angle: (angle, np.unwrap(angle), np.unwrap(angle)[0])
                        )(self._calculate_curvature_and_angle(epsilon=strain_data, alpha_dynamic=theta_dynamic)[1])
                    )(np.asarray(leaf).T))
                angle_raw = dict_recur_extract_path(bundled_tree, lambda leaf, path: leaf[0])
                angle_unwrap = dict_recur_extract_path(bundled_tree, lambda leaf, path: leaf[1])
                init_angle = dict_recur_extract_path(bundled_tree, lambda leaf, path: leaf[2])

                self.angle_results_raw[calib_shape][direct] = angle_raw
                self.angle_results_unwrap[calib_shape][direct] = angle_unwrap

                # ---------- circ / S / rot0 联合参与 phase 与 fit ----------
                if calib_shape in fit_shapes:
                    flat_init = dict_recur_flatten(init_angle)
                    for path, val in flat_init.items():
                        # ---------- inline _phase_leaf_ok ----------
                        use_for_phase = True
                        if calib_shape in ['circ', 'S']:
                            radius = path[0]
                            if radius == 0:
                                use_for_phase = False

                        if use_for_phase:
                            phase_pool.append(val)

                    flat_unwrap = dict_recur_flatten(angle_unwrap)
                    for path, curve in flat_unwrap.items():
                        fit_key = (calib_shape,) + path
                        fit_angle_data_flat[fit_key] = curve

                if verbose:
                    print(f"calib_shape: {calib_shape}")
                    print(f"initial angle: {init_angle}")

            # ---------- 联合更新全局相位 ----------
            if direct == 0 and update_phase and len(phase_pool) > 0:
                self.global_phases_set = -np.mean(phase_pool)

            # ---------- 联合做全局 / 个体分段拟合 ----------
            if len(fit_angle_data_flat) > 0:
                global_fit = self._compute_global_fit(angle_data=fit_angle_data_flat)
                individual_fits_flat = self._compute_individual_fits(angle_data=fit_angle_data_flat)

                self.global_fit_direct[direct] = global_fit
                self.individual_fits_direct[direct] = dict_recur_unflatten(individual_fits_flat)
                self.slopes_init[direct] = [-group[0] for group in global_fit]

                if verbose:
                    print(f"global phase: {self.global_phases_set}")
                    print(f"segment fit: {global_fit}")
                    slopes_detail = dict_recur_extract_path(self.individual_fits_direct[direct], lambda groups, path: [group[0] for group in groups])
                    print(f"slopes: {slopes_detail}")

    def _directional_unwrap(self, angles):
        """
        自定义的定向 unwrap 函数。
        从第三个点开始限制 unwrap 的方向：
        如果初始趋势是增加，则后续所有相位跳变只能向增加的方向 unwrap；
        如果初始趋势是减小，则只能向减小的方向 unwrap。
        """
        angles = np.asarray(angles, dtype=np.float64)
        if len(angles) < 3:
            return np.unwrap(angles)

        unwrapped = np.zeros_like(angles)
        # 前两个点使用标准 unwrap
        unwrapped[:2] = np.unwrap(angles[:2])

        # 确定初始趋势 (1 表示增加, -1 表示减小)
        # 也可以根据实际业务逻辑修改趋势的判断依据
        trend = 1 if unwrapped[1] >= unwrapped[0] else -1

        for i in range(2, len(angles)):
            # 计算当前点与前一个 unwrapped 点的原始差值
            diff = angles[i] - (unwrapped[i-1] % (2 * np.pi))

            # 将差值映射到 [-pi, pi] 之间
            diff = (diff + np.pi) % (2 * np.pi) - np.pi

            # 根据趋势强制限制跳变方向
            if trend == 1 and diff < 0:
                # 如果趋势是增加，但当前差值为负，强制加 2*pi 使其变为正向增长
                diff += 2 * np.pi
            elif trend == -1 and diff > 0:
                # 如果趋势是减小，但当前差值为正，强制减 2*pi 使其变为负向增长
                diff -= 2 * np.pi

            unwrapped[i] = unwrapped[i-1] + diff

        return unwrapped

    def _calculate_dynamic_angles(self, mode, segments, seg_slopes):
        """
        计算随传感距离变化的各重构通道动态角度。

        若无特殊说明，默认认为用于重构的外围通道在圆周上均匀分布（active_angles=None）。

        步骤：
        1. 根据 active_angles 或默认均匀分布方式，构造各重构通道的初始角度 theta_0，
        并叠加全局相位 self.global_phases_set。
        2. 若 mode == 'init'，则整条光纤上各点的动态角度恒等于 theta_0。
        3. 若 mode == 'recon'，则根据给定分段 segments 与各段扭转系数 seg_slopes，
        沿传感距离逐段累积相位变化，得到整条光纤上的动态角度分布。
        4. 对越界分段会直接跳过，不参与累积计算。

        输入：
        - mode: 计算模式
            - 'init': 初始化模式，动态角度恒等于全局相位补偿后的通道角度
            - 'recon': 重构模式，根据分段扭转系数逐段累积
        - segments: 分段列表，格式为 [(start, end), ...]，每段为索引区间
        - seg_slopes: 各分段对应的扭转系数列表

        输出：
        - theta_dynamic: 各重构通道在各采样点处的动态角度，
        形状为 (self.num_channels_recon, len(self.samp_dist))

        说明：
        - 当 active_angles 为 None 或长度与 self.num_channels_recon 不一致时，
        使用均匀分布的默认通道角度；
        - 本方法只计算参与重构的通道角度，不包含被 centre fibre 补偿后剔除的通道。
        """
        # 获取 active_angles 参数，如果不存在则为 None
        active_angles = self.active_angles
        
        # 判断是否传入了有效的 active_angles
        if active_angles is not None and len(active_angles) == self.num_channels_recon:
            # 将传入的角度(度)转换为弧度，并加上全局相位
            theta_0_list = [self.global_phases_set + np.deg2rad(ang) for ang in active_angles]
            theta_0 = np.array([theta_0_list]).T
        else:
            # 默认的均匀分布逻辑
            theta_0 = np.array([[self.global_phases_set + i * 2*np.pi/self.num_channels_recon for i in range(self.num_channels_recon)]]).T
            
        theta = np.zeros((self.num_channels_recon, len(self.samp_dist)))
        if mode == 'init':
            theta_dynamic = theta_0 + theta
        elif mode == 'recon':
            seg_frequencies = [slope for slope in seg_slopes]
            current_phase = theta_0
            last_position = self.samp_dist[0]
            # 按顺序处理每个分段
            for seg, freq in zip(segments, seg_frequencies):
                seg_start, seg_end = seg
                seg_slice = slice(seg_start, seg_end)
                
                if seg_start >= len(self.samp_dist) or seg_end > len(self.samp_dist):
                    continue

                # 计算相对距离（相对于累积基准位置）
                delta_dist = (self.samp_dist[seg_slice] - last_position)  #m
                # 计算当前分段相位（所有相同时计算）
                theta[:, seg_slice] = current_phase + np.ones((self.num_channels_recon, 1)) * (freq * delta_dist)
                # 更新基准相位和位置
                last_position = self.samp_dist[seg_end-1] if seg_end <= len(self.samp_dist) else last_position
                current_phase = theta[:, seg_end-1:seg_end] if seg_end <= len(self.samp_dist) else current_phase
            theta_dynamic = theta

        return theta_dynamic

    def _calculate_curvature_and_angle(self, epsilon, alpha_dynamic):
        """
        计算曲率向量的模与方向角，支持光纤角度动态变化。

        原理：
        - 基于各通道应变 epsilon 以及各通道对应的动态角度 alpha_dynamic，将应变投影到 x / y 两个方向上，得到曲率向量分量 kappa_x, kappa_y。
        - 再由此计算曲率模 curvature_magnitude 和方向角 curvature_angle。

        输入:
        - epsilon: 各通道应变数据，形状通常为 (self.num_channels_recon, 数据点数)
        - alpha_dynamic: 各通道动态角度数组，形状为 (self.num_channels_recon, 数据点数)

        输出:
        - curvature_magnitude: 曲率模数组
        - curvature_angle: 曲率方向角序列，单位为弧度，范围被映射到 [0, 2π)

        说明：
        - 当前实现中 curvature_magnitude 为 numpy 数组；
        - curvature_angle 经范围映射后返回为列表，其长度与数据点数一致。
        """
        # 计算曲率向量的 x 和 y 分量

        epsilon = np.asarray(epsilon, dtype=np.float64)
        alpha_dynamic = np.asarray(alpha_dynamic, dtype=np.float64)

        kappa_x = -2 / (self.num_channels_recon * self.R) * np.sum(epsilon * np.cos(alpha_dynamic), axis=0)
        kappa_y = -2 / (self.num_channels_recon * self.R) * np.sum(epsilon * np.sin(alpha_dynamic), axis=0)

        # 计算曲率模
        curvature_magnitude = np.sqrt(kappa_x**2 + kappa_y**2)

        # 计算曲率向量与坐标轴的夹角 (弧度)
        curvature_angle = np.arctan2(kappa_y, kappa_x)
        curvature_angle = [ang + 2*np.pi if ang < 0 else ang for ang in curvature_angle]    #转化到0-2pi

        return curvature_magnitude, curvature_angle
    
    def _compute_global_fit(self, angle_data):
        """
        对多个标定数据的自扭转角度-距离曲线进行分段全局线性拟合。
        支持任意段数 self.num_segments。
        """
        pooled_x = [[] for _ in range(self.num_segments)]
        pooled_y = [[] for _ in range(self.num_segments)]

        for key, angles in angle_data.items():
            # 兼容 fit_key = (calib_shape,) + path 的情况
            radius = key[1] if isinstance(key, tuple) and len(key) > 1 else key

            if radius in {0, 200, 300}:
                continue

            x_data, y_data, segments = self._segment_data(angles)

            for idx, (start, end) in enumerate(segments):
                if idx >= self.num_segments:
                    break
                pooled_x[idx].extend(x_data[start:end])
                pooled_y[idx].extend(y_data[start:end])

        combined_x = []
        combined_y = []
        merged_segments = []
        current = 0

        for xs, ys in zip(pooled_x, pooled_y):
            if len(xs) == 0:
                continue
            seg_len = len(xs)
            combined_x.extend(xs)
            combined_y.extend(ys)
            merged_segments.append((current, current + seg_len))
            current += seg_len

        if len(combined_x) == 0:
            return [(np.nan, np.nan)] * self.num_segments

        fit_results = self._piecewise_linear_fit(
            y_data=np.asarray(combined_y, dtype=float),
            x_data=np.asarray(combined_x, dtype=float),
            segments=merged_segments
        )

        # 若某些段完全没有数据，可补齐到固定长度
        if len(fit_results) < self.num_segments:
            fit_results += [(np.nan, np.nan)] * (self.num_segments - len(fit_results))

        return fit_results
    
    def _compute_individual_fits(self, angle_data):
        '''
        对每个标定半径的自扭转角度-距离数据分别进行分段线性拟合。

        说明：
        - 每个半径独立进行分段，不与其他半径的数据混合。
        - 半径为 0 的数据不参与个体拟合。
        - 分段方式由 _segment_data() 按当前 segmentation 自动生成。

        输入:
        - angle_data: 各半径对应的角度数据字典，键为半径，值为角度数组

        输出:
        - individual_fits: 字典，键为半径，值为该半径各分段拟合结果的列表
        - 每个分段拟合结果为 (斜率, 截距)
        '''
        individual_fits = {}

        for radius, angles in angle_data.items():
            if radius == 0:
                continue

            x_data, y_data, segments = self._segment_data(angles)
            individual_fits[radius] = self._piecewise_linear_fit(y_data, x_data, segments)

        return individual_fits
    
    def _piecewise_linear_fit(self, y_data, x_data, segments):
        """
        对给定数据按指定分段逐段进行线性回归。

        输入：
        - y_data: 因变量数组
        - x_data: 自变量数组
        - segments: 分段索引区间列表，格式为 [(start, end), ...]

        输出：
        - results: list of tuples
        - 每个元素为对应分段的 (斜率, 截距)
        - 若某段有效点数不足 2，则返回 (np.nan, np.nan)
        """
        results = []
        for start, end in segments:
            seg_slice = slice(int(start), int(end))
            x_seg = x_data[seg_slice]
            y_seg = y_data[seg_slice]
            if len(x_seg) < 2 or len(y_seg) < 2:
                results.append((np.nan, np.nan))
                continue
            slope, intercept, *_ = stats.linregress(x_seg, y_seg)
            results.append((float(slope), float(intercept)))
        return results

    def _segment_data(self, angles):
        """
        对单条自扭转角度-距离曲线生成拟合分段结构。
    
        基础规则：
        - self.segmentation = [n1, n2, ..., nk]
        - 先对 angles 做 np.unwrap；
        - 再根据 self.zero_init 决定是否跳过第一个采样点；
        - 再由 _calculate_segments(total_points) 生成不重叠基础分段。
    
        connect 规则：
        - 若 self.connect == False：
            直接返回不重叠基础分段 base_segments。
        - 若 self.connect == True：
            在拟合阶段生成带 1 点重叠的分段；
            首段不变，此后每一段向左拓展 1 个点，使相邻分段首尾相连。
    
        connect=True 时的拓展规则：
        - 第 0 段保持不变；
        - 从第 1 段开始，每一段 start 向左拓展 1 个点；
        - end 不变；
        - 返回区间仍采用左闭右开格式 [(start, end), ...]。
    
        输出：
        - x_data: 对应采样距离数组
        - y_data: unwrap 后的角度数组
        - segments: 用于拟合的分段索引区间列表，格式为 [(start, end), ...]
        """
        unwrapped = np.unwrap(angles)
        raw_valid_length = min(len(unwrapped), len(self.samp_dist))
    
        if self.zero_init:
            x_data = self.samp_dist[:raw_valid_length]
            y_data = unwrapped[:raw_valid_length]
        else:
            x_data = self.samp_dist[1:raw_valid_length]
            y_data = unwrapped[1:raw_valid_length]
    
        total_points = min(len(x_data), len(y_data))
        x_data = x_data[:total_points]
        y_data = y_data[:total_points]
    
        if total_points <= 0:
            return x_data, y_data, []
    
        # Step 1: 生成不重叠基础分段
        base_segments = self._calculate_segments(total_points)
    
        # Step 2: 如果不启用连接，则直接返回不重叠分段
        if not self.connect:
            return x_data, y_data, base_segments
    
        # Step 3: 启用连接时，从第二段开始，每段向左拓展 1 个点
        if len(base_segments) <= 1:
            return x_data, y_data, base_segments
    
        segments = []
    
        for idx, (start, end) in enumerate(base_segments):
            start = int(start)
            end = int(end)
    
            # 首段不变；此后每段向左拓展一个点
            if idx > 0:
                start = max(start - 1, 0)
    
            if end > start:
                segments.append((start, end))
    
        return x_data, y_data, segments

    def _calculate_segments(self, total_points):
        """
        根据 segmentation 生成最终用于重构的不重叠分段结构。

        新版 segmentation 规则：
        - self.segmentation = [n1, n2, ..., nk]
        - 列表长度 k 直接表示段数；
        - 每个元素 ni 表示对应分段的点数。

        步骤：
        1. 检查 total_points 是否合法；若 total_points <= 0，则返回 []。
        2. 检查 segmentation 是否为合法的非负整数列表。
        3. 若 segmentation 非法、为空或全为 0，则退化为单段 [(0, total_points)]。
        4. 按 segmentation 从头到尾依次生成不重叠分段。
        5. 若 segmentation 总长度小于 total_points，则将剩余点并入最后一段。
        6. 若 segmentation 总长度大于 total_points，则在 total_points 处截断。

        输入：
        - total_points: 当前数据总点数

        输出：
        - segments: 不重叠的分段区间列表，格式为 [(start, end), ...]

        说明：
        - 返回的 segments 采用左闭右开区间；
        - 分段个数通常与 len(self.segmentation) 一致；
        - 若总长度不足，则最后一段自动吸收剩余点，避免有点未被覆盖；
        - 若总长度超出，则自动裁剪到 total_points。
        """
        if total_points <= 0:
            return []

        seg_cfg = self.segmentation

        # segmentation 必须是非空 list / tuple
        if not isinstance(seg_cfg, (list, tuple)) or len(seg_cfg) == 0:
            return [(0, total_points)]

        # 尝试转成整数列表
        try:
            lengths = [int(x) for x in seg_cfg]
        except (TypeError, ValueError):
            return [(0, total_points)]

        # 不允许负数
        if any(x < 0 for x in lengths):
            return [(0, total_points)]

        # 去掉 0 长度段；如果全为 0，则退化为单段
        lengths = [x for x in lengths if x > 0]
        if len(lengths) == 0:
            return [(0, total_points)]

        segments = []
        start = 0

        for i, seg_len in enumerate(lengths):
            if start >= total_points:
                break

            # 最后一段吸收所有剩余点，避免尾部漏点
            if i == len(lengths) - 1:
                end = total_points
            else:
                end = min(start + seg_len, total_points)

            if end > start:
                segments.append((start, end))

            start = end

        return segments if segments else [(0, total_points)]
