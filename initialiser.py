import numpy as np
from scipy import stats

from sensor import Sensor

from utils import dict_recur_extract_path, dict_recur_flatten, dict_recur_unflatten

class Initialiser(Sensor):
    """
    形状重构的初始化模块
    """
    def __init__(self, discont, segmentation, connect=True, global_phases_set=None, interlaced_offset=None, unwrap_direct = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.discont = discont
        self.segmentation = segmentation # 对应原本的segments
        self.num_segments = len(segmentation)
        self.connect = connect
        self.global_phases_set = global_phases_set
        self.interlaced_offset = interlaced_offset
        self.unwrap_direct = unwrap_direct
        if not self.global_phases_set:
            self.global_phases_set = 0.0
            self.rot_seg_fit(verbose=False, update_phase=True) # 初始化正确的全局相位与全局扭转系数

    def rot_seg_fit(self, verbose, update_phase):
        """
        初始化全局相位，并基于 fit_shapes 中指定的标定数据估计初始分段扭转系数。

        当前实现中 fit_shapes = {'circ'}，因此只有 circ 数据参与：
        - global_phases_set 的估计；
        - global_fit_direct / individual_fits_direct 的分段拟合；
        - slopes_init[direct] 的初始化。

        S / rot0 / end 若存在，仍会递归计算并保存 magnitude / angle raw / unwrap / init 结果，
        但不会参与 phase_pool 或扭转系数拟合。
        """

        fit_shapes = {'circ'}

        theta_dynamic = self._calculate_dynamic_angles(
            mode='init',
            segments=None,
            seg_slopes=None
        )

        self.slopes_init = {}
        self.magnitude_results = {}
        self.angle_results_raw = {}
        self.angle_results_unwrap = {}
        self.global_fit_direct = {}
        self.individual_fits_direct = {}

        for calib_shape in ['circ', 'S', 'rot0', 'end']:
            if calib_shape in self.calib_shapes:
                self.magnitude_results[calib_shape] = {}
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

                # ---------- 递归路径展开，逐个数据计算 magnitude / angle raw / unwrap / init ----------
                target_data_flat = dict_recur_flatten(target_data)

                bundled_flat = {}

                for path, leaf in target_data_flat.items():
                    strain_data = np.asarray(leaf).T

                    magnitude, angle = self._calculate_curvature_and_angle(epsilon=strain_data, alpha_dynamic=theta_dynamic)

                    # angle_unwrap_i = np.unwrap(angle, discont=self.discont)
                    angle_unwrap_i = self._directional_unwrap(angle)
                    # print(f"calib_shape: {calib_shape}, path: {path}, \n raw angle: {angle}, \n unwrap angle: {angle_unwrap_i}")
                    init_angle_i = angle_unwrap_i[0]

                    bundled_flat[path] = (magnitude, angle, angle_unwrap_i, init_angle_i)

                # ---------- 将 flat 结果还原成原始树结构 ----------
                magnitude_raw = dict_recur_unflatten({path: bundle[0] for path, bundle in bundled_flat.items()})

                angle_raw = dict_recur_unflatten({path: bundle[1] for path, bundle in bundled_flat.items()})

                angle_unwrap = dict_recur_unflatten({path: bundle[2] for path, bundle in bundled_flat.items()})

                init_angle = dict_recur_unflatten({path: bundle[3] for path, bundle in bundled_flat.items()})

                self.magnitude_results[calib_shape][direct] = magnitude_raw
                self.angle_results_raw[calib_shape][direct] = angle_raw
                self.angle_results_unwrap[calib_shape][direct] = angle_unwrap

                # ---------- fit_shapes 中指定的标定形状参与 phase 与 fit；当前仅 circ ----------
                if calib_shape in fit_shapes:
                    for path, bundle in bundled_flat.items():
                        angle_unwrap_i = bundle[2]
                        init_angle_i = bundle[3]

                        if -np.pi <= init_angle_i <= np.pi:
                            init_angle_i = init_angle_i
                        elif init_angle_i < -np.pi:
                            init_angle_i += 2 * np.pi
                        else:
                            init_angle_i -= 2 * np.pi

                        # ---------- inline _phase_leaf_ok ----------
                        use_for_phase = True

                        if calib_shape in ['circ', 'S']:
                            radius = path[0]
                            if radius == 0:
                                use_for_phase = False

                        if use_for_phase:
                            phase_pool.append(init_angle_i)

                        fit_key = (calib_shape,) + path
                        fit_angle_data_flat[fit_key] = angle_unwrap_i

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
        # # 获取 active_angles 参数，如果不存在则为 None
        # active_angles = self.active_angles
        
        # # 判断是否传入了有效的 active_angles
        # if active_angles is not None and len(active_angles) == self.num_channels_recon:
        #     # 将传入的角度(度)转换为弧度，并加上全局相位
        #     theta_0_list = [self.global_phases_set + np.deg2rad(ang) for ang in active_angles]
        #     theta_0 = np.array([theta_0_list]).T
        # else:
        #     if self.interlaced_grating:
        #         if self.samp_dist_preset_sub is None:
        #             raise ValueError("交错刻栅模式下，samp_dist_preset_sub 不能为空")
        #         # 交错刻栅时，由于将外围通道的数据按照pre_samp_dist和sub_samp_dist的顺序合并，因此重建通道的角度也需要按照相同的顺序进行交错排列
        #         # self.num_channels_recon 已经在 Sensor 类的 __init__ 中根据是否交错刻栅进行了调整，因此这里直接使用 self.num_channels_recon 来计算均匀分布的角度即可，无需额外处理
        #         # 如果需要在全局相位基础上进行交错排列，可以在计算均匀分布的基础上添加一个交错偏移，例如：
        #         theta_0_1 = np.array([[self.global_phases_set + i * 2*np.pi/(self.num_channels_recon) for i in range(self.num_channels_recon)]]).T
        #         theta_0_2 = np.array([[self.global_phases_set + self.interlaced_offset + i * 2*np.pi/(self.num_channels_recon) for i in range(self.num_channels_recon)]]).T

        #         p1 = np.array(self.samp_dist_preset)
        #         p2 = np.array(self.samp_dist_preset_sub)
                
        #         # 按照各自的采样点数量进行水平平铺
        #         theta_0_1_full = np.repeat(theta_0_1, len(p1), axis=1)
        #         theta_0_2_full = np.repeat(theta_0_2, len(p2), axis=1)
                
        #         # 拼接后根据位置排序索引进行穿插重排
        #         theta_0_combined = np.concatenate((theta_0_1_full, theta_0_2_full), axis=1)
        #         p_combined = np.concatenate((p1, p2))
        #         sort_idx = np.argsort(p_combined)
                
        #         # 此时 theta_0 是一个形状为 (num_channels_recon, total_points) 的矩阵
        #         theta_0 = theta_0_combined[:, sort_idx]
            
        #     else:# 默认的均匀分布逻辑
        #         theta_0 = np.array([[self.global_phases_set + i * 2*np.pi/self.num_channels_recon for i in range(self.num_channels_recon)]]).T
            
        # theta = np.zeros((self.num_channels_recon, len(self.samp_dist)))
        # if mode == 'init':
        #     theta_dynamic = theta_0 + theta
        # elif mode == 'recon':
        #     seg_frequencies = [slope for slope in seg_slopes]
        #     current_phase = theta_0
        #     last_position = self.samp_dist[0]
        #     # 按顺序处理每个分段
        #     for seg, freq in zip(segments, seg_frequencies):
        #         seg_start, seg_end = seg
        #         seg_slice = slice(seg_start, seg_end)
                
        #         if seg_start >= len(self.samp_dist) or seg_end > len(self.samp_dist):
        #             continue

        #         # 计算相对距离（相对于累积基准位置）
        #         delta_dist = (self.samp_dist[seg_slice] - last_position)  #m
        #         # 计算当前分段相位（所有相同时计算）
        #         theta[:, seg_slice] = current_phase + np.ones((self.num_channels_recon, 1)) * (freq * delta_dist)
        #         # 更新基准相位和位置
        #         last_position = self.samp_dist[seg_end-1] if seg_end <= len(self.samp_dist) else last_position
        #         current_phase = theta[:, seg_end-1:seg_end] if seg_end <= len(self.samp_dist) else current_phase
        #     theta_dynamic = theta

        # 获取 active_angles 参数，如果不存在则为 None
        active_angles = self.active_angles
        
        # 判断是否传入了有效的 active_angles
        if active_angles is not None and len(active_angles) == self.num_channels_recon:
            # 将传入的角度(度)转换为弧度，并加上全局相位
            theta_0_list = [self.global_phases_set + np.deg2rad(ang) for ang in active_angles]
            theta_0 = np.array([theta_0_list]).T
        else:
            if self.interlaced_grating:
                if self.samp_dist_preset_sub is None:
                    raise ValueError("交错刻栅模式下，samp_dist_preset_sub 不能为空")
                theta_0_1 = np.array([[self.global_phases_set + i * 2*np.pi/(self.num_channels_recon) for i in range(self.num_channels_recon)]]).T
                theta_0_2 = np.array([[self.global_phases_set + self.interlaced_offset + i * 2*np.pi/(self.num_channels_recon) for i in range(self.num_channels_recon)]]).T

                p1 = np.array(self.samp_dist_preset)
                p2 = np.array(self.samp_dist_preset_sub)
                
                # 按照各自的采样点数量进行水平平铺
                theta_0_1_full = np.repeat(theta_0_1, len(p1), axis=1)
                theta_0_2_full = np.repeat(theta_0_2, len(p2), axis=1)
                
                # 拼接后根据位置排序索引进行穿插重排
                theta_0_combined = np.concatenate((theta_0_1_full, theta_0_2_full), axis=1)
                p_combined = np.concatenate((p1, p2))
                sort_idx = np.argsort(p_combined)
                
                # 此时 theta_0 是一个形状为 (num_channels_recon, total_points) 的矩阵
                theta_0 = theta_0_combined[:, sort_idx]
            
            else:# 默认的均匀分布逻辑
                theta_0 = np.array([[self.global_phases_set + i * 2*np.pi/self.num_channels_recon for i in range(self.num_channels_recon)]]).T
            
        if mode == 'init':
            # theta_0 加上 0 矩阵保持形状一致
            theta_dynamic = theta_0 + np.zeros((self.num_channels_recon, len(self.samp_dist)))
        elif mode == 'recon':
            seg_frequencies = [slope for slope in seg_slopes]
            
            # 仅记录光纤由于扭转累积的整体相位 (1, total_points)
            twist_phase = np.zeros((1, len(self.samp_dist)))
            last_position = self.samp_dist[0]
            current_twist = 0.0
            
            # 按顺序处理每个分段，累积扭转相位
            for seg, freq in zip(segments, seg_frequencies):
                seg_start, seg_end = seg
                seg_slice = slice(seg_start, seg_end)
                
                if seg_start >= len(self.samp_dist) or seg_end > len(self.samp_dist):
                    continue

                # 计算相对距离（相对于累积基准位置）
                delta_dist = (self.samp_dist[seg_slice] - last_position)  # m
                # 计算当前分段的扭转累积
                twist_phase[0, seg_slice] = current_twist + freq * delta_dist
                
                # 更新基准相位和位置
                if seg_end <= len(self.samp_dist):
                    last_position = self.samp_dist[seg_end-1]
                    current_twist = twist_phase[0, seg_end-1]
                    
            # 最终动态角度 = 固有初始角度 + 累积扭转角度
            # 利用 numpy 的 broadcast 机制：(3, 1) + (1, N) 或 (3, N) + (1, N) 都能正确相加
            theta_dynamic = theta_0 + twist_phase

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
        - curvature_angle: 曲率方向角序列，单位为弧度，范围被映射到 [-pi, pi]

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
        # curvature_angle = [ang + 2*np.pi if ang < 0 else ang for ang in curvature_angle]    #转化到0-2pi
        # 转化到-pi到pi范围内
        curvature_angle = [ang if -np.pi <= ang <= np.pi else (ang + 2*np.pi if ang < -np.pi else ang - 2*np.pi) for ang in curvature_angle]


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

            x_data, y_data, segments = self._segment_data(angles=angles)

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

        fit_results = self._piecewise_linear_fit(y_data=np.asarray(combined_y, dtype=float), x_data=np.asarray(combined_x, dtype=float), segments=merged_segments)

        # 若某些段完全没有数据，可补齐到固定长度
        if len(fit_results) < self.num_segments:
            fit_results += [(np.nan, np.nan)] * (self.num_segments - len(fit_results))

        return fit_results
    
    def _compute_individual_fits(self, angle_data):
        """
        对每个标定数据的自扭转角度-距离数据分别进行分段线性拟合。

        说明：
        - angle_data 的 key 可以是半径，也可以是 (calib_shape, ...path) 形式的完整路径键。
        - 若能从 key 中解析出半径为 0，则该组数据不参与个体拟合。
        - 每组数据独立进行分段，不与其他数据混合。
        """
        individual_fits = {}

        for radius, angles in angle_data.items():
            if radius == 0:
                continue

            x_data, y_data, segments = self._segment_data(angles=angles)
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
        - 先对 angles 做 unwrap；
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
        # unwrapped = np.unwrap(angles, discont=self.discont)
        unwrapped = self._directional_unwrap(angles)
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


    def _directional_unwrap(self, angles):
        """
        自定义的定向 unwrap 函数。
        根据 self.unwrap_direct 的值限制 unwrap 的方向：
        0  : 不限制跳变方向（标准 unwrap）
        1  : 限制所有相位跳变只能向增加的方向 unwrap
        -1 : 限制所有相位跳变只能向减小的方向 unwrap
        """
        angles = np.asarray(angles, dtype=np.float64)
        
        # 如果不限制方向，直接返回标准 unwrap 结果
        if self.unwrap_direct == 0:
            return np.unwrap(angles, discont=self.discont)

        # 如果点数少于2，直接返回原数组副本
        if len(angles) < 2:
            return angles.copy()

        unwrapped = np.zeros_like(angles)
        unwrapped[0] = angles[0]

        # 直接使用类属性指定的趋势
        trend = self.unwrap_direct

        # 既然趋势已明确，从第二个点（索引1）开始就应用方向限制
        for i in range(1, len(angles)):
            # 计算当前点与前一个 unwrapped 点的原始差值
            diff = angles[i] - (unwrapped[i-1] % (2 * np.pi))

            # 将差值映射到 [-pi, pi] 之间
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            # print(f"i: {i}, raw diff: {angles[i] - unwrapped[i-1]}, mapped diff: {diff}")
            # 根据指定的趋势强制限制跳变方向
            if trend == 1:
                if diff < 0:
                    # 强制正向增长
                    diff += 2 * np.pi
                # # 仅螺旋七芯光纤强栅情况生效    
                # # 仅前10个点（i < 10）生效：若限制跳变方向后，diff的绝对值仍小于0.9375*np.pi，则再多增加2pi
                #     if i < 10 and abs(diff) < 0.90*np.pi:
                #         diff += 2 * np.pi
                # elif diff > 0 and i < 10 and abs(diff) < 0.9*np.pi:
                #     diff += 2 * np.pi

            elif trend == -1:
                if diff > 0:
                    # 强制负向增长
                    diff -= 2 * np.pi
                #     # 仅前10个点（i < 10）生效：若限制跳变方向后，diff的绝对值仍小于0.9375*np.pi，则再多减少2pi
                #     if i < 10 and abs(diff) < 0.9*np.pi:
                #         diff -= 2 * np.pi
                # elif diff < 0 and i < 10 and abs(diff) < 0.9*np.pi:
                #     diff -= 2 * np.pi

            unwrapped[i] = unwrapped[i-1] + diff

        return unwrapped

