import numpy as np
from scipy import stats

from sensor import Sensor

class Initialiser(Sensor):
    """
    形状重构的初始化模块
    """
    def __init__(self, segmentation,  global_phases_set=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.segmentation = segmentation # 对应原本的segments
        self.global_phases_set = global_phases_set
        if self.global_phases_set == 0.0:
            self.rot_seg_fit(verbose=False, update_phase=True) # 初始化正确的全局相位与全局扭转系数

    def rot_seg_fit(self, verbose, update_phase):
        '''
        初始化全局相位与初始分段扭转系数。

        步骤：
        1. 在初始化模式下生成各通道沿传感距离的动态角度，作为曲率方向计算时的通道角度基准。
        2. 遍历可用的标定形状（当前支持按存在性处理 'circ'、'S' 与 'rot0'）以及各实验方向：
        - 对每组标定数据计算曲率方向角；
        - 对角度做 unwrap；
        - 保存原始角度与 unwrap 后角度。
        3. 当 calib_shape == 'circ' 且 direct == 0 且 update_phase=True 时，
        使用当前 circ 标定数据中除第一个半径外各半径的初始角度均值估计全局相位补偿。
        4. 当 calib_shape == 'circ' 时：
        - 对各半径角度-距离曲线做分段全局拟合；
        - 对各半径分别做分段个体拟合；
        - 将全局拟合得到的各段斜率取负，作为该方向的初始扭转系数 self.slopes_init[direct]。

        输入：
        - verbose: 是否打印初始化过程中的关键信息
        - update_phase: 是否在 circ 的 0 度方向数据上更新全局相位

        输出：
        - 无显式返回
        - 更新以下成员：
        - self.global_phases_set
        - self.slopes_init
        - self.angle_results_raw
        - self.angle_results_unwrap
        - 当存在 'circ' 标定数据时，进一步更新：
            - self.global_fit_direct
            - self.individual_fits_direct

        说明：
        - 所有存在的标定形状都会计算并保存角度结果；
        - 只有 'circ' 标定数据会进一步用于分段拟合和初始扭转系数估计。
        '''
        # ---------- 初始准备 ----------
        theta_dynamic = self._calculate_dynamic_angles(mode='init', segments=None, seg_slopes=None)
        self.slopes_init = {}
        self.angle_results_raw = {}
        self.angle_results_unwrap = {}
        self.global_fit_direct = {}
        self.individual_fits_direct = {}

        for calib_shape in ['circ', 'S', 'rot0']:
            if calib_shape in self.calib_shapes:
                self.angle_results_raw[calib_shape] = {}
                self.angle_results_unwrap[calib_shape] = {}

                for direct in self.exp_directs:
                    if direct not in self.processed_Data[calib_shape]:
                        continue

                    calib_Data = self.processed_Data[calib_shape][direct]
                    angle_raw = {}
                    angle_results = {}
                    init_angle = {}

                    # ---------- circ / S: {radius: ndarray(num_points, num_channels)} ----------
                    if calib_shape in ['circ', 'S']:
                        for radius in self.radii[direct][calib_shape]:
                            if radius not in calib_Data:
                                continue

                            strain_data = np.asarray(calib_Data[radius]).T
                            _, angle = self._calculate_curvature_and_angle(epsilon=strain_data, alpha_dynamic=theta_dynamic)
                            angle_unwrap = np.unwrap(angle)

                            angle_raw[radius] = angle
                            angle_results[radius] = angle_unwrap
                            init_angle[radius] = angle_unwrap[0]

                    # ---------- rot0: {x: {y: ndarray(num_points, num_channels)}} ----------
                    elif calib_shape == 'rot0':
                        for key1 in calib_Data:
                            angle_raw[key1] = {}
                            angle_results[key1] = {}
                            init_angle[key1] = {}

                            for key2 in calib_Data[key1]:
                                strain_data = np.asarray(calib_Data[key1][key2]).T
                                _, angle = self._calculate_curvature_and_angle(epsilon=strain_data, alpha_dynamic=theta_dynamic)
                                angle_unwrap = np.unwrap(angle)

                                angle_raw[key1][key2] = angle
                                angle_results[key1][key2] = angle_unwrap
                                init_angle[key1][key2] = angle_unwrap[0]

                    self.angle_results_raw[calib_shape][direct] = angle_raw
                    self.angle_results_unwrap[calib_shape][direct] = angle_results

                    # ---------- 仅 circ 用于全局相位与初始分段扭转系数 ----------
                    if calib_shape == 'circ':
                        if direct == 0 and update_phase:
                            self.global_phases_set = -np.array(list(init_angle.values()))[1:].mean()

                        global_fit = self._compute_global_fit(angle_data=angle_results)
                        individual_fits = self._compute_individual_fits(angle_data=angle_results)
                        self.global_fit_direct[direct] = global_fit
                        self.individual_fits_direct[direct] = individual_fits

                        self.slopes_init[direct] = [-tup[0] for tup in global_fit]
                        result = {key: [group[0] for group in groups] for key, groups in individual_fits.items()}

                    if verbose:
                        print(f"-------------------- {direct}方向初始化开始 --------------------")
                        print(f"calib_shape: {calib_shape}")
                        print(f"global phase: {self.global_phases_set}")
                        print(f"initial angle: {init_angle}")

                        if calib_shape == 'circ':
                            print(f"segment fit: {global_fit}")
                            print(f"slopes: {result}")

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
        '''
        计算随传感距离变化的各通道动态角度。
        若无特殊说明，默认用来重构的外围通道排布时均匀的（active_angles=None）。

        说明：
        - 在 init 模式下，仅使用全局相位与各方向基准角，认为沿全段不存在额外自扭转累积。
        - 在 recon 模式下，根据给定分段与各段扭转系数，逐段累积角度变化，得到整条光纤上的动态角度分布。

        输入：
        - mode: 计算模式
          - 'init': 初始化模式，动态角度恒等于全局相位补偿后的通道角度
          - 'recon': 重构模式，根据分段扭转系数逐段累积
        - segments: 分段列表，格式为 [(start, end), ...]，每段为索引区间
        - seg_slopes: 各分段对应的扭转系数列表

        输出：
        - theta_dynamic: 各通道在各采样点处的动态角度，形状为 (self.num_channels, len(self.samp_dist))
        '''
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
        '''
        对多个标定半径的自扭转角度-距离数据进行分段全局线性拟合。

        说明：
        - 仅使用半径不在 {0, 200, 300} 中的数据参与全局拟合。
        - 对每条曲线先根据当前 segmentation 进行分段。
        - 将所有半径的第 1 段、第 2 段、第 3 段数据分别聚合，再重建为一个整体数据序列进行分段线性回归。
        - 这样得到的是“各分段共享”的全局拟合结果，而不是单个半径的局部结果。

        输入：
        - angle_data: 各半径对应的角度数据字典，键为半径，值为角度数组

        输出:
        - 全局分段线性拟合结果，格式为 list of tuples
        - 每个元素为 (斜率, 截距)
        '''
        global_head_x, global_head_y = [], []
        global_mid_x, global_mid_y = [], []
        global_tail_x, global_tail_y = [], []

        for radius, angles in angle_data.items():
            if radius in {0, 200, 300}:
                continue

            x_data, y_data, segments = self._segment_data(angles)

            for idx, (start, end) in enumerate(segments):
                if idx == 0:
                    global_head_x.extend(x_data[start:end])
                    global_head_y.extend(y_data[start:end])
                elif idx == 1:
                    global_mid_x.extend(x_data[start:end])
                    global_mid_y.extend(y_data[start:end])
                elif idx == 2:
                    global_tail_x.extend(x_data[start:end])
                    global_tail_y.extend(y_data[start:end])

        combined_x = np.concatenate([global_head_x, global_mid_x, global_tail_x])
        combined_y = np.concatenate([global_head_y, global_mid_y, global_tail_y])

        # 重建全局segments
        segments = []
        current = 0

        head_len = len(global_head_x)
        if head_len > 0:
            segments.append((current, current + head_len))
            current += head_len

        mid_len = len(global_mid_x)
        if mid_len > 0:
            segments.append((current, current + mid_len))
            current += mid_len

        tail_len = len(global_tail_x)
        if tail_len > 0:
            segments.append((current, current + tail_len))

        return self._piecewise_linear_fit(combined_y, combined_x, segments)
    
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
        '''
        对单条自扭转角度-距离曲线生成带重叠的拟合分段结构。

        步骤：
        - 先对角度进行 unwrap 处理，避免 2π 跳变影响线性拟合。
        - 再根据 zero_init 决定是否跳过第一个采样点。
        - segmentation = [head_len, tail_len]：
          - 头段取前 head_len 个点
          - 尾段取后 tail_len 个点
          - 中间段与头尾段各保留 1 个点的重叠，用于拟合过渡更平滑

        输入:
        - angles: 单个半径对应的角度数组

        输出:
        - x_data: 对应采样距离数组
        - y_data: unwrap 后的角度数组
        - segments: 带重叠的分段索引区间列表，格式为 [(start, end), ...]
        '''
        unwrapped = np.unwrap(angles)
        valid_length = min(len(unwrapped), len(self.samp_dist))

        if self.zero_init:
            x_data = self.samp_dist[:valid_length]
            y_data = unwrapped[:valid_length]
        else:
            x_data = self.samp_dist[1:valid_length]
            y_data = unwrapped[1:valid_length]
            valid_length -= 1

        head = self.segmentation[0] if len(self.segmentation) > 0 else 0
        tail = self.segmentation[1] if len(self.segmentation) > 1 else 0

        head_end = min(head, valid_length)
        tail_start = max(valid_length - tail, head_end)

        segments = []

        # 头段
        if head_end > 0:
            segments.append((0, head_end))

        # 中间段（带重叠）
        mid_start = max(0, head_end - 1)
        mid_end = min(tail_start + 1, valid_length)
        if mid_end > mid_start:
            segments.append((mid_start, mid_end))

        # 尾段
        if tail_start < valid_length:
            segments.append((tail_start, valid_length))

        return x_data, y_data, segments
    
    def _calculate_segments(self, total_points):
        '''
        根据 segmentation 生成最终用于重构的不重叠分段结构。

        说明：
        - 与 _segment_data() 不同，这里生成的是“不重叠”的最终分段。
        - 该方法更适合在重构阶段或参数应用阶段使用。
        - segmentation 通常为 [head_len, tail_len]：
          - 头段从 0 开始
          - 尾段位于末尾
          - 中间段自动作为剩余过渡区域
        - 若配置非法或 total_points 无效，则退化为单段 [(0, total_points)]。

        输入:
        - total_points: 当前数据总点数

        输出:
        - segments: 不重叠的分段区间列表，格式为 [(start, end), ...]
        '''
        # --- 输入验证 ---
        if total_points <= 0:
            return []
        if not isinstance(self.segmentation, list) or len(self.segmentation) < 2 or self.segmentation[0] < 0 or self.segmentation[1] < 0:
            return [(0, total_points)]
        
        head_len = self.segmentation[0]
        tail_len = self.segmentation[1]

        # 处理 config 全为0或无效的情况
        if head_len == 0 and tail_len == 0:
            return [(0, total_points)]

        # --- 核心逻辑：根据简化规则确定所有断点 ---
        
        breakpoints = {0, total_points}

        # 计算尾部区域的理论起始点
        tail_start_point = total_points - tail_len

        # 断点1: 头段结束点 / 过渡段开始点
        # 这是头段的结束点，也是合并后的过渡段的开始点
        transition_start_point = min(head_len, tail_start_point)
        
        # 只有当 tail_len > 0 时，才有过渡段和尾段
        if tail_len > 0:
            breakpoints.add(transition_start_point)
            
            # 断点2: 过渡段结束点 / 尾段开始点
            # 只有当尾部不止一个点时，才有独立的尾段
            if tail_len > 1:
                tail_start_point_new = tail_start_point + 1
                breakpoints.add(tail_start_point_new)
        else: # tail_len == 0, 只有头段
            breakpoints.add(head_len)


        # --- 从断点生成最终分段 ---
        sorted_points = sorted(list(p for p in breakpoints if 0 <= p <= total_points))
        
        segments = []
        last_point = sorted_points[0]
        for point in sorted_points[1:]:
            if point > last_point:  # 确保不创建空分段
                segments.append((last_point, point))
            last_point = point
                
        return segments