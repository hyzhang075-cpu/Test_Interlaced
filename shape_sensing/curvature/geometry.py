"""从轴向应变计算曲率大小和方向。"""
import numpy as np

class StrainGeometryMixin:
    """Initialiser 的曲率几何实现，不改变对象初始化顺序。"""

    def _calculate_dynamic_angles(self, mode, segments, seg_slopes):
        """
        计算每个重构通道在各采样点处的动态角度。

        动态角度用于将多通道应变投影到二维曲率向量上。它由两部分组成：
        1. 每个重构通道在截面圆周上的初始角度；
        2. 沿传感距离累积的自扭转相位。

        参数：
        - mode: 计算模式。
        * 'init': 初始化模式，不考虑分段扭转累积，动态角度等于初始通道角度。
        * 'recon': 重构模式，根据 segments 和 seg_slopes 沿 self.samp_dist 累积扭转相位。
        - segments: 分段索引区间列表，格式为 [(start, end), ...]。
        仅 mode='recon' 时使用。
        - seg_slopes: 每个分段对应的扭转系数，单位通常为 rad/m。
        仅 mode='recon' 时使用。

        初始角度生成规则：
        1. 若 self.active_angles 非空且长度等于 self.num_channels_recon：
        - 使用 active_angles 指定的角度；
        - 输入单位为 degree，内部转换为 rad；
        - 每个角度加上 self.global_phases_set。
        2. 否则若 self.interlaced_grating=True：
        - 第一组采样点使用 theta_0_1；
        - 第二组采样点使用 theta_0_2 = theta_0_1 + self.interlaced_offset；
        - 根据 samp_dist_preset 和 samp_dist_preset_sub 的空间位置排序后穿插合并。
        3. 否则：
        - 默认认为所有重构通道在截面圆周上均匀分布；
        - 第 i 个通道角度为 global_phase + i * 2π / num_channels_recon。

        重构模式逻辑：
        - 在 mode='recon' 中，先初始化 twist_phase 为 0。
        - 按 segments 顺序遍历每一段，根据对应 slope 在该段内累积：
        twist_phase = current_twist + slope * delta_dist。
        - 每段结束后更新 last_position 与 current_twist，使后续分段相位连续。
        - 最终 theta_dynamic = theta_0 + twist_phase。

        输出：
        - theta_dynamic: numpy 数组，形状为 (self.num_channels_recon, len(self.samp_dist))。
        表示每个重构通道在每个采样点处的动态角度，单位 rad。

        异常：
        - interlaced_grating=True 且 self.samp_dist_preset_sub is None 时抛出 ValueError。

        说明：
        - 本方法只计算参与重构的通道角度，不包含被中心光纤补偿剔除的中心通道。
        - mode='recon' 中若某个分段越过 self.samp_dist 长度，会跳过该分段。
        - 当交错刻栅开启时，theta_0 可能本身就是二维矩阵，形状为
        (num_channels_recon, total_points)，以支持不同采样点处通道角度不同。
        """
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
        根据多通道应变和动态通道角度计算曲率模与曲率方向角。

        参数：
        - epsilon: 各重构通道的应变数据。
        形状通常为 (self.num_channels_recon, num_points)。
        - alpha_dynamic: 各重构通道在各采样点处的动态角度。
        形状通常为 (self.num_channels_recon, num_points)。

        计算公式：
        - kappa_x = -2 / (num_channels_recon * R) * Σ epsilon_i * cos(alpha_i)
        - kappa_y = -2 / (num_channels_recon * R) * Σ epsilon_i * sin(alpha_i)
        - curvature_magnitude = sqrt(kappa_x^2 + kappa_y^2)
        - curvature_angle = atan2(kappa_y, kappa_x)

        输出：
        - curvature_magnitude: numpy 数组，表示每个采样点处的曲率模。
        - curvature_angle: list，表示每个采样点处的曲率方向角，单位 rad，范围映射到 [-π, π]。

        说明：
        - epsilon 和 alpha_dynamic 会先转换为 np.float64。
        - self.R 为外围光纤到中心的半径，单位 m。
        - 当前实现假设参与重构的通道在曲率求解中等权。
        - curvature_angle 返回为 Python list，而不是 numpy 数组。
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

        # curvature_magnitude = np.full(shape=len(curvature_magnitude), fill_value=np.mean(curvature_magnitude[:-1]))
        return curvature_magnitude, curvature_angle
