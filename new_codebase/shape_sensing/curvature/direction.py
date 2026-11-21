"""动态三方向软分配；原公式从Qt回调提取，实例与状态归模块8会话。"""
import numpy as np

class DirectionSelector:
    """weights/slopes显式标定映射，base_curvature为模块3真实点粗计算回调。"""
    DIRECTION_ANCHORS = (0, 120, 240)

    def __init__(self, weights, slopes, base_curvature, *, rotation_deg=0., enabled=False,
                 reference_direct=0, min_curvature=0.5, min_consecutive=3,
                 consistency_threshold=0.6, smooth_alpha=0.15, multi_region_parameters=None):
        """输入各方向参数及回调，输出独立selector；曲率阈值1/m、旋转deg、质量无量纲。"""
        self.weights = weights
        self.slopes = slopes
        self.base_curvature = base_curvature
        self.rotation_deg = rotation_deg
        self.multi_region_parameters = multi_region_parameters
        self.direction_blend_enabled = enabled
        self.direction_reference_direct = reference_direct
        self.direction_min_curvature = min_curvature
        self.direction_min_consecutive = max(1, int(min_consecutive))
        self.direction_consistency_threshold = consistency_threshold
        self.direction_smooth_alpha = smooth_alpha
        self._validate_direction_blend_config()
        self.direction_masses_prev = self._one_hot_direction_masses(reference_direct)
        self.current_direction_masses = self.direction_masses_prev.copy()
        self.current_direction_angle_deg = float(reference_direct)
        self.current_direction_consistency = 1.0

    def select(self, signals):
        """输入真实预处理signals(P,C)，输出混合weights/slopes；状态仅更新当前会话质量。"""
        return self._get_dynamic_parameters(signals)

    def _validate_direction_blend_config(self):
        """校验仅供动态展示使用的三方向参数混合配置。"""
        if self.direction_reference_direct not in self.DIRECTION_ANCHORS:
            raise ValueError(
                f"direction_reference_direct 必须属于 {self.DIRECTION_ANCHORS}，"
                f"当前为 {self.direction_reference_direct}"
            )

        if not 0.0 < self.direction_smooth_alpha <= 1.0:
            raise ValueError("direction_smooth_alpha 必须位于 (0, 1]")

        if not 0.0 <= self.direction_consistency_threshold <= 1.0:
            raise ValueError("direction_consistency_threshold 必须位于 [0, 1]")

        if (self.direction_blend_enabled
                and self.multi_region_parameters is not None):
            raise ValueError(
                'multi-region demo 暂不支持三方向参数混合；请关闭 '
                'direction_blend_enabled 并使用已验证的 direction_reference_direct'
            )

        required = self.DIRECTION_ANCHORS if self.direction_blend_enabled else (self.direction_reference_direct,)
        missing_weights = [d for d in required if d not in self.weights]
        missing_slopes = [d for d in required if d not in self.slopes]
        if missing_weights or missing_slopes:
            raise ValueError(
                "动态方向参数不完整: "
                f"weights 缺少 {missing_weights}, slopes 缺少 {missing_slopes}"
            )

        if self.direction_blend_enabled:
            weight_shapes = {d: np.asarray(self.weights[d]).shape for d in self.DIRECTION_ANCHORS}
            slope_shapes = {d: np.asarray(self.slopes[d]).shape for d in self.DIRECTION_ANCHORS}
            if len(set(weight_shapes.values())) != 1:
                raise ValueError(f"三方向 weights 形状不一致，无法插值: {weight_shapes}")
            if len(set(slope_shapes.values())) != 1:
                raise ValueError(f"三方向 slopes 形状不一致，无法插值: {slope_shapes}")

    def _one_hot_direction_masses(self, direct):
        """输入方向int(deg)，输出三方向质量(3,)无量纲。"""
        masses = np.zeros(len(self.DIRECTION_ANCHORS), dtype=float)
        masses[self.DIRECTION_ANCHORS.index(int(direct))] = 1.0
        return masses

    def _half_cosine_direction_masses(self, angle_deg):
        """
        在圆周上只混合当前角度两侧的两套参数。

        例如 angle_deg 位于 [0, 120] 时，使用半余弦从 0 度参数平滑过渡到
        120 度参数；240 到 360 度区间会自动过渡回 0 度参数。
        """
        angle_deg = float(angle_deg) % 360.0
        sector = int(np.floor(angle_deg / 120.0)) % len(self.DIRECTION_ANCHORS)
        left_direct = self.DIRECTION_ANCHORS[sector]
        right_direct = self.DIRECTION_ANCHORS[(sector + 1) % len(self.DIRECTION_ANCHORS)]

        t = np.clip((angle_deg - left_direct) / 120.0, 0.0, 1.0)
        right_mass = 0.5 - 0.5 * np.cos(np.pi * t)
        left_mass = 1.0 - right_mass

        masses = np.zeros(len(self.DIRECTION_ANCHORS), dtype=float)
        masses[sector] = left_mass
        masses[(sector + 1) % len(self.DIRECTION_ANCHORS)] = right_mass
        return masses

    def _find_first_stable_bending_region(self, magnitude):
        """返回近端起第一个满足曲率阈值和最少连续点数的完整连续弯曲区。"""
        magnitude = np.asarray(magnitude, dtype=float)
        valid = np.isfinite(magnitude) & (magnitude >= self.direction_min_curvature)

        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            elif not is_valid and start is not None:
                if idx - start >= self.direction_min_consecutive:
                    return np.arange(start, idx, dtype=int)
                start = None

        if start is not None and len(valid) - start >= self.direction_min_consecutive:
            # print(f"bend starts at {start}, length={valid}")
            return np.arange(start, len(valid), dtype=int)
        return None

    def _estimate_direction_masses(self, calib_Data):
        """
        用指定的最佳方向参数做第一遍粗计算，并估计当前帧的方向质量。

        只使用近端第一个稳定弯曲区。区域内采用曲率加权圆周平均；低曲率、
        无稳定弯曲区或方向一致性不足时冻结上一帧质量。有效的新质量再进行
        指数平滑，以减少跨帧边界抖动。
        """
        reference_slope = np.asarray(
            self.slopes[self.direction_reference_direct][0],
            dtype=float,
        )
        _, magnitude, angle, _ = self.base_curvature(
            keys=None,
            calib_Data=calib_Data,
            base_slopes=reference_slope,
            verbose=False,
        )

        region = self._find_first_stable_bending_region(magnitude)
        if region is None:
            return self.direction_masses_prev.copy()

        region_magnitude = np.asarray(magnitude, dtype=float)[region]
        region_angle = np.asarray(angle, dtype=float)[region]
        finite = np.isfinite(region_magnitude) & np.isfinite(region_angle)
        if not np.any(finite):
            return self.direction_masses_prev.copy()

        region_magnitude = region_magnitude[finite]
        # 与现有注释中的方向判断保持同一角度约定：方向角加 rot_ang 后再映射到
        # 0/120/240 度参数坐标。软过渡应在 rot_ang 完成标定后才启用。
        region_angle = region_angle[finite] + np.deg2rad(self.rotation_deg)

        magnitude_sum = float(np.sum(region_magnitude))
        if magnitude_sum <= 0.0:
            return self.direction_masses_prev.copy()

        resultant = np.sum(region_magnitude * np.exp(1j * region_angle))
        consistency = float(np.abs(resultant) / magnitude_sum)
        self.current_direction_consistency = consistency
        if consistency < self.direction_consistency_threshold:
            return self.direction_masses_prev.copy()

        angle_deg = float(np.rad2deg(np.angle(resultant)) % 360.0)
        print(f"detected bending angle is {angle_deg}")
        raw_masses = self._half_cosine_direction_masses(angle_deg)

        alpha = self.direction_smooth_alpha
        smoothed_masses = (1.0 - alpha) * self.direction_masses_prev + alpha * raw_masses
        smoothed_masses /= np.sum(smoothed_masses)

        self.direction_masses_prev = smoothed_masses
        self.current_direction_masses = smoothed_masses.copy()
        self.current_direction_angle_deg = angle_deg
        return smoothed_masses

    def _blend_direction_parameters(self, direction_masses):
        """按当前帧方向质量混合三套完整 weights / slopes 参数。"""
        direction_masses = np.asarray(direction_masses, dtype=float)
        if direction_masses.shape != (len(self.DIRECTION_ANCHORS),):
            raise ValueError(f"direction_masses 形状错误: {direction_masses.shape}")

        reference = self.direction_reference_direct
        blended_weights = np.zeros_like(np.asarray(self.weights[reference], dtype=float))
        blended_slopes = np.zeros_like(np.asarray(self.slopes[reference], dtype=float))

        for mass, direct in zip(direction_masses, self.DIRECTION_ANCHORS):
            blended_weights += mass * np.asarray(self.weights[direct], dtype=float)
            blended_slopes += mass * np.asarray(self.slopes[direct], dtype=float)

        return blended_weights, blended_slopes

    def _get_dynamic_parameters(self, calib_Data):
        """返回当前帧参数；关闭软过渡时始终使用指定的最佳方向参数。"""
        if not self.direction_blend_enabled:
            direct = self.direction_reference_direct
            masses = self._one_hot_direction_masses(direct)
            self.current_direction_masses = masses
            self.current_direction_angle_deg = float(direct)
            return self.weights[direct], self.slopes[direct]

        masses = self._estimate_direction_masses(calib_Data=calib_Data)
        return self._blend_direction_parameters(direction_masses=masses)

