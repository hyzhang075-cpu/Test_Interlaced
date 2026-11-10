"""重构所用的理想曲率大小计算。"""
import numpy as np

class IdealCurvatureMixin:
    """理想形状曲率的既有数值实现。"""

    def _construct_ideal_magnitude_for_recon(self, keys, calib_shape, sample_dist):
        """
        为 slopes 优化阶段构造强制使用的理论曲率 magnitude。

        注意：
        - 输入 sample_dist 单位为 m；
        - keys 中的 radius / idx 通常来自 radii 配置，单位为 mm；
        - 输出曲率单位为 1/m；
        - 本函数只在 force_ideal_magnitude=True 时使用。
        """

        sample_dist = np.asarray(sample_dist, dtype=float)

        if sample_dist.ndim != 1:
            raise ValueError(f"sample_dist 应为一维数组，当前 shape={sample_dist.shape}")

        if keys is None or len(keys) == 0:
            raise ValueError(
                f"force_ideal_magnitude=True 时需要 keys 提供理论形状参数，"
                f"但当前 keys={keys}, calib_shape={calib_shape}"
            )

        # 使用相对弧长，避免裁剪后 sample_dist 不从 0 开始造成错位
        s = sample_dist - sample_dist[0]

        radius_mm = float(keys[0])

        def kappa_from_radius_mm(r_mm):
            r_mm = float(r_mm)
            if np.isclose(r_mm, 0.0):
                return 0.0
            return 1000.0 / r_mm

        # circ / end3：整段都是同一理论曲率
        if calib_shape in ["circ", "end3"]:
            kappa = kappa_from_radius_mm(radius_mm)
            return np.full(shape=len(sample_dist), fill_value=kappa, dtype=float)

        # end：前段使用主半径，末端 end_len 使用第二半径
        if calib_shape == "end":
            if len(keys) < 2:
                raise ValueError(f"end 形状需要 keys=[radius, radius_end]，当前 keys={keys}")

            radius_end_mm = float(keys[1])

            kappa_main = kappa_from_radius_mm(radius_mm)
            kappa_end = kappa_from_radius_mm(radius_end_mm)

            samp_dist = np.asarray(self.samp_dist, dtype=float)
            effective_len = float(samp_dist[-1] - samp_dist[0])
            split_pos = max(0.0, effective_len - float(self.end_len))

            return np.where(s <= split_pos, kappa_main, kappa_end).astype(float)

        # rot0：前后直线为 0，中间圆弧段为 1000 / radius
        if calib_shape == "rot0":
            if len(keys) < 2:
                raise ValueError(f"rot0 形状需要 keys=[radius, rot_pos]，当前 keys={keys}")

            radius_m = radius_mm * 0.001
            rot_pos_m = float(keys[1]) * 0.001

            if np.isclose(radius_m, 0.0):
                return np.zeros(len(sample_dist), dtype=float)

            kappa = kappa_from_radius_mm(radius_mm)
            rot_start = rot_pos_m
            rot_end = rot_pos_m + 2.0 * np.pi * radius_m

            return np.where((s >= rot_start) & (s <= rot_end), kappa, 0.0).astype(float)

        # S：前 0.1 m 直线为 0，后面两个半圆弧 magnitude 都是 1000 / radius
        if calib_shape == "S":
            radius_m = radius_mm * 0.001

            if np.isclose(radius_m, 0.0):
                return np.zeros(len(sample_dist), dtype=float)

            kappa = kappa_from_radius_mm(radius_mm)

            straight_len = 0.1
            arc_start = straight_len
            arc_end = straight_len + 2.0 * np.pi * radius_m

            return np.where((s >= arc_start) & (s <= arc_end), kappa, 0.0).astype(float)

        raise ValueError(f"force_ideal_magnitude 暂不支持 calib_shape={calib_shape}")
