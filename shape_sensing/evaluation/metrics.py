"""形状理想曲线及误差评估实现。"""
import numpy as np
import bisect
from ..common.utils import dict_recur_operate, dict_recur_extract, dict_recur_value, dict_recur_extract_path, fit_radius

class EvaluationMixin:
    """复用 Optimiser 原有理想曲线和误差方法。"""

    def _print_errors(self):
        """
        打印当前保存的各类重构误差统计结果。

        打印内容包括：
        - self.point_errors_avg: 各标定形状内部各半径（及嵌套形状下各二级参数）的平均逐点误差
        - self.end_errors: 各标定形状内部各半径（及嵌套形状下各二级参数）的末端误差
        - self.avg_point_err: 各标定形状的平均逐点误差，以及键 'calib' 对应的校准形状总体平均逐点误差
        - self.avg_end_err: 各标定形状的平均末端误差，以及键 'calib' 对应的校准形状总体平均末端误差
        - self.rot0_pitch_errors / self.rot0_radius_errors: rot0 形状的螺距误差与半径误差
        - self.avg_pitch_err / self.avg_radius_err: rot0 螺距误差与半径误差的平均值

        说明：
        - 当前实现中的 'calib' 仅汇总 circ、S、rot0 三类形状
        """
        print(f"各标定形状初始平均误差: {self.point_errors_avg}")
        print(f"各标定形状末端误差: {self.end_errors}")
        print(f"各零螺距标定形状螺距误差: {self.rot0_pitch_errors}")
        print(f"每点误差: {self.avg_point_err}")
        print(f"末端误差: {self.avg_end_err}")
        print(f"螺距误差: {self.avg_pitch_err}")
        print(f"半径误差: {self.avg_radius_err}")
        print()

    def _compute_error(self, interp_dist, rs):
        """
        计算各类标定形状下的重构结果相对于理论标定曲线的误差。

        当前版本假设 straight_trim 已经在 Sensor 阶段直接裁剪：
        - self.Data / self.processed_Data / self.plot_Data 已经只保留有效段；
        - self.samp_dist / self.ad_samp_dist 已经对应有效段；
        - 因此这里不再使用 trim_mask，也不再对重构结果做二次裁剪。

        参数：
        - interp_dist: 当前重构使用的插值弧长坐标数组，单位为 m。
        - rs: 按标定形状组织的重构结果字典，通常由 reconstruct_shapes() 生成。

        处理流程：
        1. 初始化各类误差缓存字典；
        2. 将 interp_dist 转为相对弧长，并检查有效评估长度；
        3. 对 rot0：
        - 根据 self.cur_interp 识别旋转段；
        - 估计旋转轴；
        - 计算螺距误差和半径误差；
        4. 对其他标定形状：
        - 调用 _construct_ideals() 构造理论曲线；
        - 调用 _compute_point_err() 计算归一化逐点误差；
        - 提取末端误差和平均逐点误差；
        5. 汇总 circ、S、end、end3 的 calib 平均逐点误差和末端误差。

        副作用：
        - 更新 self.point_errors；
        - 更新 self.point_errors_avg；
        - 更新 self.end_errors；
        - 更新 self.avg_point_err；
        - 更新 self.avg_end_err；
        - 更新 self.rot0_pitch_errors；
        - 更新 self.rot0_radius_errors；
        - 更新 self.avg_pitch_err；
        - 更新 self.avg_radius_err。

        异常：
        - interp_dist 为空、有效长度非正，或 rot0 段识别失败时会抛出 ValueError。
        """
        self.point_errors = {}
        self.point_errors_avg = {}
        self.end_errors = {}
        self.avg_point_err = {}
        self.avg_end_err = {}

        self.rot0_pitch_errors = None
        self.rot0_radius_errors = None
        self.avg_pitch_err = 0
        self.avg_radius_err = 0

        # 现在 interp_dist 本身已经对应裁剪后的有效段
        interp_dist_eval = np.asarray(interp_dist, dtype=float)

        if len(interp_dist_eval) == 0:
            raise ValueError("interp_dist 为空，无法计算误差")

        # 保险起见，统一从 0 开始；如果 Sensor 中已经归零，这里不会改变结果
        interp_dist_eval = interp_dist_eval - interp_dist_eval[0]

        if len(interp_dist_eval) == 1:
            val_len = 0.0
        else:
            val_len = float(interp_dist_eval[-1])

        if val_len <= 0:
            raise ValueError(
                f"有效评估长度 val_len={val_len} 非正，"
                f"请检查裁剪后的 samp_dist / interp_dist 是否至少包含两个不同位置的点。"
            )

        for calib_shape in self.calib_shapes:
            if calib_shape not in rs:
                continue

            results = rs[calib_shape]

            if calib_shape == "rot0":
                cur_interp_eval = np.asarray(self.cur_interp, dtype=float)
                rot0_errors = dict_recur_extract_path(d=results, f=lambda x, path: self._compute_rot0_err(result=x, cur_interp=cur_interp_eval, radius_m=path[-2] * 0.001))
                pitch_errors = dict_recur_extract(d=rot0_errors, f=lambda x: x[0])
                radius_errors = dict_recur_extract(d=rot0_errors, f=lambda x: x[1])

                self.rot0_pitch_errors = pitch_errors
                self.rot0_radius_errors = radius_errors

                pitch_vals = dict_recur_value(pitch_errors)
                radius_vals = dict_recur_value(radius_errors)

                self.avg_pitch_err = np.nanmean(pitch_vals)
                self.avg_radius_err = np.nanmean(radius_vals)

            else:
                ideals = self._construct_ideals(calib_shape=calib_shape, interp_dist=interp_dist_eval, val_len=val_len)
                point_errors = dict_recur_operate(d1=results, d2=ideals, f=lambda x, y: self._compute_point_err(result=x, ideal=y, interp_dist=interp_dist_eval, val_len=val_len) )
                end_errors = dict_recur_extract(d=point_errors, f=lambda x: x[-1])
                point_errors_avg = dict_recur_extract(d=point_errors, f=lambda x: np.nanmean(x))

                self.point_errors[calib_shape] = point_errors
                self.end_errors[calib_shape] = end_errors
                self.point_errors_avg[calib_shape] = point_errors_avg

                point_vals = dict_recur_value(point_errors_avg)
                end_vals = dict_recur_value(end_errors)

                self.avg_point_err[calib_shape] = np.nanmean(point_vals)
                self.avg_end_err[calib_shape] = np.nanmean(end_vals)

        calib_error_shapes = ["circ", "S", "end", "end3"]
        calib_point_errors_avg = {k: self.point_errors_avg[k] for k in calib_error_shapes if k in self.point_errors_avg}
        calib_end_errors = {k: self.end_errors[k] for k in calib_error_shapes if k in self.end_errors}

        calib_point_vals = dict_recur_value(calib_point_errors_avg)
        calib_end_vals = dict_recur_value(calib_end_errors)

        self.avg_point_err["calib"] = np.nanmean(calib_point_vals)
        self.avg_end_err["calib"] = np.nanmean(calib_end_vals)

    def _construct_ideals(self, calib_shape, interp_dist, val_len):
        """
        计算各类标定形状下的重构结果相对于理论标定曲线的误差。

        当前版本假设 straight_trim 已经在 Sensor 阶段直接裁剪：
        - self.Data / self.processed_Data / self.plot_Data 已经只保留有效段；
        - self.samp_dist / self.ad_samp_dist 已经对应有效段；
        - 因此这里不再使用 trim_mask，也不再对重构结果做二次裁剪。

        参数：
        - interp_dist: 当前重构使用的插值弧长坐标数组，单位为 m。
        - rs: 按标定形状组织的重构结果字典，通常由 reconstruct_shapes() 生成。

        处理流程：
        1. 初始化各类误差缓存字典；
        2. 将 interp_dist 转为相对弧长，并检查有效评估长度；
        3. 对 rot0：
        - 根据 self.cur_interp 识别旋转段；
        - 估计旋转轴；
        - 计算螺距误差和半径误差；
        4. 对其他标定形状：
        - 调用 _construct_ideals() 构造理论曲线；
        - 调用 _compute_point_err() 计算归一化逐点误差；
        - 提取末端误差和平均逐点误差；
        5. 汇总 circ、S、end、end3 的 calib 平均逐点误差和末端误差。

        副作用：
        - 更新 self.point_errors；
        - 更新 self.point_errors_avg；
        - 更新 self.end_errors；
        - 更新 self.avg_point_err；
        - 更新 self.avg_end_err；
        - 更新 self.rot0_pitch_errors；
        - 更新 self.rot0_radius_errors；
        - 更新 self.avg_pitch_err；
        - 更新 self.avg_radius_err。

        异常：
        - interp_dist 为空、有效长度非正，或 rot0 段识别失败时会抛出 ValueError。
        """
        ideals = {}
        for radius in self.radii[self.direct][calib_shape]:
            radius_m = radius * 0.001
            if calib_shape == 'circ':
                ideals[radius] = self._construct_ideal_shape(calib_shape=calib_shape, interp_dist=interp_dist, val_len=val_len, radius_m=radius_m, idx_m=None)
            elif calib_shape == 'S':
                ideals[radius] = self._construct_ideal_shape(calib_shape=calib_shape, interp_dist=interp_dist, val_len=2*np.pi*radius_m+0.1, radius_m=radius_m, idx_m=None)
            elif calib_shape in ['end', 'rot0', 'up', 'down']:
                ideals[radius] = {}
                for idx in self.radii[self.direct][calib_shape][radius]:
                    idx_m = idx * 0.001
                    ideals[radius][idx] = self._construct_ideal_shape(calib_shape=calib_shape, interp_dist=interp_dist, val_len=val_len, radius_m=radius_m, idx_m=idx_m)
            elif calib_shape == 'end3':
                ideals[radius] = {}
                for angle_deg in self.radii[self.direct][calib_shape][radius]:
                    ideals[radius][angle_deg] = self._construct_ideal_shape(calib_shape=calib_shape, interp_dist=interp_dist, val_len=val_len, radius_m=radius_m, idx_m=None, angle_deg=angle_deg)
        return ideals

    def _construct_ideal_shape(self, calib_shape, interp_dist, val_len, radius_m, idx_m, angle_deg=None):
        ideal_shape = []
        for arc_len in interp_dist:
            if arc_len <= val_len:
                if radius_m == 0:
                    real_point = np.array([arc_len, 0, 0])
                else:
                    if calib_shape == 'circ':
                        real_point = self._construct_ideal_circ(radius_m=radius_m, arc_len=arc_len)
                    elif calib_shape == 'S':
                        real_point = self._construct_ideal_S(radius_m=radius_m, arc_len=arc_len)
                    elif calib_shape == 'end':
                        real_point = self._construct_ideal_end(radius_m=radius_m, radius_end_m=idx_m, arc_len=arc_len)
                    elif calib_shape == 'rot0':
                        real_point = self._construct_ideal_rot0(radius_m=radius_m, rot_pos=idx_m, arc_len=arc_len)
                    elif calib_shape in ['up', 'down']:
                        real_point = self._construct_ideal_rot(radius_m=radius_m, pitch=idx_m, dir=calib_shape, arc_len=arc_len)
                    elif calib_shape == 'end3':
                        real_point = self._construct_ideal_end3(radius_m=radius_m, angle_deg=angle_deg, arc_len=arc_len)
                ideal_shape.append(real_point)
        return np.array(ideal_shape)

    def _construct_ideal_circ(self, radius_m, arc_len):
        x = radius_m * np.sin(arc_len / radius_m)
        y = radius_m * (1.0 - np.cos(arc_len / radius_m))
        real_point = np.array([x, y, 0.0])
        return real_point

    def _construct_ideal_end3(self, radius_m, angle_deg, arc_len):
        """
        构造 x-y-z-end3 的理论圆弧。

        约定：
        - x 已经作为 direct 进入外层数据结构；
        - y 是圆弧半径，已在调用处转换为 radius_m；
        - z 是圆弧平面相对 +y 方向的顺时针夹角，单位为度；
        - z=0 时退化为现有 circ 基准，即位于 x-y 平面并向 +y 弯曲。
        """
        if angle_deg is None:
            raise ValueError("end3 ideal requires angle_deg, but got None")

        base_point = self._construct_ideal_circ(radius_m=radius_m, arc_len=arc_len)
        angle_rad = np.deg2rad(float(angle_deg))

        x = base_point[0]
        lateral = base_point[1]
        y = lateral * np.cos(angle_rad)
        z = -lateral * np.sin(angle_rad)

        return np.array([x, y, z])

    def _construct_ideal_S(self, radius_m, arc_len):
        straight_len = 0.1
        half_arc_len = np.pi * radius_m
        total_S_len = 2 * half_arc_len

        if arc_len <= straight_len:
            real_point = np.array([arc_len, 0.0, 0.0])

        elif arc_len <= straight_len + half_arc_len:
            theta = (arc_len - straight_len) / radius_m
            x = straight_len + radius_m * np.sin(theta)
            y = radius_m * (1.0 - np.cos(theta))
            real_point = np.array([x, y, 0.0])

        elif arc_len <= straight_len + total_S_len:
            theta = (arc_len - straight_len - half_arc_len) / radius_m
            x = straight_len - radius_m * np.sin(theta)
            y = 2.0 * radius_m + radius_m * (1.0 - np.cos(theta))
            real_point = np.array([x, y, 0.0])
        return real_point

    def _construct_ideal_end(self, radius_m, radius_end_m, arc_len):
        if arc_len <= self.samp_dist[-1] - self.end_len:
            real_point = self._construct_ideal_circ(radius_m=radius_m, arc_len=arc_len)
        else:
            # 第一段在拼接点的末端位置
            p0 = self._construct_ideal_circ(radius_m=radius_m, arc_len=self.samp_dist[-1]-self.end_len)
            # 第二段使用局部弧长
            p_local = self._construct_ideal_circ(radius_m=radius_end_m, arc_len=arc_len-(self.samp_dist[-1]-self.end_len))
            # 将第二段局部圆弧旋转到第一段末端切向方向，再平移到拼接点
            x = p0[0] + np.cos((self.samp_dist[-1] - self.end_len) / radius_m) * p_local[0] - np.sin((self.samp_dist[-1] - self.end_len) / radius_m) * p_local[1]
            y = p0[1] + np.sin((self.samp_dist[-1] - self.end_len) / radius_m) * p_local[0] + np.cos((self.samp_dist[-1] - self.end_len) / radius_m) * p_local[1]
            real_point = np.array([x, y, 0.0])
        return real_point

    def _construct_ideal_rot0(self, radius_m, rot_pos, arc_len):
        if arc_len < rot_pos:
            real_point = np.array([arc_len, 0, 0])
        elif rot_pos <= arc_len <= rot_pos + 2*np.pi*radius_m:
            circ_point = self._construct_ideal_circ(radius_m=radius_m, arc_len=arc_len-rot_pos)
            real_point = circ_point + np.array([rot_pos, 0, 0])
        else:
            real_point = np.array([arc_len - rot_pos - 2*np.pi*radius_m, 0, 0])
        return real_point

    def _construct_ideal_rot(self, dir, radius_m, pitch, arc_len):
        theta = arc_len / np.sqrt(radius_m**2 + (pitch / (2 * np.pi))**2)
        x = radius_m * np.sin(theta)
        y = radius_m * (1.0 - np.cos(theta))
        if dir in ["up", "down"]:
            dir = 1 if dir == "up" else -1
        else:
            raise ValueError(f"Unknown dir: {dir}")
        z = dir * theta * pitch / (2 * np.pi)
        real_point = np.array([x, y, z])
        return real_point

    def _compute_point_err(self, result, ideal, interp_dist, val_len):
        eval = min(bisect.bisect_right(interp_dist, val_len), len(result), len(ideal))
        if eval <= 0:
            return np.array([np.nan])

        # trim 后用剩余区间长度归一化；无 trim 时等价于原来的 interp_dist[-1]。
        norm_len = max(float(interp_dist[eval - 1] - interp_dist[0]), 1e-12)
        return np.linalg.norm(result[:eval] - ideal[:eval], axis=1) / norm_len

    def _compute_rot0_err(self, result, cur_interp, radius_m):
        rot_start, rot_end = self._find_rot_segment(cur_interp=cur_interp)
        points = result[rot_start:rot_end+1]
        rot_axis = self._find_rot_axis(points=points)
        rot_axis /= np.linalg.norm(rot_axis)

        real_radius = real_radius = np.linalg.norm(np.cross(points - points.mean(axis=0), rot_axis), axis=1).mean()
        radius_err = np.abs(real_radius - radius_m)
        pitch_err = np.abs(np.dot(points[-1] - points[0], rot_axis))

        return pitch_err, radius_err

    def _find_rot_segment(self, cur_interp, min_straight_len=3, min_consecutive_rot=5):
        pri_straight_end = min_straight_len
        while pri_straight_end < self.interp_num - min_straight_len - min_consecutive_rot and np.all(cur_interp[:pri_straight_end] < self.straight_threshold):
            pri_straight_end += 1

        post_straight_start = self.interp_num - min_straight_len
        while post_straight_start > pri_straight_end + min_consecutive_rot and np.all(cur_interp[post_straight_start:] < self.straight_threshold):
            post_straight_start -= 1

        rot_start = pri_straight_end + 1
        while pri_straight_end < rot_start < post_straight_start - min_consecutive_rot and np.all(cur_interp[rot_start:rot_start+min_consecutive_rot] < self.rot_threshold):
            rot_start += 1

        rot_end = post_straight_start - 1
        while pri_straight_end + min_consecutive_rot < rot_end < post_straight_start and np.all(cur_interp[rot_end-min_consecutive_rot:rot_end] < self.rot_threshold):
            rot_end -= 1

        if rot_end - rot_start < min_consecutive_rot - 1:
            raise ValueError(f"螺旋段长度不足: rot_start={rot_start}, rot_end={rot_end}, length={rot_end - rot_start + 1}, min_required={min_consecutive_rot}")
        else:
            return rot_start, rot_end

    def _find_rot_axis(self, points, max_x_to_yz_ratio=0.5):
        centred_points = points - points.mean(axis=0)
        if np.allclose(centred_points, 0.0, atol=1e-9):
            raise ValueError("所有点重合，无法确定轴向")

        _, _, vh = np.linalg.svd(centred_points, full_matrices=False)

        # vh[0] 是最大方差方向，vh[-1] 是最小方差方向
        axis = vh[-1]
        axis = axis / np.linalg.norm(axis)

        x_comp, y_comp, z_comp = axis
        yz_magnitude = np.sqrt(y_comp**2 + z_comp**2)

        if yz_magnitude < 1e-12:
            ratio = np.inf
        else:
            ratio = abs(x_comp) / yz_magnitude

        # if ratio >= max_x_to_yz_ratio:
        #     print(f"警告 (PCA): 估计轴 [{x_comp:.3f}, {y_comp:.3f}, {z_comp:.3f}] 的 X 分量偏大，|x| / sqrt(y^2 + z^2) = {ratio:.3f} >= {max_x_to_yz_ratio}。")

        # 统一方向，避免 pitch 符号随机翻转
        if axis[2] < 0:
            axis = -axis

        return axis
