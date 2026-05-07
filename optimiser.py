import numpy as np
import time
import bisect
import copy
from scipy.optimize import minimize

from reconstructor import Reconstructor

from utils import dict_recur_operate, dict_recur_extract, dict_recur_value, dict_recur_extract_path, fit_radius

class Optimiser(Reconstructor):
    def __init__(self, direct=0, end_len=0.05, max_iter=5, min_loss_diff=1e-6, hyper_params=[10000, 1e-6, 1e-6], method='Powell', k_end=0.0, k_pitch=1.0, k_radius=1.0, weight_bounds=(0.5, 1.0), slope_upper_bound=20.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct = direct
        self.end_len = end_len
        self.max_iter = max_iter
        self.min_loss_diff = min_loss_diff
        self.hyper_params = hyper_params
        self.method = method
        self.k_end = k_end
        self.k_pitch = k_pitch
        self.k_radius = k_radius
        self.weight_bounds = [(weight_bounds[0] * self.init_weight, weight_bounds[1] * self.init_weight)]
        self.slope_upper_bound = slope_upper_bound
        self.slopes_shape = np.asarray(self.slopes[self.direct], dtype=float).shape

    def optim_coeff(self, coeff="slopes", verbose=True):
        """
        交替优化 slopes 和 weights。
        coeff 决定先优化哪个参数：
            coeff == "slopes": slopes -> weights -> slopes -> weights ...
            coeff == "weights": weights -> slopes -> weights -> slopes ...
        """

        if coeff not in ("slopes", "weights"):
            raise ValueError("coeff must be 'slopes' or 'weights'")

        if self.method == 'Powell':
            max_evals, xtol, ftol = self.hyper_params
            options = {"maxiter": int(max_evals), "xtol": float(xtol), "ftol": float(ftol), "disp": bool(verbose),}
        else:
            options = None

        order = ["slopes", "weights"] if coeff == "slopes" else ["weights", "slopes"]

        init_loss = None

        current_loss = init_loss
        best_loss = init_loss

        start = time.time()
        print(f"-------------------- 交替优化开始, 方向: {self.direct}, 旋转角度: {self.rot_ang}, 先优化: {coeff} --------------------")

        for outer_i in range(self.max_iter):
            print(f"========== 第{outer_i}轮交替优化开始 ==========")

            loss_before_outer = best_loss

            for c in order:
                new_loss, status, coeff_start_loss = self._optim_one_coeff_until_stop(
                    coeff=c,
                    init_loss=best_loss,
                    options=options,
                    verbose=verbose,
                )

                if loss_before_outer is None:
                    loss_before_outer = coeff_start_loss

                best_loss = new_loss
                current_loss = new_loss

                if status == "diverge":
                    print(f"{c} 优化发散，已回滚到上一轮最优变量，进入另一个参数优化")
                    continue

            if loss_before_outer is None or best_loss is None:
                print("未能得到有效初始误差，停止优化")
                break

            improvement = loss_before_outer - best_loss

            print(f"第{outer_i}轮交替优化整体提升: {improvement}")

            if improvement < 0:
                print("整体优化发生发散，停止")
                break

            if 0 <= improvement <= self.min_loss_diff:
                print("整体交替优化已收敛")
                break

            print(f"========== 第{outer_i}轮交替优化结束 ==========")
            print()

        else:
            print("交替优化已达最大轮数")

        end = time.time()
        print(f"优化耗时: {end - start}")
        print(f"-------------------- 交替优化结束, 方向: {self.direct}, 旋转角度: {self.rot_ang} --------------------")
        print()

        rs = {}
        for calib_shape in self.calib_shapes:
            r_results_postrot, interp_dist = self.reconstruct_shapes(calib_shape=calib_shape, direct=self.direct, verbose=False,)
            rs[calib_shape] = r_results_postrot

        self._compute_error(interp_dist=interp_dist, rs=rs)

        print(f"当前最优应变权重: {self.weights[self.direct].tolist()}")
        print(f"当前最优扭转系数: {self.slopes[self.direct].tolist()}")
        print(f"最小误差: {current_loss}")
        self._print_errors()      

    def _optim_one_coeff_until_stop(self, coeff, init_loss, options, verbose=True):
        """
        对单一参数 coeff 循环优化，直到该参数收敛、发散或达到最大轮数。
        """

        coeff_name = "扭转系数" if coeff == "slopes" else "应变权重"

        current_loss = init_loss
        best_loss = init_loss
        first_loss = init_loss

        print(f"-------------------- {coeff_name}连续优化开始 --------------------")

        for i in range(self.max_iter):
            prev_weights = copy.deepcopy(self.weights[self.direct])
            prev_slopes = copy.deepcopy(self.slopes[self.direct])

            slopes_flattened = np.asarray(self.slopes[self.direct], dtype=float).reshape(-1)
            slope_upper_bound = max(np.max(np.abs(slopes_flattened)), self.slope_upper_bound)
            slopes_normalised = slopes_flattened / slope_upper_bound

            new_loss, iter_init_loss = self._optim_coeff_iter(coeff=coeff, i=i, init_loss=best_loss, slopes_normalised=slopes_normalised, slope_upper_bound=slope_upper_bound, options=options)

            # 核心改动：第一次进入时，best_loss 可能还是 None
            if best_loss is None:
                best_loss = iter_init_loss
                current_loss = iter_init_loss

            if first_loss is None:
                first_loss = iter_init_loss

            prev_loss = best_loss

            if new_loss > prev_loss:
                self.weights[self.direct] = prev_weights
                self.slopes[self.direct] = prev_slopes
                current_loss = prev_loss

                print(f"{coeff_name}优化发散，已回滚")
                return current_loss, "diverge", first_loss

            current_loss = new_loss
            best_loss = new_loss

            improvement = prev_loss - current_loss

            if 0 <= improvement <= self.min_loss_diff:
                print(f"{coeff_name}优化已收敛")
                return current_loss, "converge", first_loss

        print(f"{coeff_name}优化已达最大轮数")
        return current_loss, "max_iter", first_loss   
    
    def _optim_coeff_iter(self, coeff, i, init_loss, slopes_normalised, slope_upper_bound, options):
        print(f"---------- 第{i}轮优化开始 ----------")
        start = time.time()

        if coeff == "slopes":
            loss_fn = lambda slopes: self._loss_fn(self.rot_ang, self.weights[self.direct], slopes, slope_upper_bound)
            x0 = slopes_normalised
            coeff_bounds = [(-1.0, 1.0)] * len(x0)

        elif coeff == "weights":
            loss_fn = lambda weights: self._loss_fn(self.rot_ang, weights, slopes_normalised, slope_upper_bound)
            x0 = self.weights[self.direct]
            coeff_bounds = self.weight_bounds * len(x0)

        if len(coeff_bounds) != len(x0):
            raise ValueError(f"bounds 长度({len(coeff_bounds)})与{coeff}长度({len(x0)})不一致")

        # 核心改动：允许 init_loss 为 None
        if init_loss is None:
            init_loss = float(loss_fn(x0))
        else:
            init_loss = float(init_loss)

        res = minimize(fun=loss_fn, x0=x0, method=self.method, options=options, bounds=coeff_bounds)

        current_loss = float(res.fun)

        if coeff == "slopes":
            self.slopes[self.direct] = (res.x * slope_upper_bound).reshape(self.slopes_shape)
            best_param = self.slopes[self.direct]

        elif coeff == "weights":
            self.weights[self.direct] = res.x
            best_param = self.weights[self.direct]

        end = time.time()
        print(f"第{i}轮优化耗时: {end - start}")
        print(f"当前最优: {best_param}")
        print(f"当前最小误差: {current_loss}")
        print(f"提升幅度: {init_loss - current_loss}")
        print(f"---------- 第{i}轮优化结束 ----------")
        print()

        return current_loss, init_loss

    def _print_errors(self):
        """
        打印当前保存的各类重构误差统计结果。

        打印内容包括：
        - self.point_errors_avg: 各标定形状内部各半径（及嵌套形状下各二级参数）的平均逐点误差
        - self.end_errors: 各标定形状内部各半径（及嵌套形状下各二级参数）的末端误差
        - self.avg_point_err: 各标定形状的平均逐点误差，以及键 'calib' 对应的校准形状总体平均逐点误差
        - self.avg_end_err: 各标定形状的平均末端误差，以及键 'calib' 对应的校准形状总体平均末端误差

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

    def _loss_fn(self, rot_ang, weights, slopes_normalised, slope_upper_bound):
        """
        给定当前参数，计算优化器使用的损失函数值。

        步骤：
        1. 将输入参数写回当前对象状态，包括旋转角、应变权重和扭转系数。
        2. 若提供 slope_scales，则将展平后的扭转系数恢复到原始量级；
        否则直接将输入视为原始扭转系数。
        3. 使用当前参数对所有可用标定形状执行一次形状重构。
        4. 调用 _compute_error() 计算对应的重构误差统计量。
        5. 返回当前损失：
        self.avg_point_err['calib'] + self.k * self.avg_end_err['calib']

        输入：
        - rot_ang: 当前旋转角补偿
        - weights: 当前应变权重向量
        - slopes_flattened: 展平后的扭转系数
        - slope_scales: 扭转系数对应的尺度数组；若为 None，则不做尺度恢复

        输出：
        - loss: 当前参数下的标量损失值

        说明：
        - 当前实现的损失只基于校准形状集合 'calib' 计算；
        - 'calib' 由 circ、S、rot0 三类形状的误差汇总得到。
        """
        # ---------- unpack parameters ----------
        self.rot_ang = rot_ang
        self.weights[self.direct] = weights
        self.slopes[self.direct] = (slopes_normalised * slope_upper_bound).reshape(self.slopes_shape)
        # ---------- 计算误差 ----------
        rs = {}
        for calib_shape in self.calib_shapes:
            r_results_postrot, interp_dist = self.reconstruct_shapes(calib_shape=calib_shape, direct=self.direct, verbose=False)
            rs[calib_shape] = r_results_postrot

        self._compute_error(interp_dist=interp_dist, rs=rs)

        return self.avg_point_err['calib'] + self.k_end * self.avg_end_err['calib'] + self.k_pitch * self.avg_pitch_err + self.k_radius * self.avg_radius_err

    def _compute_error(self, interp_dist, rs):
        """
        计算各类标定形状下的重构结果相对于理论标定曲线的误差。

        步骤：
        1. 根据 interp_dist 与当前 calib_shape 构造对应的理想标定曲线 ideals。
        2. 对每个标定形状，使用递归字典操作对重构结果与理想曲线逐点计算欧氏距离误差，并按 val_len 归一化。
        3. 从逐点误差中进一步提取：
            - 每组曲线的末端误差
            - 每组曲线的平均逐点误差
        4. 对每个标定形状分别汇总其内部所有曲线的平均逐点误差与平均末端误差。
        5. 进一步对 circ、S、rot0 三类校准形状汇总，得到键 'calib' 对应的整体平均逐点误差与平均末端误差。

        输入：
        - interp_dist: 插值后的弧长坐标数组
        - rs: 按标定形状组织的重构结果字典，结构为：
            - circ / S: {calib_shape: {radius: 轨迹数组}}
            - end / rot0 / up / down: {calib_shape: {key1: {key2: 轨迹数组}}}

        输出：
        - 无显式返回
        - 更新以下成员：
            - self.point_errors
            - self.point_errors_avg
            - self.end_errors
            - self.avg_point_err
            - self.avg_end_err

        说明：
        - 当前实现不再调用单独的 _compute_error_shape()，而是通过 dict_recur_operate、
        dict_recur_extract 与 dict_recur_value 递归完成误差计算与汇总；
        - self.avg_point_err 与 self.avg_end_err 中额外包含的聚合键为 'calib'，而不是 'all'；
        - 'calib' 当前仅汇总 circ、S、rot0、end 三类形状。
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

        val_len = interp_dist[-1]
        if val_len <0:
            print(f"val_len: {val_len}")

        for calib_shape in self.calib_shapes:
            results = rs[calib_shape]
            
            if calib_shape == "rot0":
                rot0_errors = dict_recur_extract_path(d=results,f=lambda x, path: self._compute_rot0_err(result=x, interp_dist=interp_dist, radius_m=path[-2] * 0.001, rot_pos=path[-1] * 0.001, val_len=val_len))

                pitch_errors = dict_recur_extract(d=rot0_errors, f=lambda x: x[0])
                radius_errors = dict_recur_extract(d=rot0_errors, f=lambda x: x[1])

                self.rot0_pitch_errors = pitch_errors
                self.rot0_radius_errors = radius_errors

                pitch_vals = dict_recur_value(pitch_errors)
                radius_vals = dict_recur_value(radius_errors)

                self.avg_pitch_err = np.nanmean(pitch_vals)
                self.avg_radius_err = np.nanmean(radius_vals)

            else:    
                point_errors = {}
                end_errors = {}
                point_errors_avg = {}
                ideals = self._construct_ideals(calib_shape=calib_shape, interp_dist=interp_dist, val_len=val_len)

                point_errors = dict_recur_operate(d1=results, d2=ideals, f=lambda x, y: self._compute_point_err(result=x, ideal=y, interp_dist=interp_dist, val_len=val_len))
                end_errors = dict_recur_extract(d=point_errors, f=lambda x: x[-1])
                point_errors_avg = dict_recur_extract(d=point_errors, f=lambda x: np.nanmean(x))

                self.point_errors[calib_shape] = point_errors
                self.end_errors[calib_shape] = end_errors
                self.point_errors_avg[calib_shape] = point_errors_avg

                # 统一处理 circ / S / end
                point_vals = dict_recur_value(point_errors_avg)
                end_vals = dict_recur_value(end_errors)
                self.avg_point_err[calib_shape] = np.nanmean(point_vals)
                self.avg_end_err[calib_shape] = np.nanmean(end_vals)
            
        calib_point_errors_avg = {k: self.point_errors_avg[k] for k in ['circ', 'S', 'end'] if k in self.point_errors_avg}
        calib_end_errors = {k: self.end_errors[k] for k in ['circ', 'S', 'end'] if k in self.end_errors}
        calib_point_vals = dict_recur_value(calib_point_errors_avg)
        calib_end_vals = dict_recur_value(calib_end_errors)
        self.avg_point_err['calib'] = np.nanmean(calib_point_vals)
        self.avg_end_err['calib'] = np.nanmean(calib_end_vals)

    def _construct_ideals(self, calib_shape, interp_dist, val_len):
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
        return ideals
                
    def _construct_ideal_shape(self, calib_shape, interp_dist, val_len, radius_m, idx_m):
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
                ideal_shape.append(real_point)
        return np.array(ideal_shape)

    def _construct_ideal_circ(self, radius_m, arc_len):
        x = radius_m * np.sin(arc_len / radius_m)
        y = radius_m * (1.0 - np.cos(arc_len / radius_m))
        real_point = np.array([x, y, 0.0])
        return real_point

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
    
    def _compute_rot0_err(self, result, interp_dist, radius_m, rot_pos, val_len):
        rot_start = bisect.bisect_left(interp_dist, rot_pos)
        rot_end = bisect.bisect_right(interp_dist, rot_pos + 2 * np.pi * radius_m) - 1

        points = result[rot_start:rot_end + 1]

        pitch_err = np.abs((result[rot_end, 2] - result[rot_start, 2]) / val_len)

        real_radius = fit_radius(points=points)
        radius_err = np.abs(real_radius - radius_m)

        return pitch_err, radius_err
        
    def _compute_point_err(self, result, ideal, interp_dist, val_len):
        eval = bisect.bisect_right(interp_dist, val_len)
        return np.linalg.norm(result[:eval] - ideal[:eval], axis=1) / val_len