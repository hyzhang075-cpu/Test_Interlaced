import numpy as np
import time
import bisect
import copy
from scipy.optimize import minimize

from reconstructor import Reconstructor

from utils import dict_recur_operate, dict_recur_extract, dict_recur_value, dict_recur_extract_path, fit_radius

class Optimiser(Reconstructor):
    def __init__(self, direct=0, end_len=0.05, max_iter=5, min_loss_diff=1e-6, hyper_params=[10000, 1e-6, 1e-6], freeze=None, method='Powell', straight_threshold=2.0, rot_threshold=4.0, k_point=1.0, k_end=0.0, k_pitch=1.0, k_radius=1.0, weight_bounds=(0.5, 1.0), slope_upper_bound=20.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct = direct
        self.end_len = end_len
        self.max_iter = max_iter
        self.min_loss_diff = min_loss_diff
        self.hyper_params = hyper_params

        self.method = method
        self.straight_threshold = straight_threshold
        self.rot_threshold = rot_threshold
        self.k_point = k_point
        self.k_end = k_end
        self.k_pitch = k_pitch
        self.k_radius = k_radius
        self.weight_bounds = [(weight_bounds[0] * self.init_weight, weight_bounds[1] * self.init_weight)]
        self.slope_upper_bound = slope_upper_bound
        self.slopes_shape = np.asarray(self.slopes[self.direct], dtype=float).shape

        self.freeze = freeze
        self._get_freeze_idx()

    def _get_freeze_idx(self):
        if self.freeze is None:
            return np.array([], dtype=int)

        freeze_idx = np.unique(np.atleast_1d(self.freeze).astype(int))

        n_weights = len(self.weights[self.direct])
        n_slopes = self.slopes_shape[0]

        if np.any(freeze_idx < 0):
            raise ValueError(f"freeze 中不应包含负 index: {freeze_idx}")

        if np.any(freeze_idx >= n_weights):
            raise ValueError(f"freeze index 超出 weights 范围: freeze={freeze_idx}, weights长度={n_weights}")

        if np.any(freeze_idx >= n_slopes):
            raise ValueError(f"freeze index 超出 slopes 范围: freeze={freeze_idx}, slopes第一维长度={n_slopes}")

        return freeze_idx

    def _get_weight_freeze_mask(self):
        mask = np.zeros(len(self.weights[self.direct]), dtype=bool)
        freeze_idx = self._get_freeze_idx()
        mask[freeze_idx] = True
        return mask

    def _get_slope_freeze_mask(self):
        mask = np.zeros(self.slopes_shape, dtype=bool)
        freeze_idx = self._get_freeze_idx()
        mask[freeze_idx, ...] = True
        return mask

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
                new_loss, status, coeff_start_loss = self._optim_one_coeff_until_stop(coeff=c, init_loss=best_loss, options=options, verbose=verbose,)

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

        base_weights = np.asarray(self.weights[self.direct], dtype=float).copy()
        base_slopes_normalised = np.asarray(slopes_normalised, dtype=float).reshape(self.slopes_shape).copy()

        if coeff == "slopes":
            freeze_mask = self._get_slope_freeze_mask()
            optimise_mask = ~freeze_mask

            x0 = base_slopes_normalised[optimise_mask]
            coeff_bounds = [(-1.0, 1.0)] * len(x0)

            def loss_fn(x):
                slopes_full = base_slopes_normalised.copy()
                slopes_full[optimise_mask] = x
                return self._loss_fn(self.rot_ang, base_weights, slopes_full.reshape(-1), slope_upper_bound)

        elif coeff == "weights":
            freeze_mask = self._get_weight_freeze_mask()
            optimise_mask = ~freeze_mask

            x0 = base_weights[optimise_mask]
            coeff_bounds = self.weight_bounds * len(x0)

            def loss_fn(x):
                weights_full = base_weights.copy()
                weights_full[optimise_mask] = x
                return self._loss_fn(self.rot_ang, weights_full, base_slopes_normalised.reshape(-1), slope_upper_bound)

        else:
            raise ValueError(f"Unknown coeff: {coeff}")

        if len(coeff_bounds) != len(x0):
            raise ValueError(f"bounds 长度({len(coeff_bounds)})与{coeff}长度({len(x0)})不一致")

        if init_loss is None:
            init_loss = float(loss_fn(x0))
        else:
            init_loss = float(init_loss)

        if len(x0) == 0:
            current_loss = init_loss
            best_param = self.slopes[self.direct] if coeff == "slopes" else self.weights[self.direct]

            end = time.time()
            print(f"{coeff} 对应变量全部被冻结，跳过本轮优化")
            print(f"第{i}轮优化耗时: {end - start}")
            print(f"当前最优: {best_param}")
            print(f"当前最小误差: {current_loss}")
            print(f"提升幅度: {init_loss - current_loss}")
            print(f"---------- 第{i}轮优化结束 ----------")
            print()

            return current_loss, init_loss

        res = minimize(fun=loss_fn, x0=x0, method=self.method, options=options, bounds=coeff_bounds)
        current_loss = float(res.fun)

        if coeff == "slopes":
            slopes_best = base_slopes_normalised.copy()
            slopes_best[optimise_mask] = res.x
            self.slopes[self.direct] = (slopes_best * slope_upper_bound).reshape(self.slopes_shape)
            best_param = self.slopes[self.direct]

        elif coeff == "weights":
            weights_best = base_weights.copy()
            weights_best[optimise_mask] = res.x
            self.weights[self.direct] = weights_best
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

    def _loss_fn(self, rot_ang, weights, slopes_normalised, slope_upper_bound):
        """
        给定当前参数，计算优化器使用的损失函数值。

        步骤：
        1. 将 rot_ang、weights 写回当前对象状态。
        2. 将 slopes_normalised 乘以 slope_upper_bound 后 reshape 回 self.slopes_shape，并写回 self.slopes[self.direct]。
        3. 使用当前参数对所有可用标定形状执行一次形状重构。
        4. 调用 _compute_error() 计算逐点误差、末端误差，以及 rot0 的螺距/半径误差。
        5. 返回加权损失：
        k_point * avg_point_err['calib']
        + k_end * avg_end_err['calib']
        + k_pitch * avg_pitch_err
        + k_radius * avg_radius_err

        输入：
        - rot_ang: 当前旋转角补偿
        - weights: 当前应变权重向量
        - slopes_normalised: 归一化后的展平扭转系数
        - slope_upper_bound: 用于恢复扭转系数原始量级的尺度

        说明：
        - 当前 avg_point_err['calib'] 与 avg_end_err['calib'] 汇总 circ / S / end。
        - rot0 不参与 calib 的逐点/末端误差汇总，而是通过 avg_pitch_err 与 avg_radius_err 进入损失。
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

        return self.k_point * self.avg_point_err['calib'] + self.k_end * self.avg_end_err['calib'] + self.k_pitch * self.avg_pitch_err + self.k_radius * self.avg_radius_err

    def _get_trim_interp_mask(self, interp_dist):
        """
        返回当前 straight_trim 对应的插值点 mask。
        straight_trim 不影响重构本身，只在误差统计阶段筛选参与评价的点。
        """
        interp_dist = np.asarray(interp_dist, dtype=float)
        trim_min = getattr(self, "trim_eval_min", interp_dist[0])
        trim_max = getattr(self, "trim_eval_max", interp_dist[-1])

        mask = (interp_dist >= trim_min) & (interp_dist <= trim_max)
        if not np.any(mask):
            raise ValueError(
                f"straight_trim 后没有可用于误差计算的插值点: "
                f"trim_eval_min={trim_min}, trim_eval_max={trim_max}, "
                f"interp范围=({interp_dist[0]}, {interp_dist[-1]})"
            )
        return mask

    def _trim_result_tree(self, results, mask):
        """递归裁剪重构/理想轨迹树。"""
        mask = np.asarray(mask, dtype=bool)
        return dict_recur_extract(results, lambda arr: arr[mask] if len(arr) == len(mask) else arr)

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
        5. 对 circ、S、end 三类常规校准形状汇总，得到键 'calib' 对应的整体平均逐点误差与平均末端误差。rot0 不进入该 calib 汇总，而是单独计算螺距误差和半径误差。

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
        - self.avg_point_err 与 self.avg_end_err 中额外包含的聚合键为 'calib'，而不是 'all'；
        - 'calib' 当前仅汇总 circ、S、end；
        - rot0 单独更新 self.rot0_pitch_errors、self.rot0_radius_errors、self.avg_pitch_err 与 self.avg_radius_err。
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

        trim_mask = self._get_trim_interp_mask(interp_dist)
        interp_dist_eval = np.asarray(interp_dist, dtype=float)[trim_mask]
        val_len = interp_dist_eval[-1]
        if val_len < 0:
            print(f"val_len: {val_len}")

        for calib_shape in self.calib_shapes:
            results = self._trim_result_tree(rs[calib_shape], trim_mask)
            
            if calib_shape == "rot0":
                cur_interp_eval = self.cur_interp[trim_mask] if len(self.cur_interp) == len(trim_mask) else self.cur_interp
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
                point_errors = {}
                end_errors = {}
                point_errors_avg = {}
                ideals = self._construct_ideals(calib_shape=calib_shape, interp_dist=interp_dist_eval, val_len=val_len)

                point_errors = dict_recur_operate(d1=results, d2=ideals, f=lambda x, y: self._compute_point_err(result=x, ideal=y, interp_dist=interp_dist_eval, val_len=val_len))
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
