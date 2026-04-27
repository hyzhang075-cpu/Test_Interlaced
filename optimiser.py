import numpy as np
import time
from scipy.optimize import minimize

from reconstructor import Reconstructor

from utils import dict_recur_operate, dict_recur_extract, dict_recur_value

class Optimiser(Reconstructor):
    def __init__(self, max_iter=5, min_loss_diff=1e-6, hyper_params=[10000, 1e-6, 1e-6], method='Powell', k=0.0, weight_bounds=(0.5, 1.0), slope_bounds=(-20.0, 20.0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct = 0
        self.max_iter = max_iter
        self.min_loss_diff = min_loss_diff
        self.hyper_params = hyper_params
        self.method = method
        self.k = k
        self.weight_bounds = [(weight_bounds[0] * self.init_weight, weight_bounds[1] * self.init_weight)] * self.num_levels if not self.end_weights else [(weight_bounds[0] * self.init_weight, weight_bounds[1] * self.init_weight)] * self.num_levels * 2
        self.slope_bounds = [slope_bounds] * (3 * self.num_levels)

    def optim_coeff(self, init_loss, verbose=True):
        """
        对当前方向的扭转系数与应变权重进行交替迭代优化。

        步骤：
        1. 以当前误差 init_loss 作为初始损失。
        2. 在每一轮外层优化开始时，将当前扭转系数直接展平为一维向量（不做归一化）。
        3. 根据当前优化方法构建 scipy.optimize.minimize 所需的选项与边界。
        4. 固定当前权重，先优化扭转系数 slopes。
        5. 固定新的扭转系数，再优化应变权重 weights。
        6. 根据本轮总损失下降幅度判断整体交替优化是否收敛；若未收敛，则进入下一轮。
        7. 优化结束后，使用最终参数对所有标定形状重新进行形状重构并计算完整误差。

        输入：
        - init_loss: 初始损失值
        - verbose: 是否将 scipy 优化器的内部输出打开

        使用的实例成员：
        - self.max_iter: 最大外层/内层迭代轮数上限
        - self.min_loss_diff: 判定收敛所需的最小损失改进量

        输出：
        - 无显式返回
        - 更新以下成员：
        - self.weights[self.direct]
        - self.slopes[self.direct]
        - 与误差统计相关的类成员（由 _compute_error() 更新）
        """
        i = 0
        prev_loss = init_loss
        current_loss = init_loss
        converge = False
        start = time.time()

        while i < self.max_iter:
            print(f"-------------------- [第{i}轮优化开始] 方向: {self.direct}, 旋转角度: {self.rot_ang} --------------------")
            start_iter = time.time()

            # 直接展平原始 slopes，不做归一化
            slopes_flattened = np.asarray(self.slopes[self.direct], dtype=float).reshape(-1)
            weights = self.weights[self.direct]

            # ---------- 初始化搜索 ----------
            if self.method == 'Powell':
                max_evals, xtol, ftol = self.hyper_params
                options = {
                    "maxiter": int(max_evals),
                    "xtol": float(xtol),
                    "ftol": float(ftol),
                    "disp": bool(verbose)
                }

            # 可选：检查 bounds 长度是否与优化变量一致
            if len(self.slope_bounds) != len(slopes_flattened):
                raise ValueError(
                    f"slope_bounds 长度({len(self.slope_bounds)})与 slopes_flattened 长度({len(slopes_flattened)})不一致"
                )

            # =========================================================
            # 1) 先优化扭转系数 slopes
            # =========================================================
            slopes_start = time.time()
            print("---------- 扭转系数优化开始 ----------")

            j = 0
            converge_slope = False
            current_loss_slope = prev_loss

            while j < self.max_iter:
                slopes_j_start = time.time()
                print(f"---------- 第{j}轮扭转系数优化开始 ----------")

                res_slopes = minimize(lambda slopes: self._loss_fn(self.rot_ang, self.weights[self.direct], slopes, None), x0=slopes_flattened, method=self.method, options=options, bounds=self.slope_bounds)

                # 不再乘 slope_scales，直接使用原始优化结果
                slopes = res_slopes.x
                self.slopes[self.direct] = slopes.reshape(self.num_levels, int(len(slopes) / self.num_levels))

                slopes_j_end = time.time()
                current_loss_j = res_slopes.fun
                loss_diff_j = current_loss_slope - current_loss_j

                print(f"第{j}轮扭转系数优化耗时: {slopes_j_end - slopes_j_start}")
                print(f"当前最小误差: {res_slopes.fun}")
                print(f"提升幅度: {loss_diff_j}")
                print(f"---------- 第{j}轮扭转系数优化结束 ----------")
                print()

                if loss_diff_j <= self.min_loss_diff:
                    converge_slope = True
                    current_loss_slope = current_loss_j
                    break
                else:
                    j += 1
                    current_loss_slope = current_loss_j
                    # 重新展平最新的原始 slopes，仍然不做归一化
                    slopes_flattened = np.asarray(self.slopes[self.direct], dtype=float).reshape(-1)

            if converge_slope:
                print("扭转系数优化已收敛")
            else:
                print("扭转系数优化已达最大轮数")

            # 用最新 slopes 重新展平，供后续 weights 优化使用
            slopes_flattened = np.asarray(self.slopes[self.direct], dtype=float).reshape(-1)

            slopes_end = time.time()
            print(f"扭转系数优化耗时: {slopes_end - slopes_start}")
            print(f"最优扭转系数: {self.slopes[self.direct].tolist()}")
            print(f"最小误差: {res_slopes.fun}")
            print("---------- 扭转系数优化结束 ----------")
            print()

            # =========================================================
            # 2) 再优化应变权重 weights
            # =========================================================
            weights_start = time.time()
            print("---------- 应变权重优化开始 ----------")

            res_weights = minimize(lambda weights: self._loss_fn(self.rot_ang, weights, slopes_flattened, None), x0=weights, method=self.method, options=options, bounds=self.weight_bounds)
            self.weights[self.direct] = res_weights.x

            current_loss = res_weights.fun

            weights_end = time.time()
            print(f"应变权重优化耗时: {weights_end - weights_start}")
            print(f"最优应变权重: {self.weights[self.direct].tolist()}")
            print(f"最小误差: {res_weights.fun}")
            print("---------- 应变权重优化结束 ----------")
            print()

            # =========================================================
            # 3) 外层收敛判断
            # =========================================================
            loss_diff = prev_loss - current_loss
            end_iter = time.time()
            print(f"本轮提升幅度: {loss_diff}")
            print(f"本轮优化耗时: {end_iter - start_iter}")
            print()

            if loss_diff <= self.min_loss_diff:
                converge = True
                break
            else:
                i += 1
                prev_loss = current_loss

        end = time.time()
        if converge:
            print("优化已收敛")
        else:
            print("优化已达最大轮数")
        print(f"优化耗时: {end - start}")
        print(f"-------------------- [应变权重与扭转系数优化结束] 方向: {self.direct}, 旋转角度: {self.rot_ang} --------------------")
        print()

        rs = {}
        for calib_shape in self.calib_shapes:
            r_results_postrot, interp_dist = self.reconstruct_shapes(calib_shape=calib_shape, direct=self.direct, verbose=False)
            rs[calib_shape] = r_results_postrot

        self._compute_error(interp_dist=interp_dist, rs=rs)
        print(f"最优应变权重: {self.weights[self.direct].tolist()}")
        print(f"最优扭转系数: {self.slopes[self.direct].tolist()}")
        print(f"最小误差: {current_loss}")
        self._print_errors()

    def _flatten_params(self, params):
        """
        将参数矩阵展平，并按元素尺度做归一化，方便优化器进行参数搜索。

        输入：
        - params: 待处理的参数数组，通常为二维数组

        输出：
        - params_flattend: 展平并归一化后的参数数组
        - param_scales: 与各参数对应的尺度数组，用于后续恢复原始量级
        """
        params_flattend = params.flatten()
        param_scales = np.array([np.abs(param) if np.abs(param) > 1.0 else 1.0 for param in params_flattend])
        params_flattend /= param_scales
        return params_flattend, param_scales

    def _print_errors(self):
        """
        打印当前保存的各类重构误差统计结果。

        打印内容包括：
        - self.point_errors_avg: 各标定形状内部各半径（及 end 形状下各 radius_end）的平均逐点误差
        - self.end_errors: 各标定形状内部各半径（及 end 形状下各 radius_end）的末端误差
        - self.avg_point_err: 各标定形状的平均逐点误差，以及键 'all' 对应的总体平均逐点误差
        - self.avg_end_err: 各标定形状的平均末端误差，以及键 'all' 对应的总体平均末端误差
        """
        print(f"各标定形状初始平均误差: {self.point_errors_avg}")
        print(f"各标定形状末端误差: {self.end_errors}")
        print(f"每点误差: {self.avg_point_err}")
        print(f"末端误差: {self.avg_end_err}")    
        print()

    def _loss_fn(self, rot_ang, weights, slopes_flattened, slope_scales=None):
        """
        给定当前参数，计算优化器使用的损失函数值。

        步骤：
        1. 将输入参数写回当前对象状态，包括旋转角、应变权重和扭转系数。
        2. 若提供 slope_scales，则将展平后的扭转系数恢复到原始量级；
        否则直接将输入视为原始扭转系数。
        3. 使用当前参数执行一次形状重构。
        4. 计算对应的重构误差。
        5. 返回总损失：
        avg_point_err + self.k * avg_end_err

        输入：
        - rot_ang: 当前旋转角补偿
        - weights: 当前应变权重向量
        - slopes_flattened: 展平后的扭转系数
        - slope_scales: 扭转系数对应的尺度数组；若为 None，则不做尺度恢复

        输出：
        - loss: 当前参数下的标量损失值
        """
        # ---------- unpack parameters ----------
        self.rot_ang = rot_ang
        self.weights[self.direct] = weights

        slopes_flattened = np.asarray(slopes_flattened, dtype=float)

        if slope_scales is not None:
            slopes = slopes_flattened * slope_scales
        else:
            slopes = slopes_flattened

        self.slopes[self.direct] = slopes.reshape(self.num_levels, int(len(slopes) / self.num_levels))

        # ---------- 计算误差 ----------
        rs = {}
        for calib_shape in self.calib_shapes:
            r_results_postrot, interp_dist = self.reconstruct_shapes(calib_shape=calib_shape, direct=self.direct, verbose=False)
            rs[calib_shape] = r_results_postrot

        self._compute_error(interp_dist=interp_dist, rs=rs)

        return self.avg_point_err['calib'] + self.k * self.avg_end_err['calib']


    def _compute_error(self, interp_dist, rs):
        """
        计算各类标定形状下的重构结果相对于理论标定曲线的误差。

        说明：
        - 对每个标定形状分别遍历其对应的半径配置；
        - 对 circ / S 形状，直接读取 rs[calib_shape][radius]；
        - 对 end 形状，进一步遍历 radius_end，读取 rs['end'][radius][radius_end]。
        - 每组重构曲线都会调用 _compute_error_shape() 逐点计算误差，并得到对应末端误差。
        - 最终分别汇总：
        - 每个标定形状内部的平均逐点误差与平均末端误差
        - 所有标定形状整体的平均逐点误差与平均末端误差

        输入：
        - interp_dist: 插值后的弧长坐标数组
        - rs: 按标定形状组织的重构结果字典，结构为：
        - circ / S: {calib_shape: {radius: 轨迹数组}}
        - end: {calib_shape: {radius: {radius_end: 轨迹数组}}}

        输出：
        - 无显式返回
        - 更新以下成员：
        - self.point_errors
        - self.point_errors_avg
        - self.end_errors
        - self.avg_point_err（包含各标定形状以及键 'all' 的总体平均误差）
        - self.avg_end_err（包含各标定形状以及键 'all' 的总体平均误差）
        """
        self.point_errors = {}
        self.point_errors_avg = {}
        self.end_errors = {}
        self.avg_point_err = {}
        self.avg_end_err = {}

        val_len = interp_dist[-1]
        if val_len <0:
            print(f"val_len: {val_len}")

        for calib_shape in self.calib_shapes:
            point_errors = {}
            end_errors = {}
            point_errors_avg = {}
            results = rs[calib_shape]
            ideals = self._construct_ideals(calib_shape=calib_shape, interp_dist=interp_dist, val_len=val_len)

            point_errors = dict_recur_operate(d1=results, d2=ideals, f=lambda x, y: np.linalg.norm(x - y, axis=1)/val_len)
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
        
        calib_point_errors_avg = {k: self.point_errors_avg[k] for k in ['circ', 'S', 'rot0'] if k in self.point_errors_avg}
        calib_end_errors = {k: self.end_errors[k] for k in ['circ', 'S', 'rot0'] if k in self.end_errors}
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
        end_len = 0.05 if radius_m == 2 else 0.15
        if arc_len <= self.ideal_len - end_len:
            real_point = self._construct_ideal_circ(radius_m=radius_m, arc_len=arc_len)
        else:
            # 第一段在拼接点的末端位置
            p0 = self._construct_ideal_circ(radius_m=radius_m, arc_len=self.ideal_len-end_len)
            # 第二段使用局部弧长
            p_local = self._construct_ideal_circ(radius_m=radius_end_m, arc_len=arc_len-(self.ideal_len-end_len))
            # 将第二段局部圆弧旋转到第一段末端切向方向，再平移到拼接点
            x = p0[0] + np.cos((self.ideal_len - end_len) / radius_m) * p_local[0] - np.sin((self.ideal_len - end_len) / radius_m) * p_local[1]
            y = p0[1] + np.sin((self.ideal_len - end_len) / radius_m) * p_local[0] + np.cos((self.ideal_len - end_len) / radius_m) * p_local[1]
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