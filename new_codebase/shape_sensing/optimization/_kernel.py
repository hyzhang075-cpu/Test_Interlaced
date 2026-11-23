import numpy as np
import time
import bisect
import copy
from scipy.optimize import minimize

from ..common.utils import dict_recur_operate, dict_recur_extract, dict_recur_value, dict_recur_extract_path, fit_radius

class OptimizationKernel:
    def _minimize(self, **kwargs):
        """运行本模块scipy搜索；测试可注入求解器验证候选边界。"""
        return minimize(**kwargs)

    def _uses_core_positions(self):
        return getattr(self, 'core_position_mode', False)

    def _uses_real_point_objective(self):
        """Support older callers that set only end5cm_point_calib."""
        return bool(getattr(self, 'real_point_calib', False)
                    or getattr(self, 'end5cm_point_calib', False))

    @staticmethod
    def _normalise_radius_key(value):
        """Use stable numeric keys when comparing configured and loaded radii."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return numeric

    def validate_calibration_data_radii(self):
        """Reject configured/data radius drift before scoring or optimization."""
        problems = []
        directs = [getattr(self, 'direct', 0)]
        for direct in directs:
            configured_shapes = self.radii.get(direct, {})
            loaded_shapes = self.processed_Data.get('circ', {}) if isinstance(
                self.processed_Data, dict) else {}
            loaded_direct = loaded_shapes.get(direct, {}) if isinstance(loaded_shapes, dict) else {}
            for shape, configured in configured_shapes.items():
                if not isinstance(configured, (list, tuple, np.ndarray)):
                    continue
                configured_keys = [self._normalise_radius_key(value) for value in configured]
                loaded = loaded_direct if shape == 'circ' else (
                    self.processed_Data.get(shape, {}).get(direct, {})
                    if isinstance(self.processed_Data, dict) else {}
                )
                if not isinstance(loaded, dict):
                    actual_keys = []
                else:
                    actual_keys = [self._normalise_radius_key(value) for value in loaded]
                configured_set = set(configured_keys)
                actual_set = set(actual_keys)
                missing = [value for value in configured_keys if value not in actual_set]
                unexpected = [value for value in actual_keys if value not in configured_set]
                if missing or unexpected:
                    problems.append(
                        f'direct={direct}, shape={shape}: '
                        f'missing configured radii={missing}; '
                        f'unexpected data radii={unexpected}'
                    )
        if problems:
            raise ValueError(
                '定标数据半径与配置不一致；请显式设置 --radii。' + ' | '.join(problems)
            )

    def _get_freeze_idx(self):
        if self.freeze is None:
            return np.array([], dtype=int)

        freeze_idx = np.unique(np.atleast_1d(self.freeze).astype(int))

        n_weights = len(self.weights[self.direct])

        if np.any(freeze_idx < 0):
            raise ValueError(f"freeze 中不应包含负 index: {freeze_idx}")

        if np.any(freeze_idx >= n_weights):
            raise ValueError(f"freeze index 超出 weights 范围: freeze={freeze_idx}, weights长度={n_weights}")

        if self._uses_core_positions():
            # Freeze indices are repeated across each spatial weight group.
            local_weight_count = n_weights
            if getattr(self, 'weight_group_ids', None) is not None:
                local_weight_count = self.num_levels
            elif self.end_weights:
                local_weight_count = n_weights // 2
            if np.any(freeze_idx >= local_weight_count):
                raise ValueError(f"freeze index 超出单组 weights 范围: freeze={freeze_idx}, weights长度={local_weight_count}")
        else:
            n_slopes = self.slopes_shape[0]
            if np.any(freeze_idx >= n_slopes):
                raise ValueError(f"freeze index 超出 slopes 范围: freeze={freeze_idx}, slopes第一维长度={n_slopes}")

        return freeze_idx

    def _get_weight_freeze_mask(self):
        if getattr(self, 'end5cm_coverage_aware_weights', False):
            mask = np.ones(len(self.weights[self.direct]), dtype=bool)
            mask[self._get_weight_optim_indices()] = False
            return mask
        mask = np.zeros(len(self.weights[self.direct]), dtype=bool)
        freeze_idx = self._get_freeze_idx()
        mask[freeze_idx] = True

        if getattr(self, 'weight_group_ids', None) is not None:
            for group in range(1, max(self.weight_group_ids)+1):
                mask[freeze_idx + group*self.num_levels] = True
        elif self.end_weights:
            mask[freeze_idx + int(len(self.weights[self.direct]) / 2)] = True

        return mask

    def _ensure_coverage_support_map(self):
        """Build one fixed support map from the production score inputs."""
        if not getattr(self, 'end5cm_coverage_aware_weights', False):
            return None
        if self._coverage_support_map is None:
            from .batch import diagnose_score_distribution
            from .coverage import build_groupwise_support_map

            report = diagnose_score_distribution(self)
            self._coverage_support_map = build_groupwise_support_map(
                report['samples'], thresholds=self.thresholds,
                group_ids=self.weight_group_ids,
                min_effective_samples=self.end5cm_min_weight_effective_samples,
                min_max_mass=self.end5cm_min_weight_max_mass,
            )
            self._coverage_support_map['policy'] = self.end5cm_uncovered_weight_policy
        return self._coverage_support_map

    def _get_weight_optim_indices(self):
        """Return flat weight indices exposed to scipy in stable group/level order."""
        if not getattr(self, 'end5cm_coverage_aware_weights', False):
            return np.flatnonzero(~self._get_weight_freeze_mask())
        support = self._ensure_coverage_support_map()
        indices = []
        for group_name in sorted(support['groups'], key=int):
            group_index = int(group_name)
            indices.extend(
                group_index * self.num_levels + int(level)
                for level in support['groups'][group_name].get('free_levels', ())
            )
        return np.asarray(indices, dtype=int)

    def _expand_coverage_weight_candidate(self, free_values):
        from .coverage import expand_groupwise_weights

        support = self._ensure_coverage_support_map()
        values = np.asarray(free_values, dtype=float).reshape(-1)
        free_weights = {}
        cursor = 0
        for group_name in sorted(support['groups'], key=int):
            levels = [int(level) for level in support['groups'][group_name].get('free_levels', ())]
            free_weights[group_name] = {
                level: values[cursor + offset] for offset, level in enumerate(levels)
            }
            cursor += len(levels)
        if cursor != values.size:
            raise ValueError(f'coverage-aware free weight count mismatch: expected {cursor}, got {values.size}')
        return expand_groupwise_weights(free_weights, support).reshape(-1)

    def _get_slope_freeze_mask(self):
        mask = np.zeros(self.slopes_shape, dtype=bool)
        freeze_idx = self._get_freeze_idx()
        mask[freeze_idx, ...] = True
        return mask

    def optim_coeff(self, verbose=True):
        """
        按固定顺序执行两阶段优化：先优化 slopes，再优化 weights。

        处理流程：
        1. 根据 self.method 和 self.hyper_params 构造 scipy.optimize.minimize 的 options；
        2. 第一阶段优化 slopes，并设置 force_ideal_magnitude=True：
        - 重构时强制使用理论曲率大小；
        - 主要让 slopes 优化集中于曲率方向、扭转系数和空间形状；
        3. 第二阶段优化 weights，并设置 force_ideal_magnitude=False：
        - 恢复正常由应变估计 magnitude 的重构流程；
        - 主要调整不同曲率等级下的应变幅值权重；
        4. 使用最终参数对所有 calib_shapes 正常重构一次；
        5. 调用 _compute_error() 计算并打印最终误差统计。

        输出：
        - 参数仍写回 self.weights/self.slopes；返回两阶段状态，供区域批量任务保存。

        说明：
        - 第二阶段 init_loss 必须重新计算，不能沿用 slopes 阶段 loss；
        - 两个阶段的 loss 定义不同，因为第一阶段强制使用理论 magnitude；
        - 最终误差始终使用正常重构，即 force_ideal_magnitude=False。
        """
        self.validate_calibration_data_radii()
        self._ensure_coverage_support_map()
        if self.method == 'Powell':
            max_evals, xtol, ftol = self.hyper_params
            options = {"maxiter": int(max_evals), "xtol": float(xtol), "ftol": float(ftol), "disp": bool(verbose),}
        else:
            options = None

        start = time.time()
        core_positions = self._uses_core_positions()
        if self._uses_real_point_objective() and not core_positions:
            # Validate data before entering a potentially expensive optimization.
            fixed_slope_loss = self._real_point_loss(direction_only=True)
            if self.optim_weights_only:
                print('真实点 weights-only：固定继承slopes，仅优化weights(真实半径曲率)')
            else:
                print('真实点模式：slopes(首个真实点方向参考) -> weights(真实半径曲率)；最终仍报告几何误差')
        else:
            fixed_slope_loss = None
        if core_positions:
            stage_order = 'fixed core positions, optimize weights'
        elif self.optim_weights_only:
            stage_order = 'fixed slopes, optimize weights only'
        else:
            stage_order = ('slopes(真实点方向) -> weights(真实点曲率幅值)' if self._uses_real_point_objective()
                           else 'slopes(强制理想曲率) -> weights(正常曲率)')
        print(f"-------------------- 优化开始, 方向: {self.direct}, 旋转角度: {self.rot_ang}, 顺序: {stage_order} --------------------")

        # ---------- 第一阶段：优化 slopes，强制使用理想 magnitude ----------
        if not core_positions:
            if self.optim_weights_only:
                slopes_loss = fixed_slope_loss
                slopes_start_loss = fixed_slope_loss
                slopes_status = 'fixed'
                slopes_recoveries = []
            else:
                slopes_loss, slopes_status, slopes_start_loss, slopes_recoveries = self._optim_one_coeff_until_stop(coeff="slopes", init_loss=None, options=options, verbose=verbose, force_ideal_magnitude=True,)

            print(f"slopes 阶段结束: status={slopes_status}, start_loss={slopes_start_loss}, final_loss={slopes_loss}")
            print()

        # ---------- 第二阶段：优化 weights，恢复正常 magnitude ----------
        # 注意：这里必须 init_loss=None，不能沿用 slopes_loss。
        # 因为 slopes 阶段和 weights 阶段使用的是不同 loss 定义。
        weights_loss, weights_status, weights_start_loss, weights_recoveries = self._optim_one_coeff_until_stop(coeff="weights", init_loss=None, options=options, verbose=verbose, force_ideal_magnitude=False,)

        print(f"weights 阶段结束: status={weights_status}, start_loss={weights_start_loss}, final_loss={weights_loss}")
        print()

        current_loss = weights_loss

        end = time.time()
        print(f"优化耗时: {end - start}")
        end_label = 'weights-only优化结束' if core_positions or self.optim_weights_only else '两阶段优化结束'
        print(f"-------------------- {end_label}, 方向: {self.direct}, 旋转角度: {self.rot_ang} --------------------")
        print()

        # ---------- 最终误差：必须使用正常重构，不强制 magnitude ----------
        rs = {}
        for calib_shape in self.calib_shapes:
            r_results_postrot, interp_dist = self.reconstruct_shapes(calib_shape=calib_shape, direct=self.direct, verbose=False, force_ideal_magnitude=False)
            rs[calib_shape] = r_results_postrot

        self._compute_error(interp_dist=interp_dist, rs=rs)

        print(f"当前最优应变权重: {self.weights[self.direct].tolist()}")
        if not core_positions:
            print(f"当前最优扭转系数: {self.slopes[self.direct].tolist()}")
        print(f"最小误差: {current_loss}")
        self._print_errors()
        # 两阶段 loss 的曲率来源不同，分别保存；最终验收误差由正常重构统计获得。
        weight_loss_definition = (
            'mean squared relative real-point curvature error'
            if self._uses_real_point_objective() else 'measured-magnitude geometry'
        )
        loss_definitions = {}
        stages = {}
        if not core_positions:
            loss_definitions['slopes'] = (
                'fixed source slopes; not optimized' if self.optim_weights_only else
                ('mean circular direction error to first real point (1-cos)'
                 if self._uses_real_point_objective() else 'ideal-magnitude geometry')
            )
            slope_stage = {'status': slopes_status, 'start_loss': slopes_start_loss,
                           'final_loss': slopes_loss, 'source_sensor_params': self.source_sensor_params}
            if slopes_recoveries:
                slope_stage['recoveries'] = slopes_recoveries
            stages['slopes'] = slope_stage

        weight_stage = {'status': weights_status, 'start_loss': weights_start_loss,
                        'final_loss': weights_loss}
        if weights_recoveries:
            weight_stage['recoveries'] = weights_recoveries
        loss_definitions['weights'] = weight_loss_definition
        stages['weights'] = weight_stage
        return {'objective': ('real_points' if self._uses_real_point_objective() else 'geometry'),
                'loss_definitions': loss_definitions, 'stages': stages}

    def _optim_one_coeff_until_stop(self, coeff, init_loss, options, verbose=True, force_ideal_magnitude=False):
        """
        对单一参数类型连续优化，直到收敛、无改善、数值发散或达到最大轮数。

        参数：
        - coeff: 要优化的参数类型，只能为 'slopes' 或 'weights'。
        - init_loss: 本阶段已知的初始 loss；为 None 时由 _optim_coeff_iter() 内部计算。
        - options: 传递给 scipy.optimize.minimize 的 options。
        - verbose: 是否打印详细优化信息。当前主要由外层 optim_coeff() 控制 options。
        - force_ideal_magnitude: 当前阶段重构时是否强制使用理论曲率大小。

        处理流程：
        1. 保存每轮开始前的 weights 和 slopes，用于发散回滚；
        2. 将当前 slopes 按 slope_upper_bound 归一化；
        3. 调用 _optim_coeff_iter() 执行一轮优化；
        4. 若出现 ``SystemError: unknown opcode``，先回滚并重试当前轮一次；
        5. 重试仍失败且已有成功轮时，回退最近成功轮并结束当前阶段；
        6. 若本轮 loss 为非有限值，则认为数值发散并回滚；
        7. 若本轮得到有限但更差的 loss，则标记无改善并回滚；
        8. 若 loss 改善量处于 [0, min_loss_diff]，则认为收敛；
        9. 否则继续下一轮，直到达到 max_iter。

        输出：
        - current_loss: 阶段结束时的最佳 loss。
        - status: 字符串，取值为 'diverge'、'no_improvement'、'converge'、
          'max_iter' 或 'runtime_error_fallback'。
        - first_loss: 本阶段第一次实际计算得到的初始 loss。
        - recoveries: 可恢复 ``unknown opcode`` 事件列表；无事件时为空列表。

        说明：
        - 该函数负责阶段级别的循环、回滚和停止判定；
        - 单轮参数向量构造、bounds 和 scipy minimize 调用由 _optim_coeff_iter() 完成。
        """
        coeff_name = "扭转系数" if coeff == "slopes" else "应变权重"

        current_loss = init_loss
        best_loss = init_loss
        first_loss = init_loss
        last_successful_iteration = None
        recoveries = []

        print(f"-------------------- {coeff_name}连续优化开始 --------------------")

        core_positions = self._uses_core_positions()
        for i in range(self.max_iter):
            prev_weights = copy.deepcopy(self.weights[self.direct])
            slope_arguments = {}
            if not core_positions:
                prev_slopes = copy.deepcopy(self.slopes[self.direct])
                slopes_flattened = np.asarray(self.slopes[self.direct], dtype=float).reshape(-1)
                slope_upper_bound = max(np.max(np.abs(slopes_flattened)), self.slope_upper_bound)
                slope_arguments = {'slopes_normalised': slopes_flattened / slope_upper_bound,
                                   'slope_upper_bound': slope_upper_bound}

            first_unknown_opcode = None
            for attempt in range(2):
                try:
                    new_loss, iter_init_loss = self._optim_coeff_iter(
                        coeff=coeff, i=i, init_loss=best_loss, options=options,
                        force_ideal_magnitude=force_ideal_magnitude, **slope_arguments,
                    )
                except SystemError as exc:
                    if 'unknown opcode' not in str(exc).lower():
                        raise
                    self.weights[self.direct] = copy.deepcopy(prev_weights)
                    if not core_positions:
                        self.slopes[self.direct] = copy.deepcopy(prev_slopes)
                    if first_unknown_opcode is None:
                        first_unknown_opcode = exc
                    if attempt == 0:
                        print(f"{coeff_name}第{i}轮遇到 SystemError: unknown opcode，已回滚并重试本轮一次")
                        continue
                    if last_successful_iteration is None:
                        print(f"{coeff_name}第{i}轮重试仍失败，且没有可回退的成功轮次")
                        raise
                    recovery = {
                        'iteration': i,
                        'exception_type': type(first_unknown_opcode).__name__,
                        'message': str(first_unknown_opcode),
                        'retry_count': 1,
                        'occurrences': 2,
                        'outcome': 'fallback_previous_iteration',
                        'fallback_iteration': last_successful_iteration,
                        'fallback_loss': float(best_loss),
                    }
                    recoveries.append(recovery)
                    print(
                        f"{coeff_name}第{i}轮重试仍遇到 unknown opcode；"
                        f"已回退第{last_successful_iteration}轮结果并结束当前阶段"
                    )
                    return best_loss, "runtime_error_fallback", first_loss, recoveries
                else:
                    if first_unknown_opcode is not None:
                        recoveries.append({
                            'iteration': i,
                            'exception_type': type(first_unknown_opcode).__name__,
                            'message': str(first_unknown_opcode),
                            'retry_count': 1,
                            'occurrences': 1,
                            'outcome': 'retry_succeeded',
                        })
                        print(f"{coeff_name}第{i}轮重试成功")
                    break

            # 核心改动：第一次进入时，best_loss 可能还是 None
            if best_loss is None:
                best_loss = iter_init_loss
                current_loss = iter_init_loss

            if first_loss is None:
                first_loss = iter_init_loss

            prev_loss = best_loss

            if not np.isfinite(new_loss):
                self.weights[self.direct] = prev_weights
                if not core_positions:
                    self.slopes[self.direct] = prev_slopes
                current_loss = prev_loss

                print(f"{coeff_name}优化出现非有限 loss，已回滚")
                return current_loss, "diverge", first_loss, recoveries

            if new_loss > prev_loss:
                self.weights[self.direct] = prev_weights
                if not core_positions:
                    self.slopes[self.direct] = prev_slopes
                current_loss = prev_loss

                print(f"{coeff_name}优化未改善，已回滚")
                return current_loss, "no_improvement", first_loss, recoveries

            current_loss = new_loss
            best_loss = new_loss
            last_successful_iteration = i

            improvement = prev_loss - current_loss

            if 0 <= improvement <= self.min_loss_diff:
                print(f"{coeff_name}优化已收敛")
                return current_loss, "converge", first_loss, recoveries

        print(f"{coeff_name}优化已达最大轮数")
        return current_loss, "max_iter", first_loss, recoveries

    def _optim_coeff_iter(self, coeff, i, init_loss, slopes_normalised=None,
                          slope_upper_bound=None, options=None, force_ideal_magnitude=False):
        """
        执行一轮指定参数类型的 scipy 优化，并将最优结果写回对象状态。

        参数：
        - coeff: 要优化的参数类型，只能为 'slopes' 或 'weights'。
        - i: 当前连续优化轮次编号，仅用于打印日志。
        - init_loss: 本轮初始 loss；若为 None，则先用当前参数计算一次 loss。
        - slopes_normalised: 当前 slopes 展平并归一化后的数组。
        - slope_upper_bound: slopes 归一化和恢复原始量级使用的尺度。
        - options: 传递给 scipy.optimize.minimize 的 options。
        - force_ideal_magnitude: 本轮 loss 计算时是否强制使用理论曲率大小。

        处理流程：
        1. 复制当前 weights 与归一化 slopes，作为本轮优化基准；
        2. 根据 coeff 生成冻结掩码和待优化掩码：
        - coeff == 'slopes': 优化未冻结的归一化 slope 元素，bounds 为 [-1, 1]；
        - coeff == 'weights': 优化未冻结的 weight 元素，bounds 为 self.weight_bounds；
        3. 构造只接收自由变量 x 的 loss_fn，并在内部还原完整 weights/slopes；
        4. 若所有变量都被冻结，则跳过 minimize 并直接返回 init_loss；
        5. 调用 scipy.optimize.minimize；
        6. 将最优自由变量写回完整 weights 或 slopes，并保存到 self。

        输出：
        - current_loss: 本轮优化结束后的 loss。
        - init_loss: 本轮优化开始时的 loss。

        异常：
        - coeff 不是 'slopes' 或 'weights' 时抛出 ValueError；
        - bounds 数量与自由变量数量不一致时抛出 ValueError。

        说明：
        - slopes 始终以归一化形式交给优化器，写回 self.slopes 时再乘以 slope_upper_bound；
        - weights 不做额外归一化，直接在实际数值边界内优化。
        """
        print(f"---------- 第{i}轮优化开始 ----------")
        start = time.time()

        base_weights = np.asarray(self.weights[self.direct], dtype=float).copy()
        core_positions = self._uses_core_positions()
        if not core_positions:
            base_slopes_normalised = np.asarray(slopes_normalised, dtype=float).reshape(self.slopes_shape).copy()

        if coeff == "slopes" and not core_positions:
            freeze_mask = self._get_slope_freeze_mask()
            optimise_mask = ~freeze_mask

            x0 = base_slopes_normalised[optimise_mask]
            coeff_bounds = [(-1.0, 1.0)] * len(x0)

            def loss_fn(x):
                slopes_full = base_slopes_normalised.copy()
                slopes_full[optimise_mask] = x
                return self._loss_fn(self.rot_ang, base_weights, slopes_full.reshape(-1), slope_upper_bound, force_ideal_magnitude=force_ideal_magnitude)

        elif coeff == "weights":
            if getattr(self, 'end5cm_coverage_aware_weights', False):
                optimise_indices = self._get_weight_optim_indices()
                x0 = base_weights[optimise_indices]
            else:
                freeze_mask = self._get_weight_freeze_mask()
                optimise_indices = None
                optimise_mask = ~freeze_mask
                x0 = base_weights[optimise_mask]
            coeff_bounds = self.weight_bounds * len(x0)

            def loss_fn(x):
                if getattr(self, 'end5cm_coverage_aware_weights', False):
                    weights_full = self._expand_coverage_weight_candidate(x)
                else:
                    weights_full = base_weights.copy()
                    weights_full[optimise_mask] = x
                if core_positions:
                    return self._loss_fn(self.rot_ang, weights_full)
                return self._loss_fn(self.rot_ang, weights_full, base_slopes_normalised.reshape(-1), slope_upper_bound, force_ideal_magnitude=force_ideal_magnitude)

        else:
            raise ValueError(f"Unknown coeff: {coeff}")

        if len(coeff_bounds) != len(x0):
            raise ValueError(f"bounds 长度({len(coeff_bounds)})与{coeff}长度({len(x0)})不一致")

        if init_loss is None:
            init_loss = float(loss_fn(x0))
        else:
            init_loss = float(init_loss)

        if len(x0) == 0:
            if coeff == "weights" and getattr(self, 'end5cm_coverage_aware_weights', False):
                self.weights[self.direct] = self._expand_coverage_weight_candidate(x0)
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

        res = self._minimize(fun=loss_fn, x0=x0, method=self.method, options=options, bounds=coeff_bounds)
        current_loss = float(res.fun)

        if coeff == "slopes":
            slopes_best = base_slopes_normalised.copy()
            slopes_best[optimise_mask] = res.x
            self.slopes[self.direct] = (slopes_best * slope_upper_bound).reshape(self.slopes_shape)
            best_param = self.slopes[self.direct]

        elif coeff == "weights":
            if getattr(self, 'end5cm_coverage_aware_weights', False):
                weights_best = self._expand_coverage_weight_candidate(res.x)
            else:
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


    def _real_point_loss(self, *, direction_only):
        """Use circular-bend measurements, excluding virtual points and straight states.

        The first real point is unaffected by candidate slopes in the existing
        phase transport; it anchors each sample's direction after global phase
        has been determined. No interpolation, direction smoothing or geometric
        error enters this loss. Each radius and each valid real point has equal
        influence. Amplitude errors are normalized by that radius's curvature.
        """
        if self.calib_shapes != ['circ']:
            raise ValueError('真实点定标目前仅支持circ圆弧数据')
        radii = np.asarray(self.radii[self.direct]['circ'], dtype=float)
        if not np.all(np.isfinite(radii)) or np.any(radii < 0) or not np.any(radii > 0):
            raise ValueError('真实点定标需要有限正半径；0仅表示直线状态')
        losses = []
        slope_arguments = {} if self._uses_core_positions() else {'slope': self.slopes[self.direct]}
        for radius in radii[radii > 0]:
            magnitude, angle = self._Strain_2_Cur(
                keys=[radius], calib_shape='circ',
                calib_Data=self.processed_Data['circ'][self.direct],
                weight=self.weights[self.direct], direct=self.direct, **slope_arguments,
            )
            magnitude = np.asarray(magnitude, dtype=float)
            angle = np.asarray(angle, dtype=float)
            if (magnitude.shape != (self.total_points,) or angle.shape != magnitude.shape
                    or not np.all(np.isfinite(magnitude)) or not np.all(np.isfinite(angle))):
                raise ValueError('真实点曲率/方向包含非法数据')
            if direction_only:
                valid = magnitude > 1e-12
                if not valid[0]:
                    raise ValueError('首个真实点无有效曲率，不能确定方向参考')
                losses.append(float(np.mean(1. - np.cos(angle[valid] - angle[0]))))
            else:
                target_curvature = 1000. / radius
                losses.append(float(np.mean((magnitude / target_curvature - 1.) ** 2)))
        return float(np.mean(losses))

    def _loss_fn(self, rot_ang, weights, slopes_normalised=None, slope_upper_bound=None,
                 force_ideal_magnitude=False):
        """候选参数只在局部副本评估，返回标量后恢复当前最优参数。"""
        previous_weights = copy.deepcopy(self.weights)
        core_positions = self._uses_core_positions()
        if not core_positions:
            previous_slopes = copy.deepcopy(self.slopes)
        missing_rotation = object()
        previous_rotation = getattr(self, 'rot_ang', missing_rotation)
        try:
            self.rot_ang = rot_ang
            self.weights[self.direct] = np.array(weights, copy=True)
            if not core_positions:
                self.slopes[self.direct] = (slopes_normalised * slope_upper_bound).reshape(self.slopes_shape).copy()
            if self._uses_real_point_objective():
                return self._real_point_loss(direction_only=force_ideal_magnitude)
            rs = {}
            for calib_shape in self.calib_shapes:
                result, distance = self.reconstruct_shapes(calib_shape=calib_shape, direct=self.direct,
                    force_ideal_magnitude=force_ideal_magnitude)
                rs[calib_shape] = result
            self._compute_error(interp_dist=distance, rs=rs)
            return (self.k_point * self.avg_point_err['calib'] + self.k_end * self.avg_end_err['calib']
                    + self.k_pitch * self.avg_pitch_err + self.k_radius * self.avg_radius_err)
        finally:
            self.weights = previous_weights
            if not core_positions:
                self.slopes = previous_slopes
            if previous_rotation is missing_rotation:
                del self.rot_ang
            else:
                self.rot_ang = previous_rotation
