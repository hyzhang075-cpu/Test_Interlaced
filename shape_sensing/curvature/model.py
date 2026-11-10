"""重构所需的曲率模型与多区域参数装配。"""
import numpy as np
from copy import deepcopy
from ..configuration.regions import validate_end_weight_config, validate_weight_group_ids
from ..common.utils import dict_recur_paths, dict_recur_unflatten, dict_recur_extract_path

class CurvatureModelMixin:
    """复用旧 Reconstructor 的曲率和区域装配方法。"""

    def _Strain_2_Cur(self, keys, calib_shape, calib_Data, weight, slope, magnitude_ratio=1.0, verbose=False, direct=None):
        """
        将一组应变数据转换为最终用于三维重构的曲率大小与方向角。

        该方法包含两次曲率计算：
        1. 使用基础扭转系数 base_slopes 计算基础 magnitude / angle，用于曲率分级；
        2. 根据曲率等级选择或混合 weight / slope 后，再重新计算最终 magnitude / angle。

        参数：
        - keys: 当前数据在 calib_Data 中的路径。
        * None: 直接使用 calib_Data.T；
        * [key1]&#58; 使用 calib_Data[key1].T，常见于 circ/S；
        * [key1, key2]&#58; 使用 calib_Data[key1][key2].T，常见于 end/end3/rot0/up/down。
        - calib_shape: 当前标定形状类型。
        - calib_Data: 当前 direct 下的标定数据字典。
        - weight: 当前 direct 下的曲率等级权重。
        - slope: 当前 direct 下的曲率等级和分段扭转系数。
        - magnitude_ratio: 用于曲率分级前修正 magnitude 的比例因子。
        - verbose: 是否打印分数、质量分配、最终权重、最终分段和最终角度等中间结果。

        处理流程：
        1. 取 slope[0] 作为基础分段扭转系数 base_slopes。
        2. 调用 _get_base() 得到：
        - strain_data；
        - 基础 magnitude；
        - 基础 angle；
        - turn_segments。
        3. 用 score_magnitude = magnitude * magnitude_ratio 作为曲率分级依据。
        4. 根据 self.soft 进行曲率等级分配：
        - soft=False: 根据 thresholds 和 min_consecutive 生成硬分级 hard_level_idx；
        - soft=True: 调用 _get_scores() 和 _alloc_mass() 生成软分配 masses。
        5. 根据 masses 与 weight 计算每个采样点的 final_weight。
        若 end_weights=True，则 end_mask 区域使用对应等级的末端权重。
        6. 用 final_weight 对 strain_data 加权，得到 weighted_strain。
        7. 构造每个等级、每个采样点对应的 slope_vecs：
        - calib_shape == 'S' 时，根据 turn_segments 区分普通段和转折段；
        - 其他形状直接按 base_segments 分配 slope。
        8. 使用 masses 对 slope_vecs 加权求和，得到逐点 slope_map。
        9. 调用 _generate_dynamic_segments() 将逐点 slope_map 压缩为动态分段和分段 slope。
        10. 调用 _calculate_dynamic_angles(mode='recon') 得到最终动态通道角度。
        11. 用 weighted_strain 和 final_theta_dynamic 重新计算最终 magnitude / angle。

        输出：
        - magnitude: 最终曲率大小数组。
        - angle: 最终曲率方向角数组，单位 rad。

        说明：
        - 曲率分级影响两个东西：应变权重 final_weight 和扭转系数 slope_map。
        - soft=True 时，不同等级之间可以平滑混合；soft=False 时，每个采样点只属于一个等级。
        - calib_shape == 'S' 时，slope 通常被解释为两套分段系数：
        前 num_segments 个用于普通段，后 num_segments 个用于转折段。
        """
        if self.multi_region_parameters is not None:
            if direct is None:
                raise ValueError('multi-region reconstruction requires direct=0/120/240')
            if keys is None:
                strain_data = np.asarray(calib_Data).T
            elif len(keys) == 1:
                strain_data = np.asarray(calib_Data[keys[0]]).T
            elif len(keys) == 2:
                strain_data = np.asarray(calib_Data[keys[0]][keys[1]]).T
            magnitude, angle, _, _, diagnostics = self._assemble_multiregion_parameters(
                strain_data, direct, magnitude_ratio=magnitude_ratio,
            )
            self.last_parameter_assembly = diagnostics
            if verbose:
                print(f'multi-region parameter assembly: {diagnostics}')
            return magnitude, angle

        # ---------- 初始准备 ----------
        base_slopes = slope[0]

        # ---------- 使用基础扭转系数计算曲率 ----------
        strain_data, magnitude, angle, turn_segments = self._get_base(keys=keys, calib_Data=calib_Data, base_slopes=base_slopes, verbose=False)

        # ---------- 进行曲率分级和分级参数计算 ----------
        score_magnitude = magnitude * magnitude_ratio
        groups = getattr(self, 'weight_group_ids', None)
        if not self.soft:
            slope_scores = None
            hard_level_idx = np.zeros(self.total_points, dtype=int)
            hard_regions = [None] * self.num_levels
            min_consecutive = max(1, int(self.min_consecutive))

            for level, threshold in enumerate(self.thresholds, start=1):
                consecutive_count = 0
                region_start = None
                region_indices = None

                for i, mag in enumerate(score_magnitude):
                    if mag > threshold:
                        if consecutive_count == 0:
                            region_start = i
                        consecutive_count += 1
                    else:
                        if consecutive_count >= min_consecutive:
                            region_indices = np.arange(region_start, i)
                            break
                        consecutive_count = 0
                        region_start = None

                if region_indices is None and consecutive_count >= min_consecutive:
                    region_indices = np.arange(region_start, self.total_points)

                if region_indices is not None and len(region_indices) > 0:
                    hard_level_idx[region_indices] = level
                    hard_regions[level] = region_indices

            slope_masses = np.zeros((self.num_levels, self.total_points))
            slope_masses[hard_level_idx, np.arange(self.total_points)] = 1.0
            weight_scores = slope_scores
            weight_masses = slope_masses
        else:
            slope_scores = self._get_scores(magnitude=score_magnitude)
            slope_masses = self._alloc_mass(scores=slope_scores)
            weight_scores = slope_scores
            weight_masses = slope_masses
            if getattr(self, 'end5cm_independent_weight_score', False) and groups is not None:
                weight_scores, _ = self._get_groupwise_weight_scores(
                    score_magnitude, groups,
                )
                weight_masses = self._alloc_mass(weight_scores)
        if verbose:
            if slope_scores is not None:
                print(f"slope scores: {slope_scores}")
            else:
                print(f"hard level index: {hard_level_idx}")
                print(f"hard regions: {hard_regions}")
            for i in range(self.num_levels-1):
                print(f"Probability mass for curvature class {i+1}: {slope_masses[i]}")

        # ---------- 计算加权后的应变 ----------
        if groups is not None:
            weight_vecs = np.asarray(weight).reshape(-1, self.num_levels)[np.asarray(groups)].T
        else:
            weight_vecs = np.array([np.where(self.end_mask, weight[i+self.num_levels] if self.end_weights else weight[i], weight[i]) for i in range(self.num_levels)])
        final_weight = np.sum(weight_masses * weight_vecs, axis=0)
        weighted_strain = strain_data * final_weight
        if verbose:
            print(f"final weight: {final_weight}")
            print(f"weighted strain: {weighted_strain}")

        # ---------- 重新分段并分配扭转系数 ----------
        slope_vecs = np.zeros((self.num_levels, self.total_points))
        if calib_shape == 'S':
            turn_masks = np.zeros(self.total_points, dtype=bool)
            for k, (t_start, t_end) in enumerate(turn_segments):
                if k % 2 != 0:
                    turn_masks[t_start:t_end] = True
            for i in range(self.num_levels):
                for j, (start, end) in enumerate(self.base_segments):
                    turn_mask = turn_masks[start:end]
                    slope_vecs[i, start:end] = np.where(turn_mask, slope[i, j + self.num_segments], slope[i, j])
        else:
            for i in range(self.num_levels):
                for j, (start, end) in enumerate(self.base_segments):
                        slope_vecs[i, start:end] = slope[i, j]

        slope_map = np.sum(slope_masses * slope_vecs, axis=0)
        final_segments, final_slope = self._generate_dynamic_segments(slope_map=slope_map)
        if verbose:
            print(f"final segments: {final_segments}")
            print(f"final slopes: {final_slope}")

        # ---------- 使用处理过的新应变、分段与扭转系数计算最终结果 ----------
        final_theta_dynamic = self._calculate_dynamic_angles(mode='recon', segments=final_segments, seg_slopes=final_slope)
        magnitude, angle = self._calculate_curvature_and_angle(epsilon=weighted_strain, alpha_dynamic=final_theta_dynamic)
        if verbose:
            print(f"final_theta_dynamic: {final_theta_dynamic}")
            print(f"magnitude: {magnitude}")
            print(f"angle: {angle}")
            print()

        return magnitude, angle

    def _get_base(self, keys, calib_Data, base_slopes, verbose=False):
        """
        使用基础扭转系数计算一组数据的基础曲率大小、方向角和转折分段。

        参数：
        - keys: 当前数据路径。
        * None: 直接使用 calib_Data.T；
        * [radius]&#58; 使用 calib_Data[radius].T；
        * [radius, idx]&#58; 使用 calib_Data[radius][idx].T。
        - calib_Data: 当前 direct 下的标定数据字典。
        - base_slopes: 基础分段扭转系数，通常为 slope[0]。
        - verbose: 是否打印 strain_data、theta_dynamic、基础 magnitude 和基础 angle。

        处理流程：
        1. 使用 self.base_segments 和 base_slopes 调用 _calculate_dynamic_angles(mode='recon')，
        得到基础动态通道角度 theta_dynamic。
        2. 根据 keys 从 calib_Data 中提取当前组应变数据，并转置为
        (num_channels_recon, num_points)。
        3. 调用 _calculate_curvature_and_angle() 得到基础 magnitude 与 angle。
        4. 对 angle 做 np.unwrap(angle, discont=self.discont)。
        5. 根据 unwrap 后相邻点角度差是否超过 self.turn_threshold 识别转折点。
        6. 根据转折点生成 turn_segments，覆盖 [0, self.total_points)。

        输出：
        - strain_data: 当前组应变矩阵，形状通常为 (num_channels_recon, total_points)。
        - magnitude: 基础曲率大小数组。
        - angle: 基础曲率方向角数组。
        - turn_segments: 根据角度突变检测得到的分段区间列表。

        说明：
        - magnitude 和 angle 主要用于后续曲率分级，不一定是最终用于重构的曲率。
        - turn_segments 主要服务于 S 形数据中普通段与转折段的 slope 选择。
        """
        theta_dynamic = self._calculate_dynamic_angles(mode='recon', segments=self.base_segments, seg_slopes=base_slopes)
        if keys is None:
            strain_data = calib_Data.T
        elif len(keys) == 1:
            radius = keys[0]
            strain_data = calib_Data[radius].T
        elif len(keys) == 2:
            radius, idx = keys
            strain_data = calib_Data[radius][idx].T
        magnitude, angle = self._calculate_curvature_and_angle(epsilon=strain_data, alpha_dynamic=theta_dynamic)
        angle_unwrap = np.unwrap(angle, discont=self.discont)
        turn_idx = [0]
        for i in range(1, len(angle)):
            if np.abs(angle_unwrap[i] - angle_unwrap[i-1]) > self.turn_threshold:
                turn_idx.append(i)
        turn_segments = [(turn_idx[i], turn_idx[i+1]) for i in range(len(turn_idx) - 1)]
        turn_segments.append((turn_idx[-1], self.total_points))
        if verbose:
            print(f"---------- radius: {'dynamic' if not keys else keys} ----------")
            print(f"strain_data: {strain_data}")
            print(f"theta_dynamic: {theta_dynamic}")
            print(f"magnitude(base): {magnitude}")
            print(f"angle(base): {angle}")
        return strain_data, magnitude, angle, turn_segments

    def _alloc_mass(self, scores):
        """
        根据各曲率等级分数分配概率质量。

        说明：
        - 从高等级到低等级逐层扣减，每一层从剩余质量 remained 中分配 score 比例的质量，最终最低等级使用剩余的全部质量。
        - 返回结果各层质量之和在每个点上应接近 1，并最终裁剪到 [0, 1]。

        输入：
        - scores: 各曲率等级的分数字段，形状通常为 (num_levels - 1, total_points)

        输出：
        - final_masses: 各等级的概率质量分布，形状为 (num_levels, total_points)
        """
        masses = []
        scores = np.asarray(scores, dtype=float)
        remained = np.ones(scores.shape[1])
        for score in scores[::-1]:
            mass = remained * score
            masses.append(mass)
            remained -= mass
        final_masses = np.vstack((remained, np.asarray(masses[::-1])))
        return np.clip(final_masses, 0.0, 1.0)

    def _get_scores(self, magnitude, *, thresholds=None, sharpness=None,
                    min_consecutive=None, boundary_mode='constant'):
        """
        根据曲率大小计算各等级的软分级分数。

        说明：
        - 对每个阈值使用 sigmoid 函数计算超过该阈值的程度。
        - sharpness 控制 sigmoid 的陡峭程度。
        - 若 min_consecutive > 1，则再对每个等级分数做一次移动平均，以增强连续性。

        输入：
        - magnitude: 曲率大小数组

        输出：
        - scores: 各等级对应的分数数组，形状通常为 (num_levels - 1, total_points)
        """
        thresholds = self.thresholds if thresholds is None else thresholds
        sharpness = self.sharpness if sharpness is None else sharpness
        min_consecutive = self.min_consecutive if min_consecutive is None else min_consecutive
        scores = []
        for threshold in thresholds:
            level_score = 1.0 / (1.0 + np.exp(-sharpness * (np.asarray(magnitude) - threshold)))
            if min_consecutive > 1:
                level_score = self._moving_average_scipy(
                    data=level_score, window_size=min_consecutive, mode=boundary_mode,
                )
            scores.append(level_score)
        return np.asarray(scores)

    def _normalise_multiregion_parameters(self, config):
        """校验并规范化多区域参数包，建立 0/120/240 方向的 NumPy 参数表。

        每个区域的 support 点用于局部评分，owned 点决定最终写入范围；所有 owned
        点必须无重叠地覆盖完整传感器，防止拼接时出现未赋值或重复赋值。
        """
        config = deepcopy(config)
        meta = config.get('multi_region', config)
        regions = config.get('regions')
        order = list(meta.get('region_order', ()))
        directions = list(meta.get('directions', ()))
        if directions != [0, 120, 240]:
            raise ValueError(f'multi-region direct 必须恰好为 0、120、240；收到 {directions}')
        if not order or not isinstance(regions, dict) or set(order) != set(regions):
            raise ValueError('multi-region region_order 与 regions 不一致')

        sensor_points = [int(point) for point in np.asarray(self.global_point_ids).tolist()]
        owned_all = []
        for name in order:
            block = regions[name]
            thresholds = np.asarray(block.get('thresholds'), dtype=float)
            segmentation = np.asarray(block.get('local_segmentation'), dtype=int)
            support = [int(point) for point in block.get('support_point_ids', ())]
            owned = [int(point) for point in block.get('owned_point_ids', ())]
            segment_ids = [int(segment) for segment in block.get('global_segment_ids', ())]
            if (thresholds.ndim != 1 or np.any(~np.isfinite(thresholds))
                    or np.any(thresholds <= 0) or np.any(np.diff(thresholds) <= 0)):
                raise ValueError(f'{name} thresholds 必须严格递增')
            if (segmentation.ndim != 1 or len(segmentation) != len(segment_ids)
                    or np.any(segmentation <= 0) or int(segmentation.sum()) != len(support)):
                raise ValueError(f'{name} local segmentation/support 不一致')
            if not set(owned).issubset(support) or not set(support).issubset(sensor_points):
                raise ValueError(f'{name} owned/support points 无效')
            levels = len(thresholds) + 1
            enabled, end_start = validate_end_weight_config(
                block.get('end_weights', False), block.get('end_start', -5), len(support))
            groups = validate_weight_group_ids(block.get('weight_group_ids'), len(support))
            expected_weights = levels * (max(groups)+1 if groups is not None else (2 if enabled else 1))
            normalized_weights = {}
            normalized_slopes = {}
            for direct in directions:
                key = str(direct) if str(direct) in block['weights'] else direct
                weights = np.asarray(block['weights'][key], dtype=float)
                slopes = np.asarray(block['slopes'][key], dtype=float)
                if weights.shape != (expected_weights,) or not np.all(np.isfinite(weights)):
                    raise ValueError(f'{direct}/{name} weights shape 应为 {(expected_weights,)}')
                if slopes.shape != (levels, len(segmentation)) or not np.all(np.isfinite(slopes)):
                    raise ValueError(
                        f'{direct}/{name} slopes shape 应为 {(levels, len(segmentation))}'
                    )
                normalized_weights[direct] = weights
                normalized_slopes[direct] = slopes
            block['thresholds'] = thresholds
            block['local_segmentation'] = segmentation
            block['support_point_ids'] = support
            block['owned_point_ids'] = owned
            block['global_segment_ids'] = segment_ids
            block['weights'] = normalized_weights
            block['end_weights'] = enabled
            block['end_start'] = end_start
            block['weight_group_ids'] = groups
            block['slopes'] = normalized_slopes
            block['sharpness'] = float(block.get('sharpness', self.sharpness))
            block['score_window'] = max(1, int(block.get('score_window', self.min_consecutive)))
            block['score_boundary_mode'] = block.get('score_boundary_mode', 'constant')
            block['weight_score_mode'] = block.get('weight_score_mode', 'spatial_window')
            if block['weight_score_mode'] not in ('spatial_window', 'group_mean'):
                raise ValueError(f'{name} weight_score_mode 不合法: {block["weight_score_mode"]!r}')
            if block['weight_score_mode'] == 'group_mean' and groups is None:
                raise ValueError(f'{name} weight_score_mode=group_mean 需要 weight_group_ids')
            owned_all.extend(owned)
        if sorted(owned_all) != sorted(sensor_points) or len(owned_all) != len(set(owned_all)):
            raise ValueError(
                f'multi-region owned points 必须无重叠覆盖全部传感点；收到 {owned_all}'
            )
        return {'region_order': order, 'directions': directions, 'regions': regions,
                'use_full_transported_phase': bool(meta.get('use_full_transported_phase', False)),
                'phase_reference_directions': list(meta.get('phase_reference_directions', directions))}

    @staticmethod
    def _local_segment_labels(segmentation):
        """把局部分段长度展开成逐点的局部分段编号。"""
        return np.repeat(np.arange(len(segmentation)), segmentation.astype(int))

    def _get_groupwise_weight_scores(self, magnitude, group_ids, *, thresholds=None, sharpness=None):
        """Score each independent weight group from its own base-magnitude mean."""
        magnitude = np.asarray(magnitude, dtype=float)
        groups = np.asarray(group_ids, dtype=int)
        if magnitude.ndim != 1 or groups.shape != magnitude.shape:
            raise ValueError('groupwise weight scoring requires one group id per magnitude point')
        if groups.size == 0:
            raise ValueError('groupwise weight scoring requires at least one point')
        thresholds = self.thresholds if thresholds is None else thresholds
        sharpness = self.sharpness if sharpness is None else sharpness
        scores = np.empty((len(thresholds), magnitude.size), dtype=float)
        group_base_magnitude = {}
        for group in sorted(set(groups.tolist())):
            members = groups == group
            mean_magnitude = float(np.mean(magnitude[members]))
            group_base_magnitude[str(int(group))] = mean_magnitude
            group_score = self._get_scores(
                np.asarray([mean_magnitude]), thresholds=thresholds,
                sharpness=sharpness, min_consecutive=1,
            )[:, 0]
            scores[:, members] = group_score[:, None]
        return scores, group_base_magnitude

    def _assemble_multiregion_parameters(self, strain_data, direct, magnitude_ratio=1.0):
        """使用各区域自己的曲率阈值和 sigmoid 配置装配逐点参数。

        先用每个区域第 0 档 slope 形成全长基础相位并估计评分曲率，再在各区域
        support 点计算等级质量；最终只把混合后的 weight 和 slope 写入该区域的
        owned 点。``direct`` 是字面方向角 0、120 或 240。
        """
        config = self.multi_region_parameters
        if config is None:
            raise ValueError('multi_region_parameters 未配置')
        if direct not in config['directions']:
            raise ValueError(f'direct 必须为 0、120、240；收到 {direct}')
        strain_data = np.asarray(strain_data, dtype=float)
        if strain_data.ndim != 2 or strain_data.shape[1] != self.total_points:
            raise ValueError(
                f'strain_data 应为 (channels, {self.total_points})，收到 {strain_data.shape}'
            )
        point_index = {int(point): index for index, point in enumerate(self.global_point_ids)}
        base_slope_map = np.empty(self.total_points, dtype=float)

        for name in config['region_order']:
            block = config['regions'][name]
            support_labels = self._local_segment_labels(block['local_segmentation'])
            support_lookup = {
                point: local_index for local_index, point in enumerate(block['support_point_ids'])
            }
            base_slopes = block['slopes'][direct][0]
            for point in block['owned_point_ids']:
                local_index = support_lookup[point]
                base_slope_map[point_index[point]] = base_slopes[support_labels[local_index]]

        fixed_bridges = (config.get('use_full_transported_phase', False)
                         and direct in config.get('phase_reference_directions', config['directions']))
        if fixed_bridges:
            from ..initialization.phase_reference import apply_full_bridge_slopes
            apply_full_bridge_slopes(base_slope_map, config, self.global_point_ids, direct)
        base_segments, base_slopes = self._generate_dynamic_segments(base_slope_map)
        base_angles = self._calculate_dynamic_angles(
            mode='recon', segments=base_segments, seg_slopes=base_slopes,
        )
        base_magnitude, _ = self._calculate_curvature_and_angle(
            epsilon=strain_data, alpha_dynamic=base_angles,
        )
        score_magnitude = np.asarray(base_magnitude, dtype=float) * magnitude_ratio

        final_weight = np.empty(self.total_points, dtype=float)
        slope_map = np.empty(self.total_points, dtype=float)
        diagnostics = {
            'direct': direct,
            'base_slope_map': base_slope_map.tolist(),
            'fixed_full_level0_bridges': bool(fixed_bridges),
            'base_magnitude': np.asarray(base_magnitude).tolist(),
        }
        for name in config['region_order']:
            block = config['regions'][name]
            support_indices = np.asarray([point_index[p] for p in block['support_point_ids']], dtype=int)
            owned_indices = np.asarray([point_index[p] for p in block['owned_point_ids']], dtype=int)
            support_lookup = {
                point: local_index for local_index, point in enumerate(block['support_point_ids'])
            }
            owned_local = np.asarray([support_lookup[p] for p in block['owned_point_ids']], dtype=int)
            support_score = score_magnitude[support_indices]
            slope_scores = self._get_scores(
                support_score,
                thresholds=block['thresholds'],
                sharpness=block['sharpness'],
                min_consecutive=block['score_window'],
                boundary_mode=block['score_boundary_mode'],
            )
            slope_masses = self._alloc_mass(slope_scores)
            if block['weight_score_mode'] == 'group_mean':
                weight_scores, group_base_magnitude = self._get_groupwise_weight_scores(
                    support_score,
                    block['weight_group_ids'],
                    thresholds=block['thresholds'],
                    sharpness=block['sharpness'],
                )
                weight_masses = self._alloc_mass(weight_scores)
            else:
                weight_scores = slope_scores
                weight_masses = slope_masses
                group_base_magnitude = None
            weights = block['weights'][direct]
            levels = weight_masses.shape[0]
            weight_vectors = np.broadcast_to(weights[:levels, None], weight_masses.shape).copy()
            if block['weight_group_ids'] is not None:
                weight_vectors = weights.reshape(-1, levels)[np.asarray(block['weight_group_ids'])].T
            elif block['end_weights']:
                # 负索引相对于区域support真实点，虚拟点随后复制补偿后的末点。
                weight_vectors[:, block['end_start']:] = weights[levels:, None]
            support_weights = np.sum(weight_masses * weight_vectors, axis=0)
            labels = self._local_segment_labels(block['local_segmentation'])
            slopes = block['slopes'][direct]
            slope_vectors = slopes[:, labels]
            support_slopes = np.sum(slope_masses * slope_vectors, axis=0)
            final_weight[owned_indices] = support_weights[owned_local]
            slope_map[owned_indices] = support_slopes[owned_local]
            diagnostics[name] = {
                'support_point_ids': list(block['support_point_ids']),
                'owned_point_ids': list(block['owned_point_ids']),
                'thresholds': block['thresholds'].tolist(),
                'score_mode': block['weight_score_mode'],
                'weight_score_mode': block['weight_score_mode'],
                'group_base_magnitude': group_base_magnitude,
                'scores': slope_scores.tolist(),
                'masses': slope_masses.tolist(),
                'slope_scores': slope_scores.tolist(),
                'slope_masses': slope_masses.tolist(),
                'weight_scores': weight_scores.tolist(),
                'weight_masses': weight_masses.tolist(),
                'mass_shape': list(weight_masses.shape),
                'end_weights': block['end_weights'],
                'end_start': block['end_start'],
                'weight_group_ids': block['weight_group_ids'],
                'effective_weights': support_weights.tolist(),
            }

        if fixed_bridges:
            apply_full_bridge_slopes(slope_map, config, self.global_point_ids, direct)
        weighted_strain = strain_data * final_weight
        final_segments, final_slopes = self._generate_dynamic_segments(slope_map)
        final_angles = self._calculate_dynamic_angles(
            mode='recon', segments=final_segments, seg_slopes=final_slopes,
        )
        magnitude, angle = self._calculate_curvature_and_angle(
            epsilon=weighted_strain, alpha_dynamic=final_angles,
        )
        diagnostics['final_weight'] = final_weight.tolist()
        diagnostics['slope_map'] = slope_map.tolist()
        diagnostics['final_magnitude'] = np.asarray(magnitude).tolist()
        diagnostics['final_angle'] = np.asarray(angle).tolist()
        diagnostics['final_theta_dynamic'] = np.asarray(final_angles).tolist()
        return magnitude, angle, final_weight, slope_map, diagnostics

    def _generate_dynamic_segments(self, slope_map):
        """
        根据逐点扭转系数映射生成动态分段结构。

        输入：
        - slope_map: 每个采样点处的扭转系数数组

        输出：
        - final_segments: 动态分段区间列表，格式为 [(start, end), ...]
        - final_slope: 每个分段对应的扭转系数列表
        """
        final_segments = []
        final_slope = []
        current_slope = slope_map[0]
        segment_start = 0
        for i in range(1, self.total_points):
            if slope_map[i] != current_slope:
                final_segments.append((segment_start, i))
                final_slope.append(current_slope)
                current_slope = slope_map[i]
                segment_start = i
        final_segments.append((segment_start, self.total_points))
        final_slope.append(current_slope)
        return final_segments, final_slope
