"""定标区域的几何与配置解析，不依赖 GUI、硬件或优化器。

局部数据使用局部坐标重构，但保留全局真实数据行号及全局空间段编号，
供后续区域参数对齐使用。这里的行号不是解调仪提供的硬件 FBG 标签。
"""

from copy import deepcopy

import numpy as np


def compute_fbg_geometry(start_value, end_value, samp_dist_preset):
    """mm 输入，返回全长(m)和未裁剪真实采样坐标(m)。

沿用 Sensor 的点序约定：预设位置的间距倒序排列。总长度只取一次
首末预设跨度，加上近端、远端两个无传感点间隙，虚拟点不计入点数。
"""
    preset = np.asarray(samp_dist_preset, dtype=float)
    gaps = np.asarray([start_value, end_value], dtype=float)
    if (preset.ndim != 1 or preset.size < 2 or not np.all(np.isfinite(preset))
            or not np.all(np.diff(preset) > 0)):
        raise ValueError('samp_dist_preset 必须包含至少两个有限、严格递增的 mm 位置')
    if not np.all(np.isfinite(gaps)) or np.any(gaps < 0):
        raise ValueError('start_value/end_value 必须是有限非负 mm 数值')
    ideal_len = float((gaps.sum() + preset[-1] - preset[0]) / 1000.0)
    positions = (gaps[0] + np.r_[0.0, np.cumsum(np.diff(preset)[::-1])]) / 1000.0
    return ideal_len, positions


def _validate_partition(segmentation, point_count, label):
    """拒绝会被旧 _calculate_segments 静默截断或补长的空间分段。"""
    values = np.asarray(segmentation, dtype=float)
    if (values.ndim != 1 or not values.size or not np.all(np.isfinite(values))
            or np.any(values <= 0) or np.any(values != np.floor(values))
            or np.sum(values) != point_count):
        raise ValueError(f'{label}分段必须为正整数列表且总和等于 {point_count} 个真实点；收到 {segmentation}')
    return values.astype(int).tolist()


def resolve_calibration_region(*, start_value, end_value, samp_dist_preset,
                               full_segmentation, profiles,
                               optimization_region, multi_region_calib):
    """在读取数据前生成单次定标配置，严格保持局部和全局空间段一致。

    multi_region_calib=False 选择一个区域独立定标或重构。
    True 允许批量选择嵌套定标长度；一次仍优化一个数据集，并不代表已联合优化。
每个局部区域必须由若干完整的全长空间段组成；不自动拆段或合并段。
"""
    ideal_len, positions = compute_fbg_geometry(start_value, end_value, samp_dist_preset)
    full_segments = _validate_partition(full_segmentation, len(positions), '全长')
    if optimization_region not in profiles:
        raise ValueError(f'未知 optimization_region={optimization_region!r}；可选 {list(profiles)}')
    profile = deepcopy(profiles[optimization_region])
    length = profile['length_m']
    if optimization_region == 'full':
        if length is not None:
            raise ValueError('full 的 length_m 必须为 None，长度由传感器几何计算')
        crop_start = 0.0
        start_index = 0
    else:
        if length is None or not np.isfinite(length) or not 0 < length <= ideal_len:
            raise ValueError(f'{optimization_region} 的 length_m 必须在 (0, {ideal_len}] m 内')
        crop_start = ideal_len - float(length)
        # 只吸收浮点舍入误差，恰好在区域边界上的真实点应被保留。
        start_index = int(np.searchsorted(positions, crop_start - 1e-12))
        if start_index == len(positions):
            raise ValueError(f'{optimization_region} 内没有真实传感点')

    boundaries = np.r_[0, np.cumsum(full_segments)]
    matches = np.flatnonzero(boundaries[:-1] == start_index)
    if not len(matches):
        raise ValueError(
            f'{optimization_region} 从全局第 {start_index + 1} 个真实点开始，切入全长分段；'
            f'全长分段={full_segments}，必须在该点之前设置独立分段边界'
        )
    first_segment = int(matches[0])
    expected_segments = full_segments[first_segment:]
    local_segments = _validate_partition(profile['segmentation'], len(positions) - start_index,
                                         optimization_region)
    if local_segments != expected_segments:
        raise ValueError(
            f'{optimization_region} 分段 {local_segments} 与全长对应分段 {expected_segments} 不一致；'
            '请同步修改全长与局部分段，不允许仅按相同点数拼接'
        )
    thresholds = np.asarray(profile['thresholds'], dtype=float)
    if (thresholds.ndim != 1 or not np.all(np.isfinite(thresholds))
            or np.any(thresholds <= 0) or np.any(np.diff(thresholds) <= 0)):
        raise ValueError(f'{optimization_region} thresholds 必须是有限、正值、严格递增列表')
    if not profile['radii'] or not profile['folder_path']:
        raise ValueError(f'{optimization_region} 缺少 radii 或 folder_path')

    samples = positions[start_index:] - crop_start
    samples[np.abs(samples) < 1e-12] = 0.0
    profile.update(
        ideal_len=ideal_len,
        straight_trim=0.0 if optimization_region == 'full' else -float(length),
        segmentation=local_segments,
        thresholds=thresholds.tolist(),
        optimization_region=optimization_region,
        global_point_ids=list(range(start_index + 1, len(positions) + 1)),
        global_segment_ids=list(range(first_segment, len(full_segments))),
        global_sample_dist_m=positions[start_index:].tolist(),
        sample_dist_m=samples.tolist(),
        crop_origin_m=crop_start,
        reconstruction_length_m=ideal_len if optimization_region == 'full' else float(length),
    )
    return profile


def validate_calibration_parameters(weights, slopes, thresholds, segmentation,
                                    directions, *, end_weights=False, s_shape=False):
    """预载参数必须匹配所选区域，防止旧 5 cm 数组被误用于新全长定标。

    None 表示让现有初始化器生成初值；此处不猜测或重采样已定标参数。
    返回独立的 NumPy 数组字典，支持重构中的 slope[level, segment] 索引，
    同时不修改 CLI/配置字典中的原始列表。
"""
    levels = len(thresholds) + 1
    expected = {
        'weights': (levels * (2 if end_weights else 1),),
        'slopes': (levels, len(segmentation) * (2 if s_shape else 1)),
    }
    normalized = []
    for name, parameters in [('weights', weights), ('slopes', slopes)]:
        if parameters is None:
            normalized.append(None)
            continue
        if not isinstance(parameters, dict):
            raise ValueError(f'{name} 必须是方向到参数数组的字典或 None')
        result = {}
        for direct in directions:
            if direct not in parameters:
                raise ValueError(f'{name} 缺少方向 {direct}')
            values = np.asarray(parameters[direct], dtype=float)
            if values.shape != expected[name] or not np.all(np.isfinite(values)):
                raise ValueError(f'{name}[{direct}] 需为有限的 {expected[name]} 数组，收到 {values.shape}；'
                                 '请提供所选区域参数或使用 None 重新初始化')
            result[direct] = values.copy()
        normalized.append(result)
    return tuple(normalized)
