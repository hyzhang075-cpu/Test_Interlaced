"""按区域执行独立定标，保存结果，并诊断实际用于选档的基础曲率。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import pprint
import re
from uuid import uuid4

import numpy as np


# 这些参数控制任务/界面/输出，不属于 Optimiser/Sensor 的构造参数。
CONTROL_KEYS = {
    'task', 'ang_step', 'ang_range', 'multi_region_calib', 'optimization_region',
    'optimization_regions', 'full_segmentation', 'calibration_regions',
    'calibration_data_root', 'calibration_output_dir', '_explicit_seed_parameters',
    '_full_segmentation_explicit',
    'total_scale', 'imu', 'imu_com', 'imu_maxlen', 'demo_recorder', 'unwrap_threshold',
    'align_threshold', 'direction_blend_enabled', 'direction_reference_direct',
    'direction_min_curvature', 'direction_min_consecutive', 'direction_consistency_threshold',
    'direction_smooth_alpha', 'demo_kalman_enabled', 'demo_kalman_Rk', 'demo_kalman_Bk',
    'anchor_constraint_enabled', 'anchor_translation_enabled', 'anchor_rotation_enabled',
    'anchor_offset_m',
}


def optimizer_parameters(args):
    return {key: deepcopy(value) for key, value in vars(args).items() if key not in CONTROL_KEYS}


def prepare_calibration_jobs(args, configure):
    """先校验全部任务，再优化；每个任务从原始 parser 配置独立复制。

不将上一个区域解析后的 folder_path、裁剪或最优参数带入下一个区域。
多区域运行的差异参数应放在各区域字典中，避免单个覆盖值误用于全部长度。
"""
    regions = args.optimization_regions if args.multi_region_calib else [args.optimization_region]
    if (not isinstance(regions, list) or not regions or
            any(not isinstance(region, str) for region in regions) or len(set(regions)) != len(regions)):
        raise ValueError('optimization_regions 必须是非空、无重复的区域名称列表')
    if len(regions) > 1:
        for key in ('folder_path', 'radii', 'thresholds', 'segmentation', 'straight_trim', 'weights', 'slopes'):
            if getattr(args, key) is not None:
                raise ValueError(f'批量任务不能共用单个 {key} 覆盖值；请写入 calibration_regions 各区域配置')
    jobs = []
    for region in regions:
        job = deepcopy(args)
        job.optimization_region = region
        context = configure(job)
        if not Path(job.folder_path).is_dir():
            raise ValueError(f'{region} 数据目录不存在: {job.folder_path}')
        jobs.append((job, context))
    return jobs


def _native(value):
    """将 NumPy 结果转成可复制的 Python/JSON 值；缺失指标以 null 表示。"""
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _save_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(_native(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def collect_calibration_result(optim, stages):
    """收集当前方向最优参数与最终正常模式误差，不将 slope 阶段 loss 当作验收误差。"""
    normal_loss = (optim.k_point * optim.avg_point_err['calib']
                   + optim.k_end * optim.avg_end_err['calib']
                   + optim.k_pitch * optim.avg_pitch_err + optim.k_radius * optim.avg_radius_err)
    return _native({
        'parameters': {
            'weights': {optim.direct: optim.weights[optim.direct]},
            'slopes': {optim.direct: optim.slopes[optim.direct]},
            'global_phases_set': optim.global_phases_set,
            'rot_ang': optim.rot_ang,
        },
        'optimized_directions': [optim.direct],
        'errors': {
            'normal_weighted_loss': normal_loss,
            'avg_point_err': optim.avg_point_err, 'avg_end_err': optim.avg_end_err,
            'avg_pitch_err': optim.avg_pitch_err, 'avg_radius_err': optim.avg_radius_err,
            'point_errors_by_sample': optim.point_errors_avg,
            'end_errors_by_sample': optim.end_errors,
        },
        'optimization': stages,
    })


def diagnose_score_distribution(optim):
    """使用生产 _get_base/_get_scores/_alloc_mass 检查选档输入和空间边界。

对应静态运行的 magnitude_ratio=1.0；不使用补偿后的 magnitude、理想曲率
替换或 Kalman 输出。processed_Data 是每个定标状态的均值，因此统计反映
空间点/定标状态差异，不代表原始帧噪声或独立重复采集分布。
"""
    if not optim.soft:
        raise ValueError('task score 当前诊断 sigmoid 软分级，请设置 soft=True')
    report = {
        'basis': 'strain base magnitude before weights; magnitude_ratio=1.0',
        'sample_semantics': 'per-calibration-state mean strain, not independent raw frames',
        'thresholds': list(optim.thresholds), 'sharpness': optim.sharpness,
        'score_window': optim.min_consecutive, 'boundary_mode': 'constant',
        'sigmoid_10_90_width_m_inv': float(2*np.log(9)/optim.sharpness) if optim.sharpness > 0 else None,
        'samples': [],
    }

    def leaves(data, path=()):
        if isinstance(data, dict):
            for key, value in data.items():
                yield from leaves(value, path + (key,))
        else:
            yield path

    for shape in optim.calib_shapes:
        data = optim.processed_Data[shape][optim.direct]
        allowed = optim.radii[optim.direct][shape]
        for path in leaves(data):
            if path[0] not in allowed:
                continue
            if len(path) == 2 and path[1] not in allowed[path[0]]:
                continue
            _, magnitude, _, _ = optim._get_base(
                keys=list(path), calib_Data=data,
                base_slopes=optim.slopes[optim.direct][0], verbose=False,
            )
            magnitude = np.asarray(magnitude, dtype=float)
            if not np.all(np.isfinite(magnitude)):
                raise ValueError(f'{shape}/{path} 基础评分曲率包含非有限值')
            scores = optim._get_scores(magnitude)
            masses = optim._alloc_mass(scores)
            theoretical = (0.0 if path[0] == 0 else 1000.0/path[0]) if shape in ('circ', 'S') else None
            report['samples'].append({
                'shape': shape, 'keys': list(path), 'direct': optim.direct,
                'theoretical_curvature_m_inv': theoretical,
                'global_point_ids': optim.global_point_ids.tolist(),
                'score_magnitude': magnitude.tolist(), 'scores': scores.tolist(),
                'masses': masses.tolist(),
                'quantiles': dict(zip(('min', 'p10', 'median', 'p90', 'max'), np.quantile(magnitude, [0,.1,.5,.9,1]).tolist())),
                'mean_level_mass': masses.mean(axis=1).tolist(),
                'tip_level_mass': masses[:, -1].tolist(),
            })
    if not report['samples']:
        raise ValueError('没有可用于评分诊断的定标样本')
    return report


def _plot_scores(report, path):
    # 科学图独立保存，避免初始化 GUI 或阻塞批量任务。
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 6), tight_layout=True)
    FigureCanvasAgg(fig)
    ax, bx = fig.subplots(1, 2)
    for index, sample in enumerate(report['samples']):
        values = sample['score_magnitude']
        ax.scatter([index]*len(values), values, s=12)
    for threshold in report['thresholds']:
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=.6)
    labels = [str(sample['keys']) for sample in report['samples']]
    ax.set(xticks=range(len(labels)), xticklabels=labels, xlabel='Calibration keys (radius in mm)', ylabel='Base curvature (1/m)')
    ax.tick_params(axis='x', rotation=90)
    means = np.array([sample['mean_level_mass'] for sample in report['samples']])
    bottom = np.zeros(len(means))
    for level in range(means.shape[1]):
        bx.bar(range(len(means)), means[:, level], bottom=bottom, label=f'level {level}')
        bottom += means[:, level]
    bx.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0,1), ylabel='Mean actual level mass')
    bx.tick_params(axis='x', rotation=90)
    bx.legend(fontsize='small')
    fig.savefig(path, dpi=150)


def run_calibration_jobs(jobs, optimizer_factory, *, output_root):
    """逐区域执行并立即落盘。异常时保留已完成项并标记失败项，然后停止。

每次启动使用独立目录，防止覆盖另一方向的定标结果。导出的参数只标记
本次 --direct 为已优化，不把其他方向的初始化值冒充最优参数。
"""
    first = jobs[0][0]
    sensor = re.sub(r'[^\w.-]+', '_', first.sensor_id).strip('.') or 'sensor'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6]
    output = Path(output_root) / f'{sensor}-{run_id}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'sensor_id': first.sensor_id, 'task': first.task, 'direct': first.direct,
              'regions': {args.optimization_region: {'status': 'pending'} for args, _ in jobs}}
    print(f'区域结果目录: {output}')
    for index, (args, context) in enumerate(jobs):
        region = args.optimization_region
        entry = report['regions'][region]
        entry.update(configuration=_native(vars(args)), context=context, status='running')
        _save_report(output / 'results.json', report)
        try:
            print(f'========== {index+1}/{len(jobs)} 区域: {region}; 方向: {args.direct} ==========')
            optim = optimizer_factory(**optimizer_parameters(args))
            optim.calibration_context = context
            if args.task == 'score':
                entry['score_distribution'] = diagnose_score_distribution(optim)
                _plot_scores(entry['score_distribution'], output / f'score-{index+1}.png')
            else:
                stages = optim.optim_coeff(verbose=True)
                entry.update(collect_calibration_result(optim, stages))
            entry['status'] = 'complete'
        except Exception as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            _save_report(output / 'results.json', report)
            print(f'{region} 失败，已完成区域结果保留在 {output}')
            raise
        _save_report(output / 'results.json', report)
        completed = {key: value['parameters'] for key, value in report['regions'].items()
                     if value['status'] == 'complete' and 'parameters' in value}
        (output / 'parameters.py').write_text(
            '# 仅包含本次 direct 的优化结果；其他方向需分别运行并补齐。\n'
            '# 以下区域相位仍是各自局部参考，不能直接当作全段部署参数包。\n'
            'region_parameters = ' + pprint.pformat(completed, width=110, sort_dicts=False) + '\n',
            encoding='utf-8',
        )
    print('========== 区域汇总 ==========')
    for region, entry in report['regions'].items():
        if 'parameters' in entry:
            print(region, 'weights=', entry['parameters']['weights'], 'slopes=', entry['parameters']['slopes'])
            print('正常重构误差:', entry['errors'])
        else:
            print(region, '评分样本数:', len(entry['score_distribution']['samples']))
    return output, report
