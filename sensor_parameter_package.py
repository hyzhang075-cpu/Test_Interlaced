"""Merge regional calibration reports into an ordered sensor parameter package."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np


FORMAT_VERSION = 2
REQUIRED_DIRECTIONS = (0, 120, 240)
REGION_ORDER = ('full', 'end15cm', 'end5cm')

PARSER_GROUPS = {
    'hardware_parameters': (
        'start_value', 'end_value', 'R', 'ideal_len', 'samp_dist_preset',
        'samp_dist_preset_sub', 'centre_fibre', 'interlaced_grating',
        'interlaced_offset',
    ),
    'calibration_experiment_parameters': (
        'interrogator', 'active_channels', 'channel_change', 'channel_layout',
        'active_angles', 'rot_dir', 'exp_directs', 'end_len', 'ss',
    ),
    'optimization_hyperparameters': (
        'freeze', 'weight_bounds', 'slope_upper_bound', 'max_iter', 'k_point',
        'k_end', 'k_pitch', 'k_radius', 'method', 'hyper_params', 'min_loss_diff',
    ),
    'dynamic_display_parameters': (
        'IP', 'PORT', 'maxsize', 'data_freq', 'use_stored_zero',
        'zero_reference_path', 'total_scale', 'imu', 'imu_com', 'imu_maxlen',
        'demo_recorder', 'unwrap_threshold', 'align_threshold',
        'direction_blend_enabled', 'direction_reference_direct',
        'direction_min_curvature', 'direction_min_consecutive',
        'direction_consistency_threshold', 'direction_smooth_alpha',
        'demo_kalman_enabled', 'demo_kalman_Rk', 'demo_kalman_Bk',
        'anchor_translation_enabled', 'anchor_rotation_enabled',
        'anchor_constraint_enabled', 'anchor_offset_m',
    ),
    'reconstruction_parameters': (
        'soft', 'reverse', 'discont', 'frame_rule', 'straight_threshold',
        'rot_threshold', 'min_consecutive', 'move_filter',
        'move_filter_for_vector', 'kalman_params', 'smoothing_params',
        'interp_num', 'curvature_interp', 'nonnegative_curvature', 'sharpness',
        'end_weights', 'end_start', 'TNB_threshold',
    ),
    'initialization_parameters': ('connect', 'turn_threshold', 'sensing_ranges'),
    'angle_optimization_parameters': ('ang_step', 'ang_range'),
    'runtime_control_parameters': (
        'sensor_params',
        'multi_region_calib', 'optimization_region', 'optimization_regions',
        'full_segmentation', 'calibration_data_root', 'calibration_output_dir',
        'folder_path', 'radii', 'direct', 'thresholds', 'straight_trim',
        'segmentation', 'weights', 'slopes', 'task',
    ),
}

SPECIAL_DESTINATIONS = ('sensor_id', 'calibration_regions', 'global_phases_set', 'rot_ang')


def _native(value):
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('参数包不能包含非有限数值')
    return value


def _parser_defaults(parser):
    return {
        action.dest: deepcopy(action.default)
        for action in parser._actions
        if action.dest != 'help'
    }


def _validate_parser_groups(defaults):
    grouped = [name for names in PARSER_GROUPS.values() for name in names]
    represented = grouped + list(SPECIAL_DESTINATIONS)
    missing = [name for name in defaults if name not in represented]
    duplicate = sorted({name for name in represented if represented.count(name) > 1})
    extra = [name for name in represented if name not in defaults]
    if missing or duplicate or extra:
        raise ValueError(
            f'parser 参数分组不完整: missing={missing}, duplicate={duplicate}, extra={extra}'
        )


def load_calibration_results(paths):
    reports = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding='utf-8'))
        direct = report.get('direct')
        if direct in seen:
            raise ValueError(f'定标方向重复: {direct}')
        seen.add(direct)
        report['_source_path'] = str(path.resolve())
        reports.append(report)
    return sorted(reports, key=lambda item: REQUIRED_DIRECTIONS.index(item['direct'])
                  if item.get('direct') in REQUIRED_DIRECTIONS else len(REQUIRED_DIRECTIONS))


def _region_context(entry, region):
    context = entry.get('context')
    if not isinstance(context, dict):
        raise ValueError(f'{region} 缺少 context')
    required = ('thresholds', 'segmentation', 'global_point_ids', 'global_segment_ids')
    if any(name not in context for name in required):
        raise ValueError(f'{region} context 不完整')
    return context


def _same_context(left, right):
    keys = (
        'optimization_region', 'length_m', 'radii', 'thresholds', 'segmentation',
        'global_point_ids', 'global_segment_ids', 'global_sample_dist_m',
        'sample_dist_m', 'crop_origin_m', 'reconstruction_length_m',
        'ideal_len', 'straight_trim',
    )
    return all(_native(left.get(key)) == _native(right.get(key)) for key in keys)


def _stage_failures(report):
    failures = []
    direct = report['direct']
    for region in REGION_ORDER:
        entry = report['regions'][region]
        stages = entry.get('optimization', {}).get('stages', {})
        for stage in ('slopes', 'weights'):
            status = stages.get(stage, {}).get('status')
            if status != 'converge':
                status_text = 'diverged' if status == 'diverge' else (status or 'missing')
                failures.append(f'{direct}/{region} {stage} {status_text}')
    return failures


def _owned_points(supports, region_index):
    inner = set()
    for later in REGION_ORDER[region_index + 1:]:
        inner.update(supports[later])
    return [point for point in supports[REGION_ORDER[region_index]] if point not in inner]


def build_sensor_package(parser, reports, *, source_commit=None, generated_at=None):
    defaults = _parser_defaults(parser)
    _validate_parser_groups(defaults)
    reports = sorted(list(reports), key=lambda item: item.get('direct', -1))
    directions = [report.get('direct') for report in reports]
    if directions != list(REQUIRED_DIRECTIONS):
        raise ValueError(f'direct 必须恰好为 0、120、240；收到 {directions}')
    sensor_ids = {report.get('sensor_id') for report in reports}
    if len(sensor_ids) != 1 or None in sensor_ids:
        raise ValueError(f'results.json 的 sensor_id 不一致: {sensor_ids}')
    sensor_id = next(iter(sensor_ids))

    for report in reports:
        if set(report.get('regions', {})) != set(REGION_ORDER):
            raise ValueError(f'{report["direct"]} 缺少 full/end15cm/end5cm 区域')

    first_entries = {region: reports[0]['regions'][region] for region in REGION_ORDER}
    contexts = {region: _region_context(first_entries[region], region) for region in REGION_ORDER}
    supports = {region: list(contexts[region]['global_point_ids']) for region in REGION_ORDER}
    region_blocks = {}
    local_phases = {}
    local_rot_angles = {}

    for region_index, region in enumerate(REGION_ORDER):
        context = contexts[region]
        thresholds = list(context['thresholds'])
        segmentation = list(context['segmentation'])
        levels = len(thresholds) + 1
        weights = {}
        slopes = {}
        errors = {}
        statuses = {}
        local_phases[region] = {}
        local_rot_angles[region] = {}
        for report in reports:
            direct = report['direct']
            entry = report['regions'][region]
            if entry.get('status') != 'complete':
                raise ValueError(f'{direct}/{region} 未完成')
            other_context = _region_context(entry, region)
            if not _same_context(context, other_context):
                raise ValueError(f'{region} 跨方向 context 不一致')
            params = entry.get('parameters', {})
            direct_key = str(direct)
            weight = np.asarray(params.get('weights', {}).get(direct_key), dtype=float)
            slope = np.asarray(params.get('slopes', {}).get(direct_key), dtype=float)
            if weight.shape != (levels,) or not np.all(np.isfinite(weight)):
                raise ValueError(f'{direct}/{region} weights 形状应为 {(levels,)}，收到 {weight.shape}')
            expected_slope = (levels, len(segmentation))
            if slope.shape != expected_slope or not np.all(np.isfinite(slope)):
                raise ValueError(f'{direct}/{region} slopes 形状应为 {expected_slope}，收到 {slope.shape}')
            weights[direct_key] = weight.tolist()
            slopes[direct_key] = slope.tolist()
            errors[direct_key] = deepcopy(entry.get('errors', {}))
            statuses[direct_key] = deepcopy(entry.get('optimization', {}))
            local_phases[region][direct_key] = params.get('global_phases_set')
            local_rot_angles[region][direct_key] = params.get('rot_ang')

        region_blocks[region] = {
            'length_m': context.get('length_m'),
            'folder_path': context.get('folder_path'),
            'radii': deepcopy(context.get('radii')),
            'thresholds': thresholds,
            'sharpness': defaults['sharpness'],
            'score_window': defaults['min_consecutive'],
            'score_boundary_mode': 'constant',
            'local_segmentation': segmentation,
            'global_segment_ids': list(context['global_segment_ids']),
            'support_point_ids': supports[region],
            'owned_point_ids': _owned_points(supports, region_index),
            'global_sample_dist_m': list(context.get('global_sample_dist_m', [])),
            'weights': weights,
            'slopes': slopes,
            'errors': errors,
            'optimization': statuses,
        }

    failures = []
    for report in reports:
        failures.extend(_stage_failures(report))
    failures.extend([
        'global phase reference is unverified',
        'score thresholds are unverified',
        'full geometry reconstruction is unverified',
    ])
    full_phase_values = list(local_phases['full'].values())
    full_rot_values = list(local_rot_angles['full'].values())
    global_phase = full_phase_values[0] if len(set(full_phase_values)) == 1 else None
    rot_ang = full_rot_values[0] if len(set(full_rot_values)) == 1 else None
    if global_phase is None:
        failures.append('full global phase differs by direction')
    if rot_ang is None:
        failures.append('full rot_ang differs by direction')
    quality_gates = {
        'optimization_complete': not any(
            ' diverged' in item or ' missing' in item or ' max_iter' in item
            for item in failures
        ),
        'phase_reference_verified': False,
        'score_verified': False,
        'full_geometry_verified': False,
        'failures': failures,
    }

    shared = {
        group: {name: _native(defaults[name]) for name in names}
        for group, names in PARSER_GROUPS.items()
    }

    package = {
        'sensor_id': sensor_id,
        'format_version': FORMAT_VERSION,
        'shared_parameters': shared,
        'multi_region': {
            'region_order': list(REGION_ORDER),
            'directions': list(REQUIRED_DIRECTIONS),
            'global_phases_set': _native(global_phase),
            'rot_ang': _native(rot_ang),
            'deployment_ready': all(value is True for key, value in quality_gates.items()
                                    if key != 'failures'),
            'quality_gates': quality_gates,
        },
        'regions': region_blocks,
        'provenance': {
            'source_commit': source_commit,
            'generated_at': generated_at,
            'parser_destination_order': list(defaults),
            'source_results': [report.get('_source_path') for report in reports],
            'local_global_phases_set': local_phases,
            'local_rot_ang': local_rot_angles,
        },
    }
    return _native(package)


def package_parameter_destinations(package):
    names = []
    for values in package['shared_parameters'].values():
        names.extend(values)
    names.extend(SPECIAL_DESTINATIONS)
    return names


def _direction_keyed(values):
    if not isinstance(values, dict):
        return deepcopy(values)
    converted = {}
    for key, value in values.items():
        try:
            direct = int(key)
        except (TypeError, ValueError):
            direct = key
        converted[direct] = deepcopy(value)
    return converted


def package_parser_defaults(package):
    """Flatten one validated package into argparse defaults.

    Region blocks are converted back to the existing ``calibration_regions``
    schema. JSON direction keys become literal integer angles so the unchanged
    optimizer and reconstructor can index them with 0/120/240.
    """
    defaults = {}
    for group in package['shared_parameters'].values():
        defaults.update(deepcopy(group))
    defaults['sensor_id'] = package['sensor_id']
    defaults['global_phases_set'] = package['multi_region']['global_phases_set']
    defaults['rot_ang'] = package['multi_region']['rot_ang']
    defaults['rot_dir'] = _direction_keyed(defaults.get('rot_dir'))
    defaults['sensing_ranges'] = _direction_keyed(defaults.get('sensing_ranges'))

    profiles = {}
    for region, block in package['regions'].items():
        profiles[region] = {
            'length_m': deepcopy(block.get('length_m')),
            'folder_path': deepcopy(block.get('folder_path')),
            'radii': deepcopy(block.get('radii')),
            'thresholds': deepcopy(block['thresholds']),
            'segmentation': deepcopy(block['local_segmentation']),
            'sharpness': deepcopy(block.get('sharpness')),
            'score_window': deepcopy(block.get('score_window')),
            'score_boundary_mode': deepcopy(block.get('score_boundary_mode', 'constant')),
            'weights': _direction_keyed(block['weights']),
            'slopes': _direction_keyed(block['slopes']),
            'global_point_ids': deepcopy(block.get('support_point_ids')),
            'global_segment_ids': deepcopy(block.get('global_segment_ids')),
            'owned_point_ids': deepcopy(block.get('owned_point_ids')),
        }
    defaults['calibration_regions'] = profiles
    return defaults


def render_calibration_comparison(reports):
    """Render a human-readable comparison without changing source reports."""
    reports = sorted(reports, key=lambda item: item['direct'])
    lines = [
        '# Multi-region calibration result comparison', '',
        '| direct | region | weighted loss (mm) | point error (mm) | end error (mm) | slopes | weights |',
        '|---:|---|---:|---:|---:|---|---|',
    ]
    for report in reports:
        direct = report['direct']
        for region in REGION_ORDER:
            entry = report['regions'][region]
            errors = entry['errors']
            stages = entry.get('optimization', {}).get('stages', {})
            slope_status = stages.get('slopes', {}).get('status', 'missing')
            weight_status = stages.get('weights', {}).get('status', 'missing')
            slope_status = 'diverged' if slope_status == 'diverge' else slope_status
            weight_status = 'diverged' if weight_status == 'diverge' else weight_status
            lines.append(
                f'| {direct} | {region} | {1000 * errors["normal_weighted_loss"]:.3f} | '
                f'{1000 * errors["avg_point_err"]["calib"]:.3f} | '
                f'{1000 * errors["avg_end_err"]["calib"]:.3f} | '
                f'{slope_status} | {weight_status} |'
            )
    lines.extend(['', '## Parameters and per-radius errors', ''])
    for report in reports:
        direct = report['direct']
        lines.extend([f'### Direction {direct}', ''])
        for region in REGION_ORDER:
            entry = report['regions'][region]
            params = entry['parameters']
            key = str(direct)
            point = entry['errors'].get('point_errors_by_sample', {}).get('circ', {})
            end = entry['errors'].get('end_errors_by_sample', {}).get('circ', {})
            radius_lines = []
            for radius, point_error in point.items():
                end_error = end.get(str(radius), end.get(radius))
                if end_error is not None:
                    radius_lines.append(
                        f'- radius {radius}: point={1000 * point_error:.3f} mm, '
                        f'end={1000 * end_error:.3f} mm'
                    )
            lines.extend([
                f'#### {region}', '',
                f'- thresholds: `{json.dumps(entry["context"]["thresholds"])}`',
                f'- local global phase: `{params.get("global_phases_set")}`',
                f'- local rot_ang: `{params.get("rot_ang")}`',
                f'- weights: `{json.dumps(params["weights"][key])}`',
                f'- slopes: `{json.dumps(params["slopes"][key])}`',
                f'- point errors: `{json.dumps(point, ensure_ascii=False)}`',
                f'- end errors: `{json.dumps(end, ensure_ascii=False)}`',
                *radius_lines,
                '',
            ])
    return '\n'.join(lines) + '\n'


def write_sensor_package(package, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(_native(package), ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def load_sensor_package(path, expected_sensor_id=None, *, require_deployable=False):
    package = json.loads(Path(path).read_text(encoding='utf-8'))
    if package.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'不支持的参数包 format_version={package.get("format_version")}')
    if expected_sensor_id is not None and package.get('sensor_id') != expected_sensor_id:
        raise ValueError(
            f'sensor_id 不一致: expected={expected_sensor_id}, actual={package.get("sensor_id")}'
        )
    if require_deployable and not package.get('multi_region', {}).get('deployment_ready'):
        failures = package.get('multi_region', {}).get('quality_gates', {}).get('failures', [])
        raise ValueError(f'parameter package is not deployment ready: {failures}')
    return package
