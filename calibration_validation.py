"""Offline full-geometry validation for a merged multi-region package."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np


REGION_ORDER = ('full', 'end15cm', 'end5cm')


def circular_angle_error_deg(actual, expected):
    """Smallest absolute angular difference between two radian phases."""
    delta = float(actual) - float(expected)
    return float(abs(np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))))


def integrate_phase_to_point(global_phase, distances, slope_map, point_index):
    """Reproduce Initialiser's discrete twist integration through one point."""
    distances = np.asarray(distances, dtype=float)
    slope_map = np.asarray(slope_map, dtype=float)
    point_index = int(point_index)
    if distances.ndim != 1 or slope_map.shape != distances.shape:
        raise ValueError('distances 和 slope_map 必须是相同长度的一维数组')
    if point_index < 0 or point_index >= len(distances):
        raise ValueError(f'point_index 越界: {point_index}')
    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(slope_map)):
        raise ValueError('phase transport 输入必须为有限数值')
    increments = slope_map[1:point_index + 1] * np.diff(distances[:point_index + 1])
    return float(global_phase + np.sum(increments))


def build_phase_transport_evidence(global_phase, distances, slope_map,
                                   local_phases, start_indices, *, tolerance_deg=30.0):
    """Compare transported full phase with regional local phase references."""
    result = {}
    for region in ('end15cm', 'end5cm'):
        predicted = integrate_phase_to_point(
            global_phase, distances, slope_map, start_indices[region],
        )
        local = float(local_phases[region])
        error = circular_angle_error_deg(predicted, local)
        result[region] = {
            'predicted_phase_rad': predicted,
            'local_phase_rad': local,
            'phase_error_deg': error,
            'tolerance_deg': float(tolerance_deg),
            'within_30_deg': bool(error <= tolerance_deg),
            'blocking': region == 'end15cm',
        }
    return result


def validate_score_evidence(package, score_report, source_path):
    """Validate one shared score report against regional package thresholds."""
    failures = []
    if score_report.get('sensor_id') != package.get('sensor_id'):
        failures.append(
            f'sensor_id 不一致: {score_report.get("sensor_id")} != {package.get("sensor_id")}'
        )
    if score_report.get('task') != 'score':
        failures.append(f'task 必须为 score，收到 {score_report.get("task")}')
    report_regions = score_report.get('regions', {})
    summaries = {}
    for region in REGION_ORDER:
        entry = report_regions.get(region)
        if not isinstance(entry, dict):
            failures.append(f'score 缺少区域 {region}')
            continue
        if entry.get('status') != 'complete':
            failures.append(f'{region} score 未完成')
        distribution = entry.get('score_distribution', {})
        expected = [float(v) for v in package.get('regions', {}).get(region, {}).get('thresholds', [])]
        actual = [float(v) for v in distribution.get('thresholds', [])]
        if actual != expected:
            failures.append(f'{region} thresholds 不一致: {actual} != {expected}')
        width = distribution.get('sigmoid_10_90_width_m_inv')
        samples = distribution.get('samples', [])
        values = []
        sample_summaries = []
        for sample in samples:
            magnitude = np.asarray(sample.get('score_magnitude', []), dtype=float)
            if magnitude.size:
                values.extend(magnitude.tolist())
            sample_summaries.append({
                'keys': deepcopy(sample.get('keys')),
                'theoretical_curvature_m_inv': sample.get('theoretical_curvature_m_inv'),
                'quantiles': deepcopy(sample.get('quantiles', {})),
            })
        numeric = np.asarray(values, dtype=float)
        if width is None or not np.isfinite(float(width)) or numeric.size == 0 or not np.all(np.isfinite(numeric)):
            failures.append(f'{region} score 包含非有限数值或空分布')
            observed_min = observed_max = None
        else:
            observed_min = float(np.min(numeric))
            observed_max = float(np.max(numeric))
            for threshold in expected:
                if not observed_min <= threshold <= observed_max:
                    failures.append(
                        f'{region} threshold {threshold} m^-1 未被实测分布跨越 '
                        f'[{observed_min}, {observed_max}]'
                    )
        summaries[region] = {
            'thresholds': actual,
            'sigmoid_10_90_width_m_inv': width,
            'observed_min_m_inv': observed_min,
            'observed_max_m_inv': observed_max,
            'samples': sample_summaries,
        }
    return {
        'score_thresholds_verified': not failures,
        'source_path': str(source_path),
        'source_direct': score_report.get('direct'),
        'regions': summaries,
        'failures': failures,
    }


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('tangent vector must be finite and nonzero')
    return vector / norm


def compute_curve_metrics(result, ideal, *, result_tangent, ideal_tangent,
                          distances=None, evaluation_start_m=None):
    result = np.asarray(result, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    if result.shape != ideal.shape or result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f'curve shapes must match as (points, 3): {result.shape}, {ideal.shape}')
    errors = np.linalg.norm(result - ideal, axis=1)
    result_tangent = _unit(result_tangent)
    ideal_tangent = _unit(ideal_tangent)
    tangent_rad = float(np.arccos(np.clip(np.dot(result_tangent, ideal_tangent), -1.0, 1.0)))
    tip_mm = float(errors[-1] * 1000)
    length_m = max(float(np.sum(np.linalg.norm(np.diff(ideal, axis=0), axis=1))), 1e-12)
    metrics = {
        'mean_point_error_mm': float(np.mean(errors) * 1000),
        'max_point_error_mm': float(np.max(errors) * 1000),
        'tip_position_error_mm': tip_mm,
        'tip_tangent_error_deg': float(np.degrees(tangent_rad)),
        'pose_error_mm': float(np.hypot(tip_mm, 1000 * length_m * tangent_rad)),
    }
    if distances is not None and evaluation_start_m is not None:
        mask = np.asarray(distances, dtype=float) >= float(evaluation_start_m) - 1e-12
        metrics['evaluated_region_point_error_mm'] = float(np.mean(errors[mask]) * 1000)
    return metrics


def straight_prefix_circular_tail(distances, bend_start_m, radius_m):
    distances = np.asarray(distances, dtype=float)
    if radius_m <= 0:
        curve = np.column_stack((distances, np.zeros((len(distances), 2))))
        return curve, np.array([1.0, 0.0, 0.0])
    curve = np.zeros((len(distances), 3), dtype=float)
    before = distances <= bend_start_m
    curve[before, 0] = distances[before]
    arc = np.maximum(0.0, distances[~before] - bend_start_m)
    curve[~before, 0] = bend_start_m + radius_m * np.sin(arc / radius_m)
    curve[~before, 1] = radius_m * (1.0 - np.cos(arc / radius_m))
    tip_arc = max(0.0, float(distances[-1] - bend_start_m))
    tangent = np.array([np.cos(tip_arc / radius_m), np.sin(tip_arc / radius_m), 0.0])
    return curve, tangent


def boundary_diagnostics(diagnostics, left_point_ids=(15, 18)):
    weights = np.asarray(diagnostics['final_weight'], dtype=float)
    slopes = np.asarray(diagnostics['slope_map'], dtype=float)
    base = np.asarray(diagnostics['base_magnitude'], dtype=float)
    magnitude = np.asarray(diagnostics['final_magnitude'], dtype=float)
    angle = np.asarray(diagnostics['final_angle'], dtype=float)
    theta = np.asarray(diagnostics['final_theta_dynamic'], dtype=float)
    vectors = magnitude[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    result = {}
    for left_id in left_point_ids:
        left = int(left_id) - 1
        right = left + 1
        if right >= len(weights):
            raise ValueError(f'boundary {left_id}->{left_id + 1} is outside point arrays')
        result[f'{left_id}->{left_id + 1}'] = {
            'weight_jump': float(abs(weights[right] - weights[left])),
            'slope_jump': float(abs(slopes[right] - slopes[left])),
            'base_curvature_jump_m_inv': float(abs(base[right] - base[left])),
            'phase_jump_rad': float(np.max(np.abs(theta[:, right] - theta[:, left]))),
            'curvature_vector_jump_m_inv': float(np.linalg.norm(vectors[right] - vectors[left])),
        }
    return result


def validation_failure_summary(report):
    full_errors = []
    tail_tangent_errors = []
    phase_jumps = []
    for regions in report['directions'].values():
        for region, radii in regions.items():
            for values in radii.values():
                if region == 'full':
                    full_errors.append(values['mean_point_error_mm'])
                else:
                    tail_tangent_errors.append(values['tip_tangent_error_deg'])
                phase_jumps.extend(
                    boundary['phase_jump_rad'] for boundary in values['boundaries'].values()
                )
    return (
        f'max full mean point error={max(full_errors):.3f} mm; '
        f'max local-test tip tangent error={max(tail_tangent_errors):.3f} deg; '
        f'max region-boundary phase jump={max(phase_jumps):.3f} rad; '
        'source optimization gates include divergence and independent pose ground truth is unavailable'
    )


def _ideal_full_circular(distances, radius_m):
    return straight_prefix_circular_tail(distances, 0.0, radius_m)


def _reconstruct_frame(full, frame, direct):
    result = full._reconstruct_frame(
        keys=None, calib_shape='circ', calib_Data=frame,
        weight=full.weights[direct], slope=full.slopes[direct],
        magnitude_ratio=1.0, init_T=[1.0, 0.0, 0.0], direct=direct,
        verbose=False,
    )
    _, _, _, interp_dist, curve, tangent, *_ = result
    rot_direction = full.rot_dir[full.calib_shapes.index('circ')]
    rotation = full._get_rot_x(direct=direct, rot_dir=rot_direction)
    return np.asarray(interp_dist), curve @ rotation.T, tangent @ rotation.T


def validate_multiregion_package(package_path):
    """Replay full and local experiments through one 22-point reconstruction."""
    import main
    from calibration_batch import optimizer_parameters
    from optimiser import Optimiser

    package_path = str(Path(package_path).resolve())
    package = json.loads(Path(package_path).read_text(encoding='utf-8'))

    def make(region, merged=False):
        argv = ['--task', 'recon', '--sensor_params', package_path,
                '--optimization_region', region,
                '--multi_region_calib', 'True' if merged else 'False']
        args = main.parse_main_args(argv)
        # Mirror main(): stored 22-point dynamic zero references are invalid for
        # cropped 7/4-point offline calibration readers.
        args.use_stored_zero = False
        context = main.configure_calibration_args(args)
        optim = Optimiser(**optimizer_parameters(args))
        optim.calibration_context = context
        return optim

    full = make('full', merged=True)
    local = {'end15cm': make('end15cm'), 'end5cm': make('end5cm')}
    report = {
        'basis': 'full 22-point reconstruction with region-owned point parameters',
        'global_phases_set': full.global_phases_set,
        'rot_ang': full.rot_ang,
        'directions': {},
    }
    plan = {
        'full': [2000, 1500, 1000, 800, 600, 400, 300, 200],
        'end15cm': [100, 75, 50, 30],
        'end5cm': [30, 20, 15],
    }
    point_index = {int(point): index for index, point in enumerate(full.global_point_ids)}
    for direct in (0, 120, 240):
        direction = {}
        straight = np.asarray(full.processed_Data['circ'][direct][0]).copy()
        for region, radii in plan.items():
            rows = {}
            for radius in radii:
                if region == 'full':
                    frame = np.asarray(full.processed_Data['circ'][direct][radius])
                    bend_start = 0.0
                else:
                    source = local[region]
                    frame = straight.copy()
                    local_frame = np.asarray(source.processed_Data['circ'][direct][radius])
                    ids = source.global_point_ids.tolist()
                    indices = [point_index[int(point)] for point in ids]
                    frame[indices] = local_frame
                    bend_start = full.ideal_len - (0.15 if region == 'end15cm' else 0.05)
                interp_dist, curve, tangent = _reconstruct_frame(full, frame, direct)
                ideal, ideal_tangent = straight_prefix_circular_tail(
                    interp_dist - interp_dist[0], bend_start,
                    0.0 if radius == 0 else radius / 1000.0,
                )
                metrics = compute_curve_metrics(
                    curve, ideal, result_tangent=tangent[-1], ideal_tangent=ideal_tangent,
                    distances=interp_dist - interp_dist[0], evaluation_start_m=bend_start,
                )
                if region != 'full':
                    source_errors = package['regions'][region]['errors'][str(direct)]
                    isolated = source_errors['point_errors_by_sample']['circ'][str(radius)] * 1000
                    metrics['isolated_calibration_point_error_mm'] = float(isolated)
                    metrics['merged_minus_isolated_point_error_mm'] = float(
                        metrics['evaluated_region_point_error_mm'] - isolated
                    )
                metrics['boundaries'] = boundary_diagnostics(full.last_parameter_assembly)
                rows[str(radius)] = metrics
            direction[region] = rows
        report['directions'][str(direct)] = direction
    report['quality_gates'] = {
        'phase_reference_verified': False,
        'rot_ang_verified': False,
        'full_geometry_verified': False,
        'reason': validation_failure_summary(report),
    }
    return report


def write_validation_report(report, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Multi-region full-geometry validation', '',
        f'- unified global phase: `{report["global_phases_set"]}`',
        f'- unified rot_ang: `{report["rot_ang"]}`',
        f'- full geometry verified: `{report["quality_gates"]["full_geometry_verified"]}`', '',
        '| direct | region | radius (mm) | region point (mm) | isolated point (mm) | delta (mm) | tip position (mm) | tip tangent (deg) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for direct, regions in report['directions'].items():
        for region, radii in regions.items():
            for radius, values in radii.items():
                lines.append(
                    f'| {direct} | {region} | {radius} | '
                    f'{values["evaluated_region_point_error_mm"]:.3f} | '
                    f'{values.get("isolated_calibration_point_error_mm", float("nan")):.3f} | '
                    f'{values.get("merged_minus_isolated_point_error_mm", float("nan")):.3f} | '
                    f'{values["tip_position_error_mm"]:.3f} | '
                    f'{values["tip_tangent_error_deg"]:.3f} |'
                )
    lines.extend(['', '## Gate result', '', report['quality_gates']['reason'], '',
                  'A negative delta means the merged full-length reconstruction reduced the '
                  'point error relative to the isolated regional reconstruction. It does not '
                  'override the tip-orientation and boundary-continuity failures.', ''])
    markdown_path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    import argparse

    cli = argparse.ArgumentParser(description='Replay a multi-region package on full geometry')
    cli.add_argument('--sensor-params', required=True)
    cli.add_argument('--json-output', default='docs/calibration-multiregion-validation.json')
    cli.add_argument('--markdown-output', default='docs/calibration-multiregion-validation.md')
    args = cli.parse_args(argv)
    report = validate_multiregion_package(args.sensor_params)
    write_validation_report(report, args.json_output, args.markdown_output)


if __name__ == '__main__':
    main()
