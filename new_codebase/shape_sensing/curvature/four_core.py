"""Schema 6 inversion on the real measurement grid."""
import numpy as np

from ..contracts import CurvatureResult

REFERENCE_LENGTH_M = 100e-6


def compute_four_core(frame, design, calibration, gains, magnitude_ratio):
    if calibration.get('schema_version') != 6:
        raise ValueError('四芯曲率计算需要 schema 6 芯位参数')
    if (not np.array_equal(frame.distance_m, design.sampling_distance_m)
            or not np.array_equal(frame.source_indices, design.source_indices)):
        raise ValueError('四芯响应的距离和全局 WL 必须与采样设计一致')
    wl_ids = (frame.source_indices + 1).tolist()
    metadata = calibration['calibration']
    if wl_ids != metadata['wl_ids'] or not np.allclose(
            frame.distance_m, calibration['scope']['local_sample_dist_m'], rtol=0, atol=1e-12):
        raise ValueError('四芯芯位参数与当前区域的 WL 或采样距离不一致')
    channels = calibration['channel_configuration']['response_channels']
    if len(channels) != 4 or set(channels) != set(frame.channel_ids):
        raise ValueError('四芯芯位参数与预处理帧的物理通道不一致')
    columns = [frame.channel_ids.index(channel) for channel in channels]
    sensitivity = float(metadata['strain_sensitivity_pm_per_ue'])
    sign = metadata['strain_projection_sign']
    if not np.isfinite(sensitivity) or sensitivity <= 0 or sign not in (-1, 1):
        raise ValueError('四芯芯位参数的灵敏度或应变符号无效')
    response = frame.signals[:, columns] / (sensitivity * sign)
    positions = []
    for section, wl in zip(calibration['sections'], wl_ids):
        if section['wl_id'] != wl or section.get('response_model') != 'four_core_common':
            raise ValueError(f'WL_{wl} 的参数编号或响应模型不一致')
        rows = []
        for channel in channels:
            core = section['cores'][f'ch{channel}']
            radius = core['r_um'] * 1e-6
            angle = core['alpha_rad']
            if radius < 0 or (angle is None and radius > 1e-12):
                raise ValueError(f'WL_{wl} 的芯位半径或角度无效')
            angle = angle if angle is not None else 0.0
            rows.append([radius * np.sin(angle), radius * np.cos(angle)])
        positions.append(rows)
    if len(positions) != len(calibration['sections']) or len(positions) != len(wl_ids):
        raise ValueError('四芯芯位截面数量与采样点不一致')
    positions = np.asarray(positions)
    gains = np.ones_like(response) if gains is None else np.asarray(gains, dtype=float)
    if (gains.shape != response.shape or not np.isfinite(gains).all()
            or np.any(gains <= 0) or not np.isfinite(positions).all()
            or not np.isfinite(response).all()):
        raise ValueError('四芯解算要求有限的芯位和响应以及有限正增益')
    matrix = np.concatenate((positions / REFERENCE_LENGTH_M, gains[:, :, None]), axis=2)
    rhs = gains * response
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if np.any(singular[:, -1] <= 4 * np.finfo(float).eps * singular[:, 0]):
        raise ValueError('四芯增广矩阵秩不足')
    projected = np.einsum('pij,pi->pj', u, rhs) / singular
    solution = np.einsum('pji,pj->pi', vh, projected)
    residual = np.einsum('pij,pj->pi', matrix, solution) - rhs
    vector = solution[:, :2] / REFERENCE_LENGTH_M
    return CurvatureResult(
        frame.distance_m.copy(), np.linalg.norm(vector, axis=1) * magnitude_ratio,
        np.arctan2(vector[:, 1], vector[:, 0]), frame.source_indices.copy(),
        dict(k_y_per_m=vector[:, 0], k_z_per_m=vector[:, 1],
             common_response_b_strain=solution[:, 2],
             augmented_condition_number=singular[:, 0]/singular[:, -1],
             corrected_strain_fit_rmse=np.sqrt(np.mean(residual**2, axis=1))),
    )
