"""Fixed local phase references and full level-zero bridge interval ownership."""
import numpy as np


def _mixed_slopes(sensor, frame, direct):
    """Estimate state-specific upstream slopes using the existing production score model."""
    _, magnitude, _, _ = sensor._get_base(keys=None, calib_Data=frame,
                                        base_slopes=sensor.slopes[direct][0], verbose=False)
    masses = sensor._alloc_mass(sensor._get_scores(magnitude=np.asarray(magnitude)))
    labels = np.empty(len(sensor.samp_dist), dtype=int)
    for index, (start, end) in enumerate(sensor.base_segments):
        labels[start:end] = index
    return np.sum(masses * np.asarray(sensor.slopes[direct])[:, labels], axis=0)


def build_fixed_phase_reference(full, upstream, context, direct):
    """Average nonzero circular states to the preceding point, then use full level zero.

    The returned first-real-point phase is fixed for the entire local optimization.
    Crop-origin phase is recorded separately because local integration starts at the
    first real point, not at the physical crop origin. All distances are in m.
    """
    ids = list(full.global_point_ids)
    first = ids.index(context['global_point_ids'][0])
    if first == 0:
        raise ValueError('local phase reference requires a preceding full point')
    left_id = ids[first-1]
    left = list(upstream.global_point_ids).index(left_id)
    labels = np.repeat(np.arange(len(full.base_segments)),
                       [end-start for start,end in full.base_segments])
    bridge = float(full.slopes[direct][0, labels[first]])
    samples = []
    for radius, frame in upstream.processed_Data['circ'][direct].items():
        if not isinstance(radius, (int,float)) or radius <= 0:
            continue
        slopes = _mixed_slopes(upstream, frame, direct)
        phase = float(upstream.global_phases_set + np.sum(
            slopes[1:left+1] * np.diff(upstream.samp_dist[:left+1])))
        if not np.isfinite(phase):
            raise ValueError('nonfinite upstream phase reference')
        samples.append((radius, phase))
    if not samples:
        raise ValueError('phase reference requires nonzero circular calibration states')
    phases = np.asarray([phase for _,phase in samples])
    mean_vector = np.mean(np.exp(1j*phases))
    if abs(mean_vector) < 1e-8:
        raise ValueError('upstream circular phase reference is ambiguous')
    mean = float(np.angle(mean_vector))
    left_position = float(full.samp_dist[first-1])
    crop_phase = mean + bridge*(float(context['crop_origin_m'])-left_position)
    first_phase = mean + bridge*(float(full.samp_dist[first])-left_position)
    return {'mode':'full_level0_fixed_bridge', 'direct':int(direct),
            'upstream_region':'full' if upstream is full else 'end15cm',
            'source_radii_mm':[radius for radius,_ in samples],
            'upstream_sample_phases_rad':[phase for _,phase in samples],
            'upstream_circular_resultant':float(abs(mean_vector)),
            'upstream_mean_phase_rad':mean, 'previous_point_id':int(left_id),
            'first_point_id':int(ids[first]), 'bridge_slope_rad_per_m':bridge,
            'phase_at_crop_origin_rad':float(crop_phase),
            'phase_at_first_point_rad':float(first_phase)}


def apply_full_bridge_slopes(slope_map, config, point_ids, direct):
    """Override only intervals ending at regional first points with full level-zero slopes."""
    full = config['regions']['full']
    labels = np.repeat(np.arange(len(full['local_segmentation'])), full['local_segmentation'])
    indices = {int(point):index for index,point in enumerate(point_ids)}
    for name in config['region_order']:
        if name == 'full':
            continue
        point = config['regions'][name]['support_point_ids'][0]
        local = full['support_point_ids'].index(point)
        slope_map[indices[point]] = full['slopes'][direct][0,labels[local]]
