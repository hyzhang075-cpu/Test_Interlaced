"""Pure coverage diagnostics for independent end5cm weight groups."""

from collections import defaultdict

import numpy as np


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def compute_groupwise_weight_coverage(
    samples, *, thresholds, weights, weight_bounds, group_ids=(0, 0, 1, 2),
    mass_threshold=0.05, magnitude_epsilon=1e-12,
):
    """Summarize theory-vs-measurement coverage for shared group scores.

    ``samples`` is the score report's sample list.  A sample contains the
    measured pointwise ``score_magnitude``, ``global_point_ids``, its
    theoretical curvature, and groupwise ``weight_masses``.  The function is
    deliberately independent of ``Optimiser`` so reports can be tested or
    regenerated without loading calibration data.
    """
    groups = np.asarray(group_ids, dtype=int)
    weights = np.asarray(weights, dtype=float).reshape(-1, len(thresholds) + 1)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError('group_ids must be a non-empty one-dimensional sequence')
    if weights.shape[0] != int(groups.max()) + 1:
        raise ValueError('weights group count does not match group_ids')
    lower, upper = (float(value) for value in weight_bounds)
    group_values = {}
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        point_ids = []
        rows = []
        for sample in samples:
            magnitudes = np.asarray(sample['score_magnitude'], dtype=float)
            ids = np.asarray(sample.get('global_point_ids', np.arange(groups.size)), dtype=int)
            if magnitudes.shape != groups.shape or ids.shape != groups.shape:
                raise ValueError('sample point arrays must match group_ids')
            selected = magnitudes[indices]
            base = float(np.mean(selected))
            theory = sample.get('theoretical_curvature_m_inv')
            theory = None if theory is None else float(theory)
            gain = None
            if (theory is not None and np.isfinite(theory)
                    and abs(theory) > magnitude_epsilon
                    and abs(base) > magnitude_epsilon):
                gain = theory / base
            masses = np.asarray(sample.get('weight_masses'), dtype=float)
            if masses.shape[1] != groups.size:
                raise ValueError('sample weight_masses point count must match group_ids')
            group_mass = masses[:, indices].mean(axis=1)
            rows.append({
                'keys': list(sample.get('keys', [])),
                'radius_mm': sample.get('keys', [None])[0] if sample.get('keys') else None,
                'point_ids': ids[indices].tolist(),
                'base_magnitude': base,
                'theoretical_curvature_m_inv': _finite_or_none(theory) if theory is not None else None,
                'required_gain': _finite_or_none(gain) if gain is not None else None,
                'level_mass': group_mass.tolist(),
            })
            if not point_ids:
                point_ids = ids[indices].tolist()
        level_rows = []
        for level, weight in enumerate(weights[group]):
            masses = np.asarray([row['level_mass'][level] for row in rows], dtype=float)
            dominant = np.asarray([
                int(np.argmax(row['level_mass'])) == level for row in rows
            ], dtype=bool)
            level_rows.append({
                'level': level,
                'weight': float(weight),
                'dominant_level_count': int(dominant.sum()),
                'mass_threshold_count': int((masses > mass_threshold).sum()),
                'effective_sample_count': float(masses.sum()),
                'max_mass': float(masses.max()) if masses.size else 0.0,
            })
        over = []
        for row in rows:
            gain = row['required_gain']
            if gain is not None and (gain < lower or gain > upper):
                over.append({
                    'keys': row['keys'], 'required_gain': gain,
                    'side': 'below_lower' if gain < lower else 'above_upper',
                })
        group_values[str(group)] = {
            'point_ids': point_ids,
            'samples': rows,
            'levels': level_rows,
            'uncovered_levels': [
                row['level'] for row in level_rows
                if row['dominant_level_count'] == 0 and row['mass_threshold_count'] == 0
            ],
            'theory_gain_over_bounds': over,
        }
    return {
        'definition': {
            'group_ids': groups.tolist(),
            'mass_threshold': float(mass_threshold),
            'coverage_rule': 'level is covered when dominant_level_count > 0 or mass_threshold_count > 0',
            'required_gain': 'theoretical_curvature_m_inv / group base magnitude',
        },
        'groups': group_values,
        'opt_bound_hits': {
            'lower': int(np.isclose(weights, lower, rtol=0, atol=1e-6).sum()),
            'upper': int(np.isclose(weights, upper, rtol=0, atol=1e-6).sum()),
            'total': int(weights.size),
        },
    }
