"""Assemble the four-core live field on the session's real WL grid."""
from dataclasses import replace

import numpy as np

from ..configuration.regions import validate_end_weight_config, validate_weight_group_ids
from ..contracts import CurvatureResult, PreparedFrame
from .service import CurvatureService

CORE_REGION_ORDER = ('full', 'end15cm', 'end5cm')


class CoreDemoCurvature:
    """Grade each region on its calibration support and place only its owned points."""

    def __init__(self, service, design, parameters, regions, direct):
        self.design = design
        self.order = tuple(regions)
        base_region = self.order[0]
        full_ids = design.source_indices + 1
        index_by_wl = {int(wl): index for index, wl in enumerate(full_ids)}
        blocks = parameters.parameters_by_region
        base_metadata = blocks.get(base_region, {}).get('core_position_metadata', {})
        hardware = base_metadata.get('hardware_geometry', {})
        if hardware.get('sensor_id', design.sensor_id) != design.sensor_id:
            raise ValueError(f'{base_region} 芯位 SensorID 与实时配置不一致')
        if ('samp_dist_preset' in hardware
                and tuple(hardware['samp_dist_preset']) != design.samp_dist_preset_mm):
            raise ValueError(f'{base_region} 芯位原始 WL 预设与实时配置不一致')
        self._regions = []
        owners = np.full(len(full_ids), -1, dtype=int)
        for ordinal, region in enumerate(self.order):
            block = blocks.get(region, {})
            metadata = block.get('core_position_metadata', {})
            if metadata.get('schema_version') != 6 or not block.get('core_position'):
                raise ValueError(f'{region} 缺少 schema 6 芯位参数')
            try:
                for key in ('hardware_geometry', 'channel_configuration', 'coordinate_frame',
                            'physical_constraints', 'acquisition_channels'):
                    if metadata.get(key) != base_metadata.get(key):
                        raise ValueError(f'{key} 与 {base_region} 不一致')
                calibration = metadata['calibration']
                for key in ('strain_sensitivity_pm_per_ue', 'strain_projection_sign',
                            'core2_azimuth_deg', 'direction_sign', 'bend_side_offset_deg'):
                    if calibration.get(key) != base_metadata['calibration'].get(key):
                        raise ValueError(f'{key} 与 {base_region} 不一致')
                wl_ids = calibration['wl_ids']
                indices = np.array([index_by_wl[wl] for wl in wl_ids], dtype=int)
                if len(indices) < 2 or np.any(np.diff(indices) <= 0):
                    raise ValueError('WL 必须按全长网格顺序排列且至少有两个真实点')
                if ordinal == 0 and not np.array_equal(wl_ids, full_ids):
                    raise ValueError('芯位必须覆盖实时采样网格')
                scope = metadata['scope']
                if scope.get('name', region) != region:
                    raise ValueError('芯位标定区域与参数块名称不一致')
                global_distance = np.asarray(scope['global_sample_dist_m'], dtype=float)
                distance = np.asarray(scope['local_sample_dist_m'], dtype=float)
                origin = scope['crop_origin_m']
                if region == 'full' and not np.isclose(origin, 0., rtol=0, atol=1e-12):
                    raise ValueError('full 芯位的全长坐标原点必须为 0')
                if (global_distance.shape != indices.shape or distance.shape != indices.shape
                        or not np.allclose(distance, global_distance - origin, rtol=0, atol=1e-12)
                        or not np.allclose(global_distance,
                                           design.sampling_distance_m[indices] + base_metadata['scope']['crop_origin_m'],
                                           rtol=0, atol=1e-12)):
                    raise ValueError('局部或全局 WL 距离与实时采样网格不一致')

                layout = dict(block['weight_layout'])
                layout['end5cm_independent_weight_score'] = layout.pop('independent_weight_score')
                layout['weight_group_ids'] = validate_weight_group_ids(layout['weight_group_ids'], len(indices))
                validate_end_weight_config(layout['end_weights'], layout['end_start'], len(indices))
                local = replace(design, source_indices=design.source_indices[indices],
                                sampling_distance_m=distance)
                regional = CurvatureService(replace(service.config, **layout))
                probe = PreparedFrame.from_design(local, np.zeros((len(indices), len(design.active_channels))),
                                                  design.active_channels)
                # Reuse module 3 validation for geometry rank, physical channels and the selected weights.
                regional.compute(probe, local, parameters, region=region, direct=direct)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{region} 实时四芯参数无效（方向 {direct}）: {exc}') from exc
            self._regions.append((region, indices, local, regional))
            owners[indices] = ordinal
        self._owners = owners
        self.source_regions = tuple(self.order[index] for index in owners)

    def compute(self, frame, parameters, *, direct):
        magnitude = np.empty_like(frame.distance_m)
        angle = np.empty_like(magnitude)
        diagnostics = {}
        for ordinal, (region, indices, local, service) in enumerate(self._regions):
            mask = self._owners[indices] == ordinal
            if not mask.any():
                continue
            regional = PreparedFrame.from_design(local, frame.signals[indices], frame.channel_ids)
            result = service.compute(regional, local, parameters, region=region, direct=direct)
            owned = indices[mask]
            magnitude[owned] = result.curvature_per_m[mask]
            angle[owned] = result.direction_rad[mask]
            for key, values in result.diagnostics.items():
                diagnostics.setdefault(key, np.empty_like(magnitude))[owned] = values[mask]
        diagnostics['source_region_by_wl'] = self.source_regions
        return CurvatureResult(frame.distance_m.copy(), magnitude, angle,
                               frame.source_indices.copy(), diagnostics)
