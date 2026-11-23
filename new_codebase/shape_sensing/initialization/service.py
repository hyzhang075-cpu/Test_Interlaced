"""模块2显式标定拟合入口。"""
from copy import deepcopy
import numpy as np
from ._kernel import InitializationKernel
from ..curvature.service import GeometryService
from ..contracts import InitializationResult

class InitializationService:
    def __init__(self, config):
        """只保存初始化配置，不读文件、不做拟合。"""
        self.config = config

    def fit(self, dataset, *, verbose=False):
        """输入模块1的(P,C)静态数据；输出相位rad、slopes rad/m及诊断。

        使用原分段拟合正文；数据、几何及诊断容器均不与调用方共享可写数组。
        """
        design = dataset.design
        kernel = InitializationKernel()
        kernel.geometry = GeometryService(design, len(dataset.channel_ids))
        kernel.processed_Data = deepcopy(dataset.data)
        kernel.radii = deepcopy(dataset.radii)
        kernel.exp_directs = list(dataset.radii)
        kernel.calib_shapes = list(dataset.radii[next(iter(dataset.radii))])
        kernel.samp_dist = design.sampling_distance_m.copy()
        kernel.total_points = len(kernel.samp_dist)
        kernel.zero_init = design.physical_start_m is not None
        kernel.straight_trim = design.straight_trim
        kernel.num_channels_recon = len(dataset.channel_ids)
        kernel.R = design.radius_m
        kernel.active_angles = design.active_angles_deg
        kernel.interlaced_grating = design.interlaced_grating
        kernel.interlaced_offset = design.interlaced_offset_rad
        kernel.samp_dist_preset = design.samp_dist_preset_mm
        kernel.samp_dist_preset_sub = design.samp_dist_preset_sub_mm
        kernel.discont = self.config.discont
        kernel.segmentation = list(self.config.segmentation)
        kernel.num_segments = len(kernel.segmentation)
        kernel.connect = self.config.connect
        kernel.global_phases_set = self.config.global_phases_set or 0.0
        if self.config.global_phases_set is None:
            kernel.rot_seg_fit(verbose=False, update_phase=True)
        kernel.rot_seg_fit(verbose=verbose, update_phase=False)
        return InitializationResult(kernel.global_phases_set, deepcopy(kernel.slopes_init),
            deepcopy(kernel.magnitude_results), deepcopy(kernel.angle_results_raw),
            deepcopy(kernel.angle_results_unwrap), deepcopy(kernel.global_fit_direct),
            deepcopy(kernel.individual_fits_direct))

    def fit_four_core(self, dataset, input_config, region):
        """Fit schema 6 from the standard module 1 CalibrationDataset."""
        from .four_core import DIRECTIONS, FourCoreFitData, FourCoreFitSettings, fit_four_cores

        circle = dataset.data.get('circ', {})
        by_direction = [{radius for radius in circle.get(direction, {}) if radius > 0}
                        for direction in DIRECTIONS]
        if not by_direction[0] or any(radii != by_direction[0] for radii in by_direction[1:]):
            raise ValueError('四芯标定需要 0、120、240 方向具有相同的非零圆弧半径')
        states = tuple((direction, radius) for direction in DIRECTIONS
                       for radius in sorted(by_direction[0]))
        design = dataset.design
        wl_ids = tuple((design.source_indices + 1).tolist())
        channels = tuple(dataset.channel_ids)
        response = np.asarray([circle[direction][radius] for direction, radius in states])
        if response.shape != (len(states), len(wl_ids), 4) or len(channels) != 4:
            raise ValueError('四芯标定需要每个真实 WL 的四路预处理响应')
        data = FourCoreFitData(response.transpose(0, 2, 1) * 1e6, states, channels, wl_ids)
        settings = FourCoreFitSettings(
            fiber_diameter_um=self.config.core_diameter_um,
            core_radius_um=self.config.core_radius_um,
            k_pm_per_ue=self.config.core_sensitivity_pm_per_ue,
            bend_side_offset_deg=self.config.core_bend_offset_deg,
        )
        result = fit_four_cores(data, settings)
        length = input_config.ideal_len
        trim = input_config.straight_trim
        origin = length + trim if trim < 0 else trim
        distances = design.sampling_distance_m
        result['hardware_geometry'] = {key: deepcopy(getattr(input_config, key)) for key in (
            'sensor_id', 'start_value', 'end_value', 'samp_dist_preset',
            'samp_dist_preset_sub', 'interlaced_grating', 'interlaced_offset')}
        result['scope'] = dict(name=region, length_m=length-origin, full_length_m=length,
            crop_origin_m=origin, boundary_margin_mm=0.0, global_wl_ids=list(wl_ids),
            global_sample_dist_m=(distances + origin).tolist(),
            local_sample_dist_m=distances.tolist(), source_data_dir=input_config.folder_path)
        for section in result["sections"]:
            section["source_region"] = region
        if input_config.interlaced_grating:
            a = np.asarray(input_config.samp_dist_preset)
            positions = np.sort(np.r_[a, input_config.samp_dist_preset_sub])
            result["acquisition_channels"] = list(input_config.active_channels)
            result["interlaced_response_mapping"] = dict(
                center_interpolation="CubicSpline", extrapolate=True,
                section_sources={str(wl): ('measured' if positions[wl-1] in a else
                    'extrapolated' if not a.min() <= positions[wl-1] <= a.max() else 'interpolated')
                    for wl in wl_ids},
            )
        return result
