"""模块2显式标定拟合入口。"""
from copy import deepcopy
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

    @staticmethod
    def fit_four_core(dataset, *, diameter_um=350.0, core_radius_um=50.0,
                      sensitivity_pm_per_ue=1.0, bend_offset_deg=-90.0,
                      size_constraint_mode="both", envelope_center_mode="free",
                      center_order_reference="envelope_center"):
        """Fit a module 1 raw four-core dataset and return a schema 6 parameter dict."""
        from .four_core import FourCoreFitSettings, fit_four_cores

        settings = FourCoreFitSettings(
            fiber_diameter_um=diameter_um,
            core_radius_um=core_radius_um,
            k_pm_per_ue=sensitivity_pm_per_ue,
            bend_side_offset_deg=bend_offset_deg,
            size_constraint_mode=size_constraint_mode,
            envelope_center_mode=envelope_center_mode,
            center_order_reference=center_order_reference,
        )
        result = fit_four_cores(dataset, settings)
        result["hardware_geometry"] = deepcopy(dataset.hardware_geometry)
        result["scope"] = deepcopy(dataset.scope)
        for section in result["sections"]:
            section["source_region"] = dataset.scope["name"]
        if dataset.interlaced_sources:
            result["acquisition_channels"] = list(dataset.acquisition_channels)
            result["interlaced_response_mapping"] = dict(
                center_interpolation="CubicSpline", extrapolate=True,
                section_sources=dataset.interlaced_sources.copy(),
            )
        return result
