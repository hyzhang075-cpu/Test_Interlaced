"""Constrained per-WL four-core calibration with an explicit common response b."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from ..input_preprocessing.four_core import DIRECTIONS


@dataclass(frozen=True)
class FourCoreFitSettings:
    fiber_diameter_um: float = 350.0
    core_radius_um: float = 50.0
    k_pm_per_ue: float = 1.0
    bend_side_offset_deg: float = -90.0
    size_constraint_mode: str = "both"
    envelope_center_mode: str = "free"
    center_order_reference: str = "envelope_center"

    # Fixture reference-up, not an assumed measured core angle.
    core2_azimuth_deg = 90.0
    direction_sign = -1
    strain_projection_sign = 1
    sensitivity_source = "explicit_effective_pm_per_ue_assumption"

    def validate(self):
        sizes_and_sensitivity = (self.fiber_diameter_um, self.core_radius_um, self.k_pm_per_ue)
        if not all(np.isfinite(value) and value > 0 for value in sizes_and_sensitivity):
            raise ValueError("Diameter, core radius, and sensitivity must be positive and finite")
        if self.fiber_diameter_um < 6 * self.core_radius_um:
            raise ValueError(
                "A centered core plus three equal outer disks require diameter >= 6 * core radius"
            )
        if not np.isfinite(self.bend_side_offset_deg):
            raise ValueError("Bend offset must be finite")
        if self.size_constraint_mode not in ("none", "pairwise", "envelope", "both"):
            raise ValueError("Unknown size constraint mode")
        if self.envelope_center_mode not in ("centroid", "free"):
            raise ValueError("Unknown envelope center mode")
        if self.center_order_reference not in ("envelope_center", "four_core_centroid"):
            raise ValueError("Unknown center order reference")


def _four_core_layout(channels):
    channels = list(channels)
    return dict(
        centre_fibre=True,
        active_channels=channels,
        center_channel=channels[0],
        outer_channels=channels[1:],
        reference_up_channel=channels[1],
        nominal_outer_angles_deg=[0.0, 120.0, 240.0],
        physical_center_core_present=True,
        response_reference="four_core_common",
        response_channels=channels,
    )


def _design_matrix(states, settings):
    rows = []
    for direction, radius in states:
        azimuth = np.deg2rad(
            settings.core2_azimuth_deg
            + settings.direction_sign * direction
            + settings.bend_side_offset_deg
        )
        unit = np.array([np.cos(azimuth), np.sin(azimuth)])
        rows.append(settings.strain_projection_sign * unit / radius)
    return np.asarray(rows)


def enclosing_circle(points):
    """Exact minimum enclosing circle for four points, via 1/2/3 support points."""
    points = np.asarray(points, dtype=float)
    candidates = list(points)
    candidates += [(points[i] + points[j]) / 2 for i, j in combinations(range(len(points)), 2)]
    for i, j, k in combinations(range(len(points)), 3):
        a, b, c = points[[i, j, k]]
        matrix = 2 * np.array([b - a, c - a])
        if abs(np.linalg.det(matrix)) > 1e-12:
            # Solve in local coordinates to avoid cancellation on translated data.
            candidates.append(
                a + np.linalg.solve(matrix, np.array([np.sum((b - a) ** 2), np.sum((c - a) ** 2)]))
            )
    radii = [np.linalg.norm(points - center, axis=1).max() for center in candidates]
    best = int(np.argmin(radii))
    return np.array(candidates[best]), float(radii[best])


def central_enclosing_circle(points):
    """Minimum enclosing radius with core 1 nearest the SAME circle center."""
    points = np.asarray(points, dtype=float)
    center, radius = enclosing_circle(points)
    distances = np.sum((points - center) ** 2, axis=1)
    if np.all(distances[1:] >= distances[0] - 1e-10):
        return center, radius
    scale = max(float(np.linalg.norm(points, axis=1).max()), 1.0)
    q = points / scale

    def constraints(z):
        d = np.sum((q - z[:2]) ** 2, axis=1)
        return np.r_[z[2] - d, d[1:] - d[0]]

    initial = np.r_[q[0], np.sum((q - q[0]) ** 2, axis=1).max()]
    result = minimize(
        lambda z: z[2],
        initial,
        jac=lambda z: np.array([0.0, 0.0, 1.0]),
        method="SLSQP",
        constraints=dict(type="ineq", fun=constraints),
        bounds=[(None, None), (None, None), (0, None)],
        options=dict(maxiter=500, ftol=1e-12),
    )
    if not result.success or not np.isfinite(result.x).all() or constraints(result.x).min() < -1e-8:
        raise RuntimeError("Could not find a common enclosing center with central core nearest")
    return result.x[:2] * scale, float(np.linalg.norm(points - result.x[:2] * scale, axis=1).max())


def fit_four_cores(data, settings):
    settings.validate()
    layout = _four_core_layout(data.channels)
    channels = layout["active_channels"]
    matrix = _design_matrix(data.states, settings)
    response = data.responses_pm
    if not np.isfinite(response).all() or np.linalg.matrix_rank(matrix) != 2:
        raise ValueError(
            "Calibration requires finite responses and two independent bend directions"
        )
    # Coordinates in micrometres -> strain -> pm, including the CSV sign.
    design = matrix * settings.k_pm_per_ue * 1000
    # Explicit joint raw-response fit: six independent position coordinates
    # (fourth position is minus the other three) and one b per bend state/WL.
    state_count = len(matrix)
    augmented = np.zeros((state_count * 4, 6 + state_count))
    for state in range(state_count):
        for channel in range(4):
            row = 4 * state + channel
            if channel < 3:
                augmented[row, 2 * channel : 2 * channel + 2] = design[state]
            else:
                augmented[row, :6] = -np.tile(design[state], 3)
            augmented[row, 6 + state] = 1.0
    fitted, _, rank, _ = np.linalg.lstsq(
        augmented, response.reshape(state_count * 4, -1), rcond=None
    )
    if rank != 6 + state_count:
        raise ValueError("Degenerate augmented calibration design")
    first_three = fitted[:6].reshape(3, 2, len(data.wl_ids)).transpose(2, 0, 1)
    target = np.concatenate((first_three, -first_three.sum(axis=1, keepdims=True)), axis=1)
    common_pm = fitted[6:]
    upper = settings.fiber_diameter_um / 2 - settings.core_radius_um
    gap = 2 * settings.core_radius_um
    pairs = list(combinations(range(4), 2))
    size_mode = settings.size_constraint_mode
    use_pairwise = size_mode in ("pairwise", "both")
    use_envelope = size_mode in ("envelope", "both")
    envelope_order = settings.center_order_reference == "envelope_center"
    free_center = (use_envelope or envelope_order) and settings.envelope_center_mode == "free"

    def variables(points):
        circle = central_enclosing_circle if envelope_order else enclosing_circle
        return np.r_[points.ravel(), circle(points)[0]] if free_center else points.ravel()

    def inequalities(z):
        points = z[:8].reshape(4, 2)
        center = z[8:] if free_center else np.zeros(2)
        radii2 = np.sum((points - center if envelope_order else points) ** 2, axis=1)
        constraints = [radii2[1:] - radii2[0]]
        if use_envelope:
            constraints.append(1 - np.sum((points - center) ** 2, axis=1))
        if use_pairwise:
            constraints.append(
                [np.sum((points[i] - points[j]) ** 2) - (gap / upper) ** 2 for i, j in pairs]
            )
        return np.concatenate(constraints)

    sections = []
    for index, wl in enumerate(data.wl_ids):
        measured = response[:, :, index]
        initial = target[index] / upper
        initial -= initial.mean(axis=0)
        if np.linalg.norm(initial) < 1e-10 or np.linalg.matrix_rank(initial) != 2:
            raise ValueError(f"WL_{wl}: unobservable or degenerate measured core geometry")
        scaled_design = design * upper
        common = common_pm[:, index, None]
        normalization = max(float(np.mean((measured - common) ** 2)), 1.0)

        def objective(z):
            residual = scaled_design @ z[:8].reshape(4, 2).T + common - measured
            return float(np.mean(residual**2) / normalization)

        def gradient(z):
            residual = scaled_design @ z[:8].reshape(4, 2).T + common - measured
            grad = (2 * (scaled_design.T @ residual).T / residual.size / normalization).ravel()
            return np.r_[grad, 0.0, 0.0] if free_center else grad

        def centered(z):
            return z[:8].reshape(4, 2).sum(axis=0)

        mode = "unconstrained_feasible"
        solution = initial
        if inequalities(variables(initial)).min() < -1e-9:
            # Include feasible triangular starts; do not independently clip cores.
            starts = [initial]
            base_angles = np.arctan2(initial[1:, 1], initial[1:, 0])
            phase = np.angle(np.mean(np.exp(1j * (base_angles - np.arange(3) * 2 * np.pi / 3))))
            for shift in (phase, phase + np.pi / 3, phase - np.pi / 3):
                a = shift + np.arange(3) * 2 * np.pi / 3
                starts.append(
                    np.vstack(([0.0, 0.0], (gap / upper) * np.column_stack((np.cos(a), np.sin(a)))))
                )
            best = None
            for start in starts:
                result = minimize(
                    objective,
                    variables(start),
                    jac=gradient,
                    method="SLSQP",
                    constraints=[
                        dict(type="eq", fun=centered),
                        dict(type="ineq", fun=inequalities),
                    ],
                    options=dict(maxiter=1000, ftol=1e-12),
                )
                if (
                    result.success
                    and np.isfinite(result.x).all()
                    and inequalities(result.x).min() >= -1e-7
                    and np.max(np.abs(centered(result.x))) < 1e-8
                ):
                    if best is None or result.fun < best.fun:
                        best = result
            if best is None:
                raise RuntimeError(f"WL_{wl}: no converged feasible four-core calibration")
            solution = best.x[:8].reshape(4, 2)
            mode = "joint_centroid_" + size_mode
        points = solution * upper
        if np.linalg.matrix_rank(points) != 2:
            raise ValueError(f"WL_{wl}: degenerate four-core positions")
        radii = np.linalg.norm(points, axis=1)
        alpha = (
            np.deg2rad(settings.core2_azimuth_deg) - np.arctan2(points[:, 1], points[:, 0])
        ) % (2 * np.pi)
        residual = design @ points.T + common - measured
        envelope_center, envelope_radius = enclosing_circle(points)
        if settings.envelope_center_mode == "centroid":
            feasible_center = np.zeros(2)
        else:
            feasible_center = (
                central_enclosing_circle(points)[0] if envelope_order else envelope_center
            )
        distances = np.linalg.norm(points - feasible_center, axis=1)
        if (use_envelope and distances.max() > upper + 1e-4) or (
            envelope_order and np.any(distances[1:] < distances[0] - 1e-4)
        ):
            raise RuntimeError(
                f"WL_{wl}: no common envelope center satisfies containment and core order"
            )
        azimuth = np.deg2rad(settings.core2_azimuth_deg)
        envelope_center_uv = (
            np.array([[np.sin(azimuth), -np.cos(azimuth)], [np.cos(azimuth), np.sin(azimuth)]])
            @ envelope_center
        )
        feasible_center_uv = (
            np.array([[np.sin(azimuth), -np.cos(azimuth)], [np.cos(azimuth), np.sin(azimuth)]])
            @ feasible_center
        )
        cores = {
            f"ch{ch}": dict(
                r_um=float(r),
                alpha_rad=float(a) if r > 1e-6 else None,
                alpha_deg=float(np.rad2deg(a)) if r > 1e-6 else None,
            )
            for ch, r, a in zip(channels, radii, alpha)
        }
        sections.append(
            dict(
                wl_id=wl,
                cores=cores,
                outer_channels=layout["outer_channels"],
                response_channels=channels,
                fit_rmse_pm=float(np.sqrt(np.mean(residual**2))),
                response_model="four_core_common",
                calibration_common_response_pm=[
                    dict(direction_deg=d, radius_mm=r, b_pm=float(b))
                    for (d, r), b in zip(data.states, common_pm[:, index])
                ],
                fit_rmse_by_channel_pm={
                    f"ch{ch}": float(v)
                    for ch, v in zip(channels, np.sqrt(np.mean(residual**2, axis=0)))
                },
                constraint_solver=mode,
                centroid_error_um=float(np.linalg.norm(points.sum(axis=0))),
                matrix_condition_number=float(np.linalg.cond(points)),
                minimum_enclosing_center_uv_um=envelope_center_uv.tolist(),
                minimum_enclosing_radius_um=envelope_radius,
                envelope_center_uv_um=feasible_center_uv.tolist(),
                envelope_containing_radius_um=float(distances.max()),
                envelope_center_identified_from_measurements=False,
                minimum_pairwise_distance_um=float(
                    min(np.linalg.norm(points[i] - points[j]) for i, j in pairs)
                ),
            )
        )
    return dict(
        schema_version=6,
        method="四芯共同项校准法",
        channel_configuration=layout,
        coordinate_frame=dict(
            origin="four_core_centroid",
            relative_positions_only=True,
            centroid_assumed_at_envelope_center=settings.envelope_center_mode == "centroid",
            undefined_angle_radius_threshold_um=1e-6,
            alpha_definition="clockwise from fixture-zero reference-up (project +Z) toward project +Y",
            project_yz_mapping="internal +X -> project +Y; internal +Y -> project +Z; axial -> project +X",
        ),
        calibration=dict(
            directions_deg=list(DIRECTIONS),
            radii_mm=data.radii_mm,
            wl_ids=list(data.wl_ids),
            nominal_wl_ids=[],
            csv_file_count=data.csv_file_count,
            wavelength_shift_unit="pm",
            input_already_straight_referenced=True,
            second_straight_rezero=False,
            center_channel_subtracted=False,
            response_reference="four_core_common",
            common_response_symbol="b",
            common_response_unit="strain",
            weight_equation="w_j * (e_j - b) = x_j*k_x + y_j*k_y",
            csv_sample_aggregation="arithmetic_mean",
            strain_sensitivity_pm_per_ue=settings.k_pm_per_ue,
            sensitivity_source=settings.sensitivity_source,
            core2_azimuth_deg=settings.core2_azimuth_deg,
            bend_side_offset_deg=settings.bend_side_offset_deg,
            direction_sign=settings.direction_sign,
            strain_projection_sign=settings.strain_projection_sign,
            physical_strain_from_csv_sign=-settings.strain_projection_sign,
        ),
        physical_constraints=dict(
            fiber_diameter_um=settings.fiber_diameter_um,
            core_radius_um=settings.core_radius_um,
            ch1_assumed_at_section_center=False,
            size_constraint_mode=size_mode,
            envelope_center_mode=settings.envelope_center_mode,
            center_order_reference=settings.center_order_reference,
            radial_reference=(
                "free_enclosing_circle"
                if settings.envelope_center_mode == "free"
                else "four_core_centroid_assumed_envelope_center"
            ),
            center_no_farther_than_each_outer=True,
            outer_center_min_radius_um=0.0,
            outer_center_max_radius_um=upper,
            minimum_pairwise_core_center_distance_um=gap,
        ),
        diagnostics=dict(
            section_count=len(sections),
            calibrated_section_count=len(sections),
            nominal_section_count=0,
            coupled_fit_sections=sum(
                s["constraint_solver"] != "unconstrained_feasible" for s in sections
            ),
            mean_fit_rmse_pm=float(np.mean([s["fit_rmse_pm"] for s in sections])),
        ),
        sections=sections,
    )
