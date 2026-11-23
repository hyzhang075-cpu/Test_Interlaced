"""Command-line adapter for four-core core-position calibration."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..initialization.service import InitializationService
from ..input_preprocessing.four_core import REGION_LENGTHS_M, load_four_core_data


def run_orthocal(argv):
    parser = argparse.ArgumentParser(
        prog="shape-sensing orthocal",
        description="Fit per-WL four-core positions from straight-referenced pm CSVs.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry-json",
        type=Path,
        required=True,
        help="Hardware geometry JSON or an existing core calibration containing hardware_geometry",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--region", choices=REGION_LENGTHS_M, default="full")
    parser.add_argument("--boundary-margin-mm", type=float, default=0.0)
    parser.add_argument(
        "--active-channels", type=int, nargs="+", help="Physical channels in acquisition order"
    )
    parser.add_argument("--radii-mm", type=int, nargs="+")
    parser.add_argument("--diameter-um", type=float, default=350.0)
    parser.add_argument("--core-radius-um", type=float, default=50.0)
    parser.add_argument("--sensitivity-pm-per-ue", type=float, default=1.0)
    parser.add_argument("--bend-offset-deg", type=float, default=-90.0)
    parser.add_argument(
        "--size-constraint-mode", choices=("none", "pairwise", "envelope", "both"), default="both"
    )
    parser.add_argument("--envelope-center-mode", choices=("centroid", "free"), default="free")
    parser.add_argument(
        "--center-order-reference",
        choices=("envelope_center", "four_core_centroid"),
        default="envelope_center",
    )
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.geometry_json.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Geometry JSON must be an object")
        geometry = payload.get("hardware_geometry", payload)
        configuration = payload.get("channel_configuration", {})
        if not isinstance(geometry, dict) or not isinstance(configuration, dict):
            raise ValueError("hardware_geometry and channel_configuration must be objects")
        if configuration.get("centre_fibre", geometry.get("centre_fibre", True)) is not True:
            raise ValueError("Orthogonal four-core calibration requires the center channel")
        channels = args.active_channels or payload.get("acquisition_channels")
        if channels is None:
            channels = configuration.get(
                "active_channels", geometry.get("active_channels", (1, 2, 3, 4))
            )
        dataset = load_four_core_data(
            args.data_dir,
            geometry,
            active_channels=channels,
            region=args.region,
            boundary_margin_mm=args.boundary_margin_mm,
            radii_mm=args.radii_mm,
        )
        result = InitializationService.fit_four_core(
            dataset,
            diameter_um=args.diameter_um,
            core_radius_um=args.core_radius_um,
            sensitivity_pm_per_ue=args.sensitivity_pm_per_ue,
            bend_offset_deg=args.bend_offset_deg,
            size_constraint_mode=args.size_constraint_mode,
            envelope_center_mode=args.envelope_center_mode,
            center_order_reference=args.center_order_reference,
        )
        _write_json(args.output_json, result)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(args.output_json)


def _write_json(destination, result):
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(content)
        except OSError:
            handle.close()
            temporary_path.unlink()
            raise
    try:
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
