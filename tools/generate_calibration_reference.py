"""Generate raw FBG samples and ideal calibration coordinates for one static set.

The generated JSON keeps the raw channel data and the 120-point theoretical
curve together for each configured circular calibration radius. Static CSV
loading, wavelength scaling, channel ordering, and negative-start trimming are
delegated to :class:`sensor.Sensor`; ideal geometry is delegated to
``Optimiser._construct_ideals``.

Run from the repository root, for example::

    python tools/generate_calibration_reference.py

Optional arguments can override the input folder, sensor parameter package,
and output JSON path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    ROOT
    / "261017_易老师数据"
    / "1016-08B5+导管两端再重新固定+0状态波长储存+重新定标"
    / "定标测试"
)
DEFAULT_SENSOR_PARAMS = ROOT / "params" / "08B5_NeTi.json"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "calibration_reference_data.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _section(package: dict, *keys: str) -> dict:
    value = package
    for key in keys:
        value = value[key]
    return value


def _create_sensor(data_dir: Path, package: dict):
    # Imports here keep --help usable without importing the reconstruction stack.
    from sensor import Sensor

    shared = _section(package, "shared_parameters")
    hardware = shared["hardware_parameters"]
    experiment = shared["calibration_experiment_parameters"]
    full_region = _section(package, "regions", "full")

    # The current files are FiberTecc static data. Keep Sensor's normal channel
    # ordering and center-fiber handling, while letting it trim negative points.
    return Sensor(
        sensor_id=package.get("sensor_id", "08B5_NeTi"),
        start_value=hardware["start_value"],
        end_value=hardware["end_value"],
        folder_path=str(data_dir),
        centre_fibre=hardware.get("centre_fibre", True),
        active_channels=experiment["active_channels"],
        rot_dir=experiment.get("rot_dir", {}).get("0", -1),
        exp_directs=[0],
        interrogator=experiment["interrogator"],
        R=hardware["R"],
        ideal_len=hardware.get("ideal_len"),
        radii={"circ": full_region["radii"]["circ"]},
        samp_dist_preset=hardware["samp_dist_preset"],
        samp_dist_preset_sub=hardware.get("samp_dist_preset_sub"),
        active_angles=experiment.get("active_angles"),
        reverse=False,
        straight_trim=0.0,
        ss=experiment.get("ss", 2.6e-3),
        channel_change=experiment.get("channel_change", False),
        channel_layout=experiment.get("channel_layout"),
        interlaced_grating=hardware.get("interlaced_grating", False),
        interlaced_offset=hardware.get("interlaced_offset", 0.0),
    )


def _build_ideal_curves(sensor, radii: list[float], interp_num: int) -> tuple[dict, np.ndarray]:
    from optimiser import Optimiser

    # Bind only the data consumed by _construct_ideals; this avoids creating an
    # optimizer or running reconstruction/optimization as a side effect.
    ideal_builder = Optimiser.__new__(Optimiser)
    ideal_builder.direct = 0
    ideal_builder.radii = {0: {"circ": radii}}

    interp_dist = np.linspace(
        float(sensor.ad_samp_dist[0]),
        float(sensor.ad_samp_dist[-1]),
        num=interp_num,
    )
    relative_interp_dist = interp_dist - interp_dist[0]
    val_len = float(relative_interp_dist[-1])
    curves = ideal_builder._construct_ideals(
        calib_shape="circ",
        interp_dist=relative_interp_dist,
        val_len=val_len,
    )
    return curves, relative_interp_dist


def generate(data_dir: Path, sensor_params: Path, output: Path) -> dict:
    package = _load_json(sensor_params)
    sensor = _create_sensor(data_dir, package)
    configured_radii = [float(radius) for radius in package["regions"]["full"]["radii"]["circ"]]

    loaded = sensor.Data.get("circ", {}).get(0, {})
    normalized_loaded = {float(radius): data for radius, data in loaded.items()}
    missing = [radius for radius in configured_radii if radius not in normalized_loaded]
    unexpected = [radius for radius in normalized_loaded if radius not in configured_radii]
    if missing or unexpected:
        raise ValueError(
            f"实际读取半径与配置不一致：missing={missing}, unexpected={unexpected}。"
            "请检查每个半径目录中的四个通道 CSV。"
        )

    reconstruction = _section(package, "shared_parameters", "reconstruction_parameters")
    interp_num = int(reconstruction["interp_num"])
    if interp_num != 120:
        raise ValueError(f"参数包 interp_num={interp_num}，与要求的 120 点不一致")
    ideal_curves, ideal_arc_len_m = _build_ideal_curves(sensor, configured_radii, interp_num)

    radius_records = []
    for radius in configured_radii:
        raw = np.asarray(normalized_loaded[radius], dtype=float)
        ideal_xyz = np.asarray(ideal_curves[radius], dtype=float)
        if raw.shape != (21, len(sensor.active_channels)):
            raise ValueError(
                f"半径 {radius:g} mm 的 raw_data 形状为 {raw.shape}，"
                f"期望 (21, {len(sensor.active_channels)})；请检查裁切和输入 CSV。"
            )
        if ideal_xyz.shape != (120, 3):
            raise ValueError(f"半径 {radius:g} mm 的理论曲线形状异常：{ideal_xyz.shape}")
        radius_records.append({
            "radius_mm": int(radius) if radius.is_integer() else radius,
            "raw_data": raw.tolist(),
            "ideal_coordinates_m": ideal_xyz.tolist(),
        })

    point_distances = np.asarray(sensor.samp_dist, dtype=float)
    raw_point_ids = np.asarray(sensor.global_point_ids, dtype=int)
    if len(point_distances) != 21:
        raise ValueError(f"裁切后传感点数为 {len(point_distances)}，预期 21")

    result = {
        "format": "static_calibration_raw_and_ideal_v1",
        "source_data_dir": str(data_dir),
        "sensor_params": str(sensor_params),
        "calibration_shape": "circ",
        "direct": 0,
        "units": {
            "raw_data": "Sensor.Data channel means, scaled by Sensor to 1e-6 units",
            "sample_distances": "m",
            "ideal_coordinates": "m",
        },
        "raw_channel_ids": list(sensor.active_channels.keys()),
        "raw_channel_order": list(sensor.active_channels.keys()),
        "raw_point_ids_1based": raw_point_ids.tolist(),
        "raw_sample_distances_m": point_distances.tolist(),
        "ideal_arc_distances_m": ideal_arc_len_m.tolist(),
        "ideal_point_count": interp_num,
        "raw_point_count": len(point_distances),
        "radii": radius_records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sensor-params", type=Path, default=DEFAULT_SENSOR_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    for path, label in ((args.data_dir, "数据目录"), (args.sensor_params, "参数包")):
        if not path.exists():
            parser.error(f"{label}不存在：{path}")
    if not args.data_dir.is_dir():
        parser.error(f"数据目录不是文件夹：{args.data_dir}")

    # Permit invocation from outside the repository root.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    result = generate(args.data_dir, args.sensor_params, args.output)
    print(
        f"已生成 {args.output}\n"
        f"半径数量: {len(result['radii'])}; "
        f"每个半径 raw_data: {result['raw_point_count']} 点; "
        f"理论坐标: {result['ideal_point_count']} 点"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
