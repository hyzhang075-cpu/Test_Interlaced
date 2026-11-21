"""导出静态定标的原始通道数据和理论曲线，不执行初始化或优化任务。

仓库根目录运行：python tools/generate_calibration_reference.py
指定方向示例：python tools/generate_calibration_reference.py --direct 0 60 120 240

输入支持“方向-半径”的平铺目录，以及“数字度/方向-半径”的两层目录；
方向从实际目录自动发现，不局限于参数包原来的 0、120、240 度。
circ 表示单段定曲率圆弧，end 表示两段相切圆弧。
end 目录中间数字为前段半径；1500 组末段长 15 cm，2000 组末段长 5 cm。

数据链路：Sensor 读取/清洗 CSV → 通道排序及负坐标裁切 → 取 Sensor.Data；
理论链路：按配置 interp_num 插值 → Optimiser._construct_ideals → 刚体旋转。
新角度定义以 +x 为初始切线、+z 为 0 度弯曲方向，正向遵循绕 +x 的右手定则。

输出保留 raw_data（当前 21×4）和 ideal_coordinates_m（当前 120×3，单位 m），
默认写到所选数据目录的 calibration_reference_data.json。
参数、分段定义及完整字段说明见 docs/calibration-reference-export.md。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "1020-08B5-导管两端固定全段+小圈定标测试"
DEFAULT_SENSOR_PARAMS = ROOT / "params" / "08B5_NeTi.json"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "calibration_reference_data.json"
# end 目录第二个数字就是前段实际半径，同时标识本次实验的末段长度：
# “1500-x”为前段半径 1500 mm、末段 15 cm；“2000-x”为前段半径 2000 mm、末段 5 cm。
DEFAULT_END_GROUPS = {
    1500: {"base_radius_mm": 1500, "end_length_m": .15, "radii_mm": [150, 100, 50]},
    2000: {"base_radius_mm": 2000, "end_length_m": .05, "radii_mm": [30, 20, 15]},
}


def _load_json(path: Path) -> dict:
    """兼容带 BOM 的 UTF-8 参数包。"""
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _finite(value, label):
    """在报错中保留数值来源，避免 NaN/inf 进入最终 JSON。"""
    if not np.all(np.isfinite(np.asarray(value, dtype=float))):
        raise ValueError(f"{label} 包含非有限值（NaN / inf）")


def _radii(values, label, *, positive=False):
    """检查半径列表并保留配置顺序；circ 的 0 表示直线，end 末段必须为正。"""
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} 必须是非空半径列表")
    _finite(values, label)
    result = [float(v) for v in values]
    if len(set(result)) != len(result) or any(v < 0 or (positive and v == 0) for v in result):
        raise ValueError(f"{label} 有重复或无效半径：{values}")
    return result


def _configuration(package, sensor_params, end_profile):
    """读取 full 区域半径、end 分组和插值点数，检查必需配置及数值有效性。

    当前参数包只有 circ 半径，因此 end 默认使用 DEFAULT_END_GROUPS。
    若包内提供 full.radii.end，则以包内列表决定需要读取哪些分组/末段半径。
    legacy 模式统一使用参数包 end_len 作为末段长度，前段半径仍取目录中间数字。
    """
    try:
        json.dumps(package, allow_nan=False)
        shared = package["shared_parameters"]
        hardware = shared["hardware_parameters"]
        experiment = shared["calibration_experiment_parameters"]
        for name in ("start_value", "end_value", "R", "samp_dist_preset"):
            _finite(hardware[name], f"hardware_parameters.{name}")
        channels = experiment["active_channels"]
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("active_channels 不能为空或重复")
        if experiment["interrogator"] not in ("FiberTecc", "Hippo"):
            raise ValueError("静态 CSV 导出仅支持 FiberTecc / Hippo")
        count = shared["reconstruction_parameters"]["interp_num"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("interp_num 必须是至少为 2 的整数")
        configured = package["regions"]["full"]["radii"]
        circ = _radii(configured["circ"], "regions.full.radii.circ")
        groups = {tag: dict(values) for tag, values in DEFAULT_END_GROUPS.items()}
        if "end" in configured:
            if not isinstance(configured["end"], dict) or not configured["end"]:
                raise ValueError("regions.full.radii.end 必须是非空分组对象")
            groups = {}
            for tag, values in configured["end"].items():
                tag = int(tag)
                if tag not in DEFAULT_END_GROUPS:
                    raise ValueError(f"end 目录标记 {tag} 未定义物理分段；请检查 DEFAULT_END_GROUPS")
                groups[tag] = dict(DEFAULT_END_GROUPS[tag], radii_mm=_radii(
                    values, f"regions.full.radii.end.{tag}", positive=True))
        if end_profile == "legacy":
            end_len = float(experiment["end_len"])
            _finite(end_len, "end_len")
            for tag in groups:
                groups[tag] = dict(groups[tag], base_radius_mm=tag, end_length_m=end_len)
        return circ, groups, count
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"请检查参数包 {sensor_params} 的必需配置或有限数值：{exc}") from exc


def _inventory(data_dir):
    """建立 (方向, 类型, 目录半径, 末段半径或 None) → 原始子目录的索引。

    角度父目录按数值排序，支持 60 度及其他整数角度；父子目录方向须一致。
    只跳过完全没有文件的角度目录；部分采集的数据继续交给后续完整性检查。
    """
    from shape_sensing.input_preprocessing.compat import Sensor

    angle_folders = sorted(
        (p for p in data_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d+度", p.name)),
        key=lambda p: int(p.name[:-1]),
    )
    roots = angle_folders or [data_dir]
    found, empty_directories = {}, []
    for root in roots:
        if angle_folders and not any(p.is_file() for p in root.rglob("*")):
            empty_directories.append(str(root))
            continue
        angle = int(root.name[:-1]) if re.fullmatch(r"\d+度", root.name) else None
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not re.fullmatch(r"\d+-\d+(?:-\d+)?", folder.name):
                continue
            direct, radius, tail, shape = Sensor._classify_folder(None, folder.name)
            if angle is not None and direct != angle:
                raise ValueError(f"方向目录 {root} 与数据目录 {folder} 的 direct={direct} 不一致")
            key = direct, shape, radius, tail
            if key in found:
                raise ValueError(f"重复的方向/半径数据：{found[key]} 和 {folder}")
            found[key] = folder
    if angle_folders and any(p.is_dir() and re.fullmatch(r"\d+-\d+(?:-\d+)?", p.name)
                             for p in data_dir.iterdir()):
        raise ValueError(f"{data_dir} 同时包含角度目录和半径目录，请选择单一数据层级")
    if not found:
        raise ValueError(f"{data_dir} 未找到 circ（角度-半径）或 end（角度-分组-末段半径）目录")
    return found, bool(angle_folders or re.fullmatch(r"\d+度", data_dir.name)), empty_directories


def _create_sensor(data_dir: Path, package: dict, directs, radii):
    """使用 Sensor 的完整静态读取流程，保持原始数据的通道和行坐标一致。

    directs 来自实际目录，不受参数包 exp_directs 原有列表限制。
    reverse=False、straight_trim=0.0 保留固定端起算的坐标；start_value 为负时，
    Sensor 仍会自动裁掉负坐标点，并用同一索引裁切 Data。
    中心光纤补偿由 Sensor 正常执行，但本工具只导出未经该补偿的 Data。
    """
    from shape_sensing.input_preprocessing.compat import Sensor

    shared = package["shared_parameters"]
    hardware = shared["hardware_parameters"]
    experiment = shared["calibration_experiment_parameters"]
    return Sensor(
        sensor_id=package.get("sensor_id", "08B5_NeTi"),
        start_value=hardware["start_value"], end_value=hardware["end_value"],
        folder_path=str(data_dir), centre_fibre=hardware.get("centre_fibre", True),
        active_channels=experiment["active_channels"],
        rot_dir=experiment.get("rot_dir", {}).get(str(directs[0]), -1),
        exp_directs=directs, interrogator=experiment["interrogator"],
        R=hardware["R"], ideal_len=hardware.get("ideal_len"), radii=radii,
        samp_dist_preset=hardware["samp_dist_preset"],
        samp_dist_preset_sub=hardware.get("samp_dist_preset_sub"),
        active_angles=experiment.get("active_angles"), reverse=False, straight_trim=0.0,
        ss=experiment.get("ss", 2.6e-3),
        channel_change=experiment.get("channel_change", False),
        channel_layout=experiment.get("channel_layout"),
        interlaced_grating=hardware.get("interlaced_grating", False),
        interlaced_offset=hardware.get("interlaced_offset", 0.0),
    )


def _build_ideal_curves(sensor, radii, interp_num, *, direct=0,
                        calibration_shape="circ", base_radius_mm=None, end_length_m=None):
    """复用 Optimiser 生成单个方向、单个形状/分组的理论坐标及相对弧长。

    circ 返回 {半径: N×3 坐标}；end 返回 {末段半径: N×3 坐标}。
    N 取参数包 interp_num，弧长范围包括 Sensor 的虚拟起点和末端点。
    此处仅装配现有几何方法所需属性，不运行 Optimiser.__init__ 或优化过程。
    """
    from shape_sensing.optimization.compat import Optimiser

    # 绕过优化器初始化，仅复用现有圆弧及分段拼接方法。
    builder = Optimiser.__new__(Optimiser)
    builder.direct = direct
    interp_dist = np.linspace(float(sensor.ad_samp_dist[0]),
                              float(sensor.ad_samp_dist[-1]), num=interp_num)
    arc = interp_dist - interp_dist[0]
    _finite(arc, "理论插值弧长")
    if arc[-1] <= 0 or not np.all(np.diff(arc) > 0):
        raise ValueError("理论插值弧长必须严格递增且总长为正；请检查采样配置")
    if calibration_shape == "end":
        if not 0 < end_length_m < arc[-1]:
            raise ValueError(f"末段长度 {end_length_m} m 必须小于理论全长 {arc[-1]} m")
        builder.radii = {direct: {"end": {base_radius_mm: radii}}}
        builder.end_len = end_length_m
        # 现有 end 方法用 samp_dist[-1] - end_len 定位拼接点。
        # 将其弧长域末端设为完整理论长度，使最后 5/15 cm 从物理末端量取；
        # 若直接传最后一个真实栅点位置，当前配置的末段会多出 6 mm。
        builder.samp_dist = np.array([arc[-1]])
    else:
        builder.radii = {direct: {"circ": radii}}
    curves = builder._construct_ideals(calib_shape=calibration_shape,
                                      interp_dist=arc, val_len=float(arc[-1]))
    return (curves[base_radius_mm] if calibration_shape == "end" else curves), arc


def _orient(curve, direct, convention):
    """把 Optimiser 的 x-y 平面曲线旋转到实验弯曲方向，不重新计算圆弧。

    z-positive：0° 向 +z，60° 朝 -y/+z，120° 朝 -y/-z，240° 朝 +y/-z。
    legacy-y：保持旧的 0° 向 +y 及顺时针角度定义，供旧数据兼容使用。
    """
    angle = np.deg2rad(float(direct))
    lateral = curve[:, 1]
    if convention == "z-positive":
        return np.column_stack((curve[:, 0], -lateral * np.sin(angle), lateral * np.cos(angle)))
    return np.column_stack((curve[:, 0], lateral * np.cos(angle), -lateral * np.sin(angle)))


def generate(data_dir: Path, sensor_params: Path, output: Path, *, directs=None,
             calibration_shape="all", angle_convention="auto", end_profile="catheter-1020") -> dict:
    """检查输入、读取数据、生成曲线并写 JSON；同时返回已写入的字典。

    默认导出目录内全部非空方向及发现的 circ/end 类型。
    单方向保持顶层 direct 为数字，多方向使用 direct=null 和 directs 列表；
    每条 radii 记录都有自己的 direct、calibration_shape 和 sample_id。
    校验或序列化失败时不覆盖原有输出文件。
    """
    # 1. 检查路径、参数包及显式选项，避免只依赖命令行入口的校验。
    data_dir, sensor_params, output = map(Path, (data_dir, sensor_params, output))
    if not data_dir.is_dir():
        raise ValueError(f"数据目录不存在或不是文件夹：{data_dir}")
    if not sensor_params.is_file():
        raise ValueError(f"参数包不存在或不是文件：{sensor_params}")
    if angle_convention not in ("auto", "legacy-y", "z-positive"):
        raise ValueError(f"未知角度定义：{angle_convention}")
    if end_profile not in ("catheter-1020", "legacy"):
        raise ValueError(f"未知 end 配置：{end_profile}")
    if calibration_shape not in ("all", "circ", "end"):
        raise ValueError(f"未知曲线类型：{calibration_shape}")
    try:
        package = _load_json(sensor_params)
    except (ValueError, OSError) as exc:
        raise ValueError(f"无法读取参数包 {sensor_params}：{exc}") from exc
    circ, groups, interp_num = _configuration(package, sensor_params, end_profile)
    # 2. 从实际文件夹发现方向，与配置半径逐项对照。已有目录缺通道不会被静默跳过。
    inventory, angle_layout, empty_directories = _inventory(data_dir)
    convention = ("z-positive" if angle_layout else "legacy-y") if angle_convention == "auto" else angle_convention
    available = sorted({k[0] for k in inventory})
    selected = sorted(set(directs)) if directs is not None else available
    if not selected or set(selected) - set(available):
        raise ValueError(f"{data_dir} 缺少方向 {sorted(set(selected) - set(available))}；可用方向 {available}")
    shapes = sorted({k[1] for k in inventory if k[0] in selected}) if calibration_shape == "all" else [calibration_shape]
    expected = {}
    for direct in selected:
        if "circ" in shapes:
            expected.update({(direct, "circ", r, None): None for r in circ})
        if "end" in shapes:
            expected.update({(direct, "end", tag, r): None
                             for tag, group in groups.items() for r in group["radii_mm"]})
    actual = {k for k in inventory if k[0] in selected and k[1] in shapes}

    def describe(keys):
        return ["-".join(str(int(v)) if float(v).is_integer() else str(v)
                         for v in (k[0], k[2], k[3]) if v is not None)
                for k in sorted(keys)]

    missing, unexpected = expected.keys() - actual, actual - expected.keys()
    if missing or unexpected:
        raise ValueError(f"{data_dir} 的配置半径与目录不一致：missing={describe(missing)}, "
                         f"unexpected={describe(unexpected)}；请检查参数包 {sensor_params} 和 end 分组配置")

    sensors, reference_sensor = {}, None
    records, arc = [], None
    sensor_radii = {"circ": circ, "end": {tag: group["radii_mm"] for tag, group in groups.items()}}
    ideal_cache = {}
    for key in expected:
        direct, shape, tag, tail = key
        folder = inventory[key]
        root = folder.parent
        if root not in sensors:
            # 同一个数据根目录只创建一次 Sensor；平铺目录可能含有多个方向。
            root_directs = sorted({k[0] for k, p in inventory.items() if p.parent == root})
            try:
                sensors[root] = _create_sensor(root, package, root_directs, sensor_radii)
            except (ValueError, KeyError, IndexError) as exc:
                raise ValueError(f"Sensor 读取失败，请检查 {root} 的通道 CSV 和参数包 {sensor_params}：{exc}") from exc
        sensor = sensors[root]
        if reference_sensor is None:
            reference_sensor = sensor
        elif (not np.array_equal(sensor.samp_dist, reference_sensor.samp_dist)
              or not np.array_equal(sensor.global_point_ids, reference_sensor.global_point_ids)):
            raise ValueError(f"{root} 的采样距离或传感点编号与其他方向不一致")
        loaded = sensor.Data.get(shape, {}).get(direct, {})
        # 3. 直接使用原始 Data；不要误用已扣除中心光纤信号的 processed_Data。
        raw = loaded.get(tag) if tail is None else loaded.get(tag, {}).get(tail)
        if raw is None:
            raise ValueError(f"{folder}（半径 {tail if tail is not None else tag} mm）未读取到有效数据；"
                             f"请检查通道 CSV 和参数包 {sensor_params}")
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (21, len(sensor.active_channels)):
            raise ValueError(f"{folder} 的 raw_data 形状 {raw.shape}，预期 (21, {len(sensor.active_channels)})；请检查负坐标裁切和 CSV")
        _finite(raw, f"{folder} 的 raw_data")
        cache_key = root, direct, shape, tag if shape == "end" else None
        # 4. 同一方向的 circ 共用一次理论构造；end 按各分组的末段长度分别构造。
        if cache_key not in ideal_cache:
            kwargs = {} if shape == "circ" else {
                "base_radius_mm": groups[tag]["base_radius_mm"],
                "end_length_m": groups[tag]["end_length_m"],
            }
            try:
                ideal_cache[cache_key] = _build_ideal_curves(
                    sensor, circ if shape == "circ" else groups[tag]["radii_mm"], interp_num,
                    direct=direct, calibration_shape=shape, **kwargs)
            except ValueError as exc:
                raise ValueError(f"{folder} 的理论曲线构造失败，请检查参数包 {sensor_params}：{exc}") from exc
        curves, arc = ideal_cache[cache_key]
        radius = tag if tail is None else tail
        xyz = np.asarray(curves[radius], dtype=float)
        if xyz.shape != (interp_num, 3):
            raise ValueError(f"{folder} 的理论曲线形状 {xyz.shape}，预期 ({interp_num}, 3)")
        xyz = _orient(xyz, direct, convention)
        _finite(xyz, f"{folder} 的 ideal_coordinates_m")
        record = {
            "radius_mm": int(radius) if float(radius).is_integer() else radius,
            "raw_data": raw.tolist(), "ideal_coordinates_m": xyz.tolist(),
            "sample_id": folder.name, "source_folder": str(folder),
            "calibration_shape": shape, "direct": direct,
        }
        if shape == "end":
            group = groups[tag]
            record.update(directory_radius_mm=tag, base_radius_mm=group["base_radius_mm"],
                          end_length_m=group["end_length_m"],
                          junction_arc_distance_m=float(arc[-1] - group["end_length_m"]))
        records.append(record)

    point_distances = np.asarray(reference_sensor.samp_dist, dtype=float)
    _finite(point_distances, f"{data_dir} 的采样距离")
    if len(point_distances) != 21:
        raise ValueError(f"{data_dir} 裁切后传感点数为 {len(point_distances)}，预期 21；请检查 {sensor_params}")
    result = {
        "format": "static_calibration_raw_and_ideal_v1",
        "source_data_dir": str(data_dir), "sensor_params": str(sensor_params),
        "calibration_shape": shapes[0] if len(shapes) == 1 else "mixed",
        "direct": selected[0] if len(selected) == 1 else None,
        "calibration_shapes": shapes, "directs": selected,
        "skipped_empty_directories": empty_directories,
        "angle_convention": convention, "end_profile": end_profile,
        "coordinate_frame": {"longitudinal_axis": "+x",
                             "positive_angle": "right_hand_about_+x" if convention == "z-positive" else "clockwise_from_+y",
                             "zero_bending_axis": "+z" if convention == "z-positive" else "+y"},
        "units": {"raw_data": "Sensor.Data channel means, scaled by Sensor to 1e-6 units",
                  "sample_distances": "m", "ideal_coordinates": "m", "radius_mm": "mm", "direct": "degree"},
        "raw_channel_ids": list(reference_sensor.active_channels),
        "raw_channel_order": list(reference_sensor.active_channels),
        "raw_point_ids_1based": np.asarray(reference_sensor.global_point_ids, dtype=int).tolist(),
        "raw_sample_distances_m": point_distances.tolist(), "ideal_arc_distances_m": arc.tolist(),
        "ideal_point_count": interp_num, "raw_point_count": len(point_distances), "radii": records,
    }
    # 5. 先完成严格序列化再写文件，保持旧字段名称和 v1 格式标识。
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    return result


def _gitignore_status(output):
    """只查询 Git 忽略状态并报告命中的规则，不修改用户的 .gitignore。"""
    try:
        status = subprocess.run(["git", "check-ignore", "-v", "--", str(output.resolve())],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return "未能检查 Git 忽略状态；未修改 .gitignore。"
    if status.returncode == 0:
        return f"输出路径被 Git 忽略：{status.stdout.strip()}（未修改 .gitignore）"
    if status.returncode == 1:
        return "输出路径未被 .gitignore 忽略；未修改现有规则。"
    return "输出路径的 Git 忽略状态无法确定；未修改 .gitignore。"


def main(argv: list[str] | None = None) -> int:
    """命令行入口：默认自动发现方向，输出路径跟随 --data-dir。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sensor-params", type=Path, default=DEFAULT_SENSOR_PARAMS)
    parser.add_argument("--output", type=Path, help="默认写入所选数据目录/calibration_reference_data.json")
    parser.add_argument("--direct", type=int, nargs="+", help="只导出指定方向，例如 0 60 120 240；默认自动发现全部方向")
    parser.add_argument("--calibration-shape", choices=["all", "circ", "end"], default="all")
    parser.add_argument("--angle-convention", choices=["auto", "z-positive", "legacy-y"], default="auto",
                        help="auto：角度父目录使用 +z 新定义，平铺目录使用旧 +y 定义")
    parser.add_argument("--end-profile", choices=["catheter-1020", "legacy"], default="catheter-1020",
                        help="catheter-1020：1500组末段15cm、2000组末段5cm；前段半径取目录中间数字；legacy：末段统一使用配置 end_len")
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    output = args.output or args.data_dir / "calibration_reference_data.json"
    try:
        result = generate(args.data_dir, args.sensor_params, output, directs=args.direct,
                          calibration_shape=args.calibration_shape, angle_convention=args.angle_convention,
                          end_profile=args.end_profile)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"已生成 {output}\n记录数量: {len(result['radii'])}; "
          f"raw_data: {result['raw_point_count']}×{len(result['raw_channel_order'])}; "
          f"ideal_coordinates_m: {result['ideal_point_count']}×3")
    counts = Counter((r["direct"], r["calibration_shape"]) for r in result["radii"])
    for direct in result["directs"]:
        print(f"方向 {direct}°: circ={counts[direct, 'circ']}; end={counts[direct, 'end']}")
    if result["skipped_empty_directories"]:
        print("以下方向目录完全没有数据文件，已跳过：" + "; ".join(result["skipped_empty_directories"]))
    print(_gitignore_status(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
