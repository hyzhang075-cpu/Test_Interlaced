"""Read straight-referenced four-core calibration CSVs in physical channel order."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from ..configuration.regions import compute_fbg_geometry


DIRECTIONS = (0, 120, 240)
REGION_LENGTHS_M = {"full": None, "end15cm": 0.15, "end5cm": 0.05}
_FOLDER = re.compile(r"(0|120|240)-(\d+)")
_WL_COLUMN = re.compile(r"WL_(\d+)")


@dataclass(frozen=True)
class FourCoreCalibrationData:
    """Raw pm responses with axes (bend state, core, real WL)."""

    responses_pm: np.ndarray
    states: tuple
    channels: tuple
    wl_ids: tuple
    hardware_geometry: dict
    scope: dict
    acquisition_channels: tuple
    interlaced_sources: dict

    @property
    def radii_mm(self):
        return [radius for direction, radius in self.states if direction == DIRECTIONS[0]]

    @property
    def csv_file_count(self):
        # Schema 6 counts channel inputs used by the selected calibration states.
        return len(self.states) * len(self.channels)


def _geometry_scope(geometry, preset, region, margin_mm):
    if region not in REGION_LENGTHS_M:
        raise ValueError(f"Unknown orthogonal region: {region}")
    if not np.isfinite(margin_mm) or margin_mm < 0 or (region == "full" and margin_mm != 0):
        raise ValueError("Boundary margin must be nonnegative and zero for full")
    length, positions = compute_fbg_geometry(geometry["start_value"], geometry["end_value"], preset)
    bend_length = length if region == "full" else REGION_LENGTHS_M[region]
    if bend_length > length or margin_mm / 1000 >= bend_length:
        raise ValueError("Region length/margin exceeds fibre length")
    origin = length - bend_length
    indices = np.flatnonzero(
        (positions >= origin + margin_mm / 1000 - 1e-12) & (positions <= length + 1e-12)
    )
    if not indices.size:
        raise ValueError("No real FBG point lies inside the selected bending region")
    return dict(
        name=region,
        length_m=bend_length,
        full_length_m=length,
        crop_origin_m=origin,
        boundary_margin_mm=margin_mm,
        global_wl_ids=(indices + 1).tolist(),
        global_sample_dist_m=positions[indices].tolist(),
        local_sample_dist_m=(positions[indices] - origin).tolist(),
    )


def _state_folders(root, radii_mm):
    folders = {}
    for path in root.iterdir():
        match = _FOLDER.fullmatch(path.name)
        if path.is_dir() and match:
            direction, radius = map(int, match.groups())
            if radius > 0:
                folders[direction, radius] = path
    radius_sets = [{r for d, r in folders if d == direction} for direction in DIRECTIONS]
    if not radius_sets[0] or any(radii != radius_sets[0] for radii in radius_sets[1:]):
        raise ValueError("0/120/240 folders must contain the same nonzero radii")
    radii = sorted(radius_sets[0] if radii_mm is None else radii_mm)
    if not radii or len(radii) != len(set(radii)) or not set(radii).issubset(radius_sets[0]):
        raise ValueError("Selected radii must be unique positive available CSV radii")
    return {
        (direction, int(radius)): folders[direction, radius]
        for direction in DIRECTIONS
        for radius in radii
    }


def _channel_means(folder, channel, wl_ids):
    files = list(folder.glob(f"Channel_{channel}_*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected one ch{channel} CSV in {folder}; got {len(files)}")
    frame = pd.read_csv(files[0])
    columns = {}
    for column in frame:
        match = _WL_COLUMN.fullmatch(column)
        if match:
            wl = int(match.group(1))
            if wl in columns:
                raise ValueError(f"Duplicate WL_{wl} in {files[0]}")
            columns[wl] = column
    missing = sorted(set(wl_ids).difference(columns))
    if missing:
        raise ValueError(f"Missing WL IDs {missing} in {files[0]}")
    numeric = frame[[columns[wl] for wl in wl_ids]].apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0, skipna=True).to_numpy(dtype=float)
    if not np.isfinite(means).all():
        bad = [wl for wl, mean in zip(wl_ids, means) if not np.isfinite(mean)]
        raise ValueError(f"Nonfinite mean for WL IDs {bad} in {files[0]}")
    return means


def _interlaced_presets(geometry):
    a = np.asarray(geometry["samp_dist_preset"], dtype=float)
    b = np.asarray(geometry.get("samp_dist_preset_sub"), dtype=float)
    if (
        a.ndim != 1
        or b.ndim != 1
        or a.size < 2
        or a.size != b.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "Interlaced A/B presets require matching lengths >= 2 and finite positions"
        )
    if np.unique(np.r_[a, b]).size != a.size + b.size:
        raise ValueError("Interlaced A/B positions must be unique and must not overlap")
    return a, b


def _merge_interlaced(responses, a, b):
    a_order = np.argsort(a)
    center_a = responses[:, 0]
    center_b = CubicSpline(a[a_order], center_a[:, a_order], axis=1)(b)
    columns = [np.concatenate((center_a, center_b), axis=1)]
    for channel in (1, 3, 5):
        columns.append(np.concatenate((responses[:, channel], responses[:, channel + 1]), axis=1))
    return np.stack(columns, axis=1)[:, :, np.argsort(np.r_[a, b])]


def load_four_core_data(
    data_dir,
    hardware_geometry,
    *,
    active_channels=(1, 2, 3, 4),
    region="full",
    boundary_margin_mm=0.0,
    radii_mm=None,
):
    """Load already-zeroed CSVs. Preserve their raw pm scale and original WL IDs."""
    geometry = deepcopy(hardware_geometry)
    interlaced = geometry.get("interlaced_grating", False)
    if not isinstance(interlaced, bool):
        raise ValueError("interlaced_grating must be a boolean")
    channels = tuple(active_channels)
    count = 7 if interlaced else 4
    if (
        len(channels) != count
        or len(set(channels)) != count
        or any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in channels)
    ):
        raise ValueError(f"Expected {count} unique positive physical channel IDs")

    preset = geometry["samp_dist_preset"]
    if interlaced:
        a, b = _interlaced_presets(geometry)
        preset = np.sort(np.r_[a, b])

    scope = _geometry_scope(geometry, preset, region, boundary_margin_mm)
    wl_ids = tuple(scope["global_wl_ids"])
    root = Path(data_dir).resolve()
    scope["source_data_dir"] = str(root)
    folders = _state_folders(root, radii_mm)
    raw_wl_ids = tuple(range(1, len(a) + 1)) if interlaced else wl_ids
    responses = np.array(
        [
            [_channel_means(folder, channel, raw_wl_ids) for channel in channels]
            for folder in folders.values()
        ]
    )
    effective_channels = channels
    interlaced_sources = {}
    if interlaced:
        # Interpolate the measured A center on the full B grid before region selection.
        responses = _merge_interlaced(responses, a, b)[:, :, np.asarray(wl_ids) - 1]
        effective_channels = channels[:1] + channels[1::2]
        for wl in wl_ids:
            position = preset[wl - 1]
            if position in a:
                source = "measured"
            else:
                source = (
                    "extrapolated" if position < a.min() or position > a.max() else "interpolated"
                )
            interlaced_sources[str(wl)] = source

    return FourCoreCalibrationData(
        responses,
        tuple(folders),
        effective_channels,
        wl_ids,
        geometry,
        scope,
        channels,
        interlaced_sources,
    )
