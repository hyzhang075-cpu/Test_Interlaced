import json
from itertools import combinations
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize(
    "entry",
    [
        ["-m", "shape_sensing"],
        ["-c", "import sys; from shape_sensing.cli import main; sys.exit(main())"],
    ],
)
def test_orthocal_cli_writes_schema6_with_global_wl_ids_and_raw_common_term(tmp_path, entry):
    channels = (7, 9, 2, 4)
    root = tmp_path / "input"
    geometry = dict(
        sensor_id="test",
        start_value=-10.0,
        end_value=6.0,
        samp_dist_preset=[100, 150, 300],
        interlaced_grating=False,
    )
    geometry_file = tmp_path / "geometry.json"
    geometry_file.write_text(json.dumps(geometry), encoding="utf-8")
    positions_um = np.array([[0, 0], [110, 0], [-55, 55 * np.sqrt(3)], [-55, -55 * np.sqrt(3)]])
    for direction, common_pm in [(0, 73), (120, -19), (240, 42)]:
        folder = root / f"{direction}-100"
        folder.mkdir(parents=True)
        angle = np.deg2rad(-direction)
        response_pm = positions_um @ (10 * np.array([np.cos(angle), np.sin(angle)]))
        for channel, response in zip(channels, response_pm):
            # WL_2 violates the pairwise constraint. WL_3 has feasible true geometry.
            first = 0.5 * response + common_pm
            second = response + common_pm
            (folder / f"Channel_{channel}_fixture.csv").write_text(
                f"WL_2,WL_3,WL_4\n{first-1},{second-1},\n{first+1},{second+1},\n",
                encoding="utf-8",
            )
    output = tmp_path / "result" / "cores.json"
    completed = subprocess.run(
        [
            sys.executable,
            *entry,
            "orthocal",
            "--data-dir",
            str(root),
            "--geometry-json",
            str(geometry_file),
            "--output-json",
            str(output),
            "--active-channels",
            *map(str, channels),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema_version"] == 6
    assert result["hardware_geometry"] == geometry
    assert result["calibration"]["wl_ids"] == [2, 3]
    assert result["calibration"]["nominal_wl_ids"] == []
    assert result["scope"]["global_sample_dist_m"] == pytest.approx([0.14, 0.19])
    assert result["channel_configuration"]["response_channels"] == list(channels)
    constrained, section = result["sections"]
    assert section["wl_id"] == 3
    assert section["response_model"] == "four_core_common"
    assert section["cores"]["ch7"] == {
        "r_um": pytest.approx(0),
        "alpha_rad": None,
        "alpha_deg": None,
    }
    assert section["cores"]["ch9"]["alpha_deg"] == pytest.approx(90)
    assert section["fit_rmse_pm"] == pytest.approx(0, abs=1e-8)
    assert [state["b_pm"] for state in section["calibration_common_response_pm"]] == pytest.approx(
        [73, -19, 42]
    )
    centers = []
    for core in constrained["cores"].values():
        angle = core["alpha_rad"] or 0
        centers.append(core["r_um"] * np.array([np.sin(angle), np.cos(angle)]))
    np.testing.assert_allclose(np.sum(centers, axis=0), [0, 0], atol=1e-6)
    assert min(np.linalg.norm(a - b) for a, b in combinations(centers, 2)) >= 100 - 1e-5
    distances = np.linalg.norm(centers - np.array(constrained["envelope_center_uv_um"]), axis=1)
    assert distances.max() <= 125 + 1e-5
    assert np.all(distances[1:] >= distances[0] - 1e-5)


def test_interlaced_service_merges_raw_channels_and_interpolates_center(tmp_path):
    from shape_sensing.initialization.service import InitializationService
    from shape_sensing.input_preprocessing.four_core import load_four_core_data

    geometry = dict(
        start_value=0,
        end_value=6,
        samp_dist_preset=[100, 200],
        samp_dist_preset_sub=[50, 150],
        interlaced_grating=True,
    )
    acquisition = (9, 1, 2, 3, 4, 5, 6)
    positions_um = np.array([[0, 0], [110, 0], [-55, 55 * np.sqrt(3)], [-55, -55 * np.sqrt(3)]])
    for direction in (0, 120, 240):
        folder = tmp_path / f"{direction}-100"
        folder.mkdir()
        angle = np.deg2rad(-direction)
        response_pm = positions_um @ (10 * np.array([np.cos(angle), np.sin(angle)]))
        for channel, core in zip(acquisition, (0, 1, 1, 2, 2, 3, 3)):
            common = [10, 20] if channel in (9, 1, 3, 5) else [5, 15]
            first, second = response_pm[core] + np.array(common)
            (folder / f"Channel_{channel}_fixture.csv").write_text(
                f"WL_1,WL_2\n{first},{second}\n", encoding="utf-8"
            )

    dataset = load_four_core_data(tmp_path, geometry, active_channels=acquisition)
    result = InitializationService.fit_four_core(dataset)
    assert result["acquisition_channels"] == list(acquisition)
    assert result["channel_configuration"]["response_channels"] == [9, 1, 3, 5]
    assert result["interlaced_response_mapping"]["section_sources"] == {
        "1": "extrapolated",
        "2": "measured",
        "3": "interpolated",
        "4": "measured",
    }
    assert [
        s["calibration_common_response_pm"][0]["b_pm"] for s in result["sections"]
    ] == pytest.approx([5, 10, 15, 20])
    assert all(s["fit_rmse_pm"] < 1e-8 for s in result["sections"])
