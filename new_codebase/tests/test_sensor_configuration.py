import json
from pathlib import Path

import pytest
import yaml

from shape_sensing.configuration import load_working_config, parameter_catalog

def _config(tmp_path, extra=None):
    data = {"task": "init", "sensor_id": "TST", "R": 0.0002, "samp_dist_preset": [1, 2], "centre_fibre": False, "start_value": 0}
    data.update(extra or {})
    path = tmp_path / "sensor.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path

def test_catalog_and_template_cover_main_parameters():
    from shape_sensing.cli import build_argument_parser

    catalog = parameter_catalog()
    parser_parameters = {action.dest for action in build_argument_parser()._actions}
    assert set(catalog) == parser_parameters - {'help', 'config', 'resume'}
    assert sum(bool(item["choices"]) for item in catalog.values()) == 9
    template = (Path(__file__).parents[1] / "configs" / "sensor_template.yaml").read_text(encoding="utf-8")
    for key, spec in catalog.items():
        if key in {"task", "sensor_params"}:
            assert key in template
            continue
        assert f"{key}:" in template
        if spec["help"]:
            assert spec["help"] in template
        if spec["choices"]:
            assert "choices:" in template

def test_preserves_false_zero_null_and_resolves_paths(tmp_path):
    path = _config(tmp_path, {"weights": None, "folder_path": "data", "calibration_regions": {"full": {"folder_path": "regions/full"}}})
    got = load_working_config(path)
    assert got["centre_fibre"] is False and got["start_value"] == 0 and got["weights"] is None
    assert got["folder_path"] == str((tmp_path / "data").resolve())
    assert got["calibration_regions"]["full"]["folder_path"] == str((tmp_path / "regions/full").resolve())

def test_rejects_missing_design_unknown_type_choice_and_nested_key(tmp_path):
    for data, phrase in [
        ({"sensor_id":"x", "R":1}, "samp_dist_preset"),
        ({"sensor_id":"x", "R":1, "samp_dist_preset":[1], "interrogator":"bad"}, "interrogator"),
        ({"sensor_id":"x", "R":"1", "samp_dist_preset":[1]}, "R"),
        ({"sensor_id":"x", "R":1, "samp_dist_preset":[1], "calibration_regions":{"full":{"bogus":1}}}, "calibration_regions.full.bogus"),
    ]:
        path=tmp_path/"bad.yaml"; path.write_text(yaml.safe_dump(data),encoding="utf-8")
        with pytest.raises(ValueError, match=phrase): load_working_config(path)

def test_duplicate_yaml_key_rejected(tmp_path):
    path=tmp_path/"dup.yaml"; path.write_text("sensor_id: A\nR: 1\nR: 2\nsamp_dist_preset: [1]\n",encoding="utf-8")
    with pytest.raises(ValueError, match="重复键"): load_working_config(path)

def test_cli_config_layer_and_explicit_override_and_conflict(tmp_path):
    import main
    path=_config(tmp_path, {"move_filter_for_vector": False})
    args=main.parse_main_args(["--config",str(path),"--start_value","5"])
    assert args.sensor_id == "TST" and args.start_value == 5 and args.move_filter_for_vector is False
    args_none=main.parse_main_args(["--config",str(path),"--weights","None"])
    assert args_none.weights is None
    with pytest.raises(ValueError, match="不能同时使用"):
        main.parse_main_args(["--config",str(path),"--sensor_params","old.json"])

def test_cli_duplicate_parameters_rejected(tmp_path):
    import main
    path=_config(tmp_path)
    with pytest.raises(SystemExit): main.parse_main_args(["--config",str(path),"--start_value","1","--start_value","2"])


def test_full_template_load_and_configure_calibration_args(tmp_path):
    import main
    source=Path(__file__).parents[1]/"configs"/"sensor_template.yaml"
    data=yaml.safe_load(source.read_text(encoding="utf-8"))
    data.update(task="init", sensor_id="TST", R=0.0001, samp_dist_preset=list(range(22)))
    data["calibration_data_root"]="fixture-root"
    path=tmp_path/"filled.yaml"
    path.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding="utf-8")
    args=main.parse_main_args(["--config",str(path)])
    assert args.task=="init" and args.sensor_id=="TST"
    assert args.calibration_regions["full"]["folder_path"]==str((tmp_path/"fixture-root"/"全段测试").resolve())
    main.configure_calibration_args(args)

def test_reject_string_bool_null_numeric_and_schema_version(tmp_path):
    for extra, message in [
        ({"centre_fibre":"false"}, "centre_fibre"),
        ({"interp_num":None}, "interp_num"),
        ({"schema_version":999}, "schema_version"),
    ]:
        path=_config(tmp_path,extra)
        with pytest.raises(ValueError,match=message): load_working_config(path)
