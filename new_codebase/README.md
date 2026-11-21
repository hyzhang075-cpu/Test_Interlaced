# 独立形状感知算法项目

本目录包含运行所需的算法代码、工具、配置、校准数据、参数和输出。主入口是 `shape_sensing/main.py`。

## 运行入口

从本目录执行：

```powershell
python shape_sensing/main.py --help
python shape_sensing/main.py --config configs/09B7_NeTi.yaml --task init
```

也支持 `python -m shape_sensing ...`。本目录的 `main.py` 保留为便利入口，实际任务实现位于包内。源码脚本入口将相对路径锚定到本项目根目录；独立安装后的包在用户工作目录解析运行路径。

完整工作流程见 [传感器算法使用说明](shape_sensing/README.md)。

## 文件归属

| 目录 | 内容 |
|---|---|
| `shape_sensing/` | 八模块实际实现、包内 main 入口、配置加载与任务编排 |
| `tools/` | 本项目独立工具副本，允许单独维护 |
| `configs/` | 传感器工作配置 |
| `data/` | 本项目使用的原始校准数据独立副本 |
| `params/`、`packages/` | 参数文件与发布包独立副本 |
| `runs/`、`calibration_results/` | 运行快照与分区域校准结果 |
| `outputs/` | 可视化输出，按传感器分目录 |
| `tests/`、`docs/` | 独立测试、使用说明和迁移记录 |

`configs/09B7_NeTi.yaml` 的数据根相对 YAML 目录指向 `../data/`，运行输出指向 `../calibration_results`。新代码和工具不导入或读取旧目录。历史报告保留源信息，用于追溯。

## 安装与分发

```powershell
python -m pip install -e ".[gui,runtime]"
python -m pytest -q
python -m pip wheel . --no-deps --wheel-dir dist
```

从本目录安装后，其他项目可导入 `shape_sensing`。运行记录、参数包、原始数据和设备配置按使用说明管理。旧项目如需调用新库，可安装本项目后使用兼容入口。

迁移映射见 [MIGRATION_DATA.json](MIGRATION_DATA.json)，原始数据副本逐文件校验见 [RAW_COPY_SHA256.json](docs/RAW_COPY_SHA256.json)。
