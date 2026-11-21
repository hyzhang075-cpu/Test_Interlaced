# 独立形状感知算法项目

本目录包含运行所需的算法代码、工具、配置、校准数据、参数和输出。主入口是 `shape_sensing/main.py`。

## 运行入口

从本目录执行：

```powershell
python shape_sensing/main.py --help
python shape_sensing/main.py --config configs/09B7_NeTi.yaml --task init
```

也支持 `python -m shape_sensing ...`。本目录的 `main.py` 保留为便利入口，实际任务实现位于包内。源码脚本入口将相对路径锚定到本项目根目录；独立安装后的包在用户工作目录解析运行路径。

初始化、优化与部署重构使用独立的模块服务，任务顺序为：

```powershell
# 1. 读取标定数据，拟合相位与扭转初值，并在 runs/<sensor_id>/ 保存初始化快照
python -m shape_sensing --config configs/09B7_NeTi.yaml --task init

# 2. 从该次 init 的目录恢复初值，执行参数搜索（把下面路径替换为实际 run 目录）
python -m shape_sensing --config configs/09B7_NeTi.yaml --task optim --resume runs/09B7_NeTi/<init-run>

# 3. 使用已导出的参数包进行静态重构或实时演示
python -m shape_sensing --sensor_params packages/<algorithm-package>.json --task recon
python -m shape_sensing --sensor_params packages/<algorithm-package>.json --task demo
```

多区域标定时，先按区域分别初始化：在 `init` 命令中指定 `--optimization_region full`、`end15cm` 或 `end5cm`，并保持 `--multi_region_calib False`。随后每个区域分别用对应的初始化 run 执行三个方向的 `optim --direct 0/120/240 --resume <该区域init-run>`，最后合并九次成功优化的 run。各区域采样点和分段不同，全长初始化的参数不能直接用于局部区域。只有 `calibration_regions` 中每个区域都已提供匹配的初始 `weights`、`slopes`、`global_phases_set`，或从完整的多区域优化报告恢复时，才可直接运行批量多区域优化；系统会在读取标定 CSV 前校验全部区域初值。

`optim`、`score`、`recon`、`demo` 不会隐式拟合初值。缺少 `weights`、`slopes` 或 `global_phases_set` 时会在任务开始报错；可通过 `--resume` 恢复匹配的初始化/优化快照，也可显式提供参数。实时演示只需设计参数、最终标定参数和实时数据流，不读取标定 CSV；零点参考仍为可选项，只有启用 `--use_stored_zero` 时才在启动数据源时读取。

公开模块服务的构造不读文件。`InputService.load_data()` 显式加载静态数据；`TaskSession.process_frame()` 是实时与无窗口回放共用的单帧计算入口。模块接口见 [模块数据接口](docs/module-data-interfaces.md)。

完整工作流程见 [传感器算法使用说明](shape_sensing/README.md)。 本次拆分的核查结果与现有局部配置限制见 [验收记录](docs/module-split-acceptance.md)。

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

## 测试产物与日志

在本目录运行 `python -m pytest -q`。pytest 缓存写入 `artifacts/testing/cache/`，运行临时文件写入 `artifacts/testing/tmp/`（下次测试会重建此临时目录）。
原顶层 pytest 目录完整归档在 `artifacts/testing/history/`，原顶层日志保存在 `artifacts/logs/`；移动清单为 `artifacts/testing/move-manifest.json`。这些目录不纳入版本管理，历史文档中的原路径保留用于追溯。
