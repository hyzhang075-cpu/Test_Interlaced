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

多区域标定时，运行 `init --multi_region_calib True`，一次按 `optimization_regions` 初始化配置中的区域。将同一个 init run 通过 `--resume` 传给三次 `optim --multi_region_calib True --direct 0/120/240`，每次优化一个方向的全部目标区域。单区域标定继续使用 `--multi_region_calib False --optimization_region <区域>`，可从单区域快照或多区域快照中恢复。详见[初始化与优化](shape_sensing/README.md#3-初始化与优化)。

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

### 11B8 交错刻栅实时 demo

使用已验收的芯位算法包（包含三方向和三个区域参数），在 PowerShell 中运行：

```powershell
Set-Location 'C:\Users\hyzha\Desktop\shape-sensing-AlgorithmPackage-08B5\new_codebase'
$env:PYTHONPATH = (Resolve-Path '..\.codex_deps').Path
python main.py --task demo --sensor_params packages/11B8_CoreCalib-full-v1.json
```

包内使用 FiberTecc，地址为 `192.168.1.75:26546`。启动采集时保持传感器直线：当前包未启用存储零点，首个完整的七通道原始快照作为零点。实时链路先对 20×7 原始 WL 信号归零，再按静态标定相同的中心芯样条插值与 A/B 通道合并得到 40×4 响应，最终输出 256 点形状。区域裁剪在交错合并之后执行；四芯模式保留中心通道。

若启用存储零点，零点文件必须描述完整原始 WL 网格和七个物理通道，不应使用四通道合并后的零点。Hippo 仍受单通道 16 个原始 WL 的协议限制，不能承载本套每通道 20 个 WL 的 11B8 数据。

验证：协议模拟覆盖 FiberTecc/Hippo、首帧/存储零点与裁剪；`tests/test_interlaced_runtime.py` 验证参数包从七通道原始帧到形状重构的链路，不连接硬件。实际设备通信与 VTK/OpenGL 窗口显示需要在有设备和可用图形上下文的机器上验证。

### 实验脚本清理

根目录的单次优化过程脚本及工具目录中的传感器专项优化实验脚本已移除。算法主体、测试和正式打包/校准/回放工具保留；最终 YAML 配置及参数 JSON 保留在 `configs/`、`params/` 和 `packages/`。历史脚本仍可从 Git 提交历史中查阅。
