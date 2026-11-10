# 算法模块封装与工作配置

## 独立库结构与兼容入口

`shape_sensing` 已包含八模块的实际函数实现，输入、初始化、曲率、优化、积分、评估、绘图与实时系统分别在对应子包维护。原算法函数按源文本迁移，保留逻辑和注释；根目录旧模块仅反向转发到新库。独立安装、模块归属、纯数组积分和程序化调用见[独立库使用说明](standalone-library.md)。`shape_sensing.cli` 负责参数解析，将任务交给 `shape_sensing.pipelines`。

`LegacyContext` 每次任务持有独立 Optimiser。配置和纯适配模块导入不创建窗口或连设备；`demo` 才会显式创建窗口、启动接收线程，退出时关闭 runtime。

## 工作配置

从 `configs/sensor_template.yaml` 复制一份到实验目录。模板保留 main parser 的110项参数元数据，其中 task 和参数包路径作为入口控制说明，不作为固定 Sensor 参数。`schema_version: 1` 是当前 YAML 版本。设置 `sensor_id`、`R`（m）与 `samp_dist_preset`（mm），并按传感器修改区域和设备参数。weights/slopes 可以保留 `null`，表示尚无最终标定值。

```powershell
Copy-Item configs/sensor_template.yaml configs/my-sensor.yaml
# 编辑 my-sensor.yaml 后执行
python main.py --config configs/my-sensor.yaml --task init
python main.py --config configs/my-sensor.yaml --task optim --direct 0
python main.py --config configs/my-sensor.yaml --task recon --direct 0
```

优先级为 parser 通用默认 < YAML < 显式 CLI。配置拒绝重复 YAML key、未知参数、类型错误、非法 choices、错误的区域嵌套和缺失传感器设计字段。数值 0、布尔 false、允许为空的 null 会保留。资源路径相对配置中的 `calibration_data_root` 解析；根路径本身相对 YAML 文件。

旧 `--sensor_params` 的 `format_version=2` JSON 包继续走旧 loader。新包也可通过 `--sensor_params` 加载；单区域任务按 region/direct 选择参数，包含 full/end15cm/end5cm 且覆盖 0/120/240 的完整包可用于多区域重构。新包验证状态默认为未验证，不能作为部署门禁通过证明。

## 任务快照、续算与发布

每个任务在 `runs/<sensor_id>/<run_id>/` 保存 `manifest.json`、`config_resolved.json` 和初始化/优化/结果阶段 JSON。manifest 记录运行时 Git commit、分支和工作区 dirty 状态；发布包沿用每个源 run 的记录。配置快照取解析后生效的参数并记录配置文件、CLI 字段和续算来源。成功和失败都会写入状态。`--resume <run目录或阶段JSON>` 必须显式给出；加载前比较传感器/几何布局和所选区域配置。命令行 weights/slopes/phase 初值优先于续算结果；weights-only 需要 slopes。

从成功的 optim 任务快照发布；每个 region/direct 独立保存，重复组合会报错，不会覆盖。初始化 run 可以导出为初值包，但会标记 `final_calibration: false`。存在 `zero_reference_path` 时导出工具会复制该资源并改写包内引用。

```powershell
python tools/export_algorithm_package.py --runs runs/08B5/<optim-run> --output packages/08B5.json
python main.py --sensor_params packages/08B5.json --task recon --optimization_region full --direct 0
```

真正的包含 full/end15cm/end5cm 及 0/120/240 方向的完整新包会组装旧 Reconstructor 可归一化的多区域结构。多区域包要求三方向与所有区域完整。旧 format_version=2 包仍由现有 `tools/sensor_parameter_package.py` 生成和加载。跨机器迁移仍需复制标定数据目录并更新工作 YAML 的数据根；参数 JSON 不携带庞大的原始标定数据。

## 尚存边界

兼容说明：`global_phases_set=0` 是合法相位值，只有 `None` 表示缺省并触发拟合；旧代码曾用真假判断把零相位当作未提供，本轮单独修正此行为（不改拟合公式）。`interlaced_offset` 名称沿用旧参数，但单位是 rad；旧 parser help 曾误写“度”，模板和当前说明已更正，默认 `π/3` 不变。

数值实现和绘图/设备实现已迁入新库。为保持原初始化与计算行为，部分算法类仍通过无构造器的职责 mixin 组合状态；`LegacyContext` 名称继续兼容已有调用。没有运行真实设备或生产数据优化；设备动态链路已通过离线/假对象验证。旧 format_version=2 多区域包继续兼容；完整的新格式包也支持多区域重构。
