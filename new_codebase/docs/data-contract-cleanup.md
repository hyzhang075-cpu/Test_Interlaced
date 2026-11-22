# 数据契约清理台账

起始分支 `AlgorithmPackage-08B5`，提交 `5448684`。两份未跟踪数据目录独立保留。适用范围内未发现 AGENTS.md 或已实施的改动。

| 阶段 | 定义和使用位置 | 引用性质 | 决定 |
|---|---|---|---|
| 1 | `contracts/__init__.py` 中的 `CurveInput`、`EvaluationSnapshot`、`OptimizationState`、`RuntimeState` | 只有定义及接口文档 | 删除；无需要迁移的测试 |
| 2 | `CalibrationSeed` 由 `initialization/service.py` 构造，`pipelines/session.py` 解包；契约和绘图测试直接构造 | 生产、测试、文档 | 拟合字段放入 `InitializationResult`；任务组装完整参数 |
| 2 | `TaskContext` 由 `pipelines/tasks.py` 构造和解包，`pipelines/__init__.py` 导出工厂；模块 4/5/6 的旧适配器曾读取 `.context.session` | 生产、兼容、测试、文档 | 任务直接传会话，保留旧引擎适配 |
| 3 | `runtime/service.py` 的 `SessionState`、`runtime/demo.py` 的 `LiveProcessingState`、`contracts` 的 `ReconstructionState`、`reconstruction/temporal.py` 的 `TemporalCurvatureState` | 生产、测试、文档 | 四者保留。服务需要容纳任意处理器的状态；两种模块 5 历史分别对应补点/积分网格。清除处理器和方向选择器的跨帧副本 |
| 4 | `PreparedCurvature`、`ShapeResult` 的状态/诊断/标架；`VisualizationInputs` 的重复几何和多余字段 | 生产、测试、文档；UDP、记录和 tools 消费原字段，旧入口提供元组 | 保留具名字段，移除状态/诊断/元组副本；时间滤波前字段在记录边界转成回放约定名 |
| 5 | `PreparedFrame` 由 `InputService`、`TaskSession.frame` 生产，由 `CurvatureService`、`DirectionSelector` 消费；`CalibrationDataset` 由模块 1 批量加载 | 生产、测试、文档 | 两类保留，生产帧共享固定几何的只读视图，方向选择直接消费已预处理帧 |
| 5 | `CurvatureResult`、`PreparedCurvature` 与八个配置类 | 生产、测试、文档 | 保留真实点和积分网格的不同约束；独立配置仍按职责投影 |

## 验证

- 改动前从 `new_codebase` 运行 `python -m pytest tests -q`。默认临时目录的父目录缺失，首次结果为 351 通过、2 失败、142 个准备错误。创建已配置的目录后重跑，493 通过、2 失败，均为 base NumPy 1.26.4 无 `np.trapezoid`。另外一套本地环境为 NumPy 1.24.4 且没有 pytest。
- 阶段 1：包导入成功，任务、兼容及冻结数值回归共 79 项通过。生产实时状态仍由模块 8 持有。基线两项失败定位到新库独立回放工具，改用既有 SciPy 依赖的同一梯形公式，回放工具测试 3 项通过。
- 阶段 2：初始化快照原有 JSON 字段、文件名和阶段状态继续由任务层写入。resume 与包合并读取的是参数字段和阶段结果，未序列化内部拟合契约。从 `new_codebase` 设置 `PYTHONUTF8=1` 后执行 `python -m pytest tests -q`，495 项通过，包含独立 wheel 构建、隔离安装及数值运算。隔离 Python 子进程使用显式 UTF-8，以匹配中文帮助输出。
- 阶段 3：运行、独立模块与数值回归用例 79 项通过。相同点数但坐标改变的同一会话网格通过均值重启回归；方向展开、Kalman 为各自上一帧保存相应网格。默认和方向混合 demo 用模拟源验证，两者均不读取标定 CSV。
- 阶段 4：运行、旧协议、记录、冻结数值、模块契约与绘图共 129 项通过。记录测试写入并重新读取 pickle，UDP 测试检查发送到套接字接口的 JSON；绘图测试实际写出 PNG，覆盖中心通道、非默认列顺序、单通道、空实验类型。
- 阶段 5：静态与动态生产帧通过统一工厂绑定 `[P,C]` 信号和只读几何视图；单独调用模块 1 仍可由其配置生成设计。固定几何视图由测试验证，原有曲率、方向、重构位置及误差断言使用同一冻结参考。最终从 `new_codebase` 设置 `PYTHONUTF8=1` 后执行 `python -m pytest tests -q`，**497 项通过，耗时 87.50 秒**。

各阶段的针对性命令均在 `new_codebase` 执行：

```powershell
python -m pytest tests/test_task_pipelines.py tests/test_packaging_acceptance.py tests/test_exact_modular_reconstruction.py -q --basetemp=artifacts/testing/tmp-stage1 --tb=short
python -m pytest tests/test_replay_end5cm_groupwise.py -q
python -m pytest tests/test_runtime_service.py tests/test_exact_modular_reconstruction.py tests/test_module_split_acceptance.py -q --basetemp=artifacts/testing/tmp-stage3
python -m pytest tests/test_runtime_service.py tests/test_exact_modular_reconstruction.py tests/test_module_contracts.py tests/test_visualization_service.py tests/test_demonstrator_anchor_pipeline.py tests/test_dynamic_reconstruction_and_wavelength.py -q --basetemp=artifacts/testing/tmp-stage4 --tb=short
```

## 真实数据及交付验收

- 三方向 09B7 真实 CSV 初始化以独立验收配置读取原数据，生成 6 张 PNG、3 段 MP4。最终代码用同一数据复验 0°，生成 2 张 PNG、1 段 MP4，FFprobe 确认视频为 H.264、180 帧。
- 初始化 JSON 经 `--resume` 恢复，使用真实 09B7 全段 0° 数据跑完两阶段优化。验收设置为 `--max_iter 1 --hyper_params "[1,1e-6,1e-6]"`。合并 init 与 optim 快照后重新加载生成的 schema v1 包，确认初始化独立保存，优化结果可用于构造生产重构会话。顺带读取仓库中既有 schema v1 与 format v2 包。
- 验收产物分别写入 `artifacts/testing/history/contract-cleanup-20261122-121511`、`contract-final-20261122-122312`、`contract-resume-20261122-122735`。首次使用默认 Powell 内层迭代上限的重复优化在有界验收成功后停止；该早期尝试的 manifest 仍标为 running。
- 实时/回放覆盖模拟源、多会话、重置、帧处理失败和网格变化。软件链路验证包括最终坐标、姿态、UDP JSON 和 pickle 记录。实际解调设备、IMU 和远端 UDP 接收端需要硬件联调。

## dc_ss 兼容验收

运行基线为 `dc_ss` 现有的 Python 3.8.20。源码和 wheel 元数据均支持 Python 3.8。测试工具独立放在 `artifacts/testing/py38-test-deps`，使用 `dc_ss` 执行 `python -m pytest tests -q`，497 项通过。真实 09B7 全段初始化成功，写出 6 张 PNG、3 段 MP4 和初始化 JSON。三个方向的相位、权重及斜率与上次真实数据验收结果在 1e-12 容差内一致。验收产物位于 `artifacts/testing/history/dc-ss-py38-20261122-132701`。

## 多区域初始化与优化续算验收

`init --multi_region_calib True` 使用区分区域与方向的快照，供批量或单区域优化恢复。已在 `dc_ss` 使用真实 09B7 配置验证。从 `artifacts/testing/history/mi-134936` 的批量初始化快照恢复，三个方向的限迭代优化共完成九项区域/方向任务。逐项核对优化输入相位、权重和斜率与初值一致，并合并、读取了验收参数包。真实初始化验收保留了 PNG，省略 MP4 编码。额外验证了批量快照恢复 end5cm 240° 单区域。全量回归 500 项通过。
