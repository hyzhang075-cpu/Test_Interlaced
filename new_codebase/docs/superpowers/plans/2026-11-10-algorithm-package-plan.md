# AlgorithmPackage-08B5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 用户已指定独立较低模型聊天执行，主聊天验收，不再申请执行方式。

**Goal:** 在 AlgorithmPackage-08B5 本地分支完成可运行的八模块适配、集中 YAML 配置、运行结果与发布包，同时保持现有算法及注释。

**Architecture:** 不复制数值算法。先围绕原类建立公开模块接口和任务上下文，再通过新配置解析和任务流水线接入旧函数；旧 CLI 和 JSON 包保持兼容。公开接口需有真实职责，不能只建目录或全用同一个无区别代理。

**Tech Stack:** 现有 Python/NumPy/SciPy/PyQt5/VTK/pytest；新增显式 PyYAML 依赖（当前环境 6.0.2 可用）。

**Spec:** docs/superpowers/specs/2026-11-10-algorithm-package-design.md

## Global Constraints
- 仅在 E:/shape_exp/支气管镜代码 的 AlgorithmPackage-08B5 分支工作；不推送，不移动或纳入原有未跟踪实验数据。
- 参数名字/解释/choices 尽量逐字保留；不重命名 discont/R/rot_ang 等原字段。保留原函数注释与算法顺序。
- 原有类先保留，适配层调用单一实现。不能改成整包导入即要求 GUI。
- 新增 YAML 不依赖 main 的 GUI 导入；必要时提取原 parser 定义并在 main 保留兼容导出，原注释连同移动。
- 每次任务有独立上下文；配置与运行状态分离。数组初值不是标定完成证明。
- 使用项目内唯一 basetemp 避开系统 Temp 权限；任何失败都报告真实信息，不放宽测试或删除已有测试。
- 每个任务完成后更新本计划与 implementation-progress.md，列变更和检查。不要自动提交实验数据。

## Review Focus
1. YAML false/0/null/重复键/未知字段不会被静默变成默认值（Task 1）。
2. 新传感器缺设计字段、区域阈值和分段不一致有清楚错误，而旧 CLI 保持兼容（Task 1, 3）。
3. 八模块不是不可执行占位，且不因导入计算接口触发 Qt/设备（Task 2）。
4. 运行中更改 YAML/默认值不能改变旧结果或发布包；并行方向输出不覆盖（Task 4, 5）。
5. 单方向、局部重优化、真实点目标和固定上游相位路径保持支持范围及失败语义（Task 3, 5, 6）。

## Task 0: 基线及执行约定
Files: docs/superpowers/plans/2026-11-10-algorithm-package-plan.md; docs/superpowers/plans/implementation-progress.md
- [ ] 阅读 spec、main.py、calibration_batch.py、calibration_regions.py、tools/sensor_parameter_package.py、README.md，以及已有 tests。
- [ ] 查看 .pytest_cache/algorithm-package-baseline.log。基线若有代码失败先报告主聊天；系统 Temp 权限问题使用项目 basetemp 解决，不改算法。
- [ ] 写 progress 文件，记录原提交 e27b245、分支与基线计数。

## Task 1: 带原说明的集中配置和 CLI 兼容
Create: shape_sensing/__init__.py; shape_sensing/configuration/{__init__,parameters,loader}.py; configs/sensor_template.yaml; tests/test_sensor_configuration.py
Modify: main.py parser/load 部分; calibration_batch.py CONTROL_KEYS; requirements.txt
Interfaces:
- build_argument_parser() 保留现有 main 导出和函数签名，可迁移至 configuration.parameters。
- load_working_config(path: str | Path) -> dict 返回原参数名的已验证映射。
- parameter_catalog() -> dict 提供 dest/type/help/choices/default/分组，供模板一致性检查；无需调用 GUI。
- parse_main_args(argv=None) 保持旧语义；新增 --config、--resume 仅作为控制字段不得漏入 Sensor。
- [ ] 先写测试：旧 CLI 与相同 YAML 得到相同关键参数；每个有效原参数在模板或入口注释中有说明及 choices；R/点位/sensor_id 缺失时拒绝新配置。
- [ ] 测试 YAML false、0、null、显式 --weights None、重复键/重复参数、未知字段、非法 choices、字符串假布尔、嵌套区域；相对路径基于文件；--config 与 --sensor_params 同用拒绝。
- [ ] 最小实现：从 parser 提取参数定义且保留全部注释，或轻量 AST 目录配合现有 parser；模板全量带中文注释，默认不重写用户配置。给通用字段默认值，设计必填显式 null。
- [ ] 运行 test_sensor_configuration.py 及既有参数、区域、包测试，记录结果。

## Task 2: 八模块公开接口和任务上下文
Create: shape_sensing/contracts/__init__.py; shape_sensing/context.py; 八个模块的 __init__.py 与必要 service.py; tests/test_algorithm_modules.py
Interfaces (可用小型 dataclass，保持字段在所有后续任务一致):
- LegacyContext(engine) 保持单任务旧对象及状态；没有全局 engine。
- InputPreprocessor(context): read/receive/preprocess 以旧输入函数为后端，必须区分离线与实时执行。
- Initializer(context).fit(verbose=False, update_phase=True) 调 rot_seg_fit。
- CurvatureEstimator(context).estimate(...) 显式适配 _Strain_2_Cur；可保留旧参数名和签名。
- ParameterOptimizer(context).optimize(verbose=True) 调 optim_coeff。
- ShapeReconstructor(context): reconstruct_shapes(...)、integrate(...)，调用旧形状/积分函数并说明输入区别。
- Evaluator(context): ideal_shape(...)、evaluate(interp_dist, rs)，调用旧理想曲线和误差函数，返回清楚指标快照。
- Visualizer(context): 按需构造旧 Visualisor，公开绘图入口。
- RealtimeSystem(context): 按需构造 Demonstrator，封装运行和关闭；不得导入就连设备。
- [ ] 先写合成数据/假 engine 调用边界测试；曲率和积分选用现有 synthetic fixture 做新旧结果 allclose，对齐有效点与姿态。
- [ ] 实现八模块，保留旧类原实现，新增中文 docstring 说明维度、单位、状态、副作用、异常。无需为了目录整齐切断所有 legacy 继承。
- [ ] 测试不同任务 context 不共享候选参数；配置与纯曲率接口可在没有 Qt/VTK 模块时导入。
- [ ] 跑已有动态重构、向量插值、末端姿态和多区域测试，记录。

## Task 3: 六任务通过模块编排执行
Create: shape_sensing/pipelines/{__init__,tasks}.py; tests/test_task_pipelines.py
Modify: main.py main 部分; configuration 控制参数；原 helper 兼容导出
Interfaces:
- execute_task(args) -> 结果/旧任务返回值，保留 init/ang/optim/score/recon/demo 旧行为与配置区域检查。
- build_context(args) -> LegacyContext，配置解析后构造，不在模块导入时读取数据。
- [ ] 先写假任务测试确认 init 不要求最终标定结果，optim 调优化，recon 调形状和评估，demo 假设备调用并保证资源释放；score 与多区域保持原 batch 行为。
- [ ] 最小迁移 main 中调度至 pipelines，用新模块接口真实组织；构造依赖先沿用旧 Optimiser，注明后续可逐步解除。保留 compute_error_ang/optim_ang 的兼容入口，避免循环导入。
- [ ] 保持真实点目标不强行调用几何积分，batch 相位依赖和现有警告行为不变。
- [ ] 跑旧入口、区域、优化状态、动态锚点测试。

## Task 4: 自动运行快照、结果与显式续算
Create: shape_sensing/artifacts/{__init__,runs}.py; tests/test_run_artifacts.py
Modify: pipelines/tasks.py; calibration_batch.py 保存配置与来源（保留现有 results schema）
Interfaces:
- create_run(sensor_id: str, task: str, resolved_config: dict, output_root: Path) -> RunStore，唯一目录，原子 JSON 写入。
- RunStore.write_result(result: dict, stage: str); finish(status: str, error=None)。保存 manifest/config_resolved，NumPy/Path 转原生值，禁止非有限 JSON 数值。
- load_resume(path, resolved_config) -> 参数初值映射；明确来源优先级：显式 CLI 初值 > 指定 resume > 工作配置初值/原初始化策略。
- [ ] 测试并行同传感器输出不覆盖，结果修改不污染配置，YAML 后续变化不修改旧快照；运行异常保存 failed 并继续抛原异常。
- [ ] 测试 init 保存初始相位/slope 和阶段，optim 保存选定结果，单区域续算及 weights-only 缺 slope 时拒绝；传感器/几何/阈值/分段不兼容 resume 拒绝。
- [ ] 对动态任务保存启动配置和结束状态，避免逐帧重复创建运行目录。
- [ ] 实现，不自动重写 YAML，不隐式选择最新结果；保留现有 calibration_results/results.json 消费者兼容。

## Task 5: 从快照导出自包含包并兼容旧包
Create: shape_sensing/artifacts/packages.py; tools/export_algorithm_package.py; tests/test_algorithm_package_export.py
Modify: main/config loading 对新包的显式格式分派；旧 tools/sensor_parameter_package.py 仅做必要适配
Interfaces:
- export_package(run_paths: list[Path], output: Path) -> dict。新包标记 package_type 与 schema_version，sensor_id、configuration、calibration、scope、provenance、validation 完整保存。
- load_algorithm_package(path) -> 已验证配置和选定标定结果；旧 format_version=2 继续走现有 loader，不能被当作新工作 YAML。
- [ ] 测试导出与当前 parser/YAML 默认值无关，导出后单方向/单区域 recon 选对参数，缺失方向拒绝错误范围；不伪造 deployment_ready。
- [ ] 测试跨传感器、布局不一致及固定上游相位来源冲突合并拒绝；只对真正共享依赖比较签名，不要求各方向 task/direct/数据路径全相等。
- [ ] 新包引用必要零点资源按相对路径明确携带或报错，包中 task 不覆盖本次用户 task。
- [ ] 检查旧包测试及 CLI 工具 --help。

## Task 6: 文档、全量回归与主聊天验收交接
Modify: README.md; docs/algorithm-packaging.md; implementation-progress.md; 本计划复选框
- [ ] 写清 A/B 八模块职责、原函数映射、注释规则、工作配置从零启动、初始化人工迭代、三方向优化、续算、发布、旧包兼容与配置复制迁移。
- [ ] 给可执行新 CLI 示例；不得写尚未实现的选项；清楚列出仍在 legacy 的实现和未进行的设备/真实数据验证。
- [ ] 使用唯一项目 basetemp 运行全部 pytest；保存 .pytest_cache/algorithm-package-final.log。执行 main.py --help、新导出工具 --help、compileall、git diff --check。
- [ ] 逐项检查原注释保留、模板 coverage、原数据未动、无临时输出被纳入改动、未改变核心数值算法。
- [ ] 在新聊天最终报告：任务完成项、变更清单、测试命令与计数、兼容限制、未完成项；主聊天将独立审查与验收。未经主聊天验收不要声称已最终验收。


## 主聊天最终交付状态

Task 0–6已按适配式封装范围完成并通过代码级验收。具体文件结构以实际实现和docs/algorithm-packaging.md为准，未强制将原函数物理搬迁；保留legacy类为本轮设计选择。
最终主聊天独立全量365项通过，入口/编译/差异检查通过，独立审查无剩余阻断项。执行明细见implementation-progress.md，最终证据见2026-11-10-algorithm-package-acceptance.md。早期步骤复选框是原始执行任务清单，完成证据以阶段记录和最终验收为准。
