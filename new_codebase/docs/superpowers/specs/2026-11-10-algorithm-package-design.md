# AlgorithmPackage-08B5 设计

日期：2026-11-10。用户已要求按本设计执行，在较低模型独立聊天实现，本聊天检验验收。

## 目的
在本地 AlgorithmPackage-08B5 分支实现渐进式八模块封装，保留原算法、函数签名、中文注释和旧入口。原基准 integrate-08B5/e27b245；09B7/ 和 1016-09B7-自研（1+3）22点86cmFBG+镍钛管全末段定标测试/ 是原有未跟踪实验数据，不添加、不改动。

## 选定方案与约束
采用单仓库、一个可编辑 YAML 配置、自动运行结果、JSON 发布包。相较独立发布多包或多仓库，此方案减少跨模块联调成本。用户决定 A 负责输入、初始化、曲率、优化，B 负责重构、评估、可视化、实时系统。
sensor_template 保留 main.py 所有有效参数的原名、解释、choices、默认行为和区域配置；新传感器专属必填字段不给其他传感器的隐式默认值。task 与配置路径为入口控制项，可以在模板注释列举而不存为固定任务。weights/slopes 可提供 null 初值兼容字段，但初始配置没有最终标定结果；初始化估计与优化完成结果明确区分。原有历史注释和算法 docstring 不删除。
不重新设计数值算法，不安装或启动真实设备，不自动执行耗时真实优化，不修改原数据，不推送远端。代码迁移与算法修复分离；例如已发现 phase=0 的旧真假判断不得悄悄混入无行为变化封装。

## 架构
shape_sensing/configuration 提供参数目录、严格 YAML 加载、现有 parser/参数包适配；shape_sensing/contracts 提供点位、曲率、形状与结果约定。
八个包分别为 input_preprocessing、initialization、curvature、optimization、reconstruction、evaluation、visualization、runtime。公开适配类调用单一原实现，不复制数值算法。每个任务独立 legacy context，避免优化候选状态污染其他任务。Qt/VTK 仅在运行需要时导入；导入配置或纯算法包不需要 GUI。
pipelines 负责 init/ang/optim/score/recon/demo 编排，main 保留命令行及必要兼容导出。迁移时可继续保留整体 _reconstruct_frame：需清楚说明接口覆盖范围及状态边界，不能把所有适配类都做成相同空壳。

## 参数和结果
configs/sensor_template.yaml 为带完整中文解释的模板；配置可按现有 PARSER_GROUPS 分组，calibration_regions 是区域参数唯一权威来源。沿用参数名字，只在适配层做单位/类型/方向字典键转换。
旧 CLI 未提供 --config 时保持原行为；新 --config 使用 defaults < 文件 < 显式 CLI（仅通用默认可补，传感器必填缺失要拒绝）。不同时隐式混合 --config 和 --sensor_params；若一起给出先明确拒绝。null 是有含义的显式值，不等同缺失，零和 false 必须保留。未知字段、非法 choices、重复参数、方向数组布局不一致需报含字段路径错误。YAML safe_load 并拒绝重复 key。相对路径以配置文件位置为基准，零点文件等资源同样处理。检查任务所需区域/方向，不强制所有任务都拥有三方向最终标定。
runs/<sensor_id>/<run_id>/ 每次唯一新建，保存 manifest.json、config_resolved.json、initialization.json 或 results.json、诊断图位置及失败状态。不重写人工 YAML，不隐式载入最新结果。支持显式 --resume 从初始化或优化结果续算，记录来源并验证设计、区域、阈值与参数布局的兼容性。
发布包自包含配置快照、选定结果、有效区域方向、代码版本、验证状态与资源引用；从保存的运行快照生成，不读取当前 parser defaults。未验证就是未验证，不能合成验证通过。现有 format_version=2 JSON 继续可读。不同传感器复制配置不复制结果；局部合并检查上游相位和布局来源。

## 注释要求
新增公开接口有中文 docstring：作用、输入输出形状、单位、状态、副作用、异常及为何复用旧函数。新增算法需写方法依据、关键步骤、参数意义、适用限制。封装本轮原则上不新增数值算法。保留旧文件有价值注释；若移动整段函数，注释一起迁移并保留旧入口转发。

## 验收
配置新旧路径等价；有效参数/说明/选项覆盖；合法零、false、null；无结果配置能初始化；旧包兼容；结果快照不受后续 YAML/parser 修改影响；八模块接口真实调用旧实现；数值曲率/形状/姿态等价；动态可用离线回放与假设备验证，不宣称设备实测。全部已有测试及新增测试通过，环境问题与代码问题分别报告。最终记录限制和仍在 legacy 中的实现位置。
