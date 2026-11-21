# 八模块拆分计划与实施台账

## 确认的边界

| 模块 | 维护职责 | 对外数据/依赖 |
|---|---|---|
| 1 输入预处理 | 静态读取、设备解包使用的输入核、通道排布、中心补偿、真实点设计 | PreparedFrame / CalibrationDataset / DesignGeometry |
| 2 初始化拟合 | 显式拟合相位和分段扭转初值，保留原拟合逻辑和注释 | 接收 Dataset，调用3的公开 GeometryService，返回 InitializationResult |
| 3 原位物理换算 | 波长信号/应变到真实采样点的曲率大小和方向、标定参数选择 | CurvatureResult；不修改点数/距离/顺序，不做重构滤波/插值 |
| 4 参数优化 | slope/weight搜索、目标函数、冻结和覆盖率策略 | 明确输入初值；通过回调调用3、5、6，不调用2 |
| 5 重构 | 曲率重构用滤波、方向展开、虚拟点、插值、时间滤波数学、积分与坐标变换 | PreparedCurvature / ShapeResult；跨帧状态显式传入返回 |
| 6 理想曲线及误差 | 理想形状/曲率、原逐点/末端/螺距/半径指标 | EvaluationReference → EvaluationResult |
| 7 可视化 | 拟合诊断、静态图、理想曲线叠图、原MP4保存及VTK显示 | 显式VisualizationInputs与曲线/理想结果 |
| 8 实时编排 | 采集生命周期、会话状态、队列、窗口、IMU、锚定、记录和通讯 | 编排1→3→5→7，最终帧统一交给各消费者 |

模块3的评分窗口只用于选择标定等级，不平滑其对外曲率；模块5才改变为重构服务的曲率表示。虚拟几何起点与存储零波长参考互相独立；后者仅demo可选启用。

## 任务调用图

- 初始化：1 → 2（调用3的基础物理接口）→ 3 → 5 → 6 → 7。
- 优化：1 → 4；4通过明确回调调用3 → 5 → 6。所有区域先校验初值，不隐式运行2。
- 重构：1 → 3 → 5 → 6 → 7。
- 实时：8管理源/状态，逐帧调用1 → 3 → 5，7负责显示。

服务构造不读取标定CSV、不拟合、不连接设备。TaskSession负责保存明确的配置、数据与参数，并组织服务；不再持有一个跨模块继承的Optimiser作为生产引擎。公开便捷API create_engine(task='init')会显式执行initialize，区别于TaskSession/服务的无I/O构造。

## 已实施步骤

| 阶段 | 实施内容 | 验收依据 |
|---|---|---|
| 1 基线和契约 | 冻结真实09B7三方向27个圆弧结果；显式配置与数据DTO | tests/fixtures/module_split_baseline；contracts；接口清单 |
| 2 重构和评估 | 原数值正文迁入5/6，比较各种插值、向量滤波、两种标架与时间滤波 | test_exact_modular_reconstruction及原向量滤波回归 |
| 3 原位物理模型 | 模块3单独接收设计、帧和参数，不补点、不展开、不插值 | 主窗口真实点/输入不变/原数值验收 |
| 4 输入、拟合、优化 | 复用原核，显式数据加载、初始化和候选回调，原初值不受候选污染 | test_input_service、test_module_split_acceptance与优化回归 |
| 5 图形及实时 | 原叠图/动画保留；会话状态、设备、记录/UDP统一处理 | test_runtime_service、test_visualization_service |
| 6 任务和兼容边界 | CLI/API切到TaskSession；旧Sensor/Initialiser/Reconstructor/Optimiser仅兼容，数值核共用 | 独立生产导入图、区域/续算/发布/独立wheel验收 |
| 7 文档与最终验收 | 完整工作流、实际数据接口和主窗口验收记录 | module-data-interfaces.md；module-split-acceptance.md |

易实施工作由较低等级模型执行，本窗口独立核对架构和真实使用路径。数值核保留原注释，新增边界说明输入、输出和状态。外部YAML字段名、选项和模板解释沿用；工作配置、生成结果和发布包仍分离管理。

## 后续维护规则

新功能修改本项目对应模块。历史兼容类不再复制一套数值正文；新生产服务不导入兼容继承链，也不读取legacy_codebase。兼容工具可调用新库的兼容入口，真正算法只维护其归属模块中的实现。请勿把旧的context.engine或通过__dict__/__new__注入跨模块状态的做法引入生产service。

实际接口、单位、状态所有权与示例见 [模块数据接口](module-data-interfaces.md)；操作命令见 [使用说明](../shape_sensing/README.md)。
