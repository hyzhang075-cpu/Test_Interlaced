# 模块数据接口

本文记录当前实现的接口，不是迁移计划。单位约定：长度 m（配置中以 `*_mm` 标出的除外）、曲率 1/m、算法角 rad、方向编号/direct 与旋转配置角度 deg、信号采用历史应变数值尺度。服务按各自说明复制输入并执行必要校验；结果 DTO 是冻结 dataclass，但其中的 NumPy 数组通常仍可写，调用方应视为只读。`PreparedFrame.from_design` 返回设计距离和索引的只读视图。`P` 表示原始真实采样点，`Q` 表示补齐虚拟点后、插值前的点数，`N` 表示模块5积分网格点（虚拟点或插值可改变长度），`C` 表示通道数。

## 八模块使用的类与关系总览

服务创建或持有其他对象属于组合，参数或回调调用属于依赖；只有类声明中的基类才是继承。数据对象放入另一个对象的字段属于数据组合，不是 Python 的类内嵌套定义；这里的主类并非通过 `class` 内再定义 `class` 来组织。

| 模块（包） | 公开服务与算法/辅助类 | 真实继承、组合或依赖 |
|---|---|---|
| 1 输入与预处理（`input_preprocessing`） | `InputService`（定义于 `__init__.py`）；`InputKernel`、`WavelengthContinuityGuard` | 服务在方法内创建 `InputKernel`；连续性保护器用于采集质量处理。核对象不是跨模块数据 DTO。 |
| 2 初始化（`initialization`） | `InitializationService`、`InitializationKernel`；依赖模块3的 `GeometryService` | 服务创建初始化核，核的 `geometry` 字段持有 `GeometryService`；通过服务调用物理几何计算，不继承模块1/3服务。 |
| 3 真实点曲率（`curvature`） | `CurvatureService`、`CurvatureKernel`、`GeometryService`、`DirectionSelector`、`_ScoringKernel`；`CurvatureModelMixin`、`StrainGeometryMixin`、`SegmentGeometry` | `CurvatureKernel` 多继承三个上述 mixin/几何类；`_ScoringKernel` 继承 `CurvatureModelMixin`。服务按调用创建核；实时处理器组合 `DirectionSelector`，后者接收同一真实点帧及上帧方向质量，通过回调计算基础曲率并返回混合参数和新质量。 |
| 4 参数优化（`optimization`） | `OptimizationService`、`OptimizationKernel` | 服务继承本模块 `OptimizationKernel`；持有预测、评估、曲率、评分回调，依赖任务编排调用3/5/6，不继承重构器。 |
| 5 重构（`reconstruction`） | `ReconstructionService`、`_ConfiguredPreparation`、`CurvaturePreparationKernel`、`ShapeIntegrator`、`ShapeIntegrationMixin`；状态类 `TemporalCurvatureState` | `_ConfiguredPreparation` 继承 `CurvaturePreparationKernel`；`ShapeIntegrator` 继承 `ShapeIntegrationMixin`。服务创建预处理核/积分器；时间滤波函数接收并返回状态，生产实时链由模块8持有它。 |
| 6 评估（`evaluation`） | `EvaluationService`、`_EvaluationKernel`、`EvaluationMixin`、`IdealCurvatureMixin` | `_EvaluationKernel` 多继承两个 mixin；服务创建评估核，接收已有形状和显式参考，不继承模块5。 |
| 7 可视化（`visualization`） | `VisualizationService`、`Visualisor`、`PlotSegments`、`VTKViewMixin`；输入数据类 `VisualizationInputs` | 服务持有 `VisualizationInputs`，绘图时创建 `Visualisor`；后者组合 `PlotSegments` 的分段方法。`VTKViewMixin` 为窗口复用的显示基类，`Visualisor` 并未继承它。 |
| 8 运行时（`runtime`） | `RuntimeService`、`LiveFrameProcessor`、`InterrogatorSource`、`RuntimeWindow`、`UdpTransport`、`protocol.DataRecorder`、`RecorderAdapter`、`ImuSource`、`EntranceAnchorConstraint`；协议 `FrameSource`/`FrameProcessor`；异常 `FrameRejected` | 服务组合处理器、源、锚定器、会话状态和输出回调；处理器持有 `TaskSession`，可组合 `DirectionSelector`。`RuntimeWindow(VTKViewMixin, QMainWindow)` 为真实多继承。两个 `Protocol` 是调用约定，均不是数据类；`FrameRejected(ValueError)` 是异常。 |

| 模块 | 输入配置类 | 参与交接的数据类（方向详见下表） |
|---|---|---|
| 1 | `InputConfig` | 输出 `DesignGeometry`、`PreparedFrame`、`CalibrationDataset`；输入信号本身仍为 `ndarray`。 |
| 2 | `InitializationConfig` | 输入 `CalibrationDataset`（含 `DesignGeometry`）；输出 `InitializationResult`，直接提供单区域相位、方向分段斜率和拟合诊断。 |
| 3 | `CurvatureConfig` | 输入 `PreparedFrame`、`DesignGeometry`、`CalibrationParameters`；输出 `CurvatureResult`；基础几何/评分接口还返回数组或元组。 |
| 4 | `OptimizationConfig`、部分 `CurvatureConfig` | 输入 `CalibrationDataset`、`CalibrationParameters`；输出候选/最终 `CalibrationParameters`；回调还传数组、树形字典和 `EvaluationResult`，阶段报告是字典。 |
| 5 | `ReconstructionConfig` | 输入 `CurvatureResult`、`DesignGeometry`、`ReconstructionState`；生成 `PreparedCurvature` 和 `ShapeResult`；返回下一帧 `ReconstructionState`，时间滤波另交接 `TemporalCurvatureState`。 |
| 6 | `EvaluationConfig` | 输入 `ShapeResult`、`EvaluationReference`，批量入口可输入 `CalibrationDataset`；输出 `EvaluationResult`；理想曲率/曲线仍为数组或树。 |
| 7 | `ViewConfig`（任务/demo用于窗口设置）；静态服务直接接收 `VisualizationInputs` | 输入 `VisualizationInputs`（含 `CalibrationDataset`/可选 `InitializationResult`）、`ShapeResult`、可选 `EvaluationResult`；旧动画入口仍接收曲线树和距离数组。 |
| 8 | 装配时消费 `InputConfig`、`ReconstructionConfig`、`ViewConfig` 和 CLI 参数；`RuntimeConfig` 是通用配置契约 | 生产帧 `RuntimeFrame`（含 `ShapeResult`、`AnchorConstraintResult`）；会话状态 `SessionState`、处理器状态 `LiveProcessingState`（含重构/时间滤波状态）。 |

`TaskSession`（`pipelines/session.py`）组合服务并持有任务数据；任务直接传递会话，`ModuleConfiguration` 组合八份模块配置。这些属于编排/契约设施，不是第9个算法模块；`LegacyContext` 属于旧引擎边界。各模块的实际交接如下。

多区域 `init` 复用优化任务的区域配置预检，为每个区域分别创建 `TaskSession`。一次运行的 `initialization.json` 以 `result.regions[region]` 保存 `parameters`、`errors`、`status` 和采样/分组 `layout`。相位单位 rad，分段斜率单位 rad/m，权重无量纲，方向键单位 deg。续算在读取优化 CSV 或启动搜索前，将各区域布局与配置逐项比较。单区域初始化继续写原扁平快照。09B7 的末端五点布局为真实点 18–22，对应三组权重索引 `[0,0,0,1,2]`。

## 跨模块数据交接表

箭头表示数据流/调用依赖，不表示继承。冻结 dataclass 不会使其中的数组、字典深度不可变；`SessionState` 本身还是可变 dataclass。私有核是服务内部计算对象，不应当列为跨模块 DTO。

| 来源 → 接收方 | 实际传递的类/非 DTO 值 | 接收方实际使用与边界 |
|---|---|---|
| 1 → 2 | `CalibrationDataset`，内含 `DesignGeometry` | `fit` 消费 `data` 信号叶子、`radii`、`channel_ids` 与设计距离/半径/通道排布/虚拟端点等；通过 `GeometryService` 完成几何计算。 |
| 1 → 3（几何也供5） | `PreparedFrame`、`DesignGeometry` | 3消费信号、距离、索引、通道数及设计几何；5另消费设计虚拟起止点。静态任务以 `TaskSession.frame` 封装 Dataset 的 `[P,C]` 叶子；动态模块 1 做完预处理后使用相同契约。两者复用设计坐标和真实点索引，方向融合直接复用当帧信号。 |
| 1 → 7 | `CalibrationDataset`，装入 `VisualizationInputs.dataset` | 绘图按 `design.active_channels` 为 `plot_data` 标注物理列（含中心通道）；同时读取 `data` 和 `radii`；`VisualizationInputs` 是模块7定义的数据组合。 |
| 1 → 4（经任务） | `CalibrationDataset` | 优化读取 `data`、`radii`、`design.sampling_distance_m/source_indices` 等构建本地搜索数据。 |
| 2 → 任务 → 3/4 | `InitializationResult` → `CalibrationParameters` | 任务层取拟合字段，按区域与方向展开等级及分组，显式 weights/slopes 优先。模块 3、4 和实时方向选择共用完整参数。 |
| 2 → 7 | `InitializationResult`，装入 `VisualizationInputs.initialization` | 绘图消费 raw/unwrap 角诊断、全局/独立拟合结果；缺诊断时初始化绘图报错。 |
| 3 → 5 | `CurvatureResult`，另传 `DesignGeometry` 和切向 `ndarray[3]` | 5消费 `distance_m/curvature_per_m/direction_rad/source_indices`；生成 `PreparedCurvature` 后在本模块积分。`PreparedCurvature` 不是3的输出。 |
| 4 → 3/5/6（经任务回调） | 候选 `CalibrationParameters`；`predict_shapes/evaluate_shapes/compute_curvature/score_inputs` 回调 | 预测经3/5返回位置树与距离数组；曲率回调返回 `(curvature, direction)` 数组；评分回调传基础量/等级分数数组；评估返回 `EvaluationResult`。并非全部包装成 DTO。 |
| 5 → 6 | 单形状 `ShapeResult`；批量路径为位置树、距离数组、曲率数组 | 单形状评估消费 `position_m/distance_m` 与 `EvaluationReference`；静态批量调用从形状提取 `position_m` 交 `evaluate_batch`，不传 ShapeResult 树。 |
| 5 → 7 | `ShapeResult`，或旧动画的 `rs` 曲线树与 `interp_dist` 数组 | `plot_shape` 读取 `position_m`；`show_reconstruction` 仍接收树/数组，并要求外传理想曲线。 |
| 5 → 8 → 下一帧5 | `ShapeResult`、`ReconstructionState`；实时滤波另传 `PreparedCurvature`/`TemporalCurvatureState` | 处理器返回 `(shape, LiveProcessingState)`；其中 `reconstruction` 保存方向展开状态、`temporal` 保存曲率滤波状态，再装入 `SessionState.processor_state`；不是传预处理核。 |
| 6 → 7/4；6 → 5（可选回调） | `EvaluationResult`；理想曲线树/数组；`distance -> curvature` 回调 | 7读 `metrics` 或显式 ideals，4读评估指标；强制理想幅值时任务把 `EvaluationService.ideal_curvature` 包成回调交5，而不是5继承评估核。 |
| 8 源/处理器 → 1/3/5 | `InterrogatorSource.read()` 返回已裁剪、历史尺度的采集差分信号 `ndarray[P,C]`；`LiveFrameProcessor` 调 `TaskSession.process_frame` | `FrameSource`/`FrameProcessor` 只是协议。源负责采集边界，1输出预处理帧；处理器可经回调传 `PreparedFrame/CalibrationParameters` 做方向选择，无需传设备/窗口对象给算法服务。 |
| 8 → 窗口/传输/记录（模块内外围） | `RuntimeFrame`，内含 `ShapeResult`、`AnchorConstraintResult` | `RuntimeWindow` 读取最终帧；`UdpTransport.publish` 接收帧；记录回调把帧拆为 `protocol.DataRecorder.save` 的时间戳、位置、姿态、信号、诊断、推进量参数。三路使用同一最终帧；`ImuSource` 向服务更新旋转数组。 |

## 历史适配与当前服务的区分

| 范围 | 旧类/适配类 | 与上表主链的关系 |
|---|---|---|
| 各包 `__init__.py` 的旧任务适配 | `InputPreprocessor`、`Initializer`、`CurvatureEstimator`、`ParameterOptimizer`、`ShapeReconstructor`、`Evaluator`、`Visualizer`、`RealtimeSystem` | 保留旧调用形式的适配入口：重构、评估、优化适配器可直接接收任务会话；旧引擎适配使用 `LegacyContext.engine`。 |
| `compat.py` 历史引擎链 | `input_preprocessing.compat.Sensor`、`initialization.compat.Initialiser`、`reconstruction.compat.Reconstructor`、`optimization.compat.Optimiser`；`runtime.compat.Demonstrator/DataRecorder` | 历史跨模块继承和旧窗口/记录实现留在兼容边界；当前八模块服务不沿这条继承链运行。 |
| `runtime/prototype.py` | `Demonstrator`、`DataRecorder` | 旧实验原型；当前 demo 的窗口、传输、记录器分别是 `RuntimeWindow`、`UdpTransport`、`runtime.protocol.DataRecorder`。 |

## 契约 DTO

`PreparedFrame` 表达一帧的信号、真实点身份和物理通道列。
`CalibrationDataset` 表达多形状、方向和半径的静态实验集合。
模块 3 对静态与动态都消费 `PreparedFrame`；设计几何由会话或数据集持有，帧可同时公开距离和索引的只读视图。
`CurvatureResult` 对应真实点，`PreparedCurvature` 对应可能包含虚拟点和插值的积分网格。

以下每张小表按字段列出实际 Python 类型/数组形状、物理意义与单位（纯软件字段明确标注）。`P` 为真实点数，`Q` 为补虚拟点后插值前点数，`N` 为积分网格点数，`C` 为通道数。

### `DesignGeometry`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `sensor_id` | `str` | 软件标识：传感器 ID；来自输入配置。 |
| `radius_m` | `float` | 感测光纤/外围纤芯到传感器中心的径向距离 R，m；不是被测曲线弯曲半径。 |
| `sampling_distance_m` | `ndarray[P]` | 当前区域/重构起点下的真实采样弧长坐标，m；局部裁剪区域已减去 `straight_trim`，不是全局坐标；全局身份由 `source_indices` 保持。 |
| `active_channels` | `tuple[int,...]` | 软件标识：激活的物理通道号，无物理单位。 |
| `channel_angles_rad` | `Mapping[int,float]` | 通道在截面内的方位角，rad；当前 `_design` 填 `{}`，不能假定已有角映射。 |
| `physical_start_m` | `float\|None` | 虚拟起点坐标，m；未启用起点时为 `None`。 |
| `physical_end_m` | `float\|None` | 虚拟末点坐标，m；未启用末点时为 `None`。 |
| `region_point_indices` | `Mapping[str,tuple[int,...]]` | 软件索引映射：区域名到真实点位置；契约无单位；当前未由 `_design` 填充。 |
| `segment_mapping` | `Mapping[str,tuple[int,...]]` | 软件索引映射：区域/分段关联；无单位；当前未由 `_design` 填充。 |
| `source_indices` | `ndarray[P,int]` | 软件索引：全局真实采样点 0-based 身份；不是距离或物理量。 |
| `straight_trim` | `float\|int` | 输入几何裁剪量；float 表示 m，int 是历史兼容语义“保留尾部点数”，两种语义不可混用。 |
| `active_angles_deg` | `tuple[float,...]\|None` | 外围通道的截面方位角，deg；`None` 表示由几何核使用默认排布。 |
| `interlaced_grating` | `bool` | 软件开关：是否采用交错光栅采样处理。 |
| `interlaced_offset_rad` | `float` | 交错光栅的方位偏移，rad。 |
| `samp_dist_preset_mm` | `tuple[float,...]` | 光栅轴向位置预设，mm。 |
| `samp_dist_preset_sub_mm` | `tuple[float,...]\|None` | B 组光栅轴向位置预设，mm；没有 B 组时为 `None`。 |

### `PreparedFrame`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `distance_m` | `ndarray[P]` | 真实点轴向弧长坐标，m；必须严格递增。会话生产帧为 DesignGeometry 的只读视图。 |
| `source_indices` | `ndarray[P,int]` | 软件索引：全局真实点 0-based 身份。会话生产帧为 DesignGeometry 的只读视图。 |
| `signals` | `ndarray[P,C]` | 解调信号/应变数值矩阵；列对应 `channel_ids`，按输入历史尺度保存，不在 `prepare` 中换算；信号由帧生产者交给当前帧。 |
| `channel_ids` | `tuple[int,...]` | 软件标签：每一列对应的物理通道号；唯一且与 C 相同长度。 |

### 标定参数 block（`CalibrationParameters.parameters_by_region[region]`）

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `weights` | `Mapping[direct,ndarray]` | 无量纲曲率等级权重。方向键为 deg。展平时按组优先，组内按等级排列，长度为组数 × 等级数。 |
| `slopes` | `Mapping[direct,ndarray[level,segment]]` | 各曲率等级、空间分段的扭转/动态角斜率，rad/m；`direct` 为方向角 deg。 |
| `global_phases_set` | `float` | 全局通道相位偏置，rad。 |
| `rot_ang` | `float` | 形状坐标绕 x 轴的安装/旋转角，配置约定 deg。 |
| `multi_region_parameters` | `Mapping[str,Any]\|None` | 软件参数包：多区域传递相位/方向等配置；无单一物理单位，子字段由多区域包定义。 |

### `CalibrationParameters`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `parameters_by_region` | `Mapping[str,Mapping[str,Any]]` | 软件索引：region 名到标定 block。region 与 block 内 direct（实验方向 deg）是两级不同索引。TaskSession 持有；模块4返回独立参数副本。 |
| `source` | `str` | 内存参数来源：`configured`（配置或恢复后的显式参数）、`initialization`、`optimization_candidate`、`optimization`。发布由任务结果、区域完成状态和产物清单判定。 |

### `CurvatureResult`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `distance_m` | `ndarray[P]` | 真实点轴向弧长坐标，m。 |
| `curvature_per_m` | `ndarray[P]` | 真实点曲率幅值，1/m；非负。 |
| `direction_rad` | `ndarray[P]` | 曲率向量在截面内的方向角，rad。 |
| `source_indices` | `ndarray[P,int]` | 软件索引：每项对应的全局真实点 0-based 身份。 |
| `diagnostics` | `Mapping[str,Any]` | 软件诊断字段包；非跨帧状态，不代表新的物理量。 |

### `PreparedCurvature`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `distance_m` | `ndarray[N]` | 模块5积分网格坐标，m；包含可能的虚拟点/插值点。 |
| `curvature_per_m` | `ndarray[N]` | 积分网格曲率幅值，1/m。 |
| `direction_rad` | `ndarray[N]` | 积分网格曲率方向角，rad。 |
| `tangent_seed` | `ndarray[3]` | 初始切向量，无量纲单位向量。 |
| `source_indices` | `ndarray[N,int]` | 软件来源索引参考：由真实索引在距离网格插值并 `.astype(int)`；插值/虚拟点处不是原始点身份保证。 |
| `diagnostics` | `Mapping[str,Any]` | 真实点、空间处理、补点和可选的时间滤波前诊断，见下表。实际积分幅角由本对象的具名字段提供。 |
| `state` | `ReconstructionState\|None` | 模块 5 返回的唯一方向展开状态，由模块 8 在有效帧后保存。 |

### `ShapeResult`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `frame_rule` | `str` | 软件配置标签：积分所用 `TNB` 或 `TNN` 标架规则。 |
| `position_m` | `ndarray[N,3]` | 中心线三维坐标，m。运行帧与锚定结果共用当前最终坐标。 |
| `tangent` | `ndarray[N,3]\|None` | 切向单位向量 T，无量纲。 |
| `normal` | `ndarray[N,3]\|None` | TNB 下主法向 N；TNN 下平行移动标架 Y 轴；无量纲。 |
| `binormal` | `ndarray[N,3]\|None` | TNB 下副法向 B；TNN 下平行移动标架 Z 轴；无量纲。 |
| `distance_m` | `ndarray[N]\|None` | 积分点轴向弧长坐标，m；仅位置的绘图输入可省略。 |
| `diagnostics` | `Mapping[str,Any]` | 积分阶段沿用预处理诊断，增加实际参与积分的幅角。 |

旧 `ShapeIntegrator.integrate`、`ShapeReconstructor.integrate` 及旧 `compat.py` 调用仍在各自边界返回位置和标架元组。生产服务读取具名字段。

| 诊断键 | 形状、单位 | 所属阶段和来源 |
|---|---|---|
| `magnitude`、`angle` | [P]，1/m、rad | 模块 3 的真实点结果进入重构时的幅角。 |
| `source_distance_m`、`adjusted_distance_m` | [P]、[Q]，m | 真实点弧长及补点后弧长。 |
| `angle_prepared`、`angle_unwrap` | [Q]，rad | 补点网格上方向展开前和展开后结果。 |
| `cur_kal`、`ang_filter` | [Q]，1/m、rad | 单帧空间处理的曲率、方向中间量，各插值模式的实际输入见模块 5。 |
| `cur_interp_before_temporal`、`ang_interp_before_temporal` | [N]，1/m、rad | 启用时间滤波时保存的滤波前积分网格幅角。记录器写为既有回放字段 `cur_interp_raw`、`ang_interp_raw`。 |
| `cur_interp`、`ang_interp` | [N]，1/m、rad | `ShapeResult` 中实际积分使用的幅角，供原 UDP 协议和记录器读取。 |

运行帧的 `before_anchor_m` 是旋转及可选曲线时间均值后的位置，`shape.position_m` 是再施加入口锚定后供显示、记录、发送的最终位置。UDP 输出边界沿用坐标到 mm 的变换与原嵌套字段；记录器用 m 保存锚定前后坐标。

### `ReconstructionState`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `previous_raw_angles_rad` | `ndarray[Q]\|None` | 上一帧、unwrap 前的曲率方向角，rad。 |
| `previous_unwrapped_angles_rad` | `ndarray[Q]\|None` | 上一帧、unwrap 后用于对齐的方向角，rad。 |
| `distance_m` | `ndarray[Q]\|None` | 前帧含虚拟点的弧长网格 m，新网格重启方向展开。 |

### `EvaluationReference`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `ideal_position_m` | `ndarray[M,3]\|None` | 理想中心线坐标，m。 |
| `ideal_curvature_per_m` | `ndarray[K]\|None` | 显式理想曲率，1/m。 |
| `direction_rad` | `array\|None` | 参考曲率方向，rad；当前参考生成路径不一定消费此字段。 |
| `radius_m` | `float\|None` | 理想形状半径，m。 |
| `shape_kind` | `str\|None` | 软件标签：理想形状类型，如 circ/end/rot0；无物理单位。 |
| `angle_deg` | `float\|None` | `end3` 圆弧平面相对 +y 方向的顺时针夹角，deg；0 表示在 xy 平面内向 +y 弯曲。 |
| `sample_coordinates_m` | `ndarray[P]\|None` | 真实采样点坐标，m；用于参考曲线按真实采样几何确定分段。 |
| `metadata` | `Mapping[str,Any]` | 软件扩展字典。已知生产键：`idx_m` 对 `end` 表示第二段弯曲半径(m)、对 `rot0` 表示起始位置(m)；`curvature_per_m` 对 `rot0` 是实际曲率数组(1/m)。其他键由调用方契约定义。 |

### `EvaluationResult`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `metrics` | `Mapping[str,Any]` | 软件指标映射；值可为标量或数组。`avg_point_err/avg_end_err/point_errors` 是归一化无量纲误差，`point_error_mean_m/end_error_m` 是绝对距离误差 m；具体键随评估入口而异。 |
| `details` | `Mapping[str,Any]` | 软件诊断映射；可含理想曲线 `[M,3]` m、误差向量或距离数组，不是固定物理量。 |

### `CalibrationDataset`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `data` | `Mapping[str,Mapping[int,dict]]` | 软件索引树：shape → direct(deg) → 半径/case → signals `[P,C]`；信号保留历史应变尺度。 |
| `radii` | `Mapping[int,Mapping[str,list\|dict]]` | 标定方向/形状对应的配置半径集合，半径键以 mm 读取/保存。 |
| `design` | `DesignGeometry` | 此数据集对应的静态传感器几何。 |
| `channel_ids` | `tuple[int,...]` | 软件列标签：dataset 信号列对应的物理通道号。 |
| `plot_data` | `Mapping[str,Mapping[int,dict]]` | 软件绘图数据树；为可视化保留的原始/派生数据，不是统一物理量。 |

### `InitializationResult`

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `global_phases_set` | `float` | 模块 2 拟合或按显式配置采用的全局相位，rad。0 表示零相位。 |
| `slopes_init` | `Mapping[int,list[float]]` | 模块 2 按实验方向（deg）索引的分段扭转斜率 [segment]，rad/m。任务层展开到等级。 |
| `magnitude_results` | `dict` | 软件诊断树：各形状/方向曲率幅值拟合结果，曲率物理量单位 1/m。 |
| `angle_results_raw` | `dict` | 软件诊断树：展开前方向角，rad。 |
| `angle_results_unwrap` | `dict` | 软件诊断树：unwrap 后方向角，rad。 |
| `global_fit_direct` | `dict` | 软件诊断树：各 direct 全局拟合结果，内部结构沿用拟合核。 |
| `individual_fits_direct` | `dict` | 软件诊断树：各 direct 分段拟合结果；内部斜率单位 rad/m。 |

### 标定 block 中多区域包的常见字段

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `multi_region_parameters.multi_region.directions` | `list[int]` | 软件方向索引列表，元素是方向角 deg。 |
| `deployment_ready` | `bool` | 软件包状态标志：是否声明可部署；不是物理参数。 |
| `direction_reference_direct` | `int` | 参考实验方向，deg。 |
| `global_phases_set` | `float` 或按 region 映射 | 全局相位偏置，rad。 |
| `slopes` / `weights` | 按 region/direct 映射的数组 | slopes 为动态角斜率 rad/m；weights 为无量纲曲率等级权重。其余包字段随 sensor parameter package 版本定义，不作通用字段推断。 |

### `RuntimeFrame`（模块8实际逐帧输出）

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `timestamp` | `float` | Unix 时间戳秒；默认由 `time.time()` 产生，也可由调用方提供。 |
| `frame_index` | `int` | 软件帧序号，无单位。 |
| `shape` | `ShapeResult` | 本帧最终重构中心线及标架。 |
| `before_anchor_m` | `ndarray[N,3]` | 锚定变换前的中心线坐标，m。 |
| `end_pose` | `ndarray[3,3]` | 末端三轴姿态矩阵/轴向量阵，无量纲；采用运行时行排列约定。 |
| `raw_signals` | `ndarray[P,C]` | 传给 RuntimeService 的采集阶段帧副本：设备数据已解包、通道排序并缩放到历史应变尺度，中心纤维补偿尚未做。 |
| `anchor` | `AnchorConstraintResult` | 软件结果 DTO：入口锚定偏移/旋转及锚定点等；坐标分量 m、变换矩阵无量纲。 |
| `insert_value_mm` | `float\|None` | 当前插入/推进量，mm；未设置时为 `None`。 |

### `SessionState`（RuntimeService 独占）

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `frame_index` | `int` | 软件帧计数，无单位。 |
| `processor_state` | `Any` | 处理器跨帧状态槽；demo 中是 `LiveProcessingState`。 |
| `curve_history` | `tuple[ndarray[N,3],...]` | 展示曲线历史，坐标 m；按窗口大小保留。 |
| `imu_rotation` | `ndarray[3,3]` | IMU/姿态旋转矩阵，无量纲，默认单位矩阵。 |
| `insert_value_mm` | `float\|None` | 插入长度，mm。 |
| `last_frame` | `RuntimeFrame\|None` | 最近提交的完整运行帧；无有效帧时为 `None`。 |

### `LiveProcessingState`（生产 demo 处理器跨帧状态）

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `reconstruction` | `ReconstructionState\|None` | 模块 5 在补点网格上的展开状态，包含展开前后方向 [Q] rad。 |
| `temporal` | `TemporalCurvatureState\|None` | 运行服务保存的曲率向量 Kalman 状态；`x` 为 `[N,2]` 1/m 分量，`p` 为同形协方差，量纲 `(1/m)^2`。 |
| `direction_masses` | `ndarray[3]\|None` | 0/120/240 deg 的无量纲方向质量。方向角和一致性在本帧计算，只有质量参与下一帧。 |
| `last_signals` | `ndarray[P,C]\|None` | 上一有效、已修复的采集帧信号，历史应变数值尺度；只用于下一帧异常行修复。 |

### `AnchorConstraintResult`（运行时入口锚定输出）

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `curve_m` | `ndarray[N,3]` | 锚定后的中心线坐标，m；未启用锚定时为输入曲线副本。 |
| `applied` | `bool` | 软件状态：本帧是否实施锚定变换。 |
| `anchor_m` | `ndarray[3]\|None` | 本会话固定入口锚点坐标，m；未建立位置锚点时为 None。 |
| `source_point_m` | `ndarray[3]\|None` | 当前曲线上按请求距离从末端反向取出的入口点，m。 |
| `anchored_point_m` | `ndarray[3]\|None` | 变换后入口点坐标，m。 |
| `translation_m` | `ndarray[3]` | 当前帧施加的平移向量，m。 |
| `rotation_matrix` | `ndarray[3,3]` | 当前帧施加的旋转矩阵，无量纲。 |
| `anchor_frame` | `ndarray[3,3]\|None` | 首帧入口参考 T/Y/Z 标架，无量纲；矩阵列为轴向量。 |
| `source_frame` | `ndarray[3,3]\|None` | 当前帧入口处源 T/Y/Z 标架，无量纲。 |
| `relative_insert_mm` | `float` | 相对本会话首个有效推进读数的插入变化量，mm。 |
| `requested_distance_m` | `float` | 从光纤末端沿曲线反向请求的入口距离，m。 |
| `used_distance_m` | `float` | 截断到当前曲线有效总长后的实际入口距离，m。 |
| `clamped` | `bool` | 软件状态：请求入口距离是否超出曲线长度并被截断。 |

### `ModuleConfiguration`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `input` | `InputConfig` | 模块1输入、采样几何与设备配置投影。 |
| `initialization` | `InitializationConfig` | 模块2拟合配置投影。 |
| `curvature` | `CurvatureConfig` | 模块3物理曲率与分级评分配置投影。 |
| `reconstruction` | `ReconstructionConfig` | 模块5展开、插值、滤波与积分配置投影。 |
| `optimization` | `OptimizationConfig` | 模块4搜索及优化目标配置投影。 |
| `evaluation` | `EvaluationConfig` | 模块6理想曲线与误差阈值配置投影。 |
| `runtime` | `RuntimeConfig` | 模块8 demo/设备/方向融合/锚定配置投影。 |
| `view` | `ViewConfig` | 模块7 viewer 显示配置投影。 |

### `TemporalCurvatureState`（demo 向量 Kalman 状态）

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `x` | `ndarray[N,2]\|None` | 每个积分网格点的曲率向量估计 `(kx,ky)`，分量 1/m；N 为模块5积分网格点数。 |
| `p` | `ndarray[N,2]\|None` | 各点两曲率分量估计的对角协方差，单位 `(1/m)^2`；Kalman 内部状态量。 |
| `distance_m` | `ndarray[N]\|None` | 前帧积分弧长网格 m，新网格重启时间滤波。 |

### `VisualizationInputs`（模块7输入 DTO）

| 字段 | 类型/形状 | 物理含义、单位；来源与约束 |
|---|---|---|
| `dataset` | `CalibrationDataset` | 软件数据依赖：模块1静态数据及设计。 |
| `initialization` | `InitializationResult\|None` | 软件诊断依赖：模块2拟合结果；初始化图需要提供。 |
| `segmentation` | `tuple[int,...]` | 软件几何分区：每段真实点数，无单位。 |
| `discont_rad` | `float` | 角度 unwrap 不连续阈值，rad。 |
| `connect` | `bool` | 软件选项：初始化分段拟合时是否连接各段。 |
| `rotation_deg` | `float` | 显示/重构的绕 x 安装旋转角，deg。 |

## 八模块服务与任务入口

1. **输入与预处理** — `InputService(InputConfig)`。`prepare(signals, distance_m, source_indices, channel_ids=None) -> PreparedFrame` 只按物理通道挑列、校验 `(P,C)/(P,)/(P,)`、有限性及距离严格递增；不换算信号。`preprocess(signals, channel_ids=None, acquired=False, design=None)` 执行历史通道排序、交错光栅处理、裁剪、中心补偿和 reverse，与可选设计校验后返回真实帧；`load_data(folder_path=None) -> CalibrationDataset` 显式读静态 CSV，`design() -> DesignGeometry` 不读标定数据。设备协议读取由 `InputKernel`/`InterrogatorSource` 负责。设备波长差分在解包边界乘 `1e-6` 转历史信号尺度；中心纤维补偿由原 `_process_centreData` 执行，不应在下游再乘一次或补偿一次。`InputPreprocessor` 是 legacy context 适配器，不是新 DTO 接口。
2. **初始化** — `InitializationService(InitializationConfig).fit(dataset, verbose=False) -> InitializationResult`。从 Dataset 的 `[P,C]` 叶子与 DesignGeometry 拟合 phase/slopes 初值；实际标定只在显式 `TaskSession.initialize()` 执行。它不做优化。
3. **真实点曲率物理模型** — `CurvatureService(CurvatureConfig).compute(frame, design, parameters, direct=0, region='full', shape_kind='circ', magnitude_ratio=1.0) -> CurvatureResult`；`base(frame, design, parameters, direct=0, region='full')` 返回基础模型结果。按 region 再按 direct 查 weights/slopes/phase；输入真实点全长不变。模块3只算真实点物理量，不做移动滤波、虚拟点或插值；评分平滑仅用于权重/等级评分，不平滑对外输出曲率。`GeometryService` 是模块内物理几何助手。
4. **参数优化** — `OptimizationService(config, curvature_config, dataset, parameters, *, region, predict_shapes, evaluate_shapes, compute_curvature, score_inputs)`；`parameters() -> CalibrationParameters`、`optimize(verbose=True) -> (CalibrationParameters, report)`。输入必须含 region 的完整 weights/slopes，缺值 `ValueError`。候选通过显式回调走模块3/5预测与模块6评估；优化不隐式调用模块2。任务 pipeline 会在读数据/启动搜索前对所有 job seeds 做 preflight；缺初值应提示 init+resume 或显式传入 seed。
5. **重构预处理与积分** — `ReconstructionService(ReconstructionConfig)`。`prepare(curvature, design, tangent_seed, state=None, ideal_curvature=None) -> PreparedCurvature` 校验输入真实曲率 `[P]`、方向/距离等长和 seed `[3]`；按 zero_init/end_point 添加显式物理虚拟点，然后执行方向 unwrap、可选平滑/向量滤波、曲率插值到 N 点网格。此处允许虚拟点、unwrap 与滤波，模块3真实曲率对象保持不变。`integrate(prepared) -> ShapeResult` 产生 T/N/B 与位置；`transform_shape(shape, *, direct, rot_dir, rot_ang)` 对位置和标架统一绕 x 变换。虚拟起点来自设计物理起点，与存储零波长参考是两个独立概念。`use_stored_zero` 只由 demo 可选启用，常规 reconstruction 不读取该文件。`tangent_seed` 是有限非零三维初始切向量，无量纲。
6. **评估** — `EvaluationService(EvaluationConfig)`。`ideal_curvature(distance_m, reference)`、`ideal_shape(distance_m, reference)` 生成显式参考；`evaluate(shape, reference) -> EvaluationResult` 接收 ShapeResult；`evaluate_batch(interp_dist, results, radii, direct, sample_distance_m, curvature_per_m)` 和 `evaluate_dataset(dataset, direct, interp_dist, results, curvature_per_m)` 提供旧批量指标布局。距离 m，positions `[N,3]` m，曲率 `[N]` 1/m。指标沿用原归一化误差定义，不用 RMSE 替代。
7. **可视化** — `VisualizationInputs(dataset, initialization=None, segmentation=(), discont_rad=pi, connect=True, rotation_deg=0)` 将静态输入与诊断交给 `VisualizationService(inputs=None)`。首点和直段裁剪来自数据集设计，理想曲线由模块 6 提供。`init_visualise(direct)` 画初始化诊断；`plot_shape(shape, ideal_position_m=None, evaluation=None, ax=None)` 返回 Figure；`show_reconstruction(direct, rs, interp_dist, showanime=False, ideals=None)` 展示旧曲线树。调用绘图方法才加载 matplotlib/VTK；动画入口必须显式传入模块6的 ideals。
8. **运行时** — `RuntimeService(processor, source=None, recorder=None, publisher=None, moving_window=1, queue_size=100, direct=0, imu=False, anchor_enabled=False, anchor_translation_enabled=False, anchor_rotation_enabled=False, anchor_offset_m=.06, rotation_deg=0, reset_processor=None)`。实时/回放都走 `process_frame`；source 生命周期由 `open/read/close` 显式管理。`RuntimeFrame` 包含 timestamp、frame_index、ShapeResult、锚定前位置 `[N,3]` m、末端姿态 `[3,3]`、raw_signals `[P,C]`、anchor 结果和插入量 mm。实际会话状态是 RuntimeService 的 `SessionState(frame_index, processor_state, curve_history, imu_rotation, insert_value_mm, last_frame)`；生产处理器返回 `LiveProcessingState(reconstruction, temporal, direction_masses, last_signals)`，由 RuntimeService 提交。`run_demo(session,args,app_factory)` 才创建窗口/设备运行链；`create_demo` 只装配 runtime/window。

## 配置 DTO 字段归属

`ModuleConfiguration.from_mapping` 只将已由 loader 校验的外部原名映射投影至以下 DTO，不改写外部名称。映射缺省字段使用 DTO 默认值。每表逐字段列类型和单位；参数说明据 `sensor_template.yaml` 与 `parameters.json`，未消费或兼容字段明确标记。

### `InputConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `sensor_id` | `str\|None` | 传感器软件标识。 |
| `start_value` | `float\|None` | 传感器首个感测点到固定件距离，mm。 |
| `end_value` | `float\|None` | 传感器末个感测点到末端距离，mm。 |
| `folder_path` | `str\|None` | 软件路径：当前标定区域数据目录。 |
| `centre_fibre` | `bool` | 构型开关：是否为 1+x 中心纤芯结构；启用后做中心纤维补偿。 |
| `active_channels` | `list[int]` | 软件通道标签及顺序；声明参与处理的物理通道号。 |
| `rot_dir` | `int\|list[int]\|dict[int,int]` | 各实验方向的旋转符号，如 -1/1；不是角度值。 |
| `exp_directs` | `list[int]` | 实验/数据方向标签，deg（如 0/120/240）。 |
| `interrogator` | `str` | 软件设备类型标签：FiberTecc/Hippo/LUNA。 |
| `R` | `float` | 感测光纤/外围纤芯到中心的径向距离，m。 |
| `ideal_len` | `float` | FBG 有效总长度，m。 |
| `radii` | `dict` | 标定形状的弯曲半径配置，键/值按数据目录约定以 mm 组织。 |
| `samp_dist_preset` | `list[float]\|None` | A 组光栅轴向位置，mm。 |
| `active_angles` | `list[float]\|None` | 外围通道截面方位角，deg；None 时几何核用默认均匀布局。 |
| `reverse` | `bool` | 软件处理开关：输出真实采样顺序是否反转。 |
| `straight_trim` | `float` | 输入几何的直段裁剪长度，m（旧输入可有兼容整型语义，见 DesignGeometry）。 |
| `ss` | `float` | LUNA 传感点间距，m。 |
| `N` | `int` | 静态均值时挑选 NaN 最少的候选行数；不是空间点数或 interp_num。 |
| `sensing_ranges` | `dict[channel,tuple[int,int]]` | LUNA 各通道读取的采样索引范围，无物理单位。 |
| `smoothing_params` | `dict` | LUNA 信号平滑的软件参数（window_length 点数、polyorder 多项式阶数）。 |
| `IP` | `str` | 软件网络地址：采集设备 IP。 |
| `PORT` | `int` | 软件网络端口号，无物理单位。 |
| `data_freq` | `int` | 采集数据频率，Hz。 |
| `maxsize` | `int` | 软件缓冲队列容量，帧数。 |
| `samp_dist_preset_sub` | `list[float]\|None` | B 组光栅轴向位置预设，mm。 |
| `interlaced_grating` | `bool` | 软件开关：光栅是否交错排列。 |
| `channel_change` | `bool` | 软件开关：通道排布是否发生变化。 |
| `channel_layout` | `dict\|None` | 软件重排表：各分段长度及通道排列顺序；索引/标签，无物理单位。 |
| `interlaced_offset` | `float` | 交错排列的角度偏移，rad。 |
| `use_stored_zero` | `bool` | demo 软件开关：是否使用文件中的绝对波长零状态；init/optim 不消费。 |
| `zero_reference_path` | `str\|None` | 软件路径：零参考 JSON；仅 use_stored_zero 的 demo 消费。 |
| `wavelength_jump_threshold_pm` | `float` | 跨帧单栅波长跳变阈值，pm。 |
| `wavelength_jump_guard_enabled` | `bool` | 软件开关：是否启用跨帧波长连续性检查。 |

### `InitializationConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `discont` | `float` | 相位展开不连续判断阈值，rad。 |
| `segmentation` | `tuple[int,...]` | 每段真实采样点数，无单位。 |
| `connect` | `bool` | 软件开关：初始化分段拟合是否连接各段。 |
| `global_phases_set` | `float\|None` | 全局通道相位初值，rad；None 时由拟合估计。 |
| `interlaced_offset` | `float` | 交错光栅角偏移，rad。 |

### `CurvatureConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `thresholds` | `tuple[float,...]` | 曲率分级阈值，1/m。 |
| `soft` | `bool` | 软件开关：是否用软分级评分。 |
| `segmentation` | `tuple[int,...]` | 区域各空间段包含的真实点数，无单位。 |
| `discont` | `float` | 方向角展开不连续阈值，rad。 |
| `turn_threshold` | `float` | 相邻 unwrap 方向角差 `abs(Δangle)` 的转折判定阈值，rad。 |
| `min_consecutive` | `int` | 分级前评分窗口要求的最少连续真实点数。 |
| `sharpness` | `float` | sigmoid 曲率软评分的斜率系数；与曲率阈值 1/m 相乘时令指数无量纲，量纲为 m；当前按历史数值尺度配置。 |
| `end_start` | `int` | 末端权重区起始负索引；相对当前真实点数组尾部，无单位。 |
| `end_weights` | `bool` | 软件开关：是否对末端区使用独立权重。 |
| `init_weight` | `float` | 初始化曲率权重值，无量纲。 |
| `weight_group_ids` | `tuple[int,...]\|None` | 软件分组索引：每真实点对应权重组，无单位。 |
| `end5cm_independent_weight_score` | `bool` | 软件评分开关：end5cm 独立权重组是否按组内基础曲率均值评分。 |

### `ReconstructionConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `frame_rule` | `str` | 软件标架选择：TNB 或 TNN。 |
| `rot_ang` | `float` | 绕 x 轴安装旋转角，deg。 |
| `interp_num` | `int` | 模块5积分插值网格目标点数，无单位。 |
| `move_filter` | `int` | 空间移动滤波窗口长度，按准备阶段的曲率采样点数 P/Q 计，不是最终积分网格 N 的预分配长度。 |
| `curvature_interp` | `str` | 软件算法选择：曲率插值方法标识。 |
| `nonnegative_curvature` | `bool` | 软件开关：是否约束插值后曲率非负。 |
| `TNB_threshold` | `float` | TNB 标架直线判定曲率阈值，1/m。 |
| `move_filter_for_vector` | `bool` | 软件开关：插值前是否对曲率向量分量做空间滤波。 |
| `kalman_params` | `Mapping[str,Mapping[str,float]]` | 模块5标量 Kalman 的噪声协方差配置：Rk 为观测噪声，Bk 为过程噪声；magnitude 项按 `(1/m)^2`，angle 项按 rad² 理解。当前准备路径实际调用 magnitude 项；angle 项保留配置，不应视为已执行角度 Kalman。 |
| `zero_init` | `bool` | 软件几何开关：是否在积分网格加入显式虚拟起点。 |
| `end_point` | `bool` | 软件几何开关：是否加入显式虚拟末点。 |
| `discont` | `float` | 空间方向 unwrap 跳变阈值，rad。 |
| `unwrap_threshold` | `float` | 跨帧方向一致性/unwrap 角差阈值，rad。 |
| `align_threshold` | `float` | 首点方向对齐逻辑用的曲率阈值，1/m（不是角度阈值）。 |

### `OptimizationConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `direct` | `int` | 标定实验方向标签，deg。 |
| `end_len` | `float` | 末端评价/搜索长度，m。 |
| `max_iter` | `int` | 优化迭代上限，轮。 |
| `min_loss_diff` | `float` | 收敛目标差阈值；复合损失尺度由原目标函数决定。 |
| `method` | `str` | 软件标签：SciPy minimize 搜索算法名。 |
| `hyper_params` | `tuple[float,...]` | 搜索器参数；当前依次对应最大函数评估数、xtol、ftol。 |
| `freeze` | `tuple[int,...]\|None` | 软件索引：冻结的参数等级索引，从 0 起；不是空间分段索引。 |
| `straight_threshold` | `float` | 理想直线段判定曲率阈值，1/m。 |
| `rot_threshold` | `float` | 理想螺旋段判定曲率阈值，1/m。 |
| `k_point` | `float` | 目标中逐点误差项权重，无量纲系数。 |
| `k_end` | `float` | 目标中末端误差项权重，无量纲系数。 |
| `k_pitch` | `float` | 目标中螺距误差项权重，无量纲系数。 |
| `k_radius` | `float` | 目标中半径误差项权重，无量纲系数。 |
| `weight_bounds` | `tuple[float,float]` | 无量纲权重搜索下/上界。 |
| `slope_upper_bound` | `float` | 动态角斜率绝对搜索上限，rad/m。 |
| `end5cm_point_calib` | `bool` | 软件任务开关：end5cm 使用真实点方向/曲率幅值校准。 |
| `real_point_calib` | `bool` | 软件任务开关：优化评分是否使用真实点。 |
| `optim_weights_only` | `bool` | 软件任务开关：仅优化权重并固定已有斜率。 |
| `end5cm_coverage_aware_weights` | `bool` | 软件任务开关：只优化有真实覆盖的 end5cm 权重档。 |
| `end5cm_min_weight_effective_samples` | `float` | coverage-aware 权重档最小有效样本数，无单位。 |
| `end5cm_min_weight_max_mass` | `float` | coverage-aware 权重档的最小最大组质量，无量纲。 |
| `end5cm_uncovered_weight_policy` | `str` | 软件策略标签：未覆盖权重档派生方法。 |
| `source_sensor_params` | `str\|None` | 软件路径/来源：传感器参数包；无物理单位。 |

### `EvaluationConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `ideal_curvature` | `bool` | 保留的软件配置字段；当前 EvaluationService 未直接读取此开关，理论曲率由调用方 `force_ideal_magnitude` 路径和显式回调控制。 |
| `end_len` | `float` | 末端误差/理想曲线长度，m。 |
| `straight_threshold` | `float` | 理想直线段识别阈值，1/m。 |
| `rot_threshold` | `float` | 理想旋转/螺旋段识别阈值，1/m。 |

### `RuntimeConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `imu` | `bool` | 软件开关：连接 IMU。 |
| `imu_com` | `str` | 软件设备路径：串行端口名。 |
| `imu_maxlen` | `int` | 软件缓存容量；IMU 历史样本数。 |
| `demo_recorder` | `bool` | 软件开关：demo 是否记录数据。 |
| `direction_blend_enabled` | `bool` | 软件开关：动态 0/120/240 参数软过渡。 |
| `direction_reference_direct` | `int` | 方向估计参考参数方向，deg。 |
| `direction_min_curvature` | `float` | 方向估计有效弯曲点最小曲率，1/m。 |
| `direction_min_consecutive` | `int` | 方向估计稳定区最少连续真实点数。 |
| `direction_consistency_threshold` | `float` | 圆周方向一致性门限，无量纲。 |
| `direction_smooth_alpha` | `float` | 方向质量指数平滑系数，范围 (0,1]，无量纲。 |
| `demo_kalman_enabled` | `bool` | 软件开关：demo 是否对插值后曲率向量做跨帧 Kalman。 |
| `demo_kalman_Rk` | `float` | demo 曲率向量 Kalman 的观测噪声协方差尺度，`(1/m)^2`；作用于两个曲率分量。 |
| `demo_kalman_Bk` | `float` | demo 曲率向量 Kalman 的过程噪声协方差增量，`(1/m)^2`；每帧加入预测协方差。 |
| `anchor_constraint_enabled` | `bool` | 兼容开关：同时启用入口平移与旋转锚定。 |
| `anchor_translation_enabled` | `bool` | 软件开关：入口位置平移锚定。 |
| `anchor_rotation_enabled` | `bool` | 软件开关：入口姿态旋转锚定。 |
| `anchor_offset_m` | `float` | 从光纤末端沿曲线反向量取的入口距离，m；运行中可调整。 |

### `ViewConfig`

| 字段 | 实际类型 | 物理含义/单位或软件用途 |
|---|---|---|
| `total_scale` | `float` | viewer 显示缩放系数，无量纲。 |
| `title` | `str` | 软件显示标题。 |
| `body_model_path` | `str` | 软件文件路径：viewer 身体模型。 |
| `max_curvature` | `float` | viewer 曲率颜色/显示阈值，1/m。 |
| `back_num` | `int` | viewer 历史回看帧数，无单位。 |

## 真实调用与生命周期

`TaskSession(parameters, task='recon', region='full', dataset=None)` 是组合入口：创建八模块配置和服务、持有 region 参数与 dataset。`task='init'` 创建时不拟合；调用 `initialize()` 才执行模块1读取和模块2拟合。非 init 的 recon/demo/ang/optim 在构造阶段校验该 region 的 `weights`、`slopes`、`global_phases_set`；缺任意项立即报错，禁止自动运行模块2。`load_data()` 显式加载静态 CSV；`process_frame(signals, state=None, direct=0, tangent_seed=None, temporal_filter=None, acquired=False, parameter_selector=None)` 编排模块1→3→5，返回 `(ShapeResult, ReconstructionState)`；`reconstruct_shapes(kind,direct,...)` 返回形状树和距离；`evaluate(...)` 委托模块6。`optim_coeff()` 创建模块4并返回搜索报告。`build_session(args, parameter_builder=..., configure_args=...)` 返回 `TaskSession`；`execute_task(...)` 管理任务快照和状态，不把 legacy Optimiser 暴露为生产注入边界。

## 索引、错误与组合示例

- `source_indices` 在 PreparedFrame/DesignGeometry 中从 0 起，表示原始真实点身份；历史兼容 `global_point_ids` 显示为 1-based。距离数组按实际方向顺序严格递增；reverse 会同时反转帧和索引。
- `direct` 是方向角（deg，常见 0/120/240），`region` 是校准区域键（如 `full/end15cm/end5cm`）。参数访问顺序为 region block → direct。区域裁剪与 global point 映射在配置/几何解析边界完成，模块3不做局部截断。
- 模块1 prepare、模块5 prepare 和模块6 evaluate 对各自数组执行维度/有限性校验，模块5另检查非负曲率和有效切向量；模块3校验真实距离对应及必需的区域/方向参数，错误通常为 `ValueError`。理论曲线缺少部分 metadata 键、基础参数缺少 direction 等底层映射错误仍可能是 `KeyError`，调用前应按接口补全，不能依赖所有失败都统一为 ValueError。模块8对无法修复的第一帧非有限采集值抛 `FrameRejected`；运行时跳过该帧且不提交会话状态。
- demo 的采集信号经解包已是历史 `1e-6` 尺度，`process_frame(..., acquired=True)` 不再套静态通道排序/信号换算；启用 demo Kalman 时滤波发生于模块5准备后、积分前，并保留每会话 temporal state。零点文件只由 demo 可选消费，不能把该行为推广到 optim 或普通 recon。

以下组合片段要求 `validated_parameters` 已完成配置/区域解析且含匹配的标定初值，`signals` 是对应采集帧；完整用户命令见 [使用说明](../shape_sensing/README.md)。

```python
from shape_sensing.pipelines.session import TaskSession

session = TaskSession(validated_parameters, task="recon", region="end5cm")
dataset = session.load_data()                 # 显式静态 CSV 读取
curves, distance_m = session.reconstruct_shapes("circ", direct=120)
metrics = session.evaluate(distance_m, {"circ": curves}, direct=120)

# 在线/replay：signals 已在历史信号尺度，state 从上一帧显式返回
state = None
shape, state = session.process_frame(signals, state=state, direct=120, acquired=True)
```


## 优化回调与树形数据细目

模块4只接收数据和可调用对象，不接收其他模块的引擎实例。TaskSession 的回调绑定规则如下。

| 回调 | 输入 | 输出/约束 |
|---|---|---|
| `predict_shapes(parameters, calib_shape, direct, force_ideal_magnitude=False)` | 独立 `CalibrationParameters` 候选、形状名、方向 deg、是否用理论曲率 | `(curve_tree, distance_m)`；每叶 `(N,3)` m，网格 `(N,)` m。内部只走模块3→5，可选显式向模块6索取理论曲率。 |
| `evaluate_shapes(interp_dist, results, direct)` | `(N,)` m；`{shape: curve_tree}`；方向 deg | 模块6 `EvaluationResult`。指标继续沿用原逐点、末端、螺距和半径定义。 |
| `compute_curvature(values, parameters, direct, kind)` | 预处理的真实信号 `(P,C)`、候选、方向、形状名 | `(curvature_per_m, direction_rad)`，均 `(P,)`，单位 1/m 与 rad，不含补点。 |
| `score_inputs(keys, data, slopes, parameters, **kwargs)` | 真实数据/路径与显式参数，或 magnitude/scores/group_ids | 模块3 base/scores/allocate/groupwise_scores 的结果；保留原搜索核适配签名，禁止触发拟合或读文件。 |

静态树的方向键是整数角度，半径键沿用实验文件的 mm 数值：`dataset.data[shape][direct][radius_mm]` 为一个信号叶子；需要第二实验位置的形状为 `...[radius_mm][case_mm]`。`dataset.radii` 外层顺序相反，为 `[direct][shape]`，用于声明应有实验及校验缺失数据。`reconstruct_shapes` 已固定 shape/direct，因此返回树不再带这两个外层；交给批量评估必须重新组成 `{shape: curve_tree}`。

`CurvatureService.base` 返回 `(strain, magnitude, angle, turning_segments)`，形状分别 `(C,P)`、`(P,)`、`(P,)` 和转折区间集合；应变无量纲、曲率 1/m、方向 rad。`scores(magnitude, *, thresholds=None, sharpness=None, min_consecutive=None, boundary_mode='constant')` 返回 `(L-1,P)` 分级分数；`allocate(scores)` 返回 `(L,P)` 等级质量；`groupwise_scores(magnitude, group_ids, **kwargs)` 在显式真实点权重组内算分数和诊断。L 是阈值数加一，weights 的具体布局还取决于 end_weights/weight_group_ids，不能一律按分段数创建。

`GeometryService(design, channel_count).dynamic_angles(*, phase_rad, mode='init', segments=None, slopes=None)` 返回 `(C,P)` rad；`curvature(strain, angles_rad)` 接收 `(C,P)` 无量纲应变与 rad，返回 `(P,)` 1/m 与 rad。初始化阶段可调用该基础物理接口，但不需要先有优化结果。

## 运行时方法和状态交接

| 入口 | 输入 | 输出与副作用 |
|---|---|---|
| `process_frame(signals, *, timestamp=None)` | 采集源阶段 `(P,C)` 差分信号，秒时间戳可省略 | `RuntimeFrame`，帧号从0起；处理后提交本会话状态，并把同一最终帧交给 recorder/publisher。它直接调用时会传播 FrameRejected。 |
| `replay(frames)` | 可迭代的采集阶段信号矩阵 | 生成器逐帧返回 RuntimeFrame，跳过 FrameRejected，不打开设备/线程。 |
| `submit(signals)` / `poll()` | submit 复制帧入队；poll 无参数 | poll 返回一个最终帧或 None；队满丢弃最旧帧；FrameRejected 被跳过。 |
| `start()` / `close()` | 无 | start 打开 source 并启动采集线程；close 关闭 source、线程与注册资源，尝试全部清理后抛首个错误。构造本身不连接设备。 |
| `reset()` | 无 | 清空状态、队列、锚点并调用 reset_processor，不改配置/标定。 |
| `set_imu_rotation(rotation)` / `set_insertion(value_mm)` | `(3,3)` 有限姿态矩阵 / 有限推进量 mm | 复制更新当前会话的姿态/推进量。 |
| `configure_anchor(**settings)` | 锚定开关、offset m、reset 等命名参数 | 更新本会话锚点约束。 |

`LiveProcessingState.reconstruction` 保存模块5的 Q 点展开状态；`temporal` 保存模块5向量 Kalman 的 x/p（积分网格 N 点、两曲率分量）；`direction_masses` 保存三方向质量；`last_signals` 是上一有效的修复后信号。RuntimeService 的会话是跨帧状态的唯一所有者。通用处理器协议允许其他处理器返回自己的状态类型。异常帧不会提交状态；缺少可修复历史时实时与回放跳过该帧。补点/积分网格变化时重启相应的方向展开/Kalman 历史，曲线均值也按积分距离重置。锚定保留会话的物理参考，直到显式重置。关闭资源后可读取最后有效帧；`reset()` 清理帧、队列及历史。设备数据在 channel reorder/设备裁剪后进入 `acquired=True` 路径，此时中心补偿尚未执行，仍由模块1处理。SessionState.curve_history 是最终时间均值计算所需的 `(N,3)` m 曲线历史，end_pose 为行排列的三根末端坐标轴。

## 可直接运行的纯数值接口例子

以下示例不需要配置文件、标定 CSV、设备或窗口，展示模块5→6的最小数据契约；它不能替代真实传感器的初始化/优化。

```python
import numpy as np
from shape_sensing.contracts import (
    DesignGeometry, CurvatureResult, ReconstructionConfig,
    EvaluationConfig, EvaluationReference,
)
from shape_sensing.reconstruction.service import ReconstructionService
from shape_sensing.evaluation.service import EvaluationService

s = np.linspace(0., .1, 12)                  # m，真实点
idx = np.arange(s.size)
design = DesignGeometry("example", .0001, s, (1, 2, 3), {}, source_indices=idx)
physical = CurvatureResult(s, np.full(s.size, 10.), np.zeros(s.size), idx)
recon = ReconstructionService(ReconstructionConfig(
    frame_rule="TNN", zero_init=False, end_point=False, interp_num=120,
))
prepared = recon.prepare(physical, design, np.array([1., 0., 0.]))
shape = recon.integrate(prepared)
reference = EvaluationReference(radius_m=.1, shape_kind="circ", sample_coordinates_m=s)
result = EvaluationService(EvaluationConfig()).evaluate(shape, reference)
print(shape.position_m.shape, result.metrics["avg_point_err"])
```
