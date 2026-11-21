# 八模块独立代码库迁移实施计划

日期：2026-11-10。基准：AlgorithmPackage-08B5 / 23e180a。用户已纠正上轮适配式封装不足，本轮要求函数实现真正迁入新库。沿用低模型实现窗口、主窗口审查验收。

## 本轮硬性完成条件
- shape_sensing 拥有所有运行所需的一方实现；新包不得 import 根目录 sensor/initialiser/reconstructor/optimiser/visualisor/demonstrator/proto_demonstrator/IMU/main/calibration_* / entrance_anchor 或 tools。
- 八模块有实际函数主体。特别将 Reconstructor 的曲率计算移至模块3、Optimiser 的理想曲线/误差移至模块6；不能整份复制旧类后继续让A/B共享混合职责，也不能仅增加wrapper。
- 旧根文件最多保留反向兼容入口：旧名称 -> 新包。新包完全不依赖旧入口。保留数值公式、函数签名、状态/调用顺序、docstring、行内说明及设备许可声明。
- python -m shape_sensing 和安装后的 shape-sensing 可使用原task/config参数；配置模板/参数JSON随包分发，保留原命名与说明。
- 构建wheel，在仓库外用该wheel安装运行，并禁止导入所有旧模块。真实曲率/积分/理想曲线计算及CLI帮助通过；外部数据由用户配置，不打包原实验数据。
- 现有365项测试保留数值断言。只依赖旧源码位置的AST测试改读新实现位置，不删除测试或放松断言。

## 目标文件与函数归属
### 1 输入预处理（A）
shape_sensing/input_preprocessing/compat.py：Sensor、WavelengthContinuityGuard及原读取/解析/通道整理实现。
shape_sensing/input_preprocessing/zero_reference.py：tools/zero_reference.py 实现。
### 2 初始化（A）
shape_sensing/initialization/compat.py：Initialiser、rot_seg_fit及拟合/分段实现。
shape_sensing/initialization/phase_reference.py：calibration_phase_reference.py。
### 3 曲率（A）
shape_sensing/curvature/geometry.py：从Initialiser移入_calculate_dynamic_angles、_calculate_curvature_and_angle，构成无构造器的StrainGeometryMixin，由Initialiser继承。
shape_sensing/curvature/model.py：从Reconstructor移入_Strain_2_Cur（实际名字_Strain里的零宽字符不得进入源码，使用原_Strain_2_Cur拼写）、_get_base、_alloc_mass、_get_scores、_normalise_multiregion_parameters、_local_segment_labels、_get_groupwise_weight_scores、_assemble_multiregion_parameters、_generate_dynamic_segments，作为CurvatureModelMixin。按原函数真实名字提取，不依据本文特殊排版创建新名字。
### 4 优化（A）
shape_sensing/optimization/compat.py：Optimiser保留优化、损失、边界、收敛等实现，继承下述EvaluationMixin和新Reconstructor。
shape_sensing/optimization/batch.py：calibration_batch.py；coverage.py：tools/end5_weight_coverage.py。
角度搜索/任务辅助函数可以保留在新cli或迁入optimization/angle_search.py，不能反向依赖旧main。
### 5 重构（B）
shape_sensing/reconstruction/compat.py：Reconstructor、完整帧流程、空间滤波、插值、角度展开。继承新CurvatureModelMixin、新Initialiser与必要理想曲率mixin。
shape_sensing/reconstruction/integration.py：实际迁入_reconstruct_from_interpolated_curvature、_F_S_frame、_parallel_frame、_build_end_pose、_get_rot_x。通过ShapeIntegrationMixin复用；公开ShapeIntegrator可只给曲率/角度/弧长直接积分，不构造Sensor、不读取文件。frame_rule/TNB_threshold等所需状态显式初始化，保留旧输出坐标/标架约定。
### 6 理想曲线/误差（B）
shape_sensing/evaluation/metrics.py：从Optimiser移入_print_errors、_compute_error、_construct_ideals、所有_construct_ideal_*、_compute_point_err、_compute_rot0_err、_find_rot_segment、_find_rot_axis等原评估实现，作为EvaluationMixin；不要移走优化专属_loss_fn/_real_point_loss。
shape_sensing/evaluation/ideal_curvature.py：Reconstructor._construct_ideal_magnitude_for_recon。
### 7 可视化（B）
shape_sensing/visualization/plots.py：Visualisor原实现。
shape_sensing/visualization/vtk_view.py：从Demonstrator移入add_ball/add_axes/add_end_pose_arrows/add_anchor_marker/_update_anchor_marker/_update_end_pose_arrows/_update_frame等纯视图方法，作为无构造器mixin供runtime窗口继承。保留Qt/VTK按需加载。
### 8 实时系统（B）
shape_sensing/runtime/compat.py：Demonstrator动态调度、通信、滤波与状态，使用新视图mixin；DataRecorder可单独recorder.py。
shape_sensing/runtime/prototype.py：proto_demonstrator原样迁入并改内部导入，保留模型路径配置。
shape_sensing/runtime/imu.py：IMU.py实际实现及许可原文；anchor.py：entrance_anchor.py。
### 公共基础与入口
shape_sensing/common/utils.py：tools.utils实际实现；common/paths.py：统一运行根/配置资源路径。
shape_sensing/configuration/regions.py：calibration_regions.py。
shape_sensing/artifacts/legacy_package.py：tools.sensor_parameter_package.py实际实现，去掉sys.path补丁，内部main导入新CLI，路径根不依赖旧文件位置。
shape_sensing/cli.py：旧main实际任务/参数入口迁入，可进一步将parser放configuration/arguments.py；所有导入都指向包内。GUI依赖延迟到需要时，旧main monkeypatch通过反向模块alias保持。
shape_sensing/__main__.py、api.py、pyproject.toml；包内resources/sensor_template.yaml。

## 执行顺序
1. 以AST定位函数但按原文本迁移（不可ast.unparse数值函数，避免丢注释）。连装饰器和相邻说明一起迁移，记录每个旧函数的新位置。先完整读取原文件一次，防止写shim后失去源；可用基准git show只读校对，不覆盖工作区。
2. 迁叶子依赖/公共配置、曲率/评估/积分mixin、核心类；更新所有局部导入，避免新循环依赖。保持super调用顺序，用无__init__mixin承接函数。
3. 迁GUI/设备/CLI/旧包工具，原root模块改成新实现反向兼容shim。__module__及co_filename必须指向新包；新包不可用动态exec加载旧文件，也不可sys.path注入旧仓库。
4. 更新现有八模块门面，使其调用新包内实现；新API可从配置创建任务引擎，ShapeIntegrator可脱离数据直接计算。修正path(__file__)假设，用户相对数据路径按既有配置约定处理。
5. 添加pyproject/setuptools包数据及可选gui/runtime依赖，构建wheel；新API、python -m和console entrypoint明确文档。用依赖已安装的当前Python执行pip wheel --no-deps --no-build-isolation，网络不必访问。
6. AST路径测试指向新实现；保留普通旧import兼容测试。新增隔离测试：只复制/安装wheel到临时目录，python -I并从仓库外运行，meta_path拒绝旧模块；实际调用曲率与积分、理想曲线及配置加载，不止--help。
7. 主窗口独立扫描新包导入、检查原函数AST/注释迁移、运行全量回归和仓库外wheel验收。通过后本地提交，原实验目录不改不纳入提交。

## 验证要求
- 新包core import不得导入Qt/VTK或连接设备；需要GUI时明确报依赖缺失。
- 数值等价：沿用旧向量插值、跨帧滤波、末端姿态、多区域拼接、真实点优化目标、相位传递回归。
- 结构等价：原函数签名/装饰器/方法体AST尽量一致；只改必需包内import/模块路径。原注释与IMU许可不得丢。
- wheel内部资源齐备，在没有旧py文件和tools目录的目录中仍可加载配置和执行计算。
- 项目内GUID --basetemp用于测试；不要删除既有目录。不要启动真实设备、覆盖实验数据或推送远端。
