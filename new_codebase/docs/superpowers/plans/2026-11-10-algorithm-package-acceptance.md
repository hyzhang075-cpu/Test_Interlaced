# 主聊天独立验收清单

分支：AlgorithmPackage-08B5；基准：e27b245；实施聊天：01a1c343-adee-7410-93e3-167cb93160a6（gpt-6-luna/high）。

## 基线
项目内独立 basetemp 的全量 pytest：330 passed, 2 failed，27.96s。
1. test_parser_and_exported_package_use_project_relative_paths：测试数据 0908-06B6 与当前 parser 默认 calibration_data_root 不一致。
2. test_main_vector_filter_defaults_off_and_accepts_explicit_true：main 实际默认 True，旧断言 False。原 Reconstructor 构造默认 False，两个入口的差异本轮不悄悄改动。
日志：.pytest_cache/algorithm-package-baseline.log。
原数值函数 AST 与注释参考：.pytest_cache/algorithm-package-reference.json。

## 主聊天验收（待实施结束后独立执行）
- [x] 查看工作区 diff、分支、原实验数据目录和未跟踪文件。
- [x] 核对 sensor_template 所有有效参数的名称、解释、choices、null/单位、区域默认值；说明与实际一致。
- [x] 实际测试无结果的新传感器配置、显式 false/0/null、非法字段/重复键及 CLI 优先级。
- [x] 八模块接口真实职责、任务调用与无 GUI 导入；原函数和注释保留对比。
- [x] 曲率、空间积分和末端姿态新旧数值比较。
- [x] 配置快照、运行失败记录、显式续算与不兼容来源拒绝。
- [x] 发布包独立于当前配置，旧 format_version=2 可用，区域/方向适用范围真实。
- [x] 全量 pytest 新运行、main/工具入口检查、compileall、git diff --check。
- [x] 明确区分代码/离线回放测试与尚未进行的真实设备和真实数据优化。

仅在上述证据齐全后报告验收结果。初期保留 legacy 实现属于设计范围，空壳目录和未接入接口不算封装完成。


## 最终验收结论（2026-11-10，主聊天）

通过代码级验收，采用保留原类和数值实现的八模块适配式封装。实现由独立 gpt-6-luna 聊天执行，主聊天运行独立测试、数值对照和只读代码审查。

- 主聊天最后一次全量：365 passed in 37.08s，退出码0。日志：.pytest_cache/algorithm-package-root-acceptance.log。
- main.py --help、导出工具脚本及模块 --help、compileall、git diff --check：均退出码0。
- 参数目录与当前parser：110个原有效字段，9组choices，字段/帮助/choices逐项一致。
- 独立验收覆盖真实旧继承构造参数签名（读取数据前主动停止）、曲率积分数值一致、理想曲线、显式零相位、配置类型、初始化结果续算、三方向三区域导出后真实规范化及完整owned点覆盖、无GUI导入、假设备正常/异常退出清理。
- 独立审查确认：batch续算逐区域恢复weights/slopes/相位/旋转角；顶层阈值和全长分段覆盖参与兼容校验；weights-only的已验证resume来源可用；内部控制字段不进入Sensor构造参数。
- 原复现：三个区域phase为0.1/0.2/0.3、rot_ang为1/2/3，恢复后实际jobs各自对应；显式CLI rot_ang=9时各区域均为9。
- 七个原核心文件函数AST和注释核对：除initialiser.__init__零相位判定修复外，原数值函数未变；原有注释无遗漏。拟合、曲率和积分公式未改。
- 旧基线的两项失败采用test-only修正：路径fixture显式指定自身root，向量滤波parser断言对齐原本实际True并保留显式False验证；未改变运行默认值或放松路径校验。
- 用户原有两个未跟踪实验目录保持原样，不纳入提交。

边界：实际计算主体和继承链仍在legacy类中，模块公开接口及任务流程现已接通；现有算法对硬件、区域、方向的适用限制继续有效。旧代码的初始化/数据读取约束仍保留，迁移工作配置时需配置原始标定数据根目录。未运行真实设备、生产数据优化或宣称部署验证通过；发布包验证状态保持未验证。

本地交付：保留AlgorithmPackage-08B5分支，不合并或推送；本次实现和验收文档保存为本地提交。
