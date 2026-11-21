# 静态定标数据与理论曲线导出

接收和使用本次 JSON 时，优先阅读数据目录中的 [JSON 数据说明（交接版）](../1020-08B5-导管两端固定全段+小圈定标测试/calibration_reference_data_README.md)，其中包含完整字段字典、采样点对应关系和可运行的读取示例。

在仓库根目录运行：

```powershell
python tools/generate_calibration_reference.py
```

默认读取 `1020-08B5-导管两端固定全段+小圈定标测试` 和
`params/08B5_NeTi.json`，在所选数据目录下生成 `calibration_reference_data.json`。
也支持 `python -m tools.generate_calibration_reference`，以及从仓库外使用脚本绝对路径运行。

```powershell
# 覆盖输入、参数包、输出；相对路径相对于当前工作目录。
python tools/generate_calibration_reference.py --data-dir "数据目录" --sensor-params "params/08B5_NeTi.json" --output "输出.json"

# 只导出指定方向、指定形状；不指定 --output 时仍写入所选数据目录。
python tools/generate_calibration_reference.py --direct 0 60 120 240 --calibration-shape end

# 旧版平铺目录（如 定标测试/0-1000），保持旧版 +y 基准坐标。
python tools/generate_calibration_reference.py --data-dir "旧数据目录" --calibration-shape circ --angle-convention legacy-y

# 新定义的数据若放在平铺目录中，需显式指定 +z 基准。
python tools/generate_calibration_reference.py --data-dir "平铺的新数据目录" --angle-convention z-positive
```

## 输入与坐标约定

- 支持平铺半径目录，以及 `0度/0-1000`、`60度/60-1000`、`120度/120-1500-100` 这样的两层目录。
  方向父目录与半径目录中的方向数字必须一致。
- 默认按实际目录自动发现方向，不局限于参数包原有的 0、120、240 度。
  因此新增 60 度不需要改动参数包的 `exp_directs` 或主体定标流程。
  每次运行会分别列出各方向的 circ/end 记录数量，便于检查数据是否全部导出。
- `--angle-convention auto` 是默认值：带 `数字度` 父目录时使用 `z-positive`；
  平铺目录使用 `legacy-y`。也可用该参数显式覆盖，文件夹布局本身并不能证明物理安装方向。
- 新定义：初始切线沿 **+x**，0° 向 **+z** 弯曲，角度正向遵循绕 **+x** 的右手定则。
  60° 朝 −y、+z，120° 朝 −y、−z，240° 朝 +y、−z。
  对 Optimiser 生成的 `(x, lateral, 0)` 作刚体旋转得到
  `(x, -lateral*sin(direct), lateral*cos(direct))`。
- 旧定义 `legacy-y`：0° 向 +y 弯曲，其他方向沿用项目旧的顺时针定义，
  得到 `(x, lateral*cos(direct), -lateral*sin(direct))`。旧版 0° circ 导出坐标保持一致。
- `circ` 目录为 `方向-半径`；半径 0 表示直线。
  期望半径与顺序来自参数包的 `regions.full.radii.circ`。

默认 `--end-profile catheter-1020` 对目录作以下物理解释，第二个数字就是前段实际半径：

| end 目录 | 前段实际半径 | 最后圆弧长度 | 末段半径 |
| --- | --- | --- | --- |
| `方向-1500-x` | 1500 mm | **0.15 m** | x mm（默认 150、100、50） |
| `方向-2000-x` | 2000 mm | **0.05 m** | x mm（默认 30、20、15） |

当前理论全长为 0.895 m，因此 1500 组的拼接点为 0.745 m，2000 组为 0.845 m。
拼接点之前，`方向-1500-x` 与同方向 `方向-1500` 的理论坐标一致，
`方向-2000-x` 则与同方向 `方向-2000` 一致；之后接入半径 x 的末段圆弧。

当前参数包仅定义 circ 半径，end 分组与末段半径使用工具中的 `DEFAULT_END_GROUPS`。
如参数包有 `regions.full.radii.end`，则按其中的分组及半径列表检查；
新定义的分组标记限于 1500、2000，未知分组报错。
旧实验可显式使用 `--end-profile legacy`，按目录第二个数字作为前段半径，
末段长度取参数包 `shared_parameters.calibration_experiment_parameters.end_len`。
角度约定和 end 分组约定可分别选择。

原始数据直接使用 `Sensor.Data`，复用静态 CSV 清洗、逐通道求均值、通道排序、
尺度转换以及负坐标裁切，不使用中心光纤补偿后的 `processed_Data`。
理论弧长覆盖 `Sensor.ad_samp_dist` 的完整范围，包括虚拟起点和末端点；
点数取参数包的 `shared_parameters.reconstruction_parameters.interp_num`（当前为 120）。

所有圆弧及分段拼接均调用现有 `Optimiser._construct_ideals()`；工具只作参数适配和坐标旋转。
现有 `_construct_ideal_end()` 的“两段圆弧按前段末端切线旋转、平移”的几何构造正确，
位置和切向连续；但原流程按 `self.samp_dist[-1] - self.end_len` 定位拼接点，且每个实例共用一个末段长度。
工具按分组分别设置末段长度，将构造器的弧长域末端设为理论全长，
使最后 5 cm／15 cm 从物理末端量取。当前末端虚拟点比最后一个真实点多 6 mm，
因此若直接使用原始 `sensor.samp_dist`，末段会多出这 6 mm。
`Sensor`、`Optimiser`、`task init`、`task optim` 均未修改。

## JSON 字段

继续使用 `static_calibration_raw_and_ideal_v1` 格式标识，保留旧字段并增加记录级元数据。
单方向、单类型导出保持顶层 `direct` 为数值、`calibration_shape` 为 `circ`／`end`；
多方向导出时顶层 `direct` 为 `null`、完整方向存入 `directs`；混合类型时
`calibration_shape` 为 `mixed`，具体类型存入 `calibration_shapes`。
多方向／混合导出的读取方应使用每条记录的 `direct` 和 `calibration_shape` 识别样本。

| 字段 | 说明 |
| --- | --- |
| `format` | 格式标识 |
| `source_data_dir`, `sensor_params` | 来源数据目录、参数包路径 |
| `direct`, `directs` | 单方向值、完整导出方向列表，单位度 |
| `calibration_shape`, `calibration_shapes` | 单类型或 mixed、完整类型列表 |
| `angle_convention`, `coordinate_frame`, `end_profile` | 角度基准、坐标约定、分组解释 |
| `raw_channel_ids`, `raw_channel_order` | Sensor 使用的通道编号与矩阵列顺序 |
| `raw_point_ids_1based` | 裁切后原始数据的行编号；当前为 2–22，不冒充硬件 SensorID |
| `raw_sample_distances_m` | 裁切后真实传感点的采样距离，m |
| `ideal_arc_distances_m` | 理论点相对于理论起点的弧长，m；当前 0–0.895 |
| `raw_point_count`, `ideal_point_count` | 原始点数、理论点数；当前为 21、120 |
| `units` | 各数据的单位或尺度说明；原始通道值是 CSV 清洗均值乘 1e-6 |
| `skipped_empty_directories` | 默认发现时跳过的完全没有文件的角度目录，命令行也会说明 |
| `radii` | 按方向、配置半径顺序排列的样本数组 |

每个 `radii` 条目：

| 字段 | 说明 |
| --- | --- |
| `radius_mm` | circ 的半径，或 end 的**末段半径**，mm |
| **`raw_data`** | **传感点 × 通道**，当前每条为 **21×4** |
| **`ideal_coordinates_m`** | 理论曲线 `(x,y,z)`，当前每条为 **120×3**，m |
| `direct`, `calibration_shape` | 此条记录的方向、circ／end 类型 |
| `sample_id`, `source_folder` | 原始半径目录名、来源路径 |
| `directory_radius_mm` | 仅 end：目录第二个数字，表示前段半径及实验分组，与 `base_radius_mm` 一致 |
| `base_radius_mm` | 仅 end：前段实际半径，mm |
| `end_length_m` | 仅 end：末段圆弧长度，m |
| `junction_arc_distance_m` | 仅 end：两段拼接点的理论弧长，m |

不同方向、不同 end 分组可能具有相同 `radius_mm`，应以 `sample_id` 识别记录。

## 检查与当前数据状态

工具检查配置与半径目录的一致性、读取后的有效数据、21 个裁切点、理论矩阵维度和有限值。
任何已开始采集的方向缺少配置半径、目录存在但通道缺失／无效、存在多余半径，均报出具体路径和半径。
只在角度父目录**完全没有文件**时默认跳过该方向；显式 `--direct` 指定该空方向时仍报错。
全部数值通过 `allow_nan=False` 序列化检查后才写入输出文件。

2026-10-21 补充数据后，工作区有 0°、60°、120°、240° 四个方向，各有 60 个 CSV，
每个方向包含 9 条 circ、6 条 end，默认共导出 **60 条记录**，没有跳过的空方向。
`60度` 内原先保留 `120-…` 名称的 15 个半径子目录已按实际方向批量改成 `60-…`；
CSV 文件名及内容未修改。脚本继续检查父子目录方向一致性，不自动改写数据目录。

脚本运行后会报告输出文件是否被 Git 忽略，**不修改 `.gitignore`**。
当前新数据目录及默认 JSON **未被现有 `.gitignore` 忽略**；旧 `261017_易老师数据` 下的输出则被现有目录规则忽略。

验证记录：实际默认命令已成功生成上述 60 条记录，全部矩阵维度正确且数值有限，
可以用 `allow_nan=False` 严格序列化。以下相关测试共 **25 项通过**，包括 60° 的
circ/end 旋转、四方向自动发现、两种末段长度及命令行各方向数量汇总：

```powershell
python -m pytest tests/test_generate_calibration_reference.py tests/test_tools_entrypoints.py tests/test_zero_reference_trimming.py -q
```

全量 `python -m pytest -q` 在收集阶段因当前环境缺少 `vtk` 中断，无法据此声明全仓库测试通过。
受影响模块为 `tests/test_demonstrator_anchor_pipeline.py`、`tests/test_demonstrator_end_pose.py`、
`tests/test_end15cm_point_calibration.py` 和 `tests/test_single_direction_recon.py`。
