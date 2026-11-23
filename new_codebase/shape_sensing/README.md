# 传感器算法使用说明

本文对应当前 `shape_sensing` 源码，覆盖准备配置、初始化、优化、合并参数、发布、离线回放和动态使用。示例使用 PowerShell，从 `new_codebase` 目录执行。

## 1. 安装与入口

先激活形状感知环境 `conda activate dc_ss`。新库支持 Python 3.8 或更高版本。若需要安装，使用该环境的 Python 在 `new_codebase` 目录执行以下命令。

```powershell
python -m pip install -e ".[gui,runtime]"
python main.py --help
```

开发时可在[模块数据接口](../docs/module-data-interfaces.md)查阅实际交接类型。曲线结果、误差和实时状态分别由重构、评估和运行服务显式返回。

基础数值功能可以用 `python -m pip install -e .` 安装。`gui` 包含 Matplotlib、PyQt5、VTK，初始化绘图、离线回放和 demo 会使用它们；`runtime` 包含串口依赖。YAML 的安装包名称是 `PyYAML`，导入名是 `yaml`。

`python main.py ...` 与 `python -m shape_sensing ...` 使用同一个任务入口。从仓库根目录也可以将下文的 `python main.py` 换成 `python .\new_codebase\shape_sensing\main.py`。源码脚本会把工作目录切换到 `new_codebase`。单独安装 wheel 后，可以使用 `python -m shape_sensing ...` 或 `shape-sensing ...`。下文的 `tools/*.py` 位于源码仓库，wheel 使用者可调用第 5 节给出的导出 API。

静态重构和实时/回放共享 `PreparedFrame`，保留单帧的物理通道列和真实点身份。固定距离和索引由设计提供。模块 3 对真实点计算曲率，模块 5 处理积分所需的网格，并返回 `ShapeResult` 的具名位置与标架字段。旧积分入口继续返回元组。

实时与回放使用同一处理路径。会话在有效帧后更新方向展开、曲率时间滤波及显示均值历史。记录、UDP 与窗口使用同一最终帧。记录中 `cur_interp` 是实际积分使用的曲率，启用时间滤波时 `cur_interp_raw` 保存同一网格上的滤波前曲率。

## 2. 每根传感器准备一份工作配置

```powershell
Copy-Item configs/sensor_template.yaml configs/my-sensor.yaml
```

编辑自己的 YAML。模板中的参数名、解释、选项继续作为完整参数参考。[参数模板](../configs/sensor_template.yaml)

| 配置内容 | 需要核对的字段 |
|---|---|
| 传感器设计 | `sensor_id`、`R`、`samp_dist_preset`、`start_value`、`end_value`、`full_segmentation` |
| 通道与设备 | `interrogator`、`active_channels`、`centre_fibre`、交错光栅和通道排布配置 |
| 定标实验 | `exp_directs`、`rot_dir`、各区域的 `folder_path`、`radii` |
| 算法与优化 | 曲率插值、滤波、`thresholds`、参数边界、`max_iter`、损失权重 |
| 任务范围 | `optimization_region`、`multi_region_calib`、`optimization_regions` |

`R` 使用米，FBG 点位、起止偏移及实验半径使用毫米。`ideal_len` 由 FBG 几何计算，可保留 `null`。`full_segmentation` 的总和必须与全长真实传感点数一致，末端区域分段必须符合区域几何。

新传感器的初值通常为：

```yaml
global_phases_set: null
weights: null
slopes: null
use_stored_zero: false
zero_reference_path: null
direction_blend_enabled: false
imu: false
```

同时检查 `calibration_regions` 各区域中的 `weights/slopes` 是否也是预期初值。只有 `init` 任务允许根据 `null` 相位执行初始化拟合；数值 `0` 表示明确提供零相位。`optim`、`score`、`recon`、`demo` 必须恢复或显式提供完整标定参数，构造不会调用模块2。

`rot_dir` 字典按实验方向角取值，例如 `{0: -1, 120: -1, 240: -1}`；历史列表按 `calib_shapes` 顺序取值，单个数值应用于全部实验。`sensing_ranges` 是 LUNA 专用参数，FBG 使用 `samp_dist_preset`。

### 路径的实际解析规则

工作 YAML 中的 `calibration_data_root` 相对 **该 YAML 所在目录**；区域 `folder_path` 相对数据根目录。`calibration_output_dir` 和 YAML 中的 `zero_reference_path` 相对该 YAML 所在目录。

当前 `configs/09B7_NeTi.yaml` 的数据根为 `../../09B7数据`。批量运行前，确认 `calibration_regions` 中每个目标区域的目录都存在。若数据放在其他位置，请调整 YAML 的数据根和各区域目录。单区域运行可用 `--folder_path` 覆盖所选区域的数据目录。

例如配置保存在 `configs/my-sensor.yaml`，数据保存在仓库根目录的 `my-data/`，可填写：

```yaml
calibration_data_root: ../my-data
calibration_output_dir: ../calibration_results
calibration_regions:
  full:
    folder_path: 全段测试
```

这只是路径字段示例，实际配置还需要保留模板中该区域的其他字段。若直接写 `calibration_output_dir: calibration_results`，结果会进入 `configs/calibration_results/`。

`runs/` 默认位于命令执行目录。后文命令行显式传入的相对路径也从命令执行目录定位。配置优先级为默认值 < YAML < 显式命令行参数。`--config` 和 `--sensor_params` 当前互斥。

## 3. 初始化与优化

### 四芯共同项芯位校准

`orthocal` 使用已经按直态归零的波长变化 CSV。输入目录包含三组方向 `0`、`120`、`240`，且每组提供相同的非零实验半径。工况目录命名为 `方向-半径`（如 `120-100`），每个激活物理通道在各目录提供一个 `Channel_<通道号>_*.csv` 文件，列名为原始 `WL_<编号>`，数值单位为 pm。程序对各列取算术平均，保留四芯原始变化量，在逐 WL 拟合中显式求出每个工况的共同响应 `b`。芯位校准使用单位权重，无需旧版中心芯扣除或二次归零。

```powershell
python -m shape_sensing orthocal --data-dir data/end15cm --geometry-json params/hardware.json --output-json params/my-end15cm-cores.json --region end15cm --active-channels 1 2 3 4
```

几何文件提供 `start_value`、`end_value` 和 `samp_dist_preset`（毫米），也可使用包含 `hardware_geometry` 的已有芯位参数 JSON。区域可选 `full`、`end15cm`、`end5cm`，默认 `full`。使用 `--radii-mm` 可指定半径子集，`--boundary-margin-mm` 可在末段区域起点处排除额外的边界点。七通道交错采集还需设置 `interlaced_grating` 和 `samp_dist_preset_sub`，并按中心、外芯 1 到 3 各自的 A/B 顺序提供物理通道。

结果为 schema 6 JSON，包含全局 WL 编号、实际距离、逐截面四芯的半径和角度，以及拟合与物理约束诊断。完整选项可通过 `python -m shape_sensing orthocal --help` 查看。

### 3.1 选择批量或单区域初始化

`--multi_region_calib True` 按 `optimization_regions` 的顺序初始化各区域。09B7 默认依次为 `full`、`end15cm`、`end5cm`，每个区域都拟合 `exp_directs` 中的方向。

```powershell
python main.py --config configs/09B7_NeTi.yaml --task init --multi_region_calib True
```

批量任务分别使用各区域的 `folder_path`、分段和阈值。程序先检查所有目标区域，再读取标定 CSV。09B7 会依次跑完 `full`、`end15cm`、`end5cm`，每个区域按 0°、120°、240° 逐一生成角度图、应变图及重构 MP4，共九组输出。视频包含重构曲线和对应的理论曲线，可在该次运行目录的 `outputs/<区域>/<方向>/` 中查看。所有区域完成后，终端会打印 `initialization.json` 的路径。该文件按区域保存相位、各方向的权重、斜率和对应的采样布局。

单区域模式的以下命令分别使用各区域对应的静态数据、阈值和分段，保存独立的初始化快照：

```powershell
python main.py --config configs/09B7_NeTi.yaml --task init --multi_region_calib False --optimization_region end5cm
```

将 `end5cm` 换成 `full` 或 `end15cm` 即可选择其他区域。单区域和多区域初始化快照均位于 `runs/<sensor_id>/<run_id>-init/initialization.json`。

模块 2 返回区域相位 `global_phases_set` 和按方向存放的分段斜率 `slopes_init`。任务层生成各等级的 `weights` 和 `slopes`，优先使用显式值。优化通过 `--resume` 读取相位和初值，无需把相位抄到 YAML。

绘图会保存图像和 MP4，MP4 导出需要可用的 FFmpeg。`init` 关闭存储零点，沿用静态数据预处理流程。要重新拟合，请运行新的 `init`；`--weights None --slopes None` 不会使优化任务执行初始化。

### 3.2 从匹配初值优化三个方向

批量初始化后，将 `$initRun` 设置为终端打印的初始化快照路径或该文件的上一级目录。三个方向都用这同一个快照，每次优化一个方向。

```powershell
$initRun = "<粘贴终端打印的 initialization.json 完整路径>"
foreach ($direct in @(0, 120, 240)) {
    python main.py --config configs/09B7_NeTi.yaml --task optim --multi_region_calib True --direct $direct --resume $initRun
    if ($LASTEXITCODE -ne 0) { throw "优化失败：$direct" }
}
```

只优化一个区域时，可使用该区域的单区域快照，也可从多区域快照中选取该区域。以下示例只优化 `end5cm` 的 240°：

```powershell
python main.py --config configs/09B7_NeTi.yaml --task optim --multi_region_calib False --optimization_region end5cm --direct 240 --resume $initRun
```

批量优化三个方向会产生 **三个 optim run**，每个 run 含该方向的所有目标区域。按区域和方向分别优化时，三区域、三方向共产生 **九个 optim run**。每次 `optim` 只优化 `--direct` 指定的方向。

续算会检查传感器、采样点、分段、曲率等级、权重分组及目标方向的初值。应使用与初始化匹配的配置。批量续算所用的快照须覆盖所有目标区域。单区域模式下显式给出的初值优先于续算值。相位 `0` 是有效初值。

### 3.3 已有完整初值时的批量任务

批量 `optim` 或 `score` 可从多区域初始化快照、已有多区域优化报告或 YAML 中各区域的完整初值读取参数。使用 YAML 初值时，请在 `calibration_regions` 的每个目标区域分别配置 `weights`、`slopes` 和 `global_phases_set`，并省略 `--resume`。

批量任务的半径、阈值、分段和初值写在各自区域中。若启用 `use_full_transported_phase`，必须已有完整初值，并按 `full → end15cm → end5cm` 记录上游相位引用；单独重做局部区域还需要匹配的来源参数包。该开关不会允许优化调用模块2。

## 4. 判断任务完成并找到对应结果

初始化结束后，查看快照目录的 `manifest.json`，其中 `status` 应为 `succeeded`。产物分布如下：

| 内容 | 路径 |
|---|---|
| 初值和重构误差 | `runs/<sensor_id>/<run_id>-init/initialization.json` |
| 多区域角度图和应变图 | 初始化快照目录下的 `outputs/<区域>/<方向>/init_plots/` |
| 多区域重构视频 | 初始化快照目录下的 `outputs/<区域>/<方向>/recon_plots/<旋转角>/<形状>/*.mp4` |
| 单区域绘图和视频 | `outputs/<sensor_id>/<通道>/<分段>/<方向>/`，分别位于 `init_plots/` 和 `recon_plots/` |

初始化快照保存误差与标定初值。三维重构曲线绘入 MP4，逐点 XYZ 坐标当前没有单独的导出文件。

每次 optim 会产生两类目录：

```text
calibration_results/<sensor_id>-<时间>-direct<方向>-<随机ID>/
    results.json
    ...各区域阶段参数、评分诊断和结果文件

runs/<sensor_id>/<时间>-<随机ID>-optim/
    manifest.json
    config_resolved.json
    optim.json
```

第一类目录实际位置取决于 `calibration_output_dir`。第二类是用于合并发布的任务快照；run 文件夹名称没有方向，方向在配置快照中。

| 文件 | 用途 |
|---|---|
| `manifest.json` | 任务类型、运行状态、代码来源 |
| `config_resolved.json` | 该次生效配置，包括 `direct` 和任务范围 |
| `optim.json` | 区域报告、最终参数、误差及来源信息 |
| `calibration_results/.../results.json` | 优化过程中逐步更新的区域报告 |

用于发布的 optim run 应为 `manifest.status == "succeeded"`，对应区域为 `status == "complete"`，且有 `weights/slopes`。完成状态与拟合误差应一起查看，优化器的收敛信息也保存在结果报告中。

可以列出当前传感器的 optim 快照。将 `my_sensor` 替换成 YAML 的 `sensor_id`：

```powershell
Get-ChildItem "runs/my_sensor" -Directory | ForEach-Object {
    $manifestPath = Join-Path $_.FullName "manifest.json"
    $configPath = Join-Path $_.FullName "config_resolved.json"
    if ((Test-Path -LiteralPath $manifestPath) -and (Test-Path -LiteralPath $configPath)) {
        $m = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $c = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        if ($m.task -eq "optim") {
            [pscustomobject]@{ Run = $_.FullName; Direct = $c.direct; Region = $c.optimization_region; MultiRegion = $c.multi_region_calib; Status = $m.status }
        }
    }
} | Format-Table -AutoSize
```

## 5. 合并参数并发布 JSON

### 5.1 从成功的 optim run 合并

导出脚本、`--runs` 和 `--output` 的相对路径均以 PowerShell 当前目录为基准。从仓库根目录操作时，请先执行 `Set-Location .\new_codebase`，再执行本节命令。

批量优化完成后，用 0°、120° 和 240° 各自的成功 run 合并。三个方向都用了多区域模式时，这三个 run 已包含全部九组区域/方向结果。`my_sensor` 是配置中的 `sensor_id`，可以与 YAML 文件名不同。

```powershell
$run0 = "runs/my_sensor/实际0方向optim目录"
$run120 = "runs/my_sensor/实际120方向optim目录"
$run240 = "runs/my_sensor/实际240方向optim目录"

python tools/export_algorithm_package.py --runs "$run0" "$run120" "$run240" --output "packages/my_sensor-v1.json"
```

`--runs` 接收包含 `manifest.json`、`config_resolved.json`、`optim.json` 的目录。直接把 `calibration_results/.../results.json` 传给该选项，会因缺少快照配套文件而失败。

逐区域单独优化时，用九个成功 run 导出完整发布包：

```powershell
$optimRuns = @(
    "runs/my_sensor/实际full-0-optim目录",
    "runs/my_sensor/实际full-120-optim目录",
    "runs/my_sensor/实际full-240-optim目录",
    "runs/my_sensor/实际end15cm-0-optim目录",
    "runs/my_sensor/实际end15cm-120-optim目录",
    "runs/my_sensor/实际end15cm-240-optim目录",
    "runs/my_sensor/实际end5cm-0-optim目录",
    "runs/my_sensor/实际end5cm-120-optim目录",
    "runs/my_sensor/实际end5cm-240-optim目录"
)
python tools/export_algorithm_package.py --runs @optimRuns --output "packages/my_sensor-multiregion-v1.json"
```

每个 **区域/方向** 组合选定一份结果，重复组合会报错。重新优化某一方向后，重新选择所需的成功快照并导出新的包。

导出工具检查 sensor_id、传感器设计、区域布局、标定模式及参数完成状态，并按区域和方向组织结果。导出 init 快照时，初值记录的 `final_calibration` 为 false。发布时选择成功的 optim 快照，demo 通过 `--sensor_params` 从导出的包读取相位、权重和斜率。

只有 wheel 或想从 Python 发布时，可使用相同 API：

```python
from shape_sensing.artifacts.packages import export_package

export_package(
    ["runs/my_sensor/实际0方向optim目录",
     "runs/my_sensor/实际120方向optim目录",
     "runs/my_sensor/实际240方向optim目录"],
    "packages/my_sensor-v1.json",
)
```

### 5.2 发布包覆盖范围

| 已完成结果 | 发布包覆盖范围 | 调用方式 |
|---|---|---|
| full 的 0/120/240 三次结果 | 全长单区域、三方向 | 单区域回放或选定方向的 demo |
| full/end15cm/end5cm 各自的 0/120/240 结果 | 三区域、三方向，共九组 | 多区域拼接回放和 demo |

新包的主要字段为 `configuration`、`calibration[region][direct]`、`regions`、`scope` 和 `provenance`。相位、旋转角、误差和源 run 路径随参数一起保存。不要将不同传感器的结果拼接在同一包中。

### 5.3 只有旧式 calibration_results 的兼容入口

仓库另有 `tools/sensor_parameter_package.py --results ...`，导出旧 `format_version=2` 格式。它要求恰好 0/120/240 三份报告，且每份都包含 full、end15cm、end5cm 的完整结果。

```powershell
python tools/sensor_parameter_package.py --results "calibration_results/实际direct0目录/results.json" "calibration_results/实际direct120目录/results.json" "calibration_results/实际direct240目录/results.json" --output "packages/my_sensor-legacy-v1.json"
```

该兼容入口以 parser 默认配置为基础，再吸收区域报告，不能传入 `--config`。自定义传感器的新工作流优先保存并使用第 5.1 节的 run 快照。旧工具的 `--base_results` 用于将三方向的局部 end15cm/end5cm 报告更新到完整旧包，接口与新包的 run 合并不同。

### 5.4 发布与跨机器迁移

用新的文件名区分每次发布，例如 `my_sensor-v1.json`、`my_sensor-v2.json`。导出成功表示参数已汇总，包内验证标记初始为 `verified: false`、`deployment_ready: false`；新格式包的多区域 demo 会显示相应验证提示，当前代码允许用于动态验证。

若导出时 `configuration.zero_reference_path` 有值，工具会复制该文件到 `<包名>_resources/` 并改写相对引用。未准备零点文件时保持 `null`，demo 阶段也可以通过命令行单独提供。

交付 JSON、存在时的资源目录、所用算法源码版本或 wheel，以及目标任务需要的数据和模型文件。新服务的构造不读取标定 CSV；demo 只加载设计、最终标定参数与动态数据源。静态 recon 在执行时仍需待重构的实验信号数据，参数包中保存的数据路径可能是绝对路径，跨机器时可以用 `--folder_path "目标机器上的全段测试目录"` 覆盖；多区域重新优化时，工作 YAML 的各区域路径应一起更新。参数 JSON 不包含原始实验数据。

### 5.5 同时交付算法库 wheel

在源码仓库根目录构建当前算法实现的 wheel：

```powershell
python -m pip wheel . --no-deps --wheel-dir dist
```

将生成的 `.whl` 一起交付，目标环境安装该文件并准备 GUI/设备依赖。参数 JSON 与算法 wheel 分别记录发布版本。若沿用同一 wheel 版本号重新构建，已安装环境需要强制重装才能使用本次代码；依赖已准备好时可执行：

```powershell
python -m pip install --force-reinstall --no-deps "dist/shape_sensing-0.1.0-py3-none-any.whl"
```

## 6. 使用发布包回放重构

### 全长单区域包

```powershell
python main.py --sensor_params packages/my_sensor-v1.json --task recon --multi_region_calib False --optimization_region full --direct 0 --exp_directs "[0]"
python main.py --sensor_params packages/my_sensor-v1.json --task recon --multi_region_calib False --optimization_region full --direct 120 --exp_directs "[120]"
python main.py --sensor_params packages/my_sensor-v1.json --task recon --multi_region_calib False --optimization_region full --direct 240 --exp_directs "[240]"
```

静态 `recon` 按 `--direct` 选出一套参数；未显式指定 `--exp_directs` 时，读取范围自动使用该方向，示例中的显式范围也可以保留。

### 完整三区域包

```powershell
python main.py --sensor_params packages/my_sensor-multiregion-v1.json --task recon --multi_region_calib True --optimization_region full --direct 0
```

改为 `--direct 120` 或 `240` 检查其他方向。多区域重构要求包内有 full、end15cm、end5cm 且覆盖三个方向。

## 7. 动态使用 demo

### 7.1 单区域包，选择一套方向参数

下面以 0 方向为例，设备地址和端口按实际连接修改。示例要求发布包的 `configuration.direction_blend_enabled` 和 `configuration.imu` 为 false，建议在工作 YAML 中设置并随运行快照发布：

```powershell
python main.py --sensor_params packages/my_sensor-v1.json --task demo --multi_region_calib False --optimization_region full --direct 0 --exp_directs "[0]" --direction_reference_direct 0 --use_stored_zero False --IP "192.168.1.75" --PORT 26546
```

选择 120 方向时，将 `direct`、`exp_directs`、`direction_reference_direct` 同时改为 120；240 方向同理。`exp_directs` 可以省略，单区域 demo 会加载该区域包内所有可用方向的参数。`direction_reference_direct` 的 parser 默认值为 240，因此选择其他方向时要显式覆盖。

新格式单区域 demo 会加载所选区域全部可用方向的参数，并使用 `direction_reference_direct` 对应的相位和旋转设置。未启用混合时，只使用参考方向参数计算帧；启用 `--direction_blend_enabled` 时，必须有 0/120/240 三套参数，即使显式设置 `--exp_directs "[0]"` 也会保留混合需要的三套参数。显式 `--weights`、`--slopes`、相位和旋转覆盖仍优先，若覆盖后缺少必要方向会清楚报错。

### 7.2 完整三区域包

```powershell
python main.py --sensor_params packages/my_sensor-multiregion-v1.json --task demo --multi_region_calib True --optimization_region full --direct 0 --direction_reference_direct 0 --use_stored_zero False
```

多区域 demo 当前要求关闭 `direction_blend_enabled`，使用选定参考方向进行区域拼接。若改用 120 或 240 参考方向，同步修改 `--direct` 和 `--direction_reference_direct`。`direction_blend_enabled` 的三方向混合功能还要求引擎同时装有三套形状兼容的参数；运行时明确拒绝多区域参数与方向混合同时启用。

demo 会创建 Qt/VTK 窗口，启动所选解调仪接收线程；启用 IMU 时还会连接相应串口。退出窗口后执行运行时清理。

### 命令行布尔参数的现有语义

`--use_stored_zero`、`--multi_region_calib` 接受显式 `True/False`。其他布尔参数的解析方式并不完全相同：

- `--direction_blend_enabled` 是 `store_true` 开关，命令行出现即开启；关闭时省略该选项，并让发布包的 `configuration.direction_blend_enabled` 为 false。`--direction_blend_enabled False` 会被解析为多余参数。
- 当前 `--imu` 使用 Python 的 `bool` 转换，字符串 `False` 也会变成 true。关闭 IMU 时在工作 YAML 中写 `imu: false`，并确认发布包的 `configuration.imu` 为 false，命令行省略 `--imu`。
- 已有包若启用了上述开关，可在用于动态运行的发布副本 `configuration` 中将它们设为 false，并记录这次运行配置调整。

### 7.3 demo 可选的存储零点

| 选择 | 设置 | 零点来源 |
|---|---|---|
| 使用首帧归零，默认 | `use_stored_zero: false`，`zero_reference_path: null` | 实时接收的首帧 |
| 使用已有零点 | `use_stored_zero: true`，填写 `zero_reference_path` | 匹配当前传感器的零点 JSON |

init、ang、optim、score、recon 都关闭存储零点。demo 仅在显式启用时要求文件，文件的 SensorID、通道和栅点数必须匹配当前传感器。

若需生成零点文件，在传感器处于预期零状态时采集原始 CSV，再按实际通道与解调仪生成：

```powershell
python tools/build_zero_reference.py "实际零状态CSV目录" --sensor-id my_sensor --interrogator FiberTecc --active-channels 1,2,3,4 --output params/my_sensor_zero_reference.json
```

在第 7.1 或 7.2 节对应命令中，将 `--use_stored_zero False` 替换成：

```text
--use_stored_zero True --zero_reference_path "params/my_sensor_zero_reference.json"
```

## 8. 常见问题与参数维护

| 情况 | 检查位置与处理 |
|---|---|
| `No module named yaml` | 在当前 Python 环境运行 `python -m pip install PyYAML` |
| 初始化要求零点文件 | 当前代码默认关闭零点并接受 `null`，检查是否运行了旧源码或旧 wheel |
| 多区域续算提示初值或布局不匹配 | 确认使用了本次多区域 init 快照，并核对 YAML、目标区域和方向。旧版单区域快照可用于原区域的单区域续算 |
| 数据目录被定位到 `configs/` 下 | 工作 YAML 的相对根路径以 YAML 目录为基准，指向仓库根目录数据时通常需要 `../` |
| `sensing_ranges` 字符串通道键 | YAML 加载器会将数字键转为整数；FBG 不依赖 LUNA 区间 |
| `rot_dir` 方向查找失败 | 字典按方向角读，列表按标定形状读，检查实际需要的方向是否有配置 |
| 合并提示缺少快照配置 | `--runs` 应指向任务快照目录，核对三个配套 JSON |
| 合并提示重复方向 | 同一区域/方向输入了多份结果，选定一份成功 run |
| demo 提示方向参数缺失 | 检查包中是否包含参考方向；混合模式必须含0/120/240三套，显式参数覆盖也须满足此范围 |
| 改 YAML 后发布包没有变化 | 已发布 JSON 是快照，重新选择结果并导出新包；demo 可用 CLI 覆盖设备参数 |

公开 API 的字段、shape、单位、输入是否被修改及状态归属见 [模块数据接口](../docs/module-data-interfaces.md)。实时运行与无窗口回放共同调用 `TaskSession.process_frame`，源连接、队列、IMU、锚定与记录归模块8管理。

数值实现按八模块维护，任务流程位于 [pipelines/tasks.py](pipelines/tasks.py)，参数解析位于 [cli.py](cli.py)，配置加载位于 [configuration/loader.py](configuration/loader.py)，新包导出位于 [artifacts/packages.py](artifacts/packages.py)。

修改源码后使用可编辑安装可直接使用工作树代码；已分发的 wheel 是构建时快照，需要重新构建、安装才能包含后续修复。


## 9. 09B7 的区域配置

| 区域 | 相对数据根的目录 | 全局真实点 | 分段 |
|---|---|---|---|
| `full` | `全段测试` | 1–22 | `[2, 5, 2, 4, 2, 2, 2, 3]` |
| `end15cm` | `末端测试/15cm` | 16–22 | `[2, 2, 3]` |
| `end5cm` | `末端测试/5cm` | 18–22 | `[2, 3]` |

`end5cm` 的三组权重对应 18–20、21、22 号点，分组索引为 `[0, 0, 0, 1, 2]`。当前布局的初值应由这套配置重新初始化生成。

已在 `dc_ss` 使用真实 09B7 数据完成批量初始化，三个区域各写出 6 张 PNG 和 3 段重构 MP4。也已验证九组区域/方向的限迭代优化，以及从批量快照恢复 `end5cm` 的 240° 单区域初值。优化验收详情见 [数据契约清理台账](../docs/data-contract-cleanup.md)。
