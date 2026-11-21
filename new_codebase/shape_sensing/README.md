# 传感器算法使用说明

本文对应当前 `shape_sensing` 源码，覆盖准备配置、初始化、优化、合并参数、发布、离线回放和动态使用。示例使用 PowerShell，从仓库根目录执行。

## 1. 安装与入口

激活实际运行算法的 Python 环境，例如 `conda activate dc_Shape`。在仓库根目录安装所需依赖：

```powershell
python -m pip install -e ".[gui,runtime]"
python main.py --help
```

基础数值功能可以用 `python -m pip install -e .` 安装。`gui` 包含 Matplotlib、PyQt5、VTK，初始化绘图、离线回放和 demo 会使用它们；`runtime` 包含串口依赖。YAML 的安装包名称是 `PyYAML`，导入名是 `yaml`。

`python main.py ...` 与 `python -m shape_sensing ...` 使用同一个任务入口。单独安装 wheel 后，可以使用 `python -m shape_sensing ...` 或 `shape-sensing ...`。下文的 `tools/*.py` 位于源码仓库，wheel 使用者可调用第 5 节给出的导出 API。

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

### 3.1 每个区域显式初始化

以下命令分别使用各区域对应的静态数据、阈值和分段，保存独立的初始化快照：

```powershell
python main.py --config configs/my-sensor.yaml --task init --multi_region_calib False --optimization_region full
python main.py --config configs/my-sensor.yaml --task init --multi_region_calib False --optimization_region end15cm
python main.py --config configs/my-sensor.yaml --task init --multi_region_calib False --optimization_region end5cm
```

只使用全长单区域时，执行第一条即可。初始化对 `exp_directs` 中的方向拟合相位及分段扭转初值，再重构、评估并绘图；每次保存 `runs/<sensor_id>/<run_id>/initialization.json`。各区域的点数和分段不同，不能将 full 初值直接用于 end15cm/end5cm。

绘图会保存图像和 MP4，MP4 导出需要可用的 FFmpeg。`init` 关闭存储零点，沿用静态数据预处理流程。要重新拟合，请运行新的 `init`；`--weights None --slopes None` 不会使优化任务执行初始化。

### 3.2 从匹配初值优化三个方向

将下列三个目录替换成刚完成的对应区域 init run。默认的独立区域工作流保持 `use_full_transported_phase: false`；已有上游相位传递流程见第 3.3 节。

```powershell
$initFull = "runs/my_sensor/实际full-init目录"
$initEnd15 = "runs/my_sensor/实际end15cm-init目录"
$initEnd5 = "runs/my_sensor/实际end5cm-init目录"
$initializations = @{ full = $initFull; end15cm = $initEnd15; end5cm = $initEnd5 }

foreach ($region in @("full", "end15cm", "end5cm")) {
    $seedRun = $initializations[$region]
    foreach ($direct in @(0, 120, 240)) {
        python main.py --config configs/my-sensor.yaml --task optim --multi_region_calib False --optimization_region $region --direct $direct --resume "$seedRun"
        if ($LASTEXITCODE -ne 0) { throw "优化失败：$region / $direct" }
    }
}
```

这会产生九个 optim run，每个区域/方向各一个。只优化全长时，将区域循环改为 `@("full")`，只需 `$initFull`。**每次 optim 只优化 `--direct` 指定的方向。** `exp_directs` 说明可加载的数据方向，不会代替三次方向优化。

续算会检查传感器、几何、区域、阈值和参数布局。显式命令行种子优先于续算参数；若显式设置 `None` 导致缺少必要初值，任务会在读取标定 CSV 前报错并指导运行 `init` 和 `--resume`。正常的零相位 `0` 不会触发重新拟合。

### 3.3 已有完整初值时的批量任务

只有 `calibration_regions` 的每个区域都已有匹配的 `weights`、`slopes`、`global_phases_set`，或恢复包含所有目标区域结果的优化报告时，才能使用 `--multi_region_calib True`。系统会先检查全部区域初值，再开始读取和优化；顶层 full 初值不会盲填到局部区域。

若已具备这些条件，可按方向执行：

```powershell
python main.py --config configs/my-sensor.yaml --task optim --multi_region_calib True --optimization_regions "['full','end15cm','end5cm']" --direct 0
python main.py --config configs/my-sensor.yaml --task optim --multi_region_calib True --optimization_regions "['full','end15cm','end5cm']" --direct 120
python main.py --config configs/my-sensor.yaml --task optim --multi_region_calib True --optimization_regions "['full','end15cm','end5cm']" --direct 240
```

批量任务的半径、阈值、分段和初值写在各自区域中。若启用 `use_full_transported_phase`，必须已有完整初值，并按 `full → end15cm → end5cm` 记录上游相位引用；单独重做局部区域还需要匹配的来源参数包。该开关不会允许优化调用模块2。

## 4. 判断任务完成并找到对应结果

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

### 5.1 从三个成功的 run 合并

将下面三个占位目录替换成实际的成功目录。`my_sensor` 是配置中的 `sensor_id`，可以与 YAML 文件名不同。

```powershell
$run0 = "runs/my_sensor/实际0方向optim目录"
$run120 = "runs/my_sensor/实际120方向optim目录"
$run240 = "runs/my_sensor/实际240方向optim目录"

python tools/export_algorithm_package.py --runs "$run0" "$run120" "$run240" --output "packages/my_sensor-v1.json"
```

`--runs` 接收包含 `manifest.json`、`config_resolved.json`、`optim.json` 的目录。直接把 `calibration_results/.../results.json` 传给该选项，会因缺少快照配套文件而失败。

按第 3.2 节分别优化三个区域时，用九个成功 run 导出完整发布包：

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

导出工具检查 sensor_id、传感器设计、区域布局、标定模式及参数完成状态，并按区域和方向组织结果。init 快照导出的是初值包，`final_calibration` 为 false；最终标定发布应选择 optim 快照。

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


## 9. 当前示例传感器的局部配置说明

`configs/09B7_NeTi.yaml` 的 full 配置可用于前述全长流程。当前原有 end5cm 配置的几何覆盖5个真实点（18–22），但局部分段为 `[2,2]`，并启用了限定4点（19–22）的独立权重模式，不能直接执行该区域的初始化/优化。需要先根据实际实验确定一致的点位、区域和分段，再重新初始化，不能沿用不同布局的参数。本次切分没有替使用者更改这些设计值。具体核查和临时验收条件见 [八模块切分验收记录](../docs/module-split-acceptance.md)。
