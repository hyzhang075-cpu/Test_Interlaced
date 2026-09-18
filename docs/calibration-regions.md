# 全长与嵌套区域定标配置

本阶段实现定标数据选择、正确几何、全局分段校验和按区域批量优化。每个区域仍使用独立 Optimiser 执行现有的先 slope、后 weight 流程；本阶段的循环不是三区域联合损失优化。

## main 中的配置

这些配置现在全部位于 `build_argument_parser()` 内，可随整套传感器 parser 参数一起复制和迁移：

```python
parser.add_argument('--multi_region_calib', default=False, ...)
parser.add_argument('--optimization_region', default='full', ...)
parser.add_argument('--optimization_regions', default=['full', 'end15cm', 'end5cm'], ...)
parser.add_argument('--full_segmentation', default=[2,3,5,2,3,3,2,2], ...)
parser.add_argument('--calibration_data_root', default='0908-06B6', ...)
parser.add_argument('--calibration_regions', default={...}, ...)
```

新传感器先使用 False/full 完成全长定标和可用性验证。`multi_region_calib=True` 且 task 为 optim 或 score 时，按 `optimization_regions` 的顺序遍历全部区域。`multi_region_calib=False` 时，init/recon 可由 `optimization_region` 单独选择 full、end15cm 或 end5cm。`multi_region_calib=True` 的 recon 使用完整参数包和全长几何。各区域的长度、数据目录、半径、阈值、局部分段和可选初值集中在 `calibration_regions` 字典；其他长度可用同样字段添加条目。

| 选择 | 数据目录（0908-06B6 下） | 真实数据行号（1-based） | 分段 | 重构长度 |
|---|---|---|---|---|
| full | 全段测试 | 1–22 | [2,3,5,2,3,3,2,2] | 877 mm |
| end15cm | 末端测试/15cm | 16–22 | [3,2,2] | 150 mm |
| end5cm | 末端测试/5cm | 19–22 | [2,2] | 50 mm |

开启区域模式会检查所有已配置长度，确保它们共享同一全长空间分段。关闭区域模式时只检查并运行当前 `optimization_region`。

全长长度计算：`(7 + (1370 - 510) + 10) / 1000 = 0.877 m`。
Sensor 与 main 共用几何计算函数，直接构造 Sensor 也会根据 FBG 几何重新计算全长。OFDR/LUNA 全长路径仍保留显式长度设置，不使用 FBG 跨度。

`straight_trim=-0.05` 从全局 0.827 m 开始，末端四个真实点的局部坐标为 7、18、29、40 mm；50 mm 为虚拟末端点，不参与空间分段。15 cm 局部坐标为 28、62、96、107、118、129、140 mm，虚拟末端为 150 mm。

全局行号跟随裁剪与反序保存，不能直接当作解调仪硬件 FBG 标签。全局行号、空间段 ID、全局坐标和裁剪原点也保存在 `optim.calibration_context`，供后续参数包使用。

## 分段错误会被拒绝

- 全长分段总和必须等于真实点数，且所有段长度为正整数。
- 裁剪起点必须落在全长空间段的开始位置。
- 局部分段必须与全长对应后缀完全相同，不能只保证总点数相同。
- 例如全长尾部采用旧的 `[2,2,3]` 时，5 cm 的起点切入其中第二段，会在 Optimiser 创建前报错。
- 若希望 5 cm 使用 `[4]`，应同步把全长改为 `[2,3,5,2,3,3,4]`，15 cm 改为 `[3,4]`，5 cm 改为 `[4]`。

## 命令行示例

在仓库目录运行：

```powershell
python main.py --task optim --multi_region_calib False --optimization_region full
python main.py --task optim --multi_region_calib True --weights None --slopes None --direct 0
python main.py --task score --multi_region_calib True --direct 0
```

三个方向完成后，可导出有序参数包：

```powershell
python tools/sensor_parameter_package.py --results `
  calibration_results/06B6_NeTi_v2_centraltube-20260913T152012Z-dbd50a/results.json `
  calibration_results/06B6_NeTi_v2_centraltube-20260913T152022Z-3aa476/results.json `
  calibration_results/06B6_NeTi_v2_centraltube-20260913T152029Z-8a1a2a/results.json `
  --output params/06B6_NeTi_v2_centraltube_multi_region.json `
  --comparison docs/calibration-results-20260913-comparison.md
```

JSON 顶层以 `sensor_id` 开始。`shared_parameters` 按 parser 顺序分成硬件、定标实验、优化超参数、动态展示、重构、初始化、角度优化和运行控制组。每个 parser 参数只有一个权威位置；显式 CLI 值覆盖包内默认值，`--task` 始终必须显式给出。

`--comparison` 输出的汇总表、逐半径 point/end error 字典及明细统一使用完整传感器长度百分比：`error (m) / full.context.ideal_len (m) × 100`。end15cm/end5cm 也使用 full 长度作分母，weighted loss 按相同基准换算；文档记录各方向使用的完整长度。原始 results.json 和参数包中的误差、优化损失仍保留原值。

单区域与合并重构命令：

```powershell
python main.py --task recon --sensor_params params/06B6_NeTi_v2_centraltube_multi_region.json --multi_region_calib False --optimization_region full --direct 0
python main.py --task recon --sensor_params params/06B6_NeTi_v2_centraltube_multi_region.json --multi_region_calib False --optimization_region end15cm --direct 0
python main.py --task recon --sensor_params params/06B6_NeTi_v2_centraltube_multi_region.json --multi_region_calib False --optimization_region end5cm --direct 0
python main.py --task recon --sensor_params params/06B6_NeTi_v2_centraltube_multi_region.json --multi_region_calib True --direct 0
```

合并重构中，full、end15cm、end5cm 各自使用自己的 thresholds、sharpness、score 窗口、weights 和 slopes。每段先在它的定标支持点范围计算 score/mass，然后只把结果写入归属点：full 1–15，end15cm 16–18，end5cm 19–22。全长只使用 full 定标的统一 `global_phases_set` 和 `rot_ang`；局部相位只作为 provenance 保留。

离线全长验证：

```powershell
python tools/calibration_validation.py --sensor-params params/06B6_NeTi_v2_centraltube_multi_region.json
```

`deployment_ready` 只在优化、统一相位、实测 score 和全长几何全部验收后为 true。当前包为 candidate：240°/full 的 slope/weight 以及 240°/end15cm weight 阶段发散，全长回放的最大平均点误差为 187.314 mm，末端测试最大切向误差 141.643°，区域边界最大相位跳变 0.906 rad，因此仍为 false。详细结果见 `calibration-multiregion-validation.md`。

第二条命令依次优化 full、end15cm、end5cm。`--weights None --slopes None` 明确要求每个区域重新初始化。`--direct` 仍控制本次循环优化的方向；0°、120°、240°需要分别运行，输出会标记本次实际优化的方向。

每次批量运行在 `calibration_output_dir` 下创建独立的“传感器ID-UTC时间-随机后缀”目录，避免覆盖已有结果。每完成一个区域就原子更新：

- `results.json`：状态、区域配置、全局点位、两阶段状态、正常重构误差和详细误差；
- `parameters.py`：便于复制的区域 weights、slopes、相位和 rot_ang。

后续区域失败时，已完成区域仍保留，失败区域会写入异常，未开始区域保持 pending。

CLI 可覆盖所选区域的 folder_path、radii、thresholds、segmentation。新增列表/字典输入使用 Python 字面量解析，例如 `--thresholds "[0.8,1.2,2,3,4]"`。`ideal_len` 与 `straight_trim` 自动计算；若手动传入冲突值则报错。

三组 thresholds 本轮保留旧值以隔离裁剪修复；支持独立修改、不同档数，但尚未重新拟合参数。评分窗口、相位策略、插值与损失函数保持原实现。

新增 `task score` 会调用生产路径的 `_get_base()`、`_get_scores()` 和 `_alloc_mass()`，输出每个区域、方向、半径和真实点的：基础选档曲率、理论曲率、分位数、逐级 score/mass、平均档位占比和末点档位占比，并生成 JSON 与 PNG。它使用 `processed_Data` 中每个定标状态的平均应变，因此可以检查半径/点位之间的评分分离和 constant 边界效应，但不能表示原始帧噪声；要评估重复性仍需保留逐次或逐帧样本。

本机 0° 实际诊断已经表明，高曲率阈值不能直接照搬理论中点：5 cm 数据的基础曲率中位数在 100/50/30/20/15 mm 时约为 7.67/15.31/25.78/31.92/34.35 m⁻¹，明显低于理论 10/20/33.33/50/66.67 m⁻¹。应基于三个方向及重复采集的实际分布再确定最终阈值。

weights/slopes 默认 None，按当前区域初始化。旧 5 cm 数值保留在 main 的注释中作为历史参考；不可认为它们适用于修正后的长度和新分段。预载参数必须为方向字典，维度对应本区域 K/J；入口会验证并转换为 NumPy 数组，不自动截断或拼接参数。

## 优化策略建议

建议先用同一空间拓扑分别定标，再通过区域参数块联合校正。重构时逐点展开 final_weight/slope_map，不代表每个点都要成为独立优化变量。

1. 全长定标先建立可用性基线。
2. 各长度分别先优化 slope、再优化 weight，得到稳定初值；保存几何、相位及数据来源。
3. 统一全局相位参考后组装区域参数。最终部署所有权为 d>150 mm、50<d<=150 mm、d<=50 mm。
4. 联合校正时仍优化区域 K×J 参数块，针对各数据集生成逐点映射；分别归一化长度、半径、方向的损失，保留全长精度约束。

直接为每个点优化独立 weight/slope 会增加自由度，目前 5 cm 只有四点，不宜先走这条路径。反过来，单独拟合后仅拼接数组又无法处理局部相位参考及交界间隔。两阶段策略可以先建立可解释的初值，再解决全局部署误差。

## 本次验证

使用三个实际目录，运行真实 Optimiser 初始化、CSV 读取、中心补偿、空间裁剪与正常模式重构，没有连接硬件、打开 GUI 或执行参数优化。

- full：3 个方向 × 9 个半径/直线状态，共 27 条。
- end15cm：3 × 13，共 39 条。
- end5cm：3 × 15，共 45 条。
- 全部 111 条形状均输出有限值，所需半径没有缺失；每条输出 120 个插值点。
- 几何、点数和末端长度符合上表，详细结果见同目录 `calibration-regions-validation.json`。

这验证配置、读取和重构链路可用，不证明重新定标后的精度提升。

修改前完整测试基线为 79 passed、3 failed。批量优化和评分诊断加入后，区域相关测试覆盖 parser 迁移、区域隔离、显式 None、失败保留、参数转换及生产评分质量。

本轮最终完整测试为 **144 passed、2 failed**。剩余两项均是既有 UDP anchor offset 测试；从关键基线提交到当前分支，`demonstrator.py`、入口约束实现和该测试没有差异。本次新增和直接相关的定向测试全部通过。
