## 1. 简介

该代码库包含形状感知任务代码，可支持多种方案的重构、参数优化与动态展示。

## 2. 依赖环境

本项目基于 Python 运行，主要依赖以下库：
- numpy
- scipy
- pandas
- matplotlib
- pySerial
- PyQt5
- vtk
在terminal中输入并运行指令pip install -r requirements.txt来进行安装

## 3. 代码结构
形状感知代码库/
├─ main.py                  # 主入口
├─ sensor.py                # 传感器参数与数据处理
├─ initialiser.py           # 初始化全局相位补偿与初始扭转系数
├─ reconstructor.py         # 形状重构
├─ optimiser.py             # 参数优化
├─ demonstrator.py          # 基础动态展示
├─ proto_demonstrator.py    # 完整样机动态展示
├─ IMU.py                   # IMU数据接受与处理文件
├─ visualisor.py            # 可视化
├─ utils.py                 # 字典递归处理方法等可以缩短代码、降低优化难度的工具
├─ model.stl                # 用于动态展示的人体模型
├─ ./tools                  # 用于部分临时任务的简易脚本
├─ requirements.txt         # 依赖环境
└─ README.md                # 当前文件
(注：如后续发现有更多参数需要手动调整，请参照当前格式更新至main.py中)

## 4. 使用说明

### 4.1. 参数优化
#### 4.1.1. 初始化
在main.py最下方的main()方法中找到硬件参数、标定实验参数以及优化使用的超参数并按照说明修改其default值。
若已有使用当前 --sensor_id 命名的文件夹，将其更名为 --sensor_id(归档)

若当前任务为LUNA数据分析，则阅读各通道结尾为gages.tsv的文件并按照以下步骤更新 --sensing_ranges 的default值:
    确保通道顺序与 --active_channels相同后，对于各个通道：
        从文件中找到传感点间距、(x轴， start)、(x轴， 1)、(x轴， 2)
        将该通道对应range改为 (((x轴， 1)-(x轴， start))/(传感点间距/1000), ((x轴， 2)-(x轴， start))/(传感点间距/1000))

    在terminal中输入并运行指令 python main.py --task init --weights None --slopes None 以生成初始Strain_Data图
    反复调整 --sensing_ranges 的default值直到 --exp_directs 中所有 方向 的 --sensor_id/--active_channels/--segmentation/方向/init_plots/Strain_Data_circ.png 中三个子图的首尾均接近水平轴
    在terminal中输入并运行指令 python main.py --task init --weights None --slopes None 进行检验

在terminal中输入并运行指令 python main.py --task init --weights None --slopes None
修改 --global_phases_set 的default值为 global phase
查看所有 --exp_directs 中的 方向 的 --sensor_id/--active_channels/--segmentation/方向/init_plots/Angle_Data.png 并将记录所有与S形弯标定形状转向位置对应处的角度跳变，修改 --turn_threshold 的default值以确保所有跳变均大于它的同时非跳变处的角度改变小于它

反复调整 --segmentation 的default值直到所有 方向 的 --sensor_id/--active_channels/--segmentation/方向/init_plots/Multi-Radius_Segmented_Linear_Fitting.png 分段合理
在terminal中输入并运行指令 python main.py --task init --weights None --slopes None 进行检验

确定 --segmentation 后检查各个 --sensor_id/--active_channels/--segmentation/方向/recon_plots/--rot_ang/形状 中最新的视频
分别尝试 --rot_dir 的两种取值并对比哪一个重构结果更合理
在terminal中输入并运行指令 python main.py --task init --weights None --slopes None 完成初始化

#### 4.1.2. 旋转角度优化
在terminal中输入并运行指令 python main.py --task ang --weights None --slopes None
若形状偏转不严重，可以先跳过该步骤，直接进入应变权重与扭转系数优化

#### 4.1.3. 应变权重与扭转系数优化
从初始化结束后的打印信息中获取：
    最优旋转角度
    最小误差
修改 --rot_ang 的 default 值为最优旋转角度。
在 len(--exp_directs) 个不同 terminal 中分别输入并运行：
    python main.py --task slopes --weights None --slopes None --direct 方向
当前版本中，优化流程内部固定为两阶段：
    1. 先优化 slopes，并强制使用理论曲率大小；
    2. 再优化 weights，并恢复正常曲率大小计算。
因此通常不再需要手动选择优先优化 slopes 或 weights。
(注：建议将这3个terminal与初始化的terminal分开)
(注: 若旋转角度差距不大，可以跳过旋转角度优化，因为这种情况下，直接从旋转角度0.0开始优化的结果会更好)


#### 4.1.4. 循环优化(大部分时候不需要)
反复执行以下操作直至 最优旋转角度 不再变化：
    从各个terminal的打印信息中获取：
        最优权重
        最优扭转系数
    修改 --weights 各方向为对应terminal的最优权重
    修改 --slopes 各方向为对应terminal的最优扭转系数
    修改 --ang_step 和 --ang_range 为原来的0.1倍(可自行判断)
    在初始化terminal中输入并运行指令 python main.py --task ang
    从打印信息中获取：
        最优旋转角度
        最小误差
    修改 --rot_ang 的default值为 最优旋转角度
    按情况在 len(--exp_directs) 个不同的terminal中分别输入并运行指令:
        python main.py --task weights --direct 方向
    或
        python main.py --task slopes --direct 方向

#### 4.1.5. 更换超参
若前面几个步骤未能给出令人满意的结果，可对优化使用的超参数进行修改。若优化结果的重构图像中各个形状的半径显然不正常，则优先修改--bounds，否则按照文件中从上到下的优先级修改其余超参数。

#### 4.1.6 效果检验
从3个terminal的打印信息中获取：
    最优权重
    最优扭转系数
修改- -weights 各方向default值为对应terminal的最优权重
修改 --slopes 各方向default值为对应terminal的最优扭转系数
选择想要检查的 方向
在terminal中输入并运行指令 python main.py --task recon --direct 方向
若效果良好，将所有parser.add_argument赋值并粘贴进以 --sensor_id 命名的txt文件中，文件储存于params文件夹中
(注：建议将这个terminal与初始化的terminal和优化的3个terminal分开)

### 4.2 重构
将对应的 params/--sensor_id.txt 中的内容复制并粘贴替代从 parser = argparse.ArgumentParser() 到 args = parser.parse_args() 的部分
选择想要重构的 方向
在terminal中输入并运行指令 python main.py --task recon --direct 方向

### 4.3 动态展示
将对应的 params/--sensor_id.txt 中的内容复制并粘贴替代从 parser = argparse.ArgumentParser() 到 args = parser.parse_args() 的部分

动态展示支持以下两种参数使用方式。

#### 4.3.1 只使用精度最高的一套方向参数

先从 0°、120°、240° 三套标定参数中选出重构精度最高的一套，并通过 `--direction_reference_direct` 指定方向。不要添加 `--direction_blend_enabled`，程序就会在整个动态过程中固定使用该方向的 `weights` 和 `slopes`。

例如，120° 参数精度最高时：

```powershell
python main.py --task demo --direction_reference_direct 120
```

如果最高精度参数是 0° 或 240°，将命令末尾改为：

```powershell
--direction_reference_direct 0
```

或：

```powershell
--direction_reference_direct 240
```

该模式也用于三方向软过渡启用前的方向坐标校准。建议先使用最高精度的单套参数检查并确定 `rot_ang`，例如：

```powershell
python main.py --task demo `
  --direction_reference_direct 120 `
  --rot_ang 旋转角度
```

其中 `旋转角度` 应替换为实际数值。

#### 4.3.2 打开 0°/120°/240° 三方向软过渡

确认三套方向参数均已写入 `--weights` 和 `--slopes`，且使用最高精度单套参数完成 `rot_ang` 校准后，添加 `--direction_blend_enabled`：

```powershell
python main.py --task demo `
  --direction_reference_direct 120 `
  --rot_ang 旋转角度 `
  --direction_blend_enabled
```

`--direction_reference_direct` 在软过渡模式中仍然需要：程序先用该方向的第 0 曲率等级 slope 粗略估计当前弯曲方向，再在相邻的 0°、120°、240° 参数之间进行半余弦插值，并用混合后的完整参数完成正式重构。

软过渡的稳定参数可按需显式设置：

```powershell
python main.py --task demo `
  --direction_reference_direct 120 `
  --rot_ang 旋转角度 `
  --direction_blend_enabled `
  --direction_min_curvature 0.5 `
  --direction_min_consecutive 3 `
  --direction_consistency_threshold 0.6 `
  --direction_smooth_alpha 0.15
```

各参数含义：

- `direction_min_curvature`：方向估计所需的最小有效曲率；低于该值时冻结上一帧方向权重。
- `direction_min_consecutive`：近端稳定弯曲区要求的最少连续点数。
- `direction_consistency_threshold`：弯曲区最低方向一致性；低于该值时冻结上一帧方向权重。
- `direction_smooth_alpha`：方向权重的跨帧指数平滑系数，越小越稳定但响应越慢，取值范围为 `(0, 1]`。

注意：

- `--direction_blend_enabled` 是开关参数，不带该参数即为关闭，不能写成 `--direction_blend_enabled False`。
- 三套 `weights` 的数组形状必须一致，三套 `slopes` 的数组形状也必须一致。
- 三方向参数需要采用相同的曲率等级数、阈值定义、分段方式和参数排列约定。
- 应先关闭软过渡校准 `rot_ang`，再打开软过渡；否则角度坐标偏差可能导致程序稳定地混合错误方向的参数。
- 单套模式和三方向软过渡都只影响动态展示，不改变静态参数优化流程。

更完整的原理和排查方法见 [动态三方向参数软过渡：原理、注意事项与踩坑记录](docs/动态三方向参数软过渡_原理_注意事项与踩坑记录.md)。

## 5.当前版本信息
版本号：v9
日期：2026.06.16
创建人：高至简
主要改动：
- 重构 straight_trim 裁剪逻辑，使数据、采样点、误差计算和可视化统一使用有效段
- 添加 end3 标定形状及其理论形状/理论曲率支持
- 将优化流程改为 slopes 强制理论曲率、weights 正常曲率的两阶段优化
- 添加完整样机动态展示原型，支持人体模型、入体长度、曲率入口识别和自动校准

## 6. 版本记录
| 版本号 | 日期 | 修改人 | 主要改动 | 备注 |
|---|---|---|---|---|
| v0 | 2026-03-18 | 高至简 | 统合重构LUNA与肠镜FBG形状重构、参数优化代码以及动态展示代码；整理统一入口与基本使用流程 | 初始版本 |
| v0.1 | 2026-03-23 | 高至简 | 添加适用于支气管镜任务的数据处理方法，并将数据处理接口按解调仪品牌分类（FiberTecc、Hippo、LUNA） | 无 |
| v1 | 2026-03-30 | 高至简 | 添加S形弯曲与末尾额外卷曲两种形状的配套算法；修复了数据未按通道排序的bug和扭转系数优化初始化使用错误初始值的bug | 无 |
| v2 | 2026-04-01 | 曹丹倩 | 修改数据处理与可视化模块以兼容任意数量的通道 | 该版本并非基于v1的拓展，而是平行的 |
| v3 | 2026-04-02 | 高至简 | 融合v1与v2的功能 | 无 |
| v4 | 2026-04-08 | 高至简 | 添加零螺距与螺旋的重构与可视化，并调整了优化顺序 | 该版本零螺距优化未使用螺距误差 |
| v4.1 | 2026-04-16 | 曹丹倩 | 添加了对Nan值与数据异常点的排除 | 当存在整个csv文件所有行都存在Nan值的情况时会有问题 |
| v4.2 | 2026-04-22 | 高至简 | 添加了扭转系数优化上下界；在某一subfolder中存在所有行都有nan值得情况下，只有该subfolder对应的标定形状会从重构与优化中被忽略；修复了_F_S_frame输出顺序与调用时不一致的bug；修改了储存初始化与重构结果的路径 | 无 |
| v5 | 2026-04-23 | 高至简 | 将global_phases_sets加入parser；修改分段逻辑使得可以兼容任意数量的分段；将bound和归一化逻辑合并；将零螺距和S形弯曲的数据也加入初始化考察范围内；进一步使用utils中的字典递归处理方法缩短代码；将优化时零螺距数据的误差标准改为螺距；修正了通道顺序问题 | 无 |
| v5.1 | 2026-04-28 | 高至简 | 将 end_len 加入args并确保end形状的基准是最后一个栅点前 end_len 长度进入末端弧圈，而非传感器末端前 end_len；添加了各分段首尾相连的初始化途径 | 无 |
| v5.2 | 2026-04-29 | 高至简 | 添加了平行标架优化；计算global_phases_set前将用于计算均值的角度等价到[-pi, pi] | 无 |
| v6 | 2026-05-07 | 高至简 | 将半径误差加入了零螺距优化；添加了动态展示时的角度跳变判定与处理 | 无 |
| v6.1 | 2026-05-08 | 高至简 | 将零螺距优化与原始版本同步，添加了PCA求轴向的过程；添加了在优化时冻结部分曲率分区相关参数的设置以方便零螺距数据优化 | 无 |
| v7 | 2026-05-20 | 高至简 | 现可指定重构一部分长度；可进行反向重构；添加了曲率分区判断前根据等曲率板形状放大的设置 | 无 |
| v8 | 2026-05-20 | 曹丹倩 | 加入交错刻栅、通道顺序改变对初始数据的重排 | 无 |
| v9 | 2026-06-16 | 高至简 | 添加交错刻栅与分段通道重排；重构 straight_trim 裁剪逻辑；添加 end3 标定形状；将优化改为 slopes 理论曲率 + weights 正常曲率的两阶段流程；添加完整样机动态展示原型 | 无 |
| v9.1 | 2026-07-07 | 曹丹倩 | 修正straight_trim剪裁后的原点问题 | 无 |
| v10 | 2026-07-22 | 曹丹倩 | 动态增加3方向参数过渡算法 | 无 |
| v10.1 | 2026-07-30 | 曹丹倩 | 修正初始化时global phase角度平均在边界处出错的问题 | 无 |
| v10.2 | 2026-08-07 | 邱国翀 | 加了动态kalman滤波 | 无 |