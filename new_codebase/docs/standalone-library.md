# 独立八模块算法库

算法实现的权威位置是 `shape_sensing/`。旧根目录模块仅为历史导入路径提供反向转发；新包不导入它们，也不从旧文件动态加载函数。源代码发行包与 wheel 不需要这些旧入口。

## 模块职责与维护范围

| 模块 | 工程师 | 实际实现文件 |
|---|---|---|
| 1 波长输入与预处理 | A | `input_preprocessing/compat.py`、`zero_reference.py` |
| 2 相位/扭转系数初始化拟合 | A | `initialization/compat.py`、`phase_reference.py` |
| 3 应变、曲率大小与方向 | A | `curvature/geometry.py`、`model.py` |
| 4 slope/weight 参数优化 | A | `optimization/compat.py`、`batch.py`、`coverage.py` |
| 5 曲率积分与形状重构 | B | `reconstruction/compat.py`、`integration.py` |
| 6 理想曲线与误差计算 | B | `evaluation/metrics.py`、`ideal_curvature.py` |
| 7 可视化 | B | `visualization/plots.py`、`vtk_view.py` |
| 8 动态执行与实时系统 | B | `runtime/compat.py`、`prototype.py`、`imu.py`、`anchor.py` |

以上路径均相对于 `shape_sensing/`。配置、结果发布、公共工具与任务编排分别位于 `configuration/`、`artifacts/`、`common/`、`pipelines/`。`cli.py` 提供命令行入口，`api.py` 提供程序化入口。

原函数按源文本迁移，保留计算公式、参数顺序、说明、行内注释和设备协议许可声明。为减少数值变化，本轮保留原算法对象的状态和初始化关系，使用不含构造器的职责 mixin 将曲率、积分、评估等方法归属到各自模块。修改算法时应修改对应新模块，不再修改旧根文件。

## 安装

在独立源代码目录执行：

```powershell
python -m pip install .
```

基础安装支持核心数值计算。需要绘图和实时设备功能时：

```powershell
python -m pip install ".[gui,runtime]"
```

也可以安装已构建的 `shape_sensing-0.1.0-py3-none-any.whl`。配置目录和运行输出属于用户工作目录；包内 JSON/YAML 模板作为安装资源分发。

## 不依赖传感器文件的积分调用

```python
import numpy as np
from shape_sensing.reconstruction import ShapeIntegrator

s_m = np.linspace(0.0, 0.10, 101)
curvature = np.full(s_m.size, 10.0)     # 单位 1/m
angle_rad = np.zeros(s_m.size)        # 曲率平面方向，单位 rad
integrator = ShapeIntegrator(frame_rule="TNN", TNB_threshold=0.0)
position_m, T, Y, Z = integrator.integrate(
    curvature, angle_rad, s_m, np.array([1.0, 0.0, 0.0])
)
```

该调用只执行包内积分实现，不创建 Sensor、不读取标定目录或连接设备。TNN 返回 `(position, T, Y, Z)`；TNB 保留旧内部积分的 `(position, N, T, B)` 顺序，使用时应按所选标架解包。

## 传感器任务

从发行源码的 `configs/sensor_template.yaml` 复制工作配置，填写设计信息、数据路径和区域配置。模板继续使用原参数名、解释与选项。

```powershell
Copy-Item configs/sensor_template.yaml sensor.yaml
# 编辑 sensor.yaml 后运行
python -m shape_sensing --task init --config sensor.yaml
python -m shape_sensing --task optim --config sensor.yaml --direct 0
```

初始化配置不要求已有最终标定数组。初始化/优化结果保存在运行目录；继续优化时显式使用 `--resume`。程序化创建引擎：

```python
from shape_sensing.api import create_engine
engine = create_engine("sensor.yaml", task="init", max_iter=3)
```

创建传感器任务引擎会读取用户配置的数据，但不会自动连接真实硬件。`create_engine` 只构造引擎，不执行任务或加载续算；续算请使用上述命令行任务的 `--resume`。原有波长差、信号缩放及标定参数约定保持原算法行为，不引入新的物理换算公式。

## 发布与迁移

使用已经完成的运行快照生成标定发布包：

```python
from shape_sensing.artifacts.packages import export_package
export_package(["runs/my_sensor/completed_optim_run"], "calibration.json")
```

```powershell
python -m shape_sensing --task recon --sensor_params calibration.json --direct 0
python -m shape_sensing --task demo --sensor_params calibration.json
```

完整多区域包需要对应区域和方向的完整结果，缺失范围会报错。参数文件可以迁移，原始标定数据、设备地址、零点资源及模型资产仍需按各任务配置提供；算法库和发行包不附带实验数据。发布包格式与上一轮已有格式继续兼容。

## 验证范围

测试包括函数实际归属检查、新包禁止导入旧模块检查、独立 wheel 安装后真实曲率/积分/理想曲线计算、两种标架与迁移前参考逐数组对照，以及原任务回归。设备测试使用假设备或离线输入，真实硬件与真实传感器调优效果需要另行验证。
