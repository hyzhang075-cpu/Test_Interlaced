# 11B8芯位校准结果

数据：1124-11B8+镍钛管全段+末段。start=177 mm，end=8 mm；A/B各20点交错刻栅，共40截面。物理通道1为中心，2/4/6为A组外芯，3/5/7为B组外芯。B组中心响应使用A组插值，最后一点含外推。

本次使用新模块完成init与0/120/240三个方向的多区域optim，并通过export_algorithm_package.py导出。优化固定芯位，仅优化权重。采用现有芯径350 μm、芯半径50 μm及灵敏度1 pm/με假设。

## 程序报告的重构误差

|区域|方向|权重优化状态|平均逐点误差 mm|平均末端误差 mm|
|---|---:|---|---:|---:|
|full|0|no_improvement|11.39|33.56|
|full|120|no_improvement|6.47|17.20|
|full|240|no_improvement|5.54|16.16|
|end15cm|0|converge|20.57|45.33|
|end15cm|120|converge|33.26|75.00|
|end15cm|240|converge|46.12|109.11|
|end5cm|0|converge|58.80|138.28|
|end5cm|120|converge|11.80|25.75|
|end5cm|240|converge|79.91|192.20|

no_improvement表示未找到更优权重，保留初始权重。converge表示权重优化达到程序收敛条件。

## 质量与验证范围

芯位拟合平均残差：full 21.58 pm，end15cm 360.51 pm，end5cm 1021.90 pm。末端残差及重构误差较大，不能视为精度验收通过。CSV中缺失及异常行按模块现有规则清洗，原始数据未修改。信息文档注明120°/240°的5 cm、R=15 mm工况有轻微扭转，本次未移除这些工况。

已用新包加载器回读，检查全部数值有限、三个成功optim运行、三区域九组方向参数及40/21/7个截面。此项仅验证产物结构与任务状态；校准和误差评价使用同一批数据，没有独立测试或全长动态验证。validation.verified及deployment_ready保持false。七通道交错实时demo尚未接入当前入口，此包可供当前静态重构流程使用。

## 复现

从new_codebase运行。Python需有numpy/scipy/pandas/PyYAML/matplotlib；本次PyYAML安装于项目.runtime_deps，需将该目录加入PYTHONPATH。

```powershell
$env:PYTHONPATH=(Resolve-Path .runtime_deps).Path
$env:MPLBACKEND="Agg"
python -m shape_sensing --config configs/11B8_CoreCalib.yaml --task init --multi_region_calib True
python tools/run_11b8_calibration.py --init-run "runs/11B8_CoreCalib/实际init目录"
```

## 本次运行来源

- C:\Users\win11\Desktop\project\1.算法开发\联网git代码\new_codebase\runs\11B8_CoreCalib\20261125-112347-784100-55c7dca37f-optim
- C:\Users\win11\Desktop\project\1.算法开发\联网git代码\new_codebase\runs\11B8_CoreCalib\20261125-112539-068508-702eb058b5-optim
- C:\Users\win11\Desktop\project\1.算法开发\联网git代码\new_codebase\runs\11B8_CoreCalib\20261125-112836-417927-902bbc8ed6-optim
