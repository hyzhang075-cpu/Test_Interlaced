# 11B8纯full校准结果

仅使用此前成功运行中由full数据独立拟合的芯位与0/120/240方向权重，未重新拟合。参数包仅含full，关闭多区域拼接，无end15cm/end5cm参数。已通过新模块实际加载，分别完成三个方向的full静态重构。

几何：start=177 mm、end=8 mm；A/B各20点交错刻栅，共40截面，全长902 mm。校准数据半径含直态0及200/300/400/600/800/1000/1500/2000 mm。

|方向|平均逐点误差 mm|平均末端误差 mm|工况数（含直态）|
|---:|---:|---:|---:|
|0°|11.39|33.56|9|
|120°|6.47|17.20|9|
|240°|5.54|16.16|9|

三个方向等权平均：逐点 7.80 mm；末端 22.31 mm。

各方向全段权重优化状态均为no_improvement，程序保留初始化权重。芯位平均拟合残差21.58 pm。

这些误差来自full校准数据上的回代，包含直态；与此前多区域包中full的报告一致，不是独立测试精度。full数据最小非零半径200 mm，尚未验证末端小半径表现。参数包validation.verified和deployment_ready保持false。

## 重构图

### 0°

![0度全段重构](C:/Users/win11/Desktop/project/1.算法开发/联网git代码/new_codebase/runs/11B8_CoreCalib/20261125-113603-365851-1665fa578a-recon/outputs/full/0/reconstruction_circ.png)

### 120°

![120度全段重构](C:/Users/win11/Desktop/project/1.算法开发/联网git代码/new_codebase/runs/11B8_CoreCalib/20261125-113638-028934-0d6d85f8e9-recon/outputs/full/120/reconstruction_circ.png)

### 240°

![240度全段重构](C:/Users/win11/Desktop/project/1.算法开发/联网git代码/new_codebase/runs/11B8_CoreCalib/20261125-113711-576225-dff6349727-recon/outputs/full/240/reconstruction_circ.png)

## 运行命令

从new_codebase目录运行；Python需有项目依赖。

```powershell
$env:PYTHONPATH=(Resolve-Path .runtime_deps).Path
$env:MPLBACKEND="Agg"
foreach ($d in @(0,120,240)) {
    python -m shape_sensing --sensor_params packages/11B8_CoreCalib-full-v1.json --task recon --multi_region_calib False --optimization_region full --direct $d
}
```

## 重构快照

- C:\Users\win11\Desktop\project\1.算法开发\联网git代码\new_codebase\runs\11B8_CoreCalib\20261125-113603-365851-1665fa578a-recon
- C:\Users\win11\Desktop\project\1.算法开发\联网git代码\new_codebase\runs\11B8_CoreCalib\20261125-113638-028934-0d6d85f8e9-recon
- C:\Users\win11\Desktop\project\1.算法开发\联网git代码\new_codebase\runs\11B8_CoreCalib\20261125-113711-576225-dff6349727-recon
