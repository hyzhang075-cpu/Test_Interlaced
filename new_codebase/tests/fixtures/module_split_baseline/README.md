# 模块拆分冻结参考

来源提交：`19c376183`，原09B7_NeTi真实标定数据。

- `real_09B7.npz`：原始真实点距离、补虚拟点距离、模块3曲率/角度、完整单帧数组与旋转后形状；326个数值数组。
- `metadata.json`：原三方向逐点与末端误差汇总、标定参数与来源。
- `reconstruction_config.json`：基线使用的原配置。
- `reference_reconstruction.txt`：来源提交原重构器源文本；测试仅提取函数AST作独立数值参考，不导入旧工程。
- `reference_temporal.txt`：来源提交原动态曲率向量Kalman函数文本。

测试不依赖.git、忽略缓存或原始CSV；生产模块不读取本目录。
