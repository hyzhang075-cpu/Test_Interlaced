# 形状感知项目

仓库分为两个独立目录。

| 目录 | 用途 |
|---|---|
| [new_codebase](new_codebase/README.md) | 当前维护的算法库，包含自己的入口、工具、配置、校准数据、参数包和运行输出 |
| [legacy_codebase](legacy_codebase/README.md) | 历史兼容入口、旧工具、历史数据和报告 |

从新项目启动：

```powershell
cd new_codebase
python shape_sensing/main.py --config configs/09B7_NeTi.yaml --task init
```

完整的初始化、优化、参数合并、发布、回放和 demo 说明见 [新库使用说明](new_codebase/shape_sensing/README.md)。目录迁移记录见 [数据迁移映射](new_codebase/MIGRATION_DATA.json)，原始数据副本校验明细见 [SHA-256 校验](new_codebase/docs/RAW_COPY_SHA256.json)。

新项目拥有独立的文件副本，运行时不读取旧目录。旧项目允许通过安装后的 `shape_sensing` 调用新算法。旧目录保留供核对，用户确认后可以删除。
