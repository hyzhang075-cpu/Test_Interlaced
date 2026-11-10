"""程序化创建独立算法引擎与纯输入形状积分器。"""
from pathlib import Path
from .reconstruction import ShapeIntegrator
from .configuration import load_working_config

def create_engine(config=None, **overrides):
    """按现有工作 YAML 构造任务引擎，或用显式 Optimiser 参数创建。

    config 为配置文件路径时复用 CLI 的解析和区域检查；任务优先级为显式
    task 覆盖、YAML task、recon。程序化覆盖会先转换并校验，再参与区域验证。
    构造会读取配置指定的数据，但不会连接硬件。
    """
    from .cli import Optimiser, build_argument_parser, configure_calibration_args, parse_main_args, optimizer_parameters
    if config is None:
        return Optimiser(**overrides)
    path=Path(config).resolve()
    loaded=load_working_config(path)
    if overrides.get('resume') is not None or loaded.get('resume') is not None:
        raise ValueError('resume is applied by the task pipeline; create_engine does not execute task resume')
    task=overrides.pop('task',None) or loaded.get('task') or 'recon'
    args=parse_main_args(['--config',str(path),'--task',str(task)])
    parser=build_argument_parser()
    actions={action.dest:action for action in parser._actions}
    for name,value in overrides.items():
        if name not in actions: raise ValueError(f"未知算法参数: {name}")
        action=actions[name]
        converter=action.type
        if converter is not None and isinstance(value,str): value=converter(value)
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"参数 {name} 的值必须属于 {action.choices}")
        setattr(args,name,value)
    # Programmatic overrides carry the same provenance as explicit CLI options;
    # downstream region selection and resume logic use these markers to preserve them.
    explicit_cli=set(getattr(args,'_explicit_cli_parameters',())) | set(overrides)
    args._explicit_cli_parameters=sorted(explicit_cli)
    explicit_seeds=set(getattr(args,'_explicit_seed_parameters',()))
    explicit_seeds.update(set(overrides) & {'weights','slopes'})
    args._explicit_seed_parameters=sorted(explicit_seeds)
    if 'full_segmentation' in overrides:
        args._full_segmentation_explicit=True
    context=configure_calibration_args(args)
    engine=Optimiser(**optimizer_parameters(args))
    engine.calibration_context=context
    return engine

__all__=["ShapeIntegrator","create_engine"]
