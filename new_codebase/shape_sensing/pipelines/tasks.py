"""通过八个独立模块编排任务；不在导入时构造GUI、设备或数据对象。"""
from pathlib import Path
import threading
import sys
from shape_sensing.context import TaskContext
from .session import TaskSession, validate_seed

def module_parameters(args, parameter_builder):
    """投影完整解析参数并保留builder的区域派生值，防止运行时/UI设置被旧构造过滤器丢掉。"""
    from dataclasses import fields
    from shape_sensing.contracts import (InputConfig, InitializationConfig, CurvatureConfig,
        ReconstructionConfig, OptimizationConfig, EvaluationConfig, RuntimeConfig, ViewConfig)
    names = {item.name for kind in (InputConfig, InitializationConfig, CurvatureConfig,
        ReconstructionConfig, OptimizationConfig, EvaluationConfig, RuntimeConfig, ViewConfig)
        for item in fields(kind)}
    values = {key: value for key, value in vars(args).items() if key in names}
    values.update(parameter_builder(args))
    return values


def build_context(args, *, parameter_builder, configure_args, optimiser_factory=None):
    """解析区域并组合八个公开模块；构造不读取CSV、不执行拟合。"""
    region = configure_args(args)
    session = TaskSession(module_parameters(args, parameter_builder), task=args.task,
                          region=getattr(args, 'optimization_region', 'full'))
    session.calibration_context = region
    return TaskContext(session)

def _visualization(session):
    from shape_sensing.visualization.service import VisualizationService, VisualizationInputs
    config = session.config
    return VisualizationService(VisualizationInputs(
        session.load_data(), session.initialization_result,
        tuple(config.initialization.segmentation), config.initialization.discont,
        config.initialization.connect, config.reconstruction.zero_init,
        config.input.straight_trim, config.reconstruction.rot_ang,
        config.input.ideal_len, config.curvature.init_weight))

def _execute_task_body(args, *, parameter_builder, configure_args, prepare_jobs, run_jobs,
                       project_root, app_factory, optimiser_factory=None,
                       visualizer_factory=None, compute_error_ang=None, optim_ang=None, system=None, session_observer=None):
    """生产任务仅编排模块service；优化不会执行初始化模块2。"""
    if args.task != 'demo':
        args.use_stored_zero = False
    if args.weights == 'None':
        args.weights = None
    if args.slopes == 'None':
        args.slopes = None
    if args.task in ('optim', 'score'):
        jobs = prepare_jobs(args, configure_args)
        # Validate every region seed before loading any calibration CSV or
        # launching the first region's search.
        for job_args, _ in jobs:
            validate_seed(module_parameters(job_args, parameter_builder), region=job_args.optimization_region)
        root = Path(args.calibration_output_dir)
        if not root.is_absolute():
            root = Path(project_root) / root
        def factory(job_args):
            session = TaskSession(module_parameters(job_args, parameter_builder), task=job_args.task,
                                  region=job_args.optimization_region)
            session.load_data()
            if session_observer is not None:
                session_observer(session)
            return session
        return run_jobs(jobs, output_root=root, session_factory=factory)
    context = build_context(args, parameter_builder=parameter_builder, configure_args=configure_args)
    session = context.session
    if session_observer is not None:
        session_observer(session)
    if args.task == 'demo':
        from shape_sensing.runtime.service import run_demo
        return run_demo(session, args, app_factory)
    if args.task == 'init':
        session.initialize(verbose=True)
        visualizer = _visualization(session)
        errors = {}
        for direct in session.dataset.radii:
            rs = {}
            for kind in session.calib_shapes:
                rs[kind], distance = session.reconstruct_shapes(kind, direct)
            result = session.evaluate(distance, rs, direct)
            errors[direct] = result.metrics['avg_point_err']['calib']
            visualizer.init_visualise(direct)
            visualizer.show_reconstruction(direct, rs, distance, showanime=True, ideals=result.details.get('ideals'))
        return {'errors': errors, 'weights': session.weights, 'slopes': session.slopes,
                'global_phases_set': session.global_phases_set}
    if args.task == 'ang':
        import numpy as np
        session.load_data()
        def score(rotation):
            block = session.parameters.parameters_by_region[session.region]
            block['rot_ang'] = float(rotation)
            errors = {}
            for direct in session.dataset.radii:
                curves = {}
                for kind in session.calib_shapes:
                    curves[kind], distance = session.reconstruct_shapes(kind, direct)
                metrics = session.evaluate(distance, curves, direct).metrics
                errors[direct] = (metrics['avg_point_err']['calib']
                    + session.config.optimization.k_end * metrics['avg_end_err']['calib']
                    + session.config.optimization.k_pitch * metrics['avg_pitch_err'])
            return errors
        best_rotation = args.rot_ang
        best_errors = score(best_rotation)
        for rotation in np.arange(args.rot_ang-args.ang_range, args.rot_ang+args.ang_range, args.ang_step):
            errors = score(rotation)
            if sum(errors.values()) < sum(best_errors.values()):
                best_rotation, best_errors = float(rotation), errors
        session.parameters.parameters_by_region[session.region]['rot_ang'] = best_rotation
        return {'rot_ang': best_rotation, 'errors': best_errors}
    if args.task == 'recon':
        session.load_data()
        rs = {}
        for kind in session.calib_shapes:
            rs[kind], distance = session.reconstruct_shapes(kind, args.direct)
        result = session.evaluate(distance, rs, args.direct)
        _visualization(session).show_reconstruction(args.direct, rs, distance, showanime=True, ideals=result.details.get('ideals'))
        return result.metrics
    raise ValueError(f'未知task: {args.task!r}')


def execute_task(args, *, optimiser_factory, visualizer_factory, app_factory,
                 configure_args, parameter_builder, prepare_jobs, run_jobs,
                 compute_error_ang, optim_ang, project_root, system=None, artifact_root=None):
    """创建运行快照，执行任务并记录结果/失败状态。"""
    from shape_sensing.artifacts.runs import create_run
    resolved=dict(vars(args))
    for key in list(resolved):
        if key.startswith('_'): resolved.pop(key,None)
    if getattr(args,'resume',None):
        from shape_sensing.artifacts.runs import load_resume
        resumed=load_resume(args.resume,resolved)
        values=resumed if isinstance(resumed,dict) else {}
        wrapped=values.get('result')
        report=(wrapped[1] if isinstance(wrapped,list) and len(wrapped)>1 and isinstance(wrapped[1],dict)
                and isinstance(wrapped[1].get('regions'),dict) else None)
        explicit=set(getattr(args,'_explicit_cli_parameters',()))
        loaded_fields=[]
        if report is not None:
            is_batch=bool(getattr(args,'multi_region_calib',False) and args.task in ('optim','score'))
            selected_regions=report.get('regions',{})
            if is_batch:
                profiles=getattr(args,'calibration_regions',None)
                if not isinstance(profiles,dict): raise ValueError('多区域续算缺少 calibration_regions 配置')
                for region,entry in selected_regions.items():
                    params=entry.get('parameters',{})
                    if entry.get('status')!='complete' or not params: raise ValueError(f'resume 区域结果不完整: {region}')
                    for name in ('weights','slopes'):
                        if name in explicit: continue
                        if params.get(name) is None: raise ValueError(f'resume {region} 缺少 {name}')
                        value=params[name]
                        if isinstance(value,dict): value={int(k) if isinstance(k,str) and k.lstrip('-').isdigit() else k:v for k,v in value.items()}
                        profiles[region][name]=value; loaded_fields.append(f'{region}.{name}')
                    if 'global_phases_set' not in explicit and 'global_phases_set' in params:
                        profiles[region]['global_phases_set']=params['global_phases_set']
                        loaded_fields.append(f'{region}.global_phases_set')
                    if 'rot_ang' in params: profiles[region]['rot_ang']=params['rot_ang']
                # prepare_calibration_jobs resolves each region from its profile; never inject the final engine.
                if 'weights' not in explicit: args.weights=None
                if 'slopes' not in explicit: args.slopes=None
            else:
                region=getattr(args,'optimization_region','full')
                entry=selected_regions.get(region)
                if entry is None or entry.get('status')!='complete': raise ValueError(f'resume 缺少区域结果: {region}')
                params=entry.get('parameters',{}); direct=str(args.direct)
                for name in ('weights','slopes'):
                    if name in explicit: continue
                    mapping=params.get(name)
                    if isinstance(mapping,dict):
                        selected=mapping.get(direct,mapping.get(int(direct)))
                        if selected is None: raise ValueError(f'resume {region} 缺少 direct={direct} 的 {name}')
                        setattr(args,name,{int(direct):selected})
                    elif mapping is not None: setattr(args,name,mapping)
                    else: raise ValueError(f'resume {region} 缺少 {name}')
                    loaded_fields.append(name)
                if 'global_phases_set' not in explicit and 'global_phases_set' in params:
                    args.global_phases_set=params['global_phases_set']; loaded_fields.append('global_phases_set')
        else:
            for name in ('weights','slopes','global_phases_set'):
                if name not in explicit and name in values:
                    setattr(args,name,values[name]); loaded_fields.append(name)
        if getattr(args,'optim_weights_only',False) and getattr(args,'slopes',None) is None and not any(x.endswith('.slopes') for x in loaded_fields):
            raise ValueError('weights-only 续算需要 resume 或显式提供 slopes')
        args._resume_seed_parameters=[name.split('.')[-1] for name in loaded_fields if name.endswith('slopes') or name in ('weights','slopes')]
        args._resume_provenance={'source':str(args.resume),'fields':loaded_fields}
        resolved['resume_provenance']=args._resume_provenance
    # Snapshot the fully merged Namespace (parser defaults < YAML < CLI), not the source YAML.
    resolved={k:v for k,v in vars(args).items() if not k.startswith('_') and k not in ('config',)}
    resolved['provenance']={'config_path':getattr(args,'_working_config',{}).get('_config_path') if getattr(args,'_working_config',None) else None,
        'cli_parameters':list(getattr(args,'_explicit_cli_parameters',())),
        'config_parameters':list(getattr(args,'_explicit_config_parameters',())),
        'resume':getattr(args,'_resume_provenance',None)}
    run=create_run(getattr(args,'sensor_id','unknown'),args.task,resolved,
                   artifact_root or (Path(project_root)/'runs'))
    state={}
    def observe(session):
        state['session'] = session
    try:
        result=_execute_task_body(args,visualizer_factory=visualizer_factory,
            app_factory=app_factory,configure_args=configure_args,parameter_builder=parameter_builder,
            prepare_jobs=prepare_jobs,run_jobs=run_jobs,compute_error_ang=compute_error_ang,optim_ang=optim_ang,
            project_root=project_root,system=system,session_observer=observe)
        payload={'task':args.task,'result':result}
        if args.task == 'init' and isinstance(result, dict):
            payload.update({key:result[key] for key in ('weights','slopes','global_phases_set') if key in result})
        engine=state.get('session')
        if engine is not None and not (args.task in ('optim','score') and getattr(args,'multi_region_calib',False)):
            for field in ('global_phases','global_phases_set','weights','slopes','rot_ang','avg_point_err','avg_end_err','avg_pitch_err','avg_radius_err'):
                if hasattr(engine,field): payload[field]=getattr(engine,field)
        stage='initialization' if args.task=='init' else 'optim' if args.task=='optim' else 'results'
        run.write_result(payload,stage); run.finish('succeeded')
        return result
    except BaseException as exc:
        run.finish('failed',error=exc)
        raise
