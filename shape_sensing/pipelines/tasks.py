"""按任务编排 legacy 算法调用；不在导入时构造 GUI、设备或数据对象。"""
from pathlib import Path
import threading
import sys
from shape_sensing.context import LegacyContext
from shape_sensing.input_preprocessing import InputPreprocessor
from shape_sensing.initialization import Initializer
from shape_sensing.optimization import ParameterOptimizer
from shape_sensing.reconstruction import ShapeReconstructor
from shape_sensing.evaluation import Evaluator
from shape_sensing.runtime import RealtimeSystem
from shape_sensing.visualization import Visualizer

def build_context(args, *, optimiser_factory, parameter_builder, configure_args):
    """由解析完成的 Namespace 构造独立旧引擎上下文；会读取对应任务数据。"""
    region=configure_args(args)
    engine=optimiser_factory(**parameter_builder(args))
    engine.calibration_context=region
    return LegacyContext(engine)

def _execute_task_body(args, *, optimiser_factory, visualizer_factory, app_factory,
                 configure_args, parameter_builder, prepare_jobs, run_jobs,
                 compute_error_ang, optim_ang, project_root, system=None):
    """执行 init/ang/optim/score/recon/demo，委托数值工作给旧算法接口。

    返回批量任务或静态任务的 legacy 返回值；demo 运行 Qt 事件循环并会接入设备。
    所有共享状态仅属于此调用创建的任务 context。
    """
    system=system or sys
    if args.task!='demo': args.use_stored_zero=False
    if args.weights=='None': args.weights=None
    if args.slopes=='None': args.slopes=None
    if args.task in ('optim','score'):
        jobs=prepare_jobs(args,configure_args)
        root=Path(args.calibration_output_dir)
        if not root.is_absolute(): root=Path(project_root)/root
        return run_jobs(jobs,optimiser_factory,output_root=root)
    context=build_context(args,optimiser_factory=optimiser_factory,parameter_builder=parameter_builder,configure_args=configure_args)
    engine=context.engine
    demo_keys=('total_scale','imu','imu_com','imu_maxlen','demo_recorder','unwrap_threshold','align_threshold',
        'direction_blend_enabled','direction_reference_direct','direction_min_curvature','direction_min_consecutive',
        'direction_consistency_threshold','direction_smooth_alpha','demo_kalman_enabled','demo_kalman_Rk','demo_kalman_Bk',
        'anchor_constraint_enabled','anchor_translation_enabled','anchor_rotation_enabled','anchor_offset_m')
    demo_params={k:getattr(args,k) for k in demo_keys}
    print(f'curvature interpolation mode: {engine.curvature_interp}')
    if args.task=='demo':
        app=app_factory(system.argv)
        runtime=RealtimeSystem(context,**demo_params)
        demo=runtime.run()
        try:
            demo.show()
            if demo.imu:
                t=threading.Thread(target=demo.receive_IMU); t.daemon=True; t.start()
                print('---------- 开始接收IMU数据 ----------')
            else: print('---------- 未启用IMU ----------')
            receiver=InputPreprocessor(context)
            t=threading.Thread(target=receiver.receive,daemon=True); t.start()
            return app.exec_()
        finally:
            runtime.close()
    vis=Visualizer(context,legacy_factory=visualizer_factory).legacy
    if args.task=='init': return compute_error_ang(optim=engine,vis=vis,init=True)
    if args.task=='ang': return optim_ang(rot_ang=args.rot_ang,ang_step=args.ang_step,ang_range=args.ang_range,optim=engine,vis=vis)
    if args.task=='optim': return ParameterOptimizer(context).optimize(verbose=True)
    if args.task=='recon':
        rs={}
        for shape in engine.calib_shapes:
            result,dist=ShapeReconstructor(context).reconstruct_shapes(calib_shape=shape,direct=args.direct,verbose=False)
            rs[shape]=result
        metrics=Evaluator(context).evaluate(interp_dist=dist,rs=rs)
        vis.show_reconstruction(direct=args.direct,rs=rs,interp_dist=dist,showanime=True)
        engine._print_errors()
        return metrics
    raise ValueError(f'未知 task: {args.task!r}')


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
    def tracked_factory(**kwargs):
        engine=optimiser_factory(**kwargs); state['engine']=engine; return engine
    try:
        result=_execute_task_body(args,optimiser_factory=tracked_factory,visualizer_factory=visualizer_factory,
            app_factory=app_factory,configure_args=configure_args,parameter_builder=parameter_builder,
            prepare_jobs=prepare_jobs,run_jobs=run_jobs,compute_error_ang=compute_error_ang,optim_ang=optim_ang,
            project_root=project_root,system=system)
        payload={'task':args.task,'result':result}
        engine=state.get('engine')
        if engine is not None and not (args.task in ('optim','score') and getattr(args,'multi_region_calib',False)):
            for field in ('global_phases','global_phases_set','weights','slopes','rot_ang','avg_point_err','avg_end_err','avg_pitch_err','avg_radius_err'):
                if hasattr(engine,field): payload[field]=getattr(engine,field)
        stage='initialization' if args.task=='init' else 'optim' if args.task=='optim' else 'results'
        run.write_result(payload,stage); run.finish('succeeded')
        return result
    except BaseException as exc:
        run.finish('failed',error=exc)
        raise
