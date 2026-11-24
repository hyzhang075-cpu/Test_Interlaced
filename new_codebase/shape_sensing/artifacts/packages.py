"""从运行快照导出自包含算法参数包。"""
from pathlib import Path
import json, shutil
from .runs import run_report
PACKAGE_TYPE='shape_sensing_algorithm_package'; SCHEMA_VERSION=1

def _run(path):
    root=Path(path) if Path(path).is_dir() else Path(path).parent
    config=json.loads((root/'config_resolved.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    stages=[root/name for name in ('optim.json','initialization.json','results.json') if (root/name).exists()]
    if not stages: raise ValueError(f'运行快照缺结果文件: {root}')
    result=json.loads(stages[0].read_text(encoding='utf-8'))
    return root,config,manifest,result

def export_package(run_paths,output):
    """按 region/direct 汇总已完成 run；优化结果要求 succeeded 且含参数报告。"""
    records=[_run(p) for p in run_paths]
    if not records: raise ValueError('至少需要一个 run 快照')
    sensors={r[2].get('sensor_id') for r in records}
    if len(sensors)!=1: raise ValueError('不能合并不同 sensor_id 的运行')
    modes = {bool(config.get('core_position_calib')) for _, config, _, _ in records}
    if len(modes) != 1:
        raise ValueError('不能混合四芯和分段 slope 标定运行')
    if True in modes:
        return _export_core_position(records, Path(output))
    base=records[0][1]; calibration={}; initialization={}; scope=[]; provenance=[]; code_sources=[]; region_meta={}; phase_values={}; rot_values={}; region_signatures={}
    for root,config,manifest,payload in records:
        task=manifest.get('task'); status=manifest.get('status')
        if status!='succeeded': raise ValueError(f'运行未成功，拒绝发布: {root} status={status}')
        for key in ('R','start_value','end_value','samp_dist_preset','samp_dist_preset_sub','active_channels','channel_layout','active_angles','centre_fibre','interlaced_grating','interlaced_offset','curvature_interp','frame_rule','move_filter','move_filter_for_vector','nonnegative_curvature','use_full_transported_phase'):
            if key in base and key in config and base[key]!=config[key]: raise ValueError(f'不同 run 传感器设计不兼容: {key}')
        provenance.append(str(root)); code_sources.append({'run_path':str(root),'code':manifest.get('code',{'commit':'unknown','branch':'unknown','working_tree_dirty':None})}); result=payload.get('result',payload)
        if task=='init':
            regional = result.get('regions')
            if isinstance(regional, dict):
                seeds = {}
                for region, entry in regional.items():
                    if entry.get('status') != 'complete' or not isinstance(entry.get('parameters'), dict):
                        raise ValueError(f'初始化区域未完成: {region}')
                    seeds[region] = entry['parameters']
            else:
                seeds = {str(config.get('optimization_region', 'full')): payload}
            for region, seed in seeds.items():
                initialization[region] = {key: seed.get(key) for key in ('weights', 'slopes', 'global_phases_set')}
                initialization[region]['run_path'] = str(root)
                scope.append({'task':'init','region':region,'final_calibration':False})
            continue
        if task!='optim': raise ValueError(f'仅支持 optim 最终结果或 init 初值包，收到 task={task!r}')
        report=None
        if isinstance(result,list) and len(result)>1 and isinstance(result[1],dict): report=result[1]
        elif isinstance(result,dict): report=result.get('report',result)
        if not isinstance(report,dict) or not isinstance(report.get('regions'),dict):
            raise ValueError(f'optim run 缺少区域结果报告: {root}')
        direct=report.get('direct',config.get('direct'))
        for region,entry in report['regions'].items():
            if entry.get('status')!='complete' or not isinstance(entry.get('parameters'),dict):
                raise ValueError(f'{direct}/{region} 优化阶段未完成')
            params=entry['parameters']
            if params.get('weights') is None: raise ValueError(f'{direct}/{region} 缺少 weights')
            if params.get('slopes') is None: raise ValueError(f'{direct}/{region} 缺少 slopes')
            if region in base.get('calibration_regions',{}) and region in config.get('calibration_regions',{}):
                for key in ('length_m','radii','thresholds','segmentation','end_weights','end_start'):
                    left=base['calibration_regions'][region].get(key); right=config['calibration_regions'][region].get(key)
                    if left!=right: raise ValueError(f'{region} 共享区域配置不兼容: {key}')
            direction=str(direct)
            context=entry.get('context') or {}
            signature={key:context.get(key,config.get(key)) for key in ('thresholds','segmentation','weight_group_ids','weight_score_mode','curvature_interp','frame_rule','move_filter','move_filter_for_vector','nonnegative_curvature')}
            previous=region_signatures.setdefault(region,signature)
            if previous != signature: raise ValueError(f'{region} 跨方向标定模式/布局不兼容')
            region_meta.setdefault(region,{'context':context,'configuration':entry.get('configuration') or config,'source_parameters':{}})
            region_meta[region]['source_parameters'][direction]=params
            phase_values.setdefault(region,{})[direction]=params.get('global_phases_set')
            rot_values.setdefault(region,{})[direction]=params.get('rot_ang')
            by_direction=calibration.setdefault(region,{})
            if direction in by_direction: raise ValueError(f'{region}/direct{direction} 重复结果，拒绝覆盖')
            by_direction[direction]={'weights':params['weights'],'slopes':params['slopes'],
                'global_phases_set':params.get('global_phases_set'),'rot_ang':params.get('rot_ang'),
                'phase_reference':params.get('phase_reference'),'errors':entry.get('errors'),
                'stage':entry.get('stages'),'run_path':str(root)}
            scope.append({'task':'optim','region':region,'direct':direct,'final_calibration':True})
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    config=dict(base)
    zero=config.get('zero_reference_path')
    resources=[]
    if zero:
        source=Path(zero)
        if not source.exists(): raise ValueError(f'zero_reference 资源不存在: {zero}')
        resource_dir=output.parent/(output.stem+'_resources'); resource_dir.mkdir(parents=True,exist_ok=True)
        destination=resource_dir/source.name
        shutil.copy2(source,destination); config['zero_reference_path']=f'{output.stem}_resources/{source.name}'
        resources.append({'path':config['zero_reference_path'],'source':str(source)})
    # 兼容 Reconstructor 的三区域结构；优先使用 batch report 保存的 resolved context。
    regions={}
    ordered=[name for name in ('full','end15cm','end5cm') if name in calibration]
    all_points=list(range(1,len(base.get('samp_dist_preset',[]))+1))
    supports={}
    for region in ordered:
        meta=region_meta[region]; ctx=meta['context']; cfg=meta['configuration']
        profile=cfg.get('calibration_regions',{}).get(region,{})
        if not ctx and not cfg.get('calibration_regions'):
            continue
        if not ctx:
            from ..configuration.regions import resolve_calibration_region
            profiles=cfg.get('calibration_regions',{})
            profiles['full']['segmentation']=list(cfg.get('full_segmentation',profiles['full'].get('segmentation',[])))
            ctx=resolve_calibration_region(start_value=cfg['start_value'],end_value=cfg['end_value'],
                samp_dist_preset=cfg['samp_dist_preset'],full_segmentation=cfg['full_segmentation'],
                profiles=profiles,optimization_region=region,multi_region_calib=False)
        segmentation=list(ctx.get('segmentation',profile.get('segmentation') or cfg.get('full_segmentation',[])))
        point_ids=list(ctx.get('global_point_ids') or ())
        supports[region]=point_ids
        thresholds=list(ctx.get('thresholds',profile.get('thresholds') or cfg.get('thresholds') or []))
        block={'calibration_objective':ctx.get('calibration_objective','geometry'),
            'calibration_geometry':{k:ctx.get(k) for k in ('optimization_region','sample_dist_m','crop_origin_m','reconstruction_length_m','ideal_len','straight_trim')},
            'weight_group_ids':ctx.get('weight_group_ids'),'weight_score_mode':ctx.get('weight_score_mode','spatial_window'),
            'length_m':ctx.get('length_m',profile.get('length_m')),'folder_path':ctx.get('folder_path',profile.get('folder_path')),
            'radii':ctx.get('radii',profile.get('radii')),'thresholds':thresholds,
            'end_weights':ctx.get('end_weights',profile.get('end_weights',False)),'end_start':ctx.get('end_start',profile.get('end_start',-5)),
            'sharpness':cfg.get('sharpness',4.8),'score_window':cfg.get('min_consecutive',3),'score_boundary_mode':'constant',
            'local_segmentation':segmentation,'global_segment_ids':list(ctx.get('global_segment_ids') or []),
            'support_point_ids':point_ids,'global_sample_dist_m':list(ctx.get('global_sample_dist_m') or ctx.get('sample_dist_m') or []),
            'weights':{},'slopes':{},'errors':{},'optimization':{},'phase_reference':{}}
        for direct,params in region_meta[region]['source_parameters'].items():
            weight_value=params['weights']; slope_value=params['slopes']
            if isinstance(weight_value,dict): weight_value=weight_value.get(direct,weight_value.get(int(direct)))
            if isinstance(slope_value,dict): slope_value=slope_value.get(direct,slope_value.get(int(direct)))
            block['weights'][direct]=weight_value; block['slopes'][direct]=slope_value
            block['errors'][direct]=params.get('errors') or {}; block['optimization'][direct]=params.get('optimization') or {}
            block['phase_reference'][direct]=params.get('phase_reference')
        regions[region]=block
    compatible_order=[name for name in ordered if name in regions]
    for idx,region in enumerate(compatible_order):
        inner=set(p for later in compatible_order[idx+1:] for p in supports[later])
        regions[region]['owned_point_ids']=[p for p in supports[region] if p not in inner]
    directions=sorted({int(d) for by_region in calibration.values() for d in by_region})
    use_fixed=bool(base.get('use_full_transported_phase',False))
    refs=[]
    if use_fixed:
        refs=[d for d in directions if all(calibration.get(region,{}).get(str(d),{}).get('phase_reference') for region in ('end15cm','end5cm'))]
        if refs != directions: raise ValueError('固定上游相位模式缺少配套 end15cm/end5cm 每方向引用')
    phases={region:phase_values.get(region,{}) for region in ordered}
    rotations={region:rot_values.get(region,{}) for region in ordered}
    package={'package_type':PACKAGE_TYPE,'schema_version':SCHEMA_VERSION,'sensor_id':next(iter(sensors),None),
        'configuration':config,'calibration':calibration,'regions':regions,
        'multi_region':{'region_order':compatible_order,'directions':directions,'use_full_transported_phase':use_fixed,
            'phase_reference_directions':refs,'global_phases_set':base.get('global_phases_set'),
            'rot_ang':base.get('rot_ang'),'deployment_ready':False},
        'initialization':initialization,'scope':scope,
        'resources':resources,'provenance':{'runs':provenance,'code_sources':code_sources},
        'validation':{'deployment_ready':False,'verified':False}}
    output.write_text(json.dumps(package,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return package

def _export_core_position(records, output):
    """Merge the optimized direction from each four-core task snapshot."""
    config = dict(records[0][1])
    calibration, blocks, initialization = {}, {}, {}
    scope, provenance = [], []
    design_keys = (
        'sensor_id', 'R', 'start_value', 'end_value', 'samp_dist_preset', 'samp_dist_preset_sub',
        'active_channels', 'channel_layout', 'centre_fibre', 'interlaced_grating', 'interlaced_offset',
        'curvature_interp', 'frame_rule', 'move_filter', 'move_filter_for_vector',
        'nonnegative_curvature', 'interp_num',
    )
    for root, resolved, manifest, payload in records:
        task = manifest.get('task')
        if manifest.get('status') != 'succeeded' or task not in ('init', 'optim'):
            raise ValueError(f'四芯参数包需要成功的 init 或 optim 运行: {root}')
        for key in design_keys:
            if config.get(key) != resolved.get(key):
                raise ValueError(f'不同四芯运行的配置不兼容: {key}')
        provenance.append({'run_path': str(root), 'code': manifest.get('code', {})})
        result = run_report(payload)
        regions = result.get('core_position_parameters')
        if not isinstance(regions, dict) or not regions:
            raise ValueError(f'运行缺少四芯区域参数: {root}')
        for region, block in regions.items():
            metadata = block.get('core_position_metadata', {})
            if metadata.get('schema_version') != 6 or not block.get('core_position'):
                raise ValueError(f'{region} 缺少 schema 6 芯位')
            geometry = {key: block[key] for key in ('core_position', 'core_position_metadata', 'weight_layout')}
            combined = blocks.setdefault(region, {**geometry, 'weights': {}})
            if any(combined[key] != value for key, value in geometry.items()):
                raise ValueError(f'{region} 芯位或权重分档配置不一致')
            if task == 'init':
                initialization[region] = {'run_path': str(root)}
                scope.append(dict(task=task, region=region, final_calibration=False))
                continue
            direct = str(resolved['direct'])
            entry = result.get('regions', {}).get(region)
            if entry is not None:
                if entry.get('status') != 'complete':
                    raise ValueError(f'{region}/{direct} 区域优化未完成')
                report = entry['optimization']
            else:
                report = result.get('reports', {}).get(region, {})
            stage = report.get('stages', {}).get('weights', {})
            weights = block.get('weights', {}).get(direct)
            if not weights or stage.get('final_loss') is None or stage.get('status') == 'diverge':
                raise ValueError(f'{region}/{direct} 缺少可用的权重优化结果')
            table = calibration.setdefault(region, {})
            if direct in table:
                raise ValueError(f'{region}/{direct} 重复优化结果')
            table[direct] = dict(weights=weights, stages=report['stages'], objective=report.get('objective'),
                                 run_path=str(root))
            if entry is not None:
                table[direct].update(errors=entry['errors'], segmentation=entry['context']['segmentation'])
            combined['weights'][direct] = weights
            scope.append(dict(task=task, region=region, direct=int(direct), final_calibration=True))

    for region, block in blocks.items():
        directions = {str(d) for d in block['core_position_metadata']['calibration']['directions_deg']}
        if set(block['weights']) != directions:
            raise ValueError(f'{region} 缺少优化方向: {sorted(directions - set(block["weights"]))}')
    resources = []
    output.parent.mkdir(parents=True, exist_ok=True)
    if config.get('zero_reference_path'):
        source = Path(config['zero_reference_path'])
        destination = output.parent / (output.stem + '_resources') / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        config['zero_reference_path'] = destination.relative_to(output.parent).as_posix()
        resources.append({'path': config['zero_reference_path'], 'source': str(source)})
    for key in ('task', 'resume', 'provenance'):
        config.pop(key, None)
    package = dict(package_type=PACKAGE_TYPE, schema_version=SCHEMA_VERSION,
        sensor_id=config['sensor_id'], configuration=config, calibration=calibration,
        core_position_parameters=blocks, initialization=initialization, scope=scope, resources=resources,
        multi_region=dict(region_order=list(blocks), directions=sorted({
            int(direct) for table in calibration.values() for direct in table}),
            deployment_ready=False),
        provenance=dict(runs=[item['run_path'] for item in provenance], code_sources=provenance),
        validation=dict(verified=False, deployment_ready=False))
    output.write_text(json.dumps(package, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return package


def load_algorithm_package(path):
    """校验并读取新包；format_version=2 委托原 loader。"""
    source=Path(path); data=json.loads(source.read_text(encoding='utf-8'))
    if data.get('format_version')==2 and data.get('package_type') is None:
        from .legacy_package import load_sensor_package
        return load_sensor_package(source)
    if 'format_version' in data and data.get('format_version') != 2:
        raise ValueError(f"不支持的 format_version={data.get('format_version')}")
    if data.get('package_type')!=PACKAGE_TYPE: raise ValueError('未知算法参数包类型')
    if data.get('schema_version')!=SCHEMA_VERSION: raise ValueError('不支持的算法参数包 schema_version')
    if not data.get('sensor_id') or not isinstance(data.get('configuration'),dict) or not isinstance(data.get('calibration'),dict):
        raise ValueError('算法参数包缺少 sensor_id/configuration/calibration')
    resource=data['configuration'].get('zero_reference_path')
    if isinstance(resource,str) and resource and not Path(resource).is_absolute():
        data['configuration']['zero_reference_path']=str((source.parent/resource).resolve())
    for region,by_direct in data['calibration'].items():
        if isinstance(by_direct,dict):
            for entry in by_direct.values():
                if isinstance(entry,dict):
                    for field in ('weights','slopes'):
                        value=entry.get(field)
                        if isinstance(value,dict): entry[field]={int(k) if str(k).lstrip('-').isdigit() else k:v for k,v in value.items()}
    return data
