"""运行配置、阶段结果和显式续算校验。"""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json, math, os, tempfile, subprocess
import numpy as np

def _native(value):
    if isinstance(value,dict): return {str(k):_native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_native(v) for v in value]
    if isinstance(value,np.ndarray): return _native(value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not math.isfinite(value): raise ValueError('JSON 不允许 NaN/Infinity')
    return value
class RunStore:
    """写入一个 run 的配置快照和阶段结果，使用原子文件替换。"""
    def __init__(self,path): self.path=Path(path)
    def _write(self,name,value):
        target=self.path/name; payload=json.dumps(_native(value),ensure_ascii=False,indent=2,allow_nan=False)
        fd,tmp=tempfile.mkstemp(dir=self.path,prefix='.tmp-')
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f: f.write(payload)
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return target
    def write_result(self,result,stage): return self._write(f'{stage}.json',result)
    def finish(self,status,error=None):
        manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        manifest.update(status=status,finished_at=datetime.now(timezone.utc).isoformat())
        if error is not None: manifest['error']=str(error)
        return self._write('manifest.json',manifest)

def _code_provenance():
    """读取 run 创建时的 Git 来源；无 Git/元数据时如实记录 unknown。"""
    root=Path.cwd()
    def git(*args):
        try: return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL,text=True).strip()
        except Exception: return None
    commit=git('rev-parse','HEAD'); branch=git('branch','--show-current')
    status=git('status','--porcelain')
    return {'commit':commit or 'unknown','branch':branch or 'unknown',
            'working_tree_dirty':None if status is None else bool(status)}

def create_run(sensor_id,task,resolved_config,output_root):
    """排他创建唯一运行目录，并保存完整 resolved 配置和 manifest。"""
    root=Path(output_root)/str(sensor_id); root.mkdir(parents=True,exist_ok=True)
    while True:
        path=root/f'{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}-{uuid4().hex[:10]}-{task}'
        try: path.mkdir(exist_ok=False); break
        except FileExistsError: continue
    store=RunStore(path)
    store._write('config_resolved.json',resolved_config)
    store._write('manifest.json',{'sensor_id':sensor_id,'task':task,'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'code':_code_provenance()})
    return store

_GEOMETRY_KEYS=('sensor_id','R','start_value','end_value','samp_dist_preset','samp_dist_preset_sub',
    'active_channels','channel_layout','active_angles','centre_fibre','interlaced_grating','interlaced_offset')
_REGION_LAYOUT_KEYS = ('radii', 'thresholds', 'segmentation', 'end_weights', 'end_start')
def _layout(value):
    if isinstance(value,dict): return {str(k):_layout(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_layout(v) for v in value]
    if isinstance(value,np.ndarray): return list(value.shape)
    return value

def initialization_layout(args, context):
    """返回单区域的 JSON 布局：分段及索引、距离 m、半径 mm、曲率等级 1/m。"""
    layout = {key: getattr(args, key, None) for key in _REGION_LAYOUT_KEYS}
    layout.update({key: context.get(key) for key in (
        'global_point_ids', 'sample_dist_m', 'weight_group_ids', 'weight_score_mode')})
    return _native(layout)


def load_resume(path,resolved_config):
    """读取显式 run 结果并检查传感器、几何、区域及阈值布局兼容性。"""
    source=Path(path)
    if source.is_dir():
        candidates=[source/'initialization.json',source/'optim.json',source/'results.json']
        source=next((p for p in candidates if p.exists()),None)
        if source is None: raise ValueError('指定 run 中没有可续算阶段结果')
    result=json.loads(source.read_text(encoding='utf-8'))
    run_dir=source.parent
    config_path=run_dir/'config_resolved.json'
    if not config_path.exists(): raise ValueError('resume 缺少 config_resolved.json，无法验证来源配置')
    prior=json.loads(config_path.read_text(encoding='utf-8'))
    core_mode = resolved_config.get('core_position_calib', False)
    if prior.get('core_position_calib', False) != core_mode:
        raise ValueError('resume 配置不兼容: core_position_calib')
    for key in _GEOMETRY_KEYS:
        if key in prior and key in resolved_config and _layout(prior[key]) != _layout(resolved_config[key]):
            raise ValueError(f'resume 配置不兼容: {key}')
    if core_mode:
        manifest = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'succeeded' or result.get('task') not in ('init', 'optim'):
            raise ValueError('四芯 resume 需要已成功完成的 init 或 optim 快照')
        blocks = result.get('result', {}).get('core_position_parameters')
        if not isinstance(blocks, dict) or not blocks:
            raise ValueError('resume 缺少四芯芯位参数')
        for key in ('core_diameter_um', 'core_radius_um', 'core_sensitivity_pm_per_ue', 'core_bend_offset_deg'):
            if prior.get(key) != resolved_config.get(key):
                raise ValueError(f'resume 配置不兼容: {key}')
        return {'core_position_parameters': blocks}
    initialization = result.get('result')
    if (result.get('task') == 'init' and isinstance(initialization, dict)
            and isinstance(initialization.get('regions'), dict)):
        manifest = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'succeeded':
            raise ValueError('resume 需要已成功完成的多区域初始化运行')
        # 已解析的布局随各区域保存，由任务层与实际选中的优化 job 比较。
        return result
    old_region=prior.get('optimization_region','full'); new_region=resolved_config.get('optimization_region','full')
    if old_region != new_region: raise ValueError('resume 配置不兼容: optimization_region')
    old_profiles=prior.get('calibration_regions',{}); new_profiles=resolved_config.get('calibration_regions',{})
    regions=set([old_region,new_region])
    if prior.get('multi_region_calib') or resolved_config.get('multi_region_calib'):
        regions.update(prior.get('optimization_regions',[])); regions.update(resolved_config.get('optimization_regions',[]))
    for region in regions:
        old_profile=dict(old_profiles.get(region,{})); new_profile=dict(new_profiles.get(region,{}))
        for key in ('thresholds','segmentation','radii','end_weights','end_start','weight_group_ids','weight_score_mode'):
            old_value=prior.get(key) if prior.get(key) is not None else old_profile.get(key)
            new_value=resolved_config.get(key) if resolved_config.get(key) is not None else new_profile.get(key)
            # Full-region segmentation falls back to the authoritative full_segmentation.
            if key=='segmentation' and region=='full':
                old_value=prior.get('segmentation') or prior.get('full_segmentation') or old_profile.get(key)
                new_value=resolved_config.get('segmentation') or resolved_config.get('full_segmentation') or new_profile.get(key)
            if old_value is not None and new_value is not None and _layout(old_value)!=_layout(new_value):
                raise ValueError(f'resume 区域配置不兼容: {region}.{key}')
    if 'weights' in result and 'slopes' not in result: raise ValueError('resume 有 weights 但没有匹配 slopes')
    for name in ('weights','slopes'):
        value=result.get(name)
        if isinstance(value,dict): result[name]={int(k) if isinstance(k,str) and k.lstrip('-').isdigit() else k:v for k,v in value.items()}
    return result
