"""Joint geometric refinement of physical per-WL core orientation, no slopes."""
import copy,json,pickle
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
root=Path(__file__).resolve().parents[1]
run=root/'runs/11B1_Cube'
seed=json.loads((run/'20261125-172521-655152-8faf677802-init/initialization.json').read_text(encoding='utf-8'))['result']['core_position_parameters']
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False
args.core_position_parameters=copy.deepcopy(seed)
for block in args.core_position_parameters.values():
    block.pop('weights',None);block.pop('weight_layout',None)
configure_calibration_args(args)
args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
dataset=session.load_data()
block=session.parameters.parameters_by_region['full']
sources={}
for direct in (0,120,240):
    best=float('inf'); selected=None
    for path in run.rglob(f'*seg_full_{direct}.pkl'):
        with path.open('rb') as stream: saved=pickle.load(stream)
        m=saved['metrics'][str(direct)]; score=m['point']+m['end']
        if score<best:
            old=saved['parameters'].parameters_by_region['full']
            selected=np.asarray(old['weights'][direct]).reshape(-1,6)[old['weight_layout']['weight_group_ids']].reshape(-1)
            best=score;sources[direct]=str(path)
    path=run/f'pointwise_core/active_full_{direct}.npz'
    if path.exists():
        with np.load(path) as candidate:
            if float(candidate['loss'])<best:
                selected=candidate['x'].copy();best=float(candidate['loss']);sources[direct]=str(path)
    block['weights'][direct]=selected
base=copy.deepcopy(session.parameters.parameters_by_region)
original=session.curvature
cache={}
for direct in (0,120,240):
    for radius,values in dataset.data['circ'][direct].items():
        frame=session.frame(values)
        cache[(direct,frame.signals.tobytes())]=original.compute(frame,session.design,session.parameters,direct=direct,region='full',shape_kind='circ')
def rotated_parameters(offsets):
    blocks=copy.deepcopy(base)
    for section,delta in zip(blocks['full']['core_position'],offsets):
        for core in section['cores'].values():
            if core['alpha_rad'] is not None:
                core['alpha_rad']+=float(delta);core['alpha_deg']=float(np.degrees(core['alpha_rad']))
        for key in ('minimum_enclosing_center_uv_um','envelope_center_uv_um'):
            if key in section:
                u,v=section[key]
                section[key]=[float(u*np.cos(delta)+v*np.sin(delta)),float(-u*np.sin(delta)+v*np.cos(delta))]
    return CalibrationParameters(blocks,source='per_WL_core_orientation_geometric_refinement')
test=np.linspace(.01,.03,18)
direct=0
values=next(v for r,v in dataset.data['circ'][direct].items() if r>0)
frame=session.frame(values)
before=cache[(direct,frame.signals.tobytes())]
after=original.compute(frame,session.design,rotated_parameters(test),direct=direct,region='full',shape_kind='circ')
delta=np.angle(np.exp(1j*(after.direction_rad-before.direction_rad)))
sign=1 if np.max(np.abs(delta-test))<np.max(np.abs(delta+test)) else -1
assert np.max(np.abs(delta-sign*test))<1e-9
assert np.max(np.abs(after.curvature_per_m-before.curvature_per_m))<1e-9
offsets=np.zeros(18)
class CachedPhysicalRotation:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        result=cache[(direct,frame.signals.tobytes())]
        return replace(result,direction_rad=result.direction_rad+sign*offsets)
session.curvature=CachedPhysicalRotation()
calls=0;best=float('inf');best_x=offsets.copy();best_metrics=None
def evaluate(x,verbose=False):
    global offsets,calls,best,best_x,best_metrics
    offsets=np.asarray(x)
    metrics={};loss=0.
    for direct in (0,120,240):
        curves,distance=session.reconstruct_shapes('circ',direct)
        m=session.evaluate(distance,{'circ':curves},direct).metrics
        metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        loss+=metrics[str(direct)]['point']+metrics[str(direct)]['end']
    calls+=1
    if loss<best:
        best=loss;best_x=np.array(x);best_metrics=copy.deepcopy(metrics)
        np.savez(run/'pointwise_core/core_orientation_active.npz',offsets_rad=x,loss=loss)
    if calls%20==0 or verbose: print('POINT_CORE_ORIENTATION',calls,'loss',loss,'metrics',metrics,flush=True)
    return loss
evaluate(offsets,True)
result=minimize(evaluate,offsets,method='L-BFGS-B',bounds=[(-np.pi/6,np.pi/6)]*18,options={'maxiter':100,'maxfun':1800,'ftol':1e-9})
session.parameters=rotated_parameters(best_x)
session.curvature=original
actual={}
for direct in (0,120,240):
    curves,distance=session.reconstruct_shapes('circ',direct)
    m=session.evaluate(distance,{'circ':curves},direct).metrics
    actual[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    assert abs(actual[str(direct)]['point']-best_metrics[str(direct)]['point'])<1e-9
    assert abs(actual[str(direct)]['end']-best_metrics[str(direct)]['end'])<1e-9
def native(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [native(v) for v in x]
    return x
report={'metrics':actual,'offsets_rad':best_x,'source_gains':sources,'spatial_partition_search':False,'parameters':session.parameters.parameters_by_region}
(run/'pointwise_core/core_orientation_refinement.json').write_text(json.dumps(native(report),indent=2),encoding='utf-8')
print('POINT_CORE_ORIENTATION_VERIFIED',actual,flush=True)
