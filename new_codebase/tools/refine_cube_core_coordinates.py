"""Joint physical four-core coordinate/gain refinement, no spatial partition."""
import copy,json,pickle,itertools,os
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters,CurvatureResult
from shape_sensing.initialization.four_core import enclosing_circle
from shape_sensing.curvature.service import CurvatureService
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube';out=run/'pointwise_core'
region=os.environ.get('CUBE_PHYSICAL_REGION','full')
if region!='full':
    out=run/f'pointwise_core_physical_{region}'
    out.mkdir(exist_ok=True)
seed=json.loads((run/'20261125-172521-655152-8faf677802-init/initialization.json').read_text(encoding='utf-8'))['result']['core_position_parameters']
candidate_path=os.environ.get('CUBE_PHYSICAL_SEED')
candidate=json.loads(Path(candidate_path).read_text(encoding='utf-8')) if candidate_path else None
target_excess=os.environ.get('CUBE_PHYSICAL_OBJECTIVE')=='target_excess'
if candidate:
    seed.update(copy.deepcopy(candidate['parameters']))
    out=run/'pointwise_core_target_refinement'
    out.mkdir(exist_ok=True)
if os.environ.get('CUBE_PHYSICAL_OUTPUT'):
    out=root/os.environ['CUBE_PHYSICAL_OUTPUT'];out.mkdir(exist_ok=True)
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.core_position_parameters=copy.deepcopy(seed)
args.optimization_region=region
if os.environ.get('CUBE_INTERPOLATION'):args.curvature_interp=os.environ['CUBE_INTERPOLATION']
for block in args.core_position_parameters.values():block.pop('weights',None);block.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region=region);dataset=session.load_data()
block=session.parameters.parameters_by_region[region];sources={}
if candidate:
    layout=dict(candidate['parameters'][region]['weight_layout'])
    layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
    session.config=replace(session.config,curvature=replace(session.config.curvature,**layout))
    session.curvature=CurvatureService(session.config.curvature)
    block['weight_layout']=copy.deepcopy(candidate['parameters'][region]['weight_layout'])
levels=len(session.config.curvature.thresholds)+1
for direct in (0,120,240):
    if candidate:
        block['weights'][direct]=np.asarray(candidate['parameters']['full']['weights'][str(direct)])
        sources[direct]=candidate_path
        warm=run/'pointwise_core_physical_seed_balanced'/f'active_full_{direct}.npz'
        if not warm.exists():warm=run/'pointwise_core_physical_seed'/f'active_full_{direct}.npz'
        if warm.exists() and os.environ.get('CUBE_PHYSICAL_NO_WARM')!='1':
            with np.load(warm) as active:block['weights'][direct]=active['x'].copy()
            sources[direct]=str(warm)
        continue
    best=float('inf')
    for path in run.rglob(f'*seg_{region}_{direct}.pkl'):
        with path.open('rb') as stream:saved=pickle.load(stream)
        m=saved['metrics'][str(direct)];score=m['point']+m['end']
        if score<best:
            old=saved['parameters'].parameters_by_region[region];best=score;sources[direct]=str(path)
            block['weights'][direct]=np.asarray(old['weights'][direct]).reshape(-1,levels)[old['weight_layout']['weight_group_ids']].reshape(-1)
    path=run/'pointwise_core'/f'active_{region}_{direct}.npz'
    if path.exists():
        with np.load(path) as active:
            if float(active['loss'])<best:block['weights'][direct]=active['x'].copy();sources[direct]=str(path)
    if region!='full':
        path=run/'pointwise_core_balanced'/f'active_{region}_{direct}.npz'
        if path.exists():
            with np.load(path) as active:block['weights'][direct]=active['x'].copy()
            sources[direct]=str(path)
base=copy.deepcopy(session.parameters.parameters_by_region);original=session.curvature
(out/'physical_coordinates_seed.json').write_text(
    json.dumps({'parameters':base,'source_gains':sources,'region':region},
               indent=2,default=lambda value:value.tolist()),encoding='utf-8')
sections=block['core_position'];p=len(sections)
coordinates=np.asarray([[[c['r_um']*np.sin(c['alpha_rad'] or 0),c['r_um']*np.cos(c['alpha_rad'] or 0)] for c in section['cores'].values()] for section in sections])/100.
centres=np.asarray([s['envelope_center_uv_um'] for s in sections])/100.
x0=np.r_[coordinates[:,:3].reshape(-1),centres.reshape(-1),np.zeros(3*p)]
resume_x=None
if os.environ.get('CUBE_PHYSICAL_RESUME'):
    with np.load(os.environ['CUBE_PHYSICAL_RESUME']) as checkpoint:resume_x=checkpoint['x'].copy()
pairs=list(itertools.combinations(range(4),2));states=[];frames=[];lookup={}
for direct in (0,120,240):
    for radius,values in dataset.data['circ'][direct].items():
        frame=session.frame(values);lookup[(direct,frame.signals.tobytes())]=len(frames);frames.append(frame);states.append((direct,radius))
metadata=block['core_position_metadata']['calibration'];sensitivity=metadata['strain_sensitivity_pm_per_ue'];sign=metadata['strain_projection_sign']
responses=np.asarray([f.signals for f in frames])/(sensitivity*sign)
def unpack(x):
    first=x[:6*p].reshape(p,3,2);points=np.concatenate((first,-first.sum(axis=1,keepdims=True)),axis=1)
    return points,x[6*p:8*p].reshape(p,2),x[8*p:].reshape(3,p)
def constraints(x):
    points,centre,log_gain=unpack(x);r2=np.sum((points-centre[:,None,:])**2,axis=2)
    separation=np.stack([np.sum((points[:,i]-points[:,j])**2,axis=1)-1 for i,j in pairs],axis=1)
    return np.r_[separation.reshape(-1),(1.25**2-r2).reshape(-1),(r2[:,1:]-r2[:,:1]).reshape(-1)]
assert constraints(x0).min()>-1e-5
current=[];current_gains=[]
def build_fields(x):
    global current,current_gains
    points,centre,log_gain=unpack(x)
    matrix=np.concatenate((points,np.ones((p,4,1))),axis=2)
    inverse=np.linalg.pinv(matrix)
    vectors=np.einsum('pij,spj->spi',inverse[:,:2,:],responses)/1e-4
    magnitude=np.linalg.norm(vectors,axis=2);angle=np.arctan2(vectors[:,:,1],vectors[:,:,0]);current=[];current_gains=[]
    for index,(direct,radius) in enumerate(states):
        gain=original._effective_core_weights(magnitude[index],np.asarray(base[region]['weights'][direct]))*np.exp(log_gain[(0,120,240).index(direct)])
        current_gains.append(gain)
        current.append(CurvatureResult(frames[index].distance_m,magnitude[index]*gain,angle[index],frames[index].source_indices))
class CachedPhysicalInverse:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):return current[lookup[(direct,frame.signals.tobytes())]]
session.curvature=CachedPhysicalInverse();calls=0;best=float('inf');best_x=x0.copy();best_metrics=None
target_direct=os.environ.get('CUBE_CORE_TARGET_DIRECTION');metric_cache={}
def evaluate(x,verbose=False):
    global calls,best,best_x,best_metrics
    key=np.asarray(x).tobytes()
    if not verbose and key in metric_cache:return metric_cache[key][0]
    build_fields(x);metrics={};loss=0.
    for direct in (0,120,240):
        curves,distance=session.reconstruct_shapes('circ',direct);m=session.evaluate(distance,{'circ':curves},direct).metrics
        metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        loss+=metrics[str(direct)]['point']+.5*metrics[str(direct)]['end']
    if target_direct:loss=metrics[target_direct]['point']+.5*metrics[target_direct]['end']
    if target_excess:
        values=np.asarray([value for m in metrics.values() for value in m.values()])
        target=.0095 if region=='full' else .095
        loss=float(np.sum(np.maximum(values-target,0)**2)/target+.001*values.sum())
    metric_cache[key]=(loss,copy.deepcopy(metrics))
    if len(metric_cache)>1024:del metric_cache[next(iter(metric_cache))]
    calls+=1;feasible=float(constraints(x).min())
    passed_guard=(os.environ.get('CUBE_KEEP_PASSED_FULL')!='1' or
                  all(metrics[str(d)]['point']<.01 and metrics[str(d)]['end']<.01 for d in (0,120)))
    if feasible>=-1e-6 and passed_guard and loss<best:
        best=loss;best_x=np.array(x);best_metrics=copy.deepcopy(metrics)
        np.savez(out/'physical_coordinates_active.npz',x=x,loss=loss)
    if calls%100==0 or verbose:print('PHYSICAL_CORES',calls,'loss',loss,'constraint_min',feasible,'best_feasible',best,'metrics',metrics,flush=True)
    return loss
evaluate(x0,True)
# Verify the cached forward path against the unmodified physical inverse at seed.
session.curvature=original
for direct in (0,120,240):
    curves,distance=session.reconstruct_shapes('circ',direct);m=session.evaluate(distance,{'circ':curves},direct).metrics
    assert abs(float(m['avg_point_err']['calib'])-best_metrics[str(direct)]['point'])<1e-9
session.curvature=CachedPhysicalInverse()
start=x0
if os.environ.get('CUBE_PHYSICAL_RESUME'):
    start=resume_x
    if start.shape!=x0.shape:raise ValueError('Physical checkpoint shape does not match region')
    evaluate(start,True)
bounds=[(-2.,2.)]*(8*p)+[(-.8,.8)]*(3*p)
if os.environ.get('CUBE_FIX_CENTRAL_CORE')=='1':
    for point in range(p):
        for axis in range(2):bounds[6*point+axis]=(0.,0.);bounds[6*p+2*point+axis]=(0.,0.)
constraint_list=[{'type':'ineq','fun':constraints}]
if target_direct:
    def passed_accuracy_constraints(x):
        evaluate(x);m=metric_cache[np.asarray(x).tobytes()][1]
        return np.asarray([.00999-m[str(d)][k] for d in (0,120) for k in ('point','end')])
    constraint_list.append({'type':'ineq','fun':passed_accuracy_constraints})
result=minimize(evaluate,start,method='SLSQP',bounds=bounds,constraints=constraint_list,options={'maxiter':int(os.environ.get('CUBE_PHYSICAL_MAXITER','80')),'ftol':1e-8,'disp':True})
np.savez(out/'physical_coordinates_last_trial.npz',x=result.x)
if constraints(result.x).min() < -1e-7 or (target_direct and passed_accuracy_constraints(result.x).min() < -1e-8):
    # Restore physical feasibility before evaluating a near-boundary final trial.
    trial=result.x.copy()
    projection=minimize(lambda x:float(np.sum((x-trial)**2)),trial,
        jac=lambda x:2*(x-trial),method='SLSQP',
        bounds=bounds,
        constraints=constraint_list,
        options={'maxiter':200,'ftol':1e-12})
    if constraints(projection.x).min()>=-1e-7:evaluate(projection.x,True)
assert constraints(best_x).min()>=-1e-6
evaluate(best_x,True);points,centres,log_gain=unpack(best_x);blocks=copy.deepcopy(base)
for index,section in enumerate(blocks[region]['core_position']):
    section['signal_fit_diagnostics_provenance']='initialization_before_geometry_refinement; not recomputed for refined coordinates'
    xy=points[index]*100.;centre=centres[index]*100.
    for core,point in zip(section['cores'].values(),xy):
        radius=float(np.linalg.norm(point));alpha=float(np.arctan2(point[0],point[1])%(2*np.pi))
        core.update(r_um=radius,alpha_rad=alpha if radius>1e-8 else None,alpha_deg=float(np.degrees(alpha)) if radius>1e-8 else None)
    enclosing_center,enclosing_radius=enclosing_circle(xy)
    section.update(envelope_center_uv_um=centre.tolist(),envelope_containing_radius_um=float(np.linalg.norm(xy-centre,axis=1).max()),minimum_enclosing_center_uv_um=enclosing_center.tolist(),minimum_enclosing_radius_um=float(enclosing_radius),minimum_pairwise_distance_um=float(min(np.linalg.norm(xy[i]-xy[j]) for i,j in pairs)),centroid_error_um=float(np.linalg.norm(xy.mean(axis=0))),constraint_solver='joint_physical_geometry_slsqp')
for di,direct in enumerate((0,120,240)):
    blocks[region]['weights'][direct]=(np.asarray(base[region]['weights'][direct]).reshape(p,-1)*np.exp(log_gain[di,:,None])).reshape(-1)
session.parameters=CalibrationParameters(blocks,source='joint_physical_core_geometry_and_point_gain');session.curvature=original;actual={}
for direct in (0,120,240):
    curves,distance=session.reconstruct_shapes('circ',direct);m=session.evaluate(distance,{'circ':curves},direct).metrics
    actual[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    assert abs(actual[str(direct)]['point']-best_metrics[str(direct)]['point'])<1e-8
    assert abs(actual[str(direct)]['end']-best_metrics[str(direct)]['end'])<1e-8
def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    return x
report={'metrics':actual,'parameters':session.parameters.parameters_by_region,'physical_constraint_min':float(constraints(best_x).min()),'optimizer_success':bool(result.success),'source_gains':sources,'spatial_partition_search':False,'objective':'squared_excess_over_target' if target_excess else 'point_plus_0.5_end','region':region,'k_point':1.0,'k_end':0.5,'interp_num':256}
(out/'physical_coordinate_refinement.json').write_text(json.dumps(native(report),indent=2),encoding='utf-8')
print('PHYSICAL_CORES_VERIFIED',actual,flush=True)
