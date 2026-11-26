import os,sys,pickle,copy,json
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')];os.chdir(root/'new_codebase')
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
folder=Path('runs/11B8_core_point_refine_k1_end05')
base=None
for direct in (0,120,240):
 with (folder/f'full_{direct}.pkl').open('rb') as f:r=pickle.load(f)
 b=r['parameters'].parameters_by_region['full']
 if base is None:base=copy.deepcopy(b)
 base['weights'][direct]=b['weights'][direct]
a=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
a.multi_region_calib=False;a.optimization_region='full';a.calibration_regions['full']['weight_group_ids']=list(range(40));a.core_position_parameters={'full':base}
configure_calibration_args(a);a.exp_directs=[0,120,240]
s=TaskSession(module_parameters(a,optimizer_parameters),task='optim',region='full');s.load_data()
n=0;best=float('inf');best_x=np.zeros(16);last_metrics={}
def evaluate(x,verbose=False):
 global n,best,best_x,last_metrics
 b=copy.deepcopy(base)
 angles=np.interp(s.design.sampling_distance_m,np.linspace(s.design.sampling_distance_m[0],s.design.sampling_distance_m[-1],8),x[:8])
 scales=np.exp(np.interp(s.design.sampling_distance_m,np.linspace(s.design.sampling_distance_m[0],s.design.sampling_distance_m[-1],8),x[8:]))
 for section,delta,scale in zip(b['core_position'],angles,scales):
  for core in section['cores'].values():
   if core['alpha_rad'] is not None:
    core['r_um']*=float(scale);core['alpha_rad']+=float(delta);core['alpha_deg']=float(np.degrees(core['alpha_rad']))
 s.parameters=CalibrationParameters({'full':b},source='shared_core_angle_refinement')
 m={};loss=0.
 for direct in (0,120,240):
  curves,d=s.reconstruct_shapes('circ',direct);v=s.evaluate(d,{'circ':curves},direct).metrics
  m[str(direct)]={'point':float(v['avg_point_err']['calib']),'end':float(v['avg_end_err']['calib'])}
  loss+=m[str(direct)]['point']+0.5*m[str(direct)]['end']
 last_metrics=m
 n+=1
 if loss<best:
  best=float(loss);best_x=np.array(x)
  with (folder/'shared_core_constrained_best.pkl').open('wb') as f:pickle.dump({'parameters':s.parameters,'metrics':m,'loss':loss,'offsets':x},f)
 if n%20==0 or verbose:print('ANGLE',n,'loss',loss,'best',best,'metrics',m,flush=True)
 return loss

with (folder/'shared_core_angle_best.pkl').open('rb') as f:prev=pickle.load(f)
x0=np.r_[prev['offsets'],np.zeros(8)]
cache={}
def state(x):
 key=np.asarray(x).tobytes()
 if key not in cache:
  value=evaluate(x)
  cache.clear();cache[key]=(value,copy.deepcopy(last_metrics))
 return cache[key]
def constraints(x):
 _,metrics=state(x)
 return np.array([1.-v[k]/0.0098 for v in metrics.values() for k in ('point','end')])
evaluate(x0,True)
result=minimize(lambda x:state(x)[0],x0,method='SLSQP',constraints=[{'type':'ineq','fun':constraints}],bounds=[(-np.pi/12,np.pi/12)]*8+[(-0.05,0.05)]*8,options={'maxiter':100,'ftol':1e-9})
evaluate(result.x,True)
with (folder/'shared_core_constrained_final.pkl').open('wb') as f:pickle.dump({'parameters':s.parameters,'metrics':last_metrics,'result':result},f)
print('CONSTRAINED_COMPLETE',result.success,result.message,flush=True)
