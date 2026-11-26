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
n=0;best=float('inf');best_x=np.zeros(8)
def evaluate(x,verbose=False):
 global n,best,best_x
 b=copy.deepcopy(base)
 angles=np.interp(s.design.sampling_distance_m,np.linspace(s.design.sampling_distance_m[0],s.design.sampling_distance_m[-1],8),x)
 for section,delta in zip(b['core_position'],angles):
  for core in section['cores'].values():
   if core['alpha_rad'] is not None:
    core['alpha_rad']+=float(delta);core['alpha_deg']=float(np.degrees(core['alpha_rad']))
 s.parameters=CalibrationParameters({'full':b},source='shared_core_angle_refinement')
 m={};loss=0.
 for direct in (0,120,240):
  curves,d=s.reconstruct_shapes('circ',direct);v=s.evaluate(d,{'circ':curves},direct).metrics
  m[str(direct)]={'point':float(v['avg_point_err']['calib']),'end':float(v['avg_end_err']['calib'])}
  loss+=m[str(direct)]['point']+0.5*m[str(direct)]['end']
 n+=1
 if loss<best:
  best=float(loss);best_x=np.array(x)
  with (folder/'shared_core_angle_best.pkl').open('wb') as f:pickle.dump({'parameters':s.parameters,'metrics':m,'loss':loss,'offsets':x},f)
 if n%20==0 or verbose:print('ANGLE',n,'loss',loss,'best',best,'metrics',m,flush=True)
 return loss
evaluate(np.zeros(8),True)
minimize(evaluate,np.zeros(8),method='L-BFGS-B',bounds=[(-np.pi/12,np.pi/12)]*8,options={'maxiter':80,'maxfun':1000,'ftol':1e-10})
evaluate(best_x,True)
print('ANGLE_COMPLETE',flush=True)
