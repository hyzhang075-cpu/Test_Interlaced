import sys,os,pickle,copy,json
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')];os.chdir(root/'new_codebase')
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
out=Path('runs/11B8_core_point_refine_k1_end05')
for region,count in [('end15cm',21),('end5cm',7)]:
 with (out/f'{region}_consistent_core.pkl').open('rb') as f:rec=pickle.load(f)
 b=copy.deepcopy(rec['parameters'].parameters_by_region[region])
 a=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
 a.optimization_region=region;a.multi_region_calib=False;a.end15cm_point_calib=False;a.end5cm_point_calib=False
 a.calibration_regions[region]['weight_group_ids']=list(range(count));a.core_position_parameters={region:b}
 configure_calibration_args(a);a.exp_directs=[0,120,240]
 with (out/f'{region}_dataset_cache.pkl').open('rb') as f:dataset=pickle.load(f)
 s=TaskSession(module_parameters(a,optimizer_parameters),task='optim',region=region,dataset=dataset)
 shape=(count,len(b['weight_layout']['thresholds'])+1)
 initial=np.asarray(b['weights'][240]).reshape(shape).copy()
 mask=np.zeros(shape,dtype=bool);mask[:,6:]=True
 calls=0;best=float('inf');best_x=initial[mask].copy()
 def evaluate(x,verbose=False):
  global calls,best,best_x
  candidate=copy.deepcopy(b);w=initial.copy();w[mask]=x;candidate['weights'][240]=w.reshape(-1)
  s.parameters=CalibrationParameters({region:candidate})
  curves,dist=s.reconstruct_shapes('circ',240);m=s.evaluate(dist,{'circ':curves},240).metrics
  point=float(m['avg_point_err']['calib']);end=float(m['avg_end_err']['calib']);loss=point+.5*end
  calls+=1
  if loss<best:
   best=loss;best_x=x.copy()
   with (out/f'{region}_highcurve_best.pkl').open('wb') as f:pickle.dump({'parameters':s.parameters,'point':point,'end':end,'loss':loss},f)
  if calls%50==0 or verbose:print('HIGH',region,calls,'point',point,'end',end,'best',best,flush=True)
  return loss
 evaluate(best_x,True)
 minimize(evaluate,best_x,method='L-BFGS-B',bounds=[(.1,10.)]*len(best_x),options={'maxiter':100,'maxfun':5000,'ftol':1e-11})
 evaluate(best_x,True)
 print('HIGH_COMPLETE',region,flush=True)
