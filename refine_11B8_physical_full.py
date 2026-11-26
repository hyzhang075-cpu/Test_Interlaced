import sys,os,pickle,copy
from pathlib import Path
from dataclasses import replace
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')];os.chdir(root/'new_codebase')
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.core_regions import CoreDemoCurvature
out=Path('runs/11B8_core_point_refine_k1_end05')
with (out/'final_candidate.pkl').open('rb') as f:blocks=pickle.load(f)['blocks']
a=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
a.multi_region_calib=False;a.optimization_region='full';a.calibration_regions['full']['weight_group_ids']=list(range(40));a.core_position_parameters=copy.deepcopy(blocks)
configure_calibration_args(a);a.exp_directs=[0,120,240]
with (out/'full_dataset_cache.pkl').open('rb') as f:dataset=pickle.load(f)
s=TaskSession(module_parameters(a,optimizer_parameters),task='optim',region='full',dataset=dataset)
for direct in (0,120,240):
 initial=np.asarray(blocks['full']['weights'][direct]).reshape(40,6).copy();calls=0;best_v=float('inf');best_loss=float('inf');best_record=None;cache={}
 def evaluate(x):
  global calls,best_v,best_loss,best_record
  key=x.tobytes()
  if key in cache:return cache[key]
  b=copy.deepcopy(blocks)
  scale=np.exp(np.interp(s.design.sampling_distance_m,np.linspace(s.design.sampling_distance_m[0],s.design.sampling_distance_m[-1],8),x))
  b['full']['weights'][direct]=(initial*scale[:,None]).reshape(-1)
  params=CalibrationParameters(b);s.parameters=params
  curves,dist=s.reconstruct_shapes('circ',direct);m=s.evaluate(dist,{'circ':curves},direct).metrics
  metrics={'full':{'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}}
  assembler=CoreDemoCurvature(s.curvature,s.design,params,('full','end15cm','end5cm'),direct);curves={}
  core=b['full']['core_position_metadata']['calibration']
  for radius,signals in s.dataset.data['circ'][direct].items():
   p=assembler.compute(s.frame(signals),params,direct=direct);angle=p.direction_rad-np.deg2rad(core['core2_azimuth_deg']+core['bend_side_offset_deg']+core['direction_sign']*direct)
   p=replace(p,direction_rad=np.arctan2(np.sin(angle),np.cos(angle)));prepared=s.reconstruction.prepare(p,s.design,np.array([1.,0.,0.]));shape=s.reconstruction.integrate(prepared);curves[radius]=shape.position_m;s.cur_interp=prepared.curvature_per_m
  m=s.evaluate(shape.distance_m,{'circ':curves},direct).metrics;metrics['stitched']={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
  loss=sum(v['point']+.5*v['end'] for v in metrics.values());c=np.array([1-v[k]/.00995 for v in metrics.values() for k in ('point','end')]);violation=max(0.,float(-c.min()));calls+=1
  if violation<best_v or (violation==best_v and loss<best_loss):
   best_v=violation;best_loss=loss;best_record=copy.deepcopy(b)
   with (out/f'physical_full_{direct}_best.pkl').open('wb') as f:pickle.dump({'blocks':b,'metrics':metrics,'violation':violation},f)
  if calls%20==0:print('PHYSICAL_FULL',direct,calls,metrics,'violation',violation,flush=True)
  cache.clear();cache[key]=(loss,c)
  return loss,c
 result=minimize(lambda x:evaluate(x)[0],np.zeros(8),method='SLSQP',constraints=[{'type':'ineq','fun':lambda x:evaluate(x)[1]}],bounds=[(-.5,.5)]*8,options={'maxiter':100,'ftol':1e-10})
 blocks=best_record
 print('DIRECTION_COMPLETE',direct,'violation',best_v,flush=True)
with (out/'physical_final_candidate.pkl').open('wb') as f:pickle.dump({'blocks':blocks},f)
print('PHYSICAL_COMPLETE',flush=True)
