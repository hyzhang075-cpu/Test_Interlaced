import os,sys,pickle,copy
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
from shape_sensing.initialization.four_core import enclosing_circle
out=Path('runs/11B8_core_point_refine_k1_end05')
def load(name):
 with (out/name).open('rb') as f:return pickle.load(f)
full=load('shared_core_point_constrained_final.pkl')['parameters'].parameters_by_region['full']
mid=load('end15cm_highcurve_best.pkl')['parameters'].parameters_by_region['end15cm']
base=load('end5cm_highcurve_best.pkl')['parameters'].parameters_by_region['end5cm']
def session(region,b,count):
 a=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
 a.optimization_region=region;a.multi_region_calib=False;a.end15cm_point_calib=False;a.end5cm_point_calib=False;a.calibration_regions[region]['weight_group_ids']=list(range(count));a.core_position_parameters={region:b}
 configure_calibration_args(a);a.exp_directs=[0,120,240]
 return TaskSession(module_parameters(a,optimizer_parameters),task='optim',region=region,dataset=load(f'{region}_dataset_cache.pkl'))
s=session('end5cm',base,7);f=session('full',full,40)
calls=0;best_violation=float('inf');cache={}
def evaluate(x):
 global calls,best_violation
 key=x.tobytes()
 if key in cache:return cache[key]
 b=copy.deepcopy(base);geometric=[]
 for i,section in enumerate(b['core_position']):
  angle,a,d,shear=x.reshape(7,4)[i]
  transform=np.array([[np.exp(a),shear],[0.,np.exp(d)]])
  rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  points=[]
  for core in section['cores'].values():
   alpha=core['alpha_rad'] or 0.;p=np.array([core['r_um']*np.sin(alpha),core['r_um']*np.cos(alpha)])
   points.append(rotation@transform@p)
  points=np.array(points);spacing=np.linalg.norm(points[:,None]-points[None,:],axis=2)[np.triu_indices(4,1)].min()
  scale=100.000001/spacing;points*=scale
  center,radius=enclosing_circle(points);geometric.append(1.-radius/125.)
  for core,p in zip(section['cores'].values(),points):core['r_um']=float(np.linalg.norm(p));core['alpha_rad']=float(np.arctan2(p[0],p[1]));core['alpha_deg']=float(np.degrees(core['alpha_rad']))
  for direct in (0,120,240):
   w=np.asarray(b['weights'][direct]).reshape(7,-1).copy();w[i]*=scale;b['weights'][direct]=w.reshape(-1)
 metrics={};loss=0.;constraints=list(geometric)
 s.parameters=CalibrationParameters({'end5cm':b})
 for direct in (0,120,240):
  curves,dist=s.reconstruct_shapes('circ',direct);m=s.evaluate(dist,{'circ':curves},direct).metrics
  point=float(m['avg_point_err']['calib']);end=float(m['avg_end_err']['calib']);metrics[str(direct)]={'point':point,'end':end}
  loss+=point+.5*end;constraints.extend([1-point/.099,1-end/.099])
 blocks={'full':full,'end15cm':mid,'end5cm':b};params=CalibrationParameters(blocks)
 for direct in (0,120,240):
  assembler=CoreDemoCurvature(f.curvature,f.design,params,('full','end15cm','end5cm'),direct);curves={}
  core=full['core_position_metadata']['calibration']
  for radius,signals in f.dataset.data['circ'][direct].items():
   p=assembler.compute(f.frame(signals),params,direct=direct)
   angle=p.direction_rad-np.deg2rad(core['core2_azimuth_deg']+core['bend_side_offset_deg']+core['direction_sign']*direct)
   p=replace(p,direction_rad=np.arctan2(np.sin(angle),np.cos(angle)))
   prepared=f.reconstruction.prepare(p,f.design,np.array([1.,0.,0.]));shape=f.reconstruction.integrate(prepared);curves[radius]=shape.position_m;f.cur_interp=prepared.curvature_per_m
  m=f.evaluate(shape.distance_m,{'circ':curves},direct).metrics
  point=float(m['avg_point_err']['calib']);end=float(m['avg_end_err']['calib']);metrics['stitched_'+str(direct)]={'point':point,'end':end}
  loss+=point+.5*end;constraints.extend([1-point/.00995,1-end/.00995])
 calls+=1;c=np.array(constraints);violation=max(0.,float(-c.min()))
 if violation<best_violation:
  best_violation=violation
  with (out/'end5_joint_best.pkl').open('wb') as stream:pickle.dump({'parameters':params,'metrics':metrics,'x':x,'violation':violation,'loss':loss},stream)
 if calls%20==0:print('JOINT',calls,'loss',loss,'violation',violation,'metrics',metrics,flush=True)
 cache.clear();cache[key]=(loss,c)
 return loss,c
x0=np.zeros(28)
result=minimize(lambda x:evaluate(x)[0],x0,method='SLSQP',constraints=[{'type':'ineq','fun':lambda x:evaluate(x)[1]}],bounds=[(-np.pi/4,np.pi/4),(-.3,.3),(-.3,.3),(-.3,.3)]*7,options={'maxiter':100,'ftol':1e-9})
print('JOINT_COMPLETE',result.success,result.message,flush=True)
