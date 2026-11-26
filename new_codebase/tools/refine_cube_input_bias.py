"""Experimental static input-offset calibration; all 18 measured WLs retained."""
import copy,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters,CurvatureResult
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.curvature.four_core import compute_four_core
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube';out=run/'experimental_input_bias240_kend05';out.mkdir(exist_ok=True)
record=json.loads((run/'physical_refinement_target240_round2_kend05/physical_coordinate_refinement.json').read_text())
blocks=copy.deepcopy(record['parameters']);block=blocks['full']
block['weights']={int(d):np.asarray(w) for d,w in block['weights'].items()};initial=block['weights'][240].copy()
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');s.load_data()
layout=dict(block['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
service=CurvatureService(replace(s.config.curvature,**layout));s.parameters=CalibrationParameters(blocks)
calibration={**block['core_position_metadata'],'sections':block['core_position']};meta=calibration['calibration'];p=len(block['core_position'])
positions=np.asarray([[[c['r_um']*np.sin(c['alpha_rad'] or 0),c['r_um']*np.cos(c['alpha_rad'] or 0)] for c in section['cores'].values()] for section in block['core_position']])/100.
inverse=np.linalg.pinv(np.concatenate((positions,np.ones((p,4,1))),axis=2))[:,:2,:]
frames=[];lookup={}
for radius,values in s.dataset.data['circ'][240].items():
    f=s.frame(values);lookup[f.signals.tobytes()]=len(frames);frames.append(f)
signals=np.asarray([f.signals for f in frames]);current=[];cache={};calls=0;selected=np.zeros(4*p);selected_rank=(float('inf'),)*3
def unpack(x):
    first=x[:3*p].reshape(p,3);bias=np.column_stack((first,-first.sum(axis=1)))*50.
    return bias,x[3*p:]
def build(x):
    global current
    bias,scalar=unpack(x);response=(signals-bias[None,:,:]*1e-6)/(meta['strain_sensitivity_pm_per_ue']*meta['strain_projection_sign'])
    vector=np.einsum('pij,spj->spi',inverse,response)/1e-4;magnitude=np.linalg.norm(vector,axis=2);angle=np.arctan2(vector[:,:,1],vector[:,:,0]);current=[]
    for i,f in enumerate(frames):
        gain=service._effective_core_weights(magnitude[i],initial)*np.exp(scalar)
        current.append(CurvatureResult(f.distance_m,magnitude[i]*gain,angle[i],f.source_indices))
class Cached:
    def compute(self,frame,design,parameters,**kwargs):return current[lookup[frame.signals.tobytes()]]
s.curvature=Cached()
def evaluate(x,verbose=False):
    global calls,selected,selected_rank
    key=np.asarray(x).tobytes()
    if not verbose and key in cache:return cache[key][0]
    build(x);curves,distance=s.reconstruct_shapes('circ',240);m=s.evaluate(distance,{'circ':curves},240).metrics
    point=float(m['avg_point_err']['calib']);end=float(m['avg_end_err']['calib']);loss=point+.5*end
    cache[key]=(loss,point,end)
    if len(cache)>1024:del cache[next(iter(cache))]
    bias,scalar=unpack(x);violation=np.maximum(np.array([point,end])-.00999,0)
    rank=(float(violation.max()),float(violation.sum()),loss)
    if np.abs(bias).max()<=50+1e-5 and rank<selected_rank:
        selected_rank=rank;selected=x.copy();np.savez(out/'active.npz',x=x,point=point,end=end)
    calls+=1
    if calls%100==0 or verbose:print('STATIC_INPUT_BIAS',calls,'point',point,'end',end,'max_bias_pm',float(np.abs(bias).max()),flush=True)
    return loss
def constraints(x):
    evaluate(x);loss,point,end=cache[np.asarray(x).tobytes()];bias,scalar=unpack(x)
    return np.r_[50-bias.reshape(-1),50+bias.reshape(-1),.00999-point,.00999-end]
x0=np.zeros(4*p);evaluate(x0,True)
opt=minimize(evaluate,x0,method='SLSQP',bounds=[(-1.,1.)]*(3*p)+[(-.8,.8)]*p,constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':100,'ftol':1e-9})
evaluate(selected,True);expected=cache[selected.tobytes()];bias,scalar=unpack(selected)
block['weights'][240]=(initial.reshape(p,-1)*np.exp(scalar[:,None])).reshape(-1);block['signal_bias_pm']={240:bias}
class Physical:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        corrected=replace(frame,signals=frame.signals-(bias*1e-6 if direct==240 else 0))
        return service.compute(corrected,design,parameters,direct=direct,region='full')
s.curvature=Physical();metrics={}
for direct in (0,120,240):
    curves,distance=s.reconstruct_shapes('circ',direct);m=s.evaluate(distance,{'circ':curves},direct).metrics
    metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
assert abs(metrics['240']['point']-expected[1])<1e-8 and abs(metrics['240']['end']-expected[2])<1e-8
report={'metrics':metrics,'parameters':blocks,'experimental_adapter':True,'all_18_measurements_retained':True,'signal_bias_bounds_pm':[-50,50],'k_point':1.,'k_end':.5,'interp_num':256,'source_zero_state_replaced':False}
(out/'candidate.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist()),encoding='utf-8')
print('STATIC_INPUT_BIAS_VERIFIED',metrics,flush=True)
