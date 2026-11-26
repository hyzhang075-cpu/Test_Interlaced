"""Experimental static channel sensitivity calibration in the four-core inverse."""
import copy,json,os
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
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube'
out=root/os.environ.get('CUBE_CHANNEL_OUTPUT','runs/11B1_Cube/static_channel_sensitivity_kend05');out.mkdir(exist_ok=True)
source=root/os.environ.get('CUBE_CHANNEL_SOURCE','runs/11B1_Cube/physical_refinement_target240_round2_kend05/physical_coordinate_refinement.json')
record=json.loads(source.read_text(encoding='utf-8'));blocks=copy.deepcopy(record['parameters'])
for b in blocks.values():
    if 'weights' in b:b['weights']={int(d):np.asarray(w) for d,w in b['weights'].items()}
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');dataset=s.load_data()
layout=dict(blocks['full']['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
s.config=replace(s.config,curvature=replace(s.config.curvature,**layout));original=CurvatureService(s.config.curvature)
s.parameters=CalibrationParameters(blocks);s.curvature=original
base=copy.deepcopy(blocks);p=len(blocks['full']['core_position']);states=[];frames=[];lookup={}
for d in (0,120,240):
    for radius,values in dataset.data['circ'][d].items():
        f=s.frame(values);lookup[(d,f.signals.tobytes())]=len(frames);frames.append(f);states.append((d,radius))
block=blocks['full'];meta=block['core_position_metadata']['calibration']
positions=np.asarray([[[c['r_um']*np.sin(c['alpha_rad'] or 0),c['r_um']*np.cos(c['alpha_rad'] or 0)] for c in section['cores'].values()] for section in block['core_position']])/100.
response=np.asarray([f.signals for f in frames])/(meta['strain_sensitivity_pm_per_ue']*meta['strain_projection_sign'])
current=[]
def unpack(x):
    first=x[:3*p].reshape(p,3)
    logs=np.column_stack((first,-first.sum(axis=1)))
    return np.exp(logs),x[3*p:].reshape(3,p),logs
def build(x):
    global current
    relative,scalar,logs=unpack(x)
    matrix=np.concatenate((positions,relative[:,:,None]),axis=2)
    inverse=np.linalg.pinv(matrix)
    vector=np.einsum('pij,spj->spi',inverse[:,:2,:],response*relative[None,:,:])/1e-4
    magnitude=np.linalg.norm(vector,axis=2);angle=np.arctan2(vector[:,:,1],vector[:,:,0]);current=[]
    for i,(d,r) in enumerate(states):
        gain=original._effective_core_weights(magnitude[i],base['full']['weights'][d])*np.exp(scalar[(0,120,240).index(d)])
        current.append(CurvatureResult(frames[i].distance_m,magnitude[i]*gain,angle[i],frames[i].source_indices))
class Cached:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):return current[lookup[(direct,frame.signals.tobytes())]]
def metrics():
    result={}
    for d in (0,120,240):
        curves,distance=s.reconstruct_shapes('circ',d);m=s.evaluate(distance,{'circ':curves},d).metrics
        result[str(d)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    return result
expected=metrics();s.curvature=Cached();x0=np.zeros(6*p);build(x0)
for d,m in metrics().items():
    for k,v in m.items():assert abs(v-expected[d][k])<1e-8
calls=0;best=float('inf');best_x=x0.copy();cache={};limit=np.log(2.)
def evaluate(x,verbose=False):
    global calls,best,best_x
    key=np.asarray(x).tobytes()
    if not verbose and key in cache:return cache[key][0]
    build(x);m=metrics();loss=m['240']['point']+.5*m['240']['end'];cache[key]=(loss,m)
    if len(cache)>1024:del cache[next(iter(cache))]
    relative,scalar,logs=unpack(x)
    passed=all(m[str(d)][k]<.01 for d in (0,120) for k in ('point','end'))
    if passed and np.abs(logs).max()<=limit+1e-7 and loss<best:
        best=loss;best_x=x.copy();np.savez(out/'active.npz',x=x,loss=loss)
    calls+=1
    if calls%100==0 or verbose:print('CHANNEL_SENSITIVITY',calls,'loss',loss,'selected',best,'metrics',m,flush=True)
    return loss
def constraints(x):
    evaluate(x);m=cache[np.asarray(x).tobytes()][1];relative,scalar,logs=unpack(x)
    return np.r_[limit-logs.reshape(-1),limit+logs.reshape(-1),[.00999-m[str(d)][k] for d in (0,120) for k in ('point','end')]]
evaluate(x0,True)
opt=minimize(evaluate,x0,method='SLSQP',bounds=[(-limit,limit)]*(3*p)+[(-.8,.8)]*(3*p),constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':100,'ftol':1e-8})
evaluate(best_x,True);expected=copy.deepcopy(cache[best_x.tobytes()][1]);relative,scalar,logs=unpack(best_x)
for i,d in enumerate((0,120,240)):blocks['full']['weights'][d]=(base['full']['weights'][d].reshape(p,-1)*np.exp(scalar[i,:,None])).reshape(-1)
blocks['full']['core_channel_gains']=relative
calibration={**block['core_position_metadata'],'sections':block['core_position']}
class PhysicalSensitivity:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        raw=compute_four_core(frame,design,calibration,relative,1.)
        gain=original._effective_core_weights(raw.curvature_per_m,blocks['full']['weights'][direct])
        return compute_four_core(frame,design,calibration,relative*gain[:,None],1.)
s.curvature=PhysicalSensitivity();actual=metrics()
for d,m in actual.items():
    for k,v in m.items():assert abs(v-expected[d][k])<1e-8
report={'metrics':actual,'parameters':blocks,'k_point':1.,'k_end':.5,'interp_num':256,'source':str(source),'experimental_adapter':True,'relative_channel_gain_bounds':[.5,2.],'optimizer_success':bool(opt.success),'spatial_partition_search':False}
(out/'static_sensitivity_candidate.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist()),encoding='utf-8')
print('CHANNEL_SENSITIVITY_VERIFIED',actual,flush=True)
