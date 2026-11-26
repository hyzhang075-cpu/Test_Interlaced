"""All eighteen real measurements, explicit physical affine calibration, no imputation."""
import copy,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.curvature.core_regions import CoreDemoCurvature
from experimental_cube_core_adapter import ExperimentalCoreAdapter
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube';out=run/'all18_affine_core_candidate';out.mkdir(exist_ok=True)
prior=json.loads((run/'channel_bias_levels_all18_experimental/candidate.json').read_text());blocks={'full':copy.deepcopy(prior['parameters']['full'])}
local_prior=json.loads((run/'pointwise_core_fusion_joint_low_kend05/fusion_gain_refinement.json').read_text())
for r in ('end15cm','end5cm'):blocks[r]=copy.deepcopy(local_prior['parameters'][r])
for b in blocks.values():b['weights']={int(d):np.asarray(w) for d,w in b['weights'].items()}
blocks['full']['channel_weights']={int(d):np.asarray(w) for d,w in blocks['full']['channel_weights'].items()}
assert not blocks['full'].get('quality_imputation')
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');s.load_data()
layout=dict(blocks['full']['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
service=CurvatureService(replace(s.config.curvature,**layout));s.parameters=CalibrationParameters(blocks)
runtime=ExperimentalCoreAdapter(service,s.design,s.parameters,allow_imputation=False)
factors={d:np.ones(18) for d in (0,120,240)}
class Scaled:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        c=runtime.channel_compute(frame,design,parameters,direct=direct)
        return replace(c,curvature_per_m=c.curvature_per_m*factors[direct])
def metrics(direct):
    fusion=CoreDemoCurvature(service,s.design,s.parameters,('full','end15cm','end5cm'),direct)
    name,indices,design,_=fusion._regions[0];fusion._regions[0]=(name,indices,design,Scaled())
    class Fused:
        def compute(self,frame,design,parameters,*,direct=0,**kwargs):return fusion.compute(frame,parameters,direct=direct)
    result={}
    for name,adapter in [('single_full',Scaled()),('fusion',Fused())]:
        s.curvature=adapter;curves,distance=s.reconstruct_shapes('circ',direct);m=s.evaluate(distance,{'circ':curves},direct).metrics
        result[name]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    return result
all_metrics={}
for direct in (0,120,240):
    calls=0;cache={};selected=np.zeros(18);best_rank=(float('inf'),)*3;selected_metrics=None
    def objective(x):
        global calls,selected,best_rank,selected_metrics
        key=np.asarray(x).tobytes()
        if key in cache:return cache[key][0]
        factors[direct]=np.exp(x);m=metrics(direct);loss=sum(v['point']+.5*v['end'] for v in m.values())
        cache[key]=(loss,copy.deepcopy(m))
        if len(cache)>1024:del cache[next(iter(cache))]
        violation=np.maximum(np.asarray([v for r in m.values() for v in r.values()])-.00999,0)
        rank=(float(violation.max()),float(violation.sum()),loss)
        if rank<best_rank:best_rank=rank;selected=x.copy();selected_metrics=copy.deepcopy(m)
        calls+=1
        if calls%100==0:print('ALL18_GEOMETRY_POLISH',direct,calls,m,flush=True)
        return loss
    def gates(x):
        objective(x);m=cache[np.asarray(x).tobytes()][1]
        return np.asarray([.00999-v for r in m.values() for v in r.values()])
    objective(np.zeros(18))
    fit=minimize(objective,np.zeros(18),method='SLSQP',bounds=[(-.15,.15)]*18,constraints=[{'type':'ineq','fun':gates}],options={'maxiter':80,'ftol':1e-10})
    blocks['full']['channel_weights'][direct]*=np.exp(selected)[:,None,None];factors[direct]=np.ones(18)
    actual=metrics(direct)
    for mode in actual:
        for key in actual[mode]:assert abs(actual[mode][key]-selected_metrics[mode][key])<1e-8
    all_metrics[str(direct)]=actual;print('ALL18_FULL_VERIFIED',direct,actual,flush=True)
for region in ('end15cm','end5cm'):
    args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
    args.multi_region_calib=False;args.optimization_region=region;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
    for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
    configure_calibration_args(args);args.exp_directs=[0,120,240]
    local=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region=region)
    layout=dict(blocks[region]['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
    local.curvature=CurvatureService(replace(local.config.curvature,**layout));local.parameters=CalibrationParameters(blocks);local.load_data()
    for direct in (0,120,240):
        curves,distance=local.reconstruct_shapes('circ',direct);m=local.evaluate(distance,{'circ':curves},direct).metrics
        all_metrics[str(direct)][region]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        print('ALL18_LOCAL_VERIFIED',region,direct,all_metrics[str(direct)][region],flush=True)
passed=all(all(v['point']<(.01 if r in ('single_full','fusion') else .1) and v['end']<(.01 if r in ('single_full','fusion') else .1) for r,v in m.items()) for m in all_metrics.values())
report={'metrics':all_metrics,'parameters':blocks,'all_gates_passed':passed,'all_18_measurements_retained':True,'measurement_imputation':False,'raw_data_modified':False,'experimental_adapter':True,'interp_num':256,'curvature_interp':'vector_linear','k_point':1.,'k_end':.5,'relative_input_bias_reference_channel':4}
(out/'candidate.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist()),encoding='utf-8')
print('ALL18_ALL_GATES',passed,all_metrics,flush=True)
