"""Reviewable conditional candidate; WL imputation requires an explicit user choice."""
import copy,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
import diagnose_cube_quality_imputation as lab
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.curvature.core_regions import CoreDemoCurvature
s=lab.s;blocks=lab.blocks;factors={d:np.ones(len(lab.good)) for d in (0,120,240)}
class Scaled:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        c=lab.Channels().compute(frame,design,parameters,direct=direct)
        return replace(c,curvature_per_m=c.curvature_per_m*factors[direct])
def evaluate_modes(direct):
    fusion=CoreDemoCurvature(lab.service,s.design,s.parameters,('full','end15cm','end5cm'),direct)
    name,indices,design,_=fusion._regions[0];fusion._regions[0]=(name,indices,design,Scaled())
    result={}
    for name,adapter in [('single_full',lab.Imputed(Scaled())),('fusion',lab.Imputed(fusion,True))]:
        s.curvature=adapter;curves,distance=s.reconstruct_shapes('circ',direct)
        m=s.evaluate(distance,{'circ':curves},direct).metrics
        result[name]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    return result
metrics={}
for direct in (0,120,240):
    count=0;best=float('inf');selected=np.zeros(len(lab.good));selected_metrics=None
    def objective(x):
        global count,best,selected,selected_metrics
        factors[direct]=np.exp(x);m=evaluate_modes(direct)
        value=sum(v['point']+.5*v['end'] for v in m.values());count+=1
        if all(v['point']<.01 and v['end']<.01 for v in m.values()) and value<best:
            best=value;selected=x.copy();selected_metrics=copy.deepcopy(m)
        if count%100==0:print('CONDITIONAL_GEOMETRY_POLISH',direct,count,m,flush=True)
        return value
    objective(np.zeros(len(lab.good)))
    assert selected_metrics is not None
    opt=minimize(objective,np.zeros(len(lab.good)),method='L-BFGS-B',bounds=[(-.08,.08) if g else (0.,0.) for g in lab.good],options={'maxiter':60,'ftol':1e-10,'maxfun':1500})
    lab.channels[direct]=lab.channels[direct]*np.exp(selected)[:,None,None]
    factors[direct]=np.ones(len(lab.good))
    actual=evaluate_modes(direct)
    for mode in actual:
        for key in actual[mode]:assert abs(actual[mode][key]-selected_metrics[mode][key])<1e-8
    metrics[str(direct)]=actual
    print('CONDITIONAL_FULL_VERIFIED',direct,actual,flush=True)
blocks['full']['channel_weights']=lab.channels
blocks['full']['quality_imputation']={'wl_ids':lab.bad,'rule':'calibration vector RMSE max over directions >0.3 per m','method':'linear vector interpolation, nearest endpoint extrapolation','real_points_directly_used':int(lab.good.sum())}
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
        metrics[str(direct)][region]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        print('CONDITIONAL_LOCAL_VERIFIED',region,direct,metrics[str(direct)][region],flush=True)
passed=all(all(v['point']<(.01 if r in ('single_full','fusion') else .1) and v['end']<(.01 if r in ('single_full','fusion') else .1) for r,v in m.items()) for m in metrics.values())
report={'metrics':metrics,'parameters':blocks,'thresholds_passed_under_imputation_assumption':passed,'conditional_on_measurement_imputation':True,'requires_user_choice':True,'not_full_18_point_direct_inversion':True,'interp_num':256,'curvature_interp':'vector_linear','k_point':1.,'k_end':.5,'experimental_adapter':True,'raw_data_modified':False,'limitations':['WL1 and WL6 replaced in reconstruction only','initial 0.314 m depends on nearest valid WL extrapolation','reported errors are on supplied calibration states; arbitrary-shape accuracy not established']}
folder=lab.run/'conditional_quality_candidate';folder.mkdir(exist_ok=True)
(folder/'candidate.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist()),encoding='utf-8')
print('CONDITIONAL_ALL_GATES',passed,metrics,flush=True)
