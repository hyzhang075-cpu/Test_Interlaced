"""Production pipeline replay, channel-order/compatibility assertions and time subsets."""
import copy,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.artifacts.packages import load_algorithm_package
from shape_sensing.curvature.core_regions import CoreDemoCurvature
root=Path(__file__).resolve().parents[1];folder=root/'runs/11B1_Cube/all18_affine_core_candidate'
path=folder/'algorithm_package_candidate.json';package=load_algorithm_package(path)
record=json.loads((folder/'candidate.json').read_text());blocks=package['core_position_parameters']
args=parse_main_args(['--sensor_params',str(path),'--task','recon'])
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='recon',region='full');dataset=s.load_data()
s.parameters=CalibrationParameters(blocks);service=s.curvature
assert s.config.reconstruction.interp_num==256 and s.config.optimization.k_point==1. and s.config.optimization.k_end==.5
assert all(not b.get('quality_imputation') for b in blocks.values())
frame=s.frame(dataset.data['circ'][0][300]);before=frame.signals.copy()
reference=service.compute(frame,s.design,s.parameters,direct=0)
order=[2,0,3,1]
permuted=replace(frame,signals=frame.signals[:,order],channel_ids=tuple(frame.channel_ids[i] for i in order))
other=service.compute(permuted,s.design,s.parameters,direct=0)
assert np.allclose(reference.curvature_per_m,other.curvature_per_m,rtol=0,atol=1e-10)
assert np.allclose(np.exp(1j*reference.direction_rad),np.exp(1j*other.direction_rad),rtol=0,atol=1e-10)
assert np.array_equal(frame.signals,before)
legacy=copy.deepcopy(blocks);b=legacy['full'];b.pop('channel_weights');b.pop('signal_bias_pm');b.pop('channel_grading_signal',None)
legacy_params=CalibrationParameters(legacy);old=service.compute(frame,s.design,legacy_params,direct=0)
mapping=b['weights'];w=np.asarray(mapping.get(0,mapping.get('0'))).reshape(18,-1)
equivalent=copy.deepcopy(legacy);equivalent['full']['channel_weights']={0:np.repeat(w[:,:,None],4,axis=2)}
equivalent['full']['signal_bias_pm']={0:np.zeros((18,4))}
new=service.compute(frame,s.design,CalibrationParameters(equivalent),direct=0)
assert np.allclose(old.curvature_per_m,new.curvature_per_m,rtol=0,atol=1e-10)
assert np.allclose(np.exp(1j*old.direction_rad),np.exp(1j*new.direction_rad),rtol=0,atol=1e-10)
for mode in ('zero_gain','nan_bias','bad_shape'):
    invalid=copy.deepcopy(blocks)
    if mode=='nan_bias':invalid['full']['signal_bias_pm']['0'][0][0]=float('nan')
    elif mode=='bad_shape':invalid['full']['channel_weights']['0']=invalid['full']['channel_weights']['0'][:-1]
    else:invalid['full']['channel_weights']['0'][0][0][0]=0.
    try:service.compute(frame,s.design,CalibrationParameters(invalid),direct=0)
    except ValueError:pass
    else:raise AssertionError(mode)
def evaluate(fused):
    result={}
    for direct in (0,120,240):
        if fused:
            fusion=CoreDemoCurvature(service,s.design,s.parameters,('full','end15cm','end5cm'),direct)
            class Fused:
                def compute(self,frame,design,parameters,*,direct=0,**kwargs):return fusion.compute(frame,parameters,direct=direct)
            s.curvature=Fused()
            diagnostic=fusion.compute(frame,s.parameters,direct=direct).diagnostics['calibrated_channel_gains']
            assert diagnostic.shape==(18,4) and np.isfinite(diagnostic[:7]).all() and np.isnan(diagnostic[7:]).all()
        else:s.curvature=service
        curves,distance=s.reconstruct_shapes('circ',direct);m=s.evaluate(distance,{'circ':curves},direct).metrics
        result[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    return result
actual={}
for name,fused in [('single_full',False),('fusion',True)]:
    actual[name]=evaluate(fused)
    for d,m in actual[name].items():
        for k,v in m.items():assert abs(v-record['metrics'][d][name][k])<1e-8
    print('PRODUCTION_REPLAY_VERIFIED',name,actual[name],flush=True)
quarter_data={name:copy.deepcopy(dataset.data) for name in ('first_quarter','last_quarter')}
data_root=root/'1124-11B1+导管0状态＋定标测试/全段测试'
for direct in (0,120,240):
    for radius,expected in dataset.data['circ'][direct].items():
        full=[];first=[];last=[]
        for channel in (1,2,3,4):
            paths=sorted((data_root/f'{direct}-{radius}').glob(f'Channel_{channel}_*.csv'))
            table=pd.read_csv(paths[0]);values=table.iloc[:,2:20].apply(pd.to_numeric,errors='coerce').to_numpy()
            values=values[np.isfinite(values).all(axis=1)&(np.abs(values)<9000).all(axis=1)]
            n=max(1,len(values)//4);full.append(values.mean(axis=0));first.append(values[:n].mean(axis=0));last.append(values[-n:].mean(axis=0))
        assert np.allclose(np.asarray(full).T*1e-6,expected,rtol=0,atol=1e-12)
        quarter_data['first_quarter']['circ'][direct][radius]=np.asarray(first).T*1e-6
        quarter_data['last_quarter']['circ'][direct][radius]=np.asarray(last).T*1e-6
stability={}
for name,data in quarter_data.items():
    s.dataset=replace(dataset,data=data);stability[name]={}
    for mode,fused in [('single_full',False),('fusion',True)]:
        stability[name][mode]=evaluate(fused)
        print('PRODUCTION_TEMPORAL_SUBSET',name,mode,stability[name][mode],flush=True)
for region in ('end15cm','end5cm'):
    args=parse_main_args(['--sensor_params',str(path),'--task','recon','--optimization_region',region])
    configure_calibration_args(args);args.exp_directs=[0,120,240]
    local=TaskSession(module_parameters(args,optimizer_parameters),task='recon',region=region);local.load_data();actual[region]={}
    for direct in (0,120,240):
        curves,distance=local.reconstruct_shapes('circ',direct);m=local.evaluate(distance,{'circ':curves},direct).metrics
        actual[region][str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        for k,v in actual[region][str(direct)].items():assert abs(v-record['metrics'][str(direct)][region][k])<1e-8
    print('PRODUCTION_LOCAL_VERIFIED',region,actual[region],flush=True)
passed=all(m['point']<(.01 if mode in ('single_full','fusion') else .1) and m['end']<(.01 if mode in ('single_full','fusion') else .1) for mode,rows in actual.items() for m in rows.values())
stable=all(m['point']<.01 and m['end']<.01 for subset in stability.values() for rows in subset.values() for m in rows.values())
report={'metrics':actual,'all_gates_passed':passed,'temporal_subsets':stability,'temporal_subset_full_gates_passed':stable,'all_18_measurements_retained':True,'measurement_imputation':False,'parameter_file':str(path),'legacy_scalar_equivalence_verified':True,'channel_order_invariance_verified':True,'invalid_parameter_rejection_verified':True,'raw_frame_immutability_verified':True}
(folder/'production_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PRODUCTION_ALL_GATES',passed,'TEMPORAL_SUBSETS',stable,flush=True)
