"""Independent JSON replay and temporal-subset robustness for conditional candidate."""
import copy,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.service import CurvatureService
from experimental_cube_core_adapter import ExperimentalCoreAdapter
root=Path(__file__).resolve().parents[1];folder=root/'runs/11B1_Cube/conditional_quality_candidate'
record=json.loads((folder/'candidate.json').read_text(encoding='utf-8'));blocks=record['parameters']
for b in blocks.values():b['weights']={int(d):np.asarray(w) for d,w in b['weights'].items()}
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');dataset=s.load_data()
layout=dict(blocks['full']['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
service=CurvatureService(replace(s.config.curvature,**layout));s.parameters=CalibrationParameters(blocks)
try:
    ExperimentalCoreAdapter(service,s.design,s.parameters)
except ValueError:pass
else:raise AssertionError('Candidate must not silently enable WL imputation')
def evaluate(fusion):
    s.curvature=ExperimentalCoreAdapter(service,s.design,s.parameters,fusion=fusion,allow_imputation=True)
    result={}
    for direct in (0,120,240):
        curves,distance=s.reconstruct_shapes('circ',direct);m=s.evaluate(distance,{'circ':curves},direct).metrics
        result[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    return result
replayed={}
for name,fusion in [('single_full',False),('fusion',True)]:
    replayed[name]=evaluate(fusion)
    for d,m in replayed[name].items():
        for k,v in m.items():assert abs(v-record['metrics'][d][name][k])<1e-8
    print('INDEPENDENT_CANDIDATE_REPLAY',name,replayed[name],flush=True)
quarter_data={name:copy.deepcopy(dataset.data) for name in ('first_quarter','last_quarter')}
data_root=root/'1124-11B1+导管0状态＋定标测试/全段测试'
for direct in (0,120,240):
    for radius,expected in dataset.data['circ'][direct].items():
        full=[];first=[];last=[]
        for channel in (1,2,3,4):
            paths=sorted((data_root/f'{direct}-{radius}').glob(f'Channel_{channel}_*.csv'))
            frame=pd.read_csv(paths[0]);values=frame.iloc[:,2:20].apply(pd.to_numeric,errors='coerce').to_numpy()
            values=values[np.isfinite(values).all(axis=1)&(np.abs(values)<9000).all(axis=1)]
            count=max(1,len(values)//4)
            full.append(values.mean(axis=0));first.append(values[:count].mean(axis=0));last.append(values[-count:].mean(axis=0))
        assert np.allclose(np.asarray(full).T*1e-6,expected,rtol=0,atol=1e-12),'CSV preprocessing mismatch'
        quarter_data['first_quarter']['circ'][direct][radius]=np.asarray(first).T*1e-6
        quarter_data['last_quarter']['circ'][direct][radius]=np.asarray(last).T*1e-6
stability={}
for name,data in quarter_data.items():
    s.dataset=replace(dataset,data=data);stability[name]={}
    for mode,fusion in [('single_full',False),('fusion',True)]:
        stability[name][mode]=evaluate(fusion)
        print('TEMPORAL_SUBSET_ROBUSTNESS',name,mode,stability[name][mode],flush=True)
report={'independent_replay':replayed,'temporal_subset_checks':stability,'conditional_on_wl_imputation':True,'temporal_checks_are_not_independent_holdout_training':True,'default_imputation_opt_in_guard_verified':True}
(folder/'independent_runtime_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
