"""Sensitivity-assumption diagnostics with all original geometric constraints."""
import copy,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import os
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.initialization.service import InitializationService
root=Path(__file__).resolve().parents[1]
folder=root/'runs/11B1_Cube/segment_search'
record=json.loads((folder/'7seg_full.json').read_text(encoding='utf-8'))
args=parse_main_args(['--config','configs/11B1_Cube_core.yaml','--task','optim'])
args.multi_region_calib=False
args.full_segmentation=record['layout']
args.calibration_regions['full']['weight_group_ids']=np.repeat(np.arange(7),record['layout']).tolist()
block=copy.deepcopy(record['parameters']['full'])
block['weights']={int(k):np.asarray(v) for k,v in block['weights'].items()}
args.core_position_parameters={'full':block}
configure_calibration_args(args)
args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
dataset=session.load_data()
subtract_zero = os.environ.get('CUBE_SUBTRACT_ZERO') == '1'
if subtract_zero:
    signals=copy.deepcopy(dataset.data)
    for direct,states in signals['circ'].items():
        zero=np.array(states[0],copy=True)
        for radius in states:
            states[radius]=states[radius]-zero
    dataset=replace(dataset,data=signals)
    session.dataset=dataset
results=[]
for sensitivity in ((1.,) if subtract_zero else (.5,.75,1.,1.25,1.5)):
    print('SENSITIVITY_FIT_START',sensitivity,flush=True)
    config=replace(session.config.initialization,core_sensitivity_pm_per_ue=sensitivity)
    fitted=InitializationService(config,input_config=session.config.input,region='full').fit(dataset)
    candidate=copy.deepcopy(block)
    candidate['core_position']=fitted.core_position
    candidate['core_position_metadata']=fitted.core_position_metadata
    session.parameters=CalibrationParameters({'full':candidate},source='sensitivity_diagnostic')
    metrics={}
    for direct in (0,120,240):
        curves,distance=session.reconstruct_shapes('circ',direct)
        m=session.evaluate(distance,{'circ':curves},direct).metrics
        metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    row={'sensitivity':sensitivity,'metrics':metrics,'mean_signal_rmse_pm':float(np.mean([s['fit_rmse_pm'] for s in fitted.core_position]))}
    results.append(row)
    print('SENSITIVITY_RESULT',row,flush=True)
    def native(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [native(v) for v in x]
        return x
    prefix='zero_corrected_sensitivity' if subtract_zero else 'sensitivity'
    (folder/f'{prefix}_{sensitivity}.json').write_text(json.dumps(native({'result':row,'parameters':session.parameters.parameters_by_region}),indent=2),encoding='utf-8')
(folder/('zero_corrected_comparison.json' if subtract_zero else 'sensitivity_comparison.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
