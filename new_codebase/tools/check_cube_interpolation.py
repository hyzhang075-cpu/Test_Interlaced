"""Compare production interpolation settings on identical four-core parameters."""
import copy,json,pickle
from dataclasses import replace
from pathlib import Path
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.reconstruction.service import ReconstructionService
root=Path(__file__).resolve().parents[1]
folder=root/'runs/11B1_Cube/segment_search'
data=json.loads((folder/'5seg_full.json').read_text(encoding='utf-8'))
args=parse_main_args(['--config','configs/11B1_Cube_core.yaml','--task','optim'])
args.multi_region_calib=False
args.full_segmentation=data['layout']
args.calibration_regions['full']['weight_group_ids']=np.repeat(np.arange(len(data['layout'])),data['layout']).tolist()
block=copy.deepcopy(data['parameters']['full'])
block['weights']={int(k):np.asarray(v) for k,v in block['weights'].items()}
args.core_position_parameters={'full':block}
configure_calibration_args(args)
args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
session.load_data()
results=[]
import os
if os.environ.get('CUBE_IDEAL_ONLY') == '1':
    for direct in (0,120,240):
        curves,distance=session.reconstruct_shapes('circ',direct,force_ideal_magnitude=True)
        m=session.evaluate(distance,{'circ':curves},direct).metrics
        row={'direct':direct,'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        results.append(row)
        print('IDEAL_MAGNITUDE_DIAGNOSTIC',row,flush=True)
    (folder/'ideal_magnitude_diagnostic.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    raise SystemExit
for method in ('vector_pchip','vector_linear','hermite','scalar_pchip'):
    for filtered in (True,False):
        config=replace(session.config.reconstruction,curvature_interp=method,move_filter_for_vector=filtered,move_filter=3 if filtered else 1)
        session.reconstruction=ReconstructionService(config)
        for direct in (0,120,240):
            curves,distance=session.reconstruct_shapes('circ',direct)
            metrics=session.evaluate(distance,{'circ':curves},direct).metrics
            row={'method':method,'filtered':filtered,'direct':direct,'point':float(metrics['avg_point_err']['calib']),'end':float(metrics['avg_end_err']['calib'])}
            results.append(row)
            print('INTERPOLATION',row,flush=True)
(folder/'interpolation_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
