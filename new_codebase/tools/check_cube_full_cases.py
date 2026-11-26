"""Read-only per-radius evaluation of the latest k_end=0.5 core candidate."""
import copy,json
from pathlib import Path
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
root=Path(__file__).resolve().parents[1]
folder=root/'runs/11B1_Cube/pointwise_core_single_kend05'
record=json.loads((folder/'full.json').read_text(encoding='utf-8'))
blocks=record['parameters']
for b in blocks.values():
    if 'weights' in b:b['weights']={int(d):np.asarray(w) for d,w in b['weights'].items()}
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
s.parameters=CalibrationParameters(blocks);s.load_data();rows={}
for direct in (0,120,240):
    curves,distance=s.reconstruct_shapes('circ',direct)
    m=s.evaluate(distance,{'circ':curves},direct).metrics
    rows[str(direct)]={k:m[k] for k in ('point_errors_avg','end_errors')}
    print('FULL_CASES',direct,rows[str(direct)],flush=True)
(folder/'per_radius_errors.json').write_text(json.dumps(rows,indent=2,default=lambda v:v.tolist()),encoding='utf-8')
