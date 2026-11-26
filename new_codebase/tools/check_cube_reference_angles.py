"""Diagnostic of one constant fixture reference angle per calibration direction."""
import copy,json,pickle,os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
root=Path(__file__).resolve().parents[1]
run=root/'runs/11B1_Cube'
seed=json.loads((run/'20261125-172521-655152-8faf677802-init/initialization.json').read_text(encoding='utf-8'))['result']['core_position_parameters']
physical_path=os.environ.get('CUBE_REFERENCE_CORE_SEED')
physical=json.loads(Path(physical_path).read_text(encoding='utf-8')) if physical_path else None
if physical:seed.update(copy.deepcopy(physical['parameters']))
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False
args.optimization_region='full'
args.core_position_parameters=copy.deepcopy(seed)
for block in args.core_position_parameters.values():
    block.pop('weights',None);block.pop('weight_layout',None)
configure_calibration_args(args)
args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
session.load_data()
output=[]
for direct in (0,120,240):
    best=float('inf'); selected=None; source=None
    if physical:
        best=-1.;source=physical_path
        selected=np.asarray(physical['parameters']['full']['weights'][str(direct)])
        warm=run/f'pointwise_core_physical_seed/active_full_{direct}.npz'
        if warm.exists():
            with np.load(warm) as state:selected=state['x'].copy()
            source=str(warm)
    for checkpoint in run.rglob(f'*seg_full_{direct}.pkl'):
        with checkpoint.open('rb') as stream: saved=pickle.load(stream)
        metrics=saved['metrics'][str(direct)]
        score=metrics['point']+metrics['end']
        if score<best:
            best=score;source=str(checkpoint)
            old=saved['parameters'].parameters_by_region['full']
            selected=np.asarray(old['weights'][direct]).reshape(-1,6)[old['weight_layout']['weight_group_ids']].reshape(-1)
    active=run/f'pointwise_core/active_full_{direct}.npz'
    if active.exists():
        with np.load(active) as candidate:
            if float(candidate['loss'])<best:
                best=float(candidate['loss']);selected=candidate['x'].copy();source=str(active)
    session.parameters.parameters_by_region['full']['weights'][direct]=selected
    curves,distance=session.reconstruct_shapes('circ',direct)
    def evaluate(angle):
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
        transformed={radius:points@rotation.T for radius,points in curves.items()}
        metrics=session.evaluate(distance,{'circ':transformed},direct).metrics
        p=float(metrics['avg_point_err']['calib']);e=float(metrics['avg_end_err']['calib'])
        return p+e,p,e
    before=evaluate(0.)
    result=minimize_scalar(lambda angle:evaluate(angle)[0],bounds=(-np.pi/12,np.pi/12),method='bounded',options={'xatol':1e-8})
    after=evaluate(float(result.x))
    row={'direct':direct,'source':source,'constant_angle_deg':float(np.degrees(result.x)),'before':{'point':before[1],'end':before[2]},'after':{'point':after[1],'end':after[2]},'diagnostic_only':True}
    output.append(row)
    print('REFERENCE_ANGLE',row,flush=True)
(run/('pointwise_core/reference_angle_physical_diagnostic.json' if physical else 'pointwise_core/reference_angle_diagnostic.json')).write_text(json.dumps(output,indent=2),encoding='utf-8')
