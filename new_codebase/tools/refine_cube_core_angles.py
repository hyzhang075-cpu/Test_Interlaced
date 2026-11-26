"""Refine shared sectional core orientation against all three directions."""
import copy,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
root=Path(__file__).resolve().parents[1]
folder=root/'runs/11B1_Cube/segment_search'
record=json.loads((folder/'6seg_full.json').read_text(encoding='utf-8'))
layout=record['layout']
args=parse_main_args(['--config','configs/11B1_Cube_core.yaml','--task','optim'])
args.multi_region_calib=False
args.full_segmentation=layout
groups=np.repeat(np.arange(len(layout)),layout)
args.calibration_regions['full']['weight_group_ids']=groups.tolist()
block=copy.deepcopy(record['parameters']['full'])
block['weights']={int(k):np.asarray(v) for k,v in block['weights'].items()}
args.core_position_parameters={'full':block}
configure_calibration_args(args)
args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
session.load_data()
base=copy.deepcopy(session.parameters.parameters_by_region)
calls=0
best=float('inf')
best_x=None
def evaluate(x,verbose=False):
    global calls,best,best_x
    blocks=copy.deepcopy(base)
    for section,group in zip(blocks['full']['core_position'],groups):
        delta=float(x[group])
        for core in section['cores'].values():
            if core['alpha_rad'] is not None:
                core['alpha_rad']+=delta
                core['alpha_deg']=float(np.degrees(core['alpha_rad']))
        for key in ('minimum_enclosing_center_uv_um','envelope_center_uv_um'):
            if key in section:
                u,v=section[key]
                section[key]=[u*np.cos(delta)-v*np.sin(delta),u*np.sin(delta)+v*np.cos(delta)]
    session.parameters=CalibrationParameters(blocks,source='shared_core_orientation_refinement')
    metrics={}
    loss=0.
    for direct in (0,120,240):
        curves,distance=session.reconstruct_shapes('circ',direct)
        m=session.evaluate(distance,{'circ':curves},direct).metrics
        metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        loss+=metrics[str(direct)]['point']+metrics[str(direct)]['end']
    calls+=1
    if loss<best:
        best=loss
        best_x=np.array(x)
        np.savez(folder/'core_angle_refinement_active.npz',x=x,loss=loss)
    if calls%20==0 or verbose:
        print('CORE_ANGLE',calls,'loss',loss,'metrics',metrics,flush=True)
    return loss
evaluate(np.zeros(len(layout)),True)
result=minimize(evaluate,np.zeros(len(layout)),method='L-BFGS-B',bounds=[(-np.pi/4,np.pi/4)]*len(layout),options={'maxiter':100,'maxfun':1200,'ftol':1e-9})
evaluate(best_x,True)
def native(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [native(v) for v in x]
    return x
(folder/'core_angle_refinement.json').write_text(json.dumps(native({'layout':layout,'offsets_rad':best_x,'loss':best,'parameters':session.parameters.parameters_by_region}),indent=2),encoding='utf-8')
