"""Fit the supplied central-plus-three physical topology, without twist slopes."""
import copy,json
from pathlib import Path
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
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
states=[(d,r) for d in (0,120,240) for r in sorted(dataset.data['circ'][d]) if r>0]
response=np.asarray([dataset.data['circ'][d][r] for d,r in states])*1e6
design=np.asarray([[np.cos(np.deg2rad(-d)),np.sin(np.deg2rad(-d))] for d,r in states])
design*=np.asarray([1000/r for d,r in states])[:,None]
common=response.mean(axis=2)
targets=np.linalg.lstsq(design,(response-common[:,:,None]).reshape(len(states),-1),rcond=None)[0].reshape(2,18,4).transpose(1,2,0)
for handedness in (-1,1):
    template=np.vstack((np.zeros((1,2)),np.column_stack((np.cos(handedness*np.arange(3)*2*np.pi/3),np.sin(handedness*np.arange(3)*2*np.pi/3)))))
    candidate=copy.deepcopy(block)
    rms=[]
    for index,(section,target) in enumerate(zip(candidate['core_position'],targets)):
        u,_,vt=np.linalg.svd(template.T@target)
        correction=np.diag([1.,np.linalg.det(u@vt)])
        rotation=u@correction@vt
        unit=template@rotation
        radius=float(np.clip(np.sum(unit*target)/3,100.,125.))
        points=radius*unit
        angles=(np.pi/2-np.arctan2(points[:,1],points[:,0]))%(2*np.pi)
        for j,(channel,core) in enumerate(section['cores'].items()):
            core['r_um']=0. if j==0 else radius
            core['alpha_rad']=None if j==0 else float(angles[j])
            core['alpha_deg']=None if j==0 else float(np.degrees(angles[j]))
        residual=design@points.T+common[:,index,None]-response[:,index]
        section['fit_rmse_pm']=float(np.sqrt(np.mean(residual**2)))
        section['constraint_solver']='nominal_central_plus_three_procrustes'
        section['minimum_enclosing_center_uv_um']=[0.,0.]
        section['envelope_center_uv_um']=[0.,0.]
        section['minimum_enclosing_radius_um']=radius
        section['envelope_containing_radius_um']=radius
        section['minimum_pairwise_distance_um']=radius
        section['centroid_error_um']=0.
        rms.append(section['fit_rmse_pm'])
    session.parameters=CalibrationParameters({'full':candidate},source='nominal_topology_core_fit')
    metrics={}
    for direct in (0,120,240):
        curves,distance=session.reconstruct_shapes('circ',direct)
        m=session.evaluate(distance,{'circ':curves},direct).metrics
        metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    print('NOMINAL_CORE',handedness,'signal_rmse',np.mean(rms),'metrics',metrics,flush=True)
    def native(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [native(v) for v in x]
        return x
    (folder/f'nominal_core_{handedness}.json').write_text(json.dumps(native({'metrics':metrics,'signal_rmse':float(np.mean(rms)),'parameters':session.parameters.parameters_by_region}),indent=2),encoding='utf-8')
