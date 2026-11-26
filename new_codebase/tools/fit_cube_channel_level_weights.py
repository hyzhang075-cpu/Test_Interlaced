"""Experimental positive channel/curvature-level weights inside four-core inversion."""
import copy,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.curvature.four_core import compute_four_core
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube'
folder=run/'channel_level_weights_experimental';folder.mkdir(exist_ok=True)
record=json.loads((run/'physical_refinement_target240_round2_kend05/physical_coordinate_refinement.json').read_text(encoding='utf-8'))
blocks=copy.deepcopy(record['parameters']);block=blocks['full']
block['weights']={int(d):np.asarray(w) for d,w in block['weights'].items()}
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');s.load_data()
layout=dict(block['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
config=replace(s.config.curvature,**layout);service=CurvatureService(config)
s.parameters=CalibrationParameters(blocks);s.curvature=service
calibration={**block['core_position_metadata'],'sections':block['core_position']};meta=calibration['calibration']
p=len(block['core_position']);levels=len(config.thresholds)+1
positions=np.asarray([[[c['r_um']*np.sin(c['alpha_rad'] or 0),c['r_um']*np.cos(c['alpha_rad'] or 0)] for c in section['cores'].values()] for section in block['core_position']])/100.
fitted={};fit_metrics={}
for direct in (0,120,240):
    signals=[];masses=[];targets=[]
    offset=meta['core2_azimuth_deg']+meta['bend_side_offset_deg']+meta['direction_sign']*direct
    unit=np.array([np.cos(np.deg2rad(offset)),np.sin(np.deg2rad(offset))])
    for radius,values in s.dataset.data['circ'][direct].items():
        frame=s.frame(values);raw=compute_four_core(frame,s.design,calibration,None,1.)
        signals.append(frame.signals/(meta['strain_sensitivity_pm_per_ue']*meta['strain_projection_sign']))
        masses.append(service.allocate(service.scores(raw.curvature_per_m)))
        targets.append(np.zeros(2) if radius==0 else unit*1000/radius)
    signals=np.asarray(signals);masses=np.asarray(masses);target=np.asarray(targets)
    initial=block['weights'][direct].reshape(p,levels);weights=[];errors=[]
    scale=max(float(np.sqrt(np.mean(target**2))),1e-6)
    for point in range(p):
        basis=masses[:,:,point];response=signals[:,point,:];pos=positions[point]
        start=np.log(np.maximum(initial[point],.004))[:,None]*np.ones((1,4))
        def prediction(x):
            gains=basis@np.exp(x.reshape(levels,4))
            matrix=np.concatenate((np.broadcast_to(pos,(len(response),4,2)),gains[:,:,None]),axis=2)
            inv=np.linalg.pinv(matrix)
            return np.einsum('sij,sj->si',inv[:,:2,:],gains*response)/1e-4
        def residual(x):
            difference=(prediction(x)-target)/scale
            regularization=1e-5*(x-start.reshape(-1))
            return np.r_[difference.reshape(-1),regularization]
        result=least_squares(residual,start.reshape(-1),bounds=(np.log(.004),np.log(30.)),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        weights.append(np.exp(result.x.reshape(levels,4)))
        errors.append(float(np.sqrt(np.mean((prediction(result.x)-target)**2))))
        print('CHANNEL_LEVEL_FIT',direct,'WL',point+1,'vector_rmse',errors[-1],flush=True)
    fitted[direct]=np.asarray(weights);fit_metrics[str(direct)]=errors
    np.savez(folder/f'channel_weights_{direct}.npz',weights=fitted[direct])
block['channel_weights']=fitted
class ChannelLevelInverse:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        # No radius, ideal shape or ideal curvature is available in this forward path.
        raw=compute_four_core(frame,design,calibration,None,1.)
        mass=service.allocate(service.scores(raw.curvature_per_m))
        gains=np.einsum('lp,plc->pc',mass,fitted[direct])
        return compute_four_core(frame,design,calibration,gains,1.)
s.curvature=ChannelLevelInverse();metrics={}
for direct in (0,120,240):
    curves,distance=s.reconstruct_shapes('circ',direct);m=s.evaluate(distance,{'circ':curves},direct).metrics
    metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    print('CHANNEL_LEVEL_FORWARD_VERIFIED',direct,metrics[str(direct)],flush=True)
report={'metrics':metrics,'parameters':blocks,'vector_fit_rmse':fit_metrics,'experimental_adapter':True,'gain_bounds':[.004,30.],'interp_num':256,'curvature_interp':'vector_linear','k_point':1.,'k_end':.5,'fit_objective':'physical curvature-vector least squares; geometry metrics reported separately','spatial_partition_search':False}
(folder/'channel_level_candidate.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist()),encoding='utf-8')
