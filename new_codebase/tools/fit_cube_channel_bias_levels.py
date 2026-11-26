"""All-18-WL experimental affine channel calibration inside physical inversion."""
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
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube';folder=run/'channel_bias_levels_all18_experimental';folder.mkdir(exist_ok=True)
prior=json.loads((run/'channel_level_weights_experimental/channel_level_candidate.json').read_text())
blocks=copy.deepcopy(prior['parameters']);block=blocks['full'];block.pop('quality_imputation',None)
block['weights']={int(d):np.asarray(w) for d,w in block['weights'].items()}
old_channels={int(d):np.asarray(w) for d,w in block['channel_weights'].items()}
bias_seed=json.loads((run/'experimental_input_bias240_kend05/candidate.json').read_text())['parameters']['full']['signal_bias_pm']['240']
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');s.load_data()
layout=dict(block['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
service=CurvatureService(replace(s.config.curvature,**layout));s.parameters=CalibrationParameters(blocks)
calibration={**block['core_position_metadata'],'sections':block['core_position']};meta=calibration['calibration']
p=len(block['core_position']);levels=len(service.config.thresholds)+1
positions=np.asarray([[[c['r_um']*np.sin(c['alpha_rad'] or 0),c['r_um']*np.cos(c['alpha_rad'] or 0)] for c in section['cores'].values()] for section in block['core_position']])/100.
channel_weights={};biases={};fit_errors={}
for direct in (0,120,240):
    response=[];mass=[];targets=[]
    angle=np.deg2rad(meta['core2_azimuth_deg']+meta['bend_side_offset_deg']+meta['direction_sign']*direct)
    for radius,values in s.dataset.data['circ'][direct].items():
        f=s.frame(values);raw=compute_four_core(f,s.design,calibration,None,1.)
        response.append(f.signals/(meta['strain_sensitivity_pm_per_ue']*meta['strain_projection_sign']))
        mass.append(service.allocate(service.scores(raw.curvature_per_m)))
        targets.append(np.zeros(2) if radius==0 else np.array([np.cos(angle),np.sin(angle)])*1000/radius)
    response=np.asarray(response);mass=np.asarray(mass);target=np.asarray(targets);scale=max(float(np.sqrt(np.mean(target**2))),1e-6)
    fitted=[];offsets=[];errors=[]
    for point in range(p):
        start_log=np.log(np.clip(old_channels[direct][point],.004,30.)).reshape(-1)
        start_bias=np.zeros(3)
        if direct==240 and point==0:
            b=np.asarray(bias_seed[point]);start_bias=np.clip((b[:3]-b[3])/100.,-1.,1.)
        start=np.r_[start_bias,start_log];basis=mass[:,:,point];raw_response=response[:,point,:];pos=positions[point]
        def unpack(x):return np.r_[x[:3]*100.,0.],np.exp(x[3:].reshape(levels,4))
        def predict(x):
            bias,weights=unpack(x);gains=basis@weights
            corrected=raw_response-bias[None,:]*1e-6/(meta['strain_sensitivity_pm_per_ue']*meta['strain_projection_sign'])
            matrix=np.concatenate((np.broadcast_to(pos,(len(corrected),4,2)),gains[:,:,None]),axis=2)
            return np.einsum('sij,sj->si',np.linalg.pinv(matrix)[:,:2,:],gains*corrected)/1e-4
        def residual(x):return np.r_[((predict(x)-target)/scale).reshape(-1),1e-5*(x[3:]-start_log),1e-5*x[:3]]
        before=float(np.sqrt(np.mean((predict(np.r_[np.zeros(3),start_log])-target)**2)))
        if before>.001:
            lower=np.r_[np.full(3,-1.),np.full(4*levels,np.log(.004))]
            upper=np.r_[np.full(3,1.),np.full(4*levels,np.log(30.))]
            fit=least_squares(residual,start,bounds=(lower,upper),max_nfev=800,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            x=fit.x
        else:x=np.r_[np.zeros(3),start_log]
        bias,weights=unpack(x);fitted.append(weights);offsets.append(bias);errors.append(float(np.sqrt(np.mean((predict(x)-target)**2))))
        print('ALL18_AFFINE_CHANNEL_FIT',direct,'WL',point+1,'vector_rmse',errors[-1],'bias_pm',np.round(bias,3).tolist(),flush=True)
    channel_weights[direct]=np.asarray(fitted);biases[direct]=np.asarray(offsets);fit_errors[str(direct)]=errors
    np.savez(folder/f'weights_bias_{direct}.npz',weights=channel_weights[direct],bias=biases[direct])
block['channel_weights']=channel_weights;block['signal_bias_pm']=biases
block['channel_grading_signal']='raw_unit_gain_core_curvature_before_input_bias'
class Physical:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        raw=compute_four_core(frame,design,calibration,None,1.)
        masses=service.allocate(service.scores(raw.curvature_per_m));gains=np.einsum('lp,plc->pc',masses,channel_weights[direct])
        corrected=replace(frame,signals=frame.signals-biases[direct]*1e-6)
        return compute_four_core(corrected,design,calibration,gains,1.)
s.curvature=Physical();metrics={}
for direct in (0,120,240):
    curves,distance=s.reconstruct_shapes('circ',direct);m=s.evaluate(distance,{'circ':curves},direct).metrics
    metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    print('ALL18_AFFINE_CHANNEL_FORWARD',direct,metrics[str(direct)],flush=True)
report={'metrics':metrics,'parameters':blocks,'vector_fit_rmse':fit_errors,'all_18_measurements_retained':True,'no_measurement_imputation':True,'experimental_adapter':True,'relative_channel_bias_bounds_pm':[-100.,100.],'bias_reference_channel':4,'initialization_fit_objective':'curvature vector; geometry polishing follows','interp_num':256,'k_point':1.,'k_end':.5}
(folder/'candidate.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist()),encoding='utf-8')
