"""Explicit measurement-imputation diagnostic; not accepted as 18-point inversion."""
import copy,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.curvature.four_core import compute_four_core
from shape_sensing.curvature.core_regions import CoreDemoCurvature
root=Path(__file__).resolve().parents[1];run=root/'runs/11B1_Cube'
record=json.loads((run/'channel_level_weights_experimental/channel_level_candidate.json').read_text())
blocks={'full':copy.deepcopy(record['parameters']['full'])}
local=json.loads((run/'pointwise_core_fusion_joint_low_kend05/fusion_gain_refinement.json').read_text())
for r in ('end15cm','end5cm'):blocks[r]=copy.deepcopy(local['parameters'][r])
for b in blocks.values():b['weights']={int(d):np.asarray(w) for d,w in b['weights'].items()}
block=blocks['full'];channels={int(d):np.asarray(w) for d,w in block['channel_weights'].items()}
maximum=np.max(np.asarray(list(record['vector_fit_rmse'].values())),axis=0)
good=maximum<=.3;bad=(np.flatnonzero(~good)+1).tolist()
assert good.sum()>=2
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');s.load_data()
layout=dict(block['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
service=CurvatureService(replace(s.config.curvature,**layout));s.parameters=CalibrationParameters(blocks)
calibration={**block['core_position_metadata'],'sections':block['core_position']}
class Channels:
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        raw=compute_four_core(frame,design,calibration,None,1.)
        mass=service.allocate(service.scores(raw.curvature_per_m));gains=np.einsum('lp,plc->pc',mass,channels[direct])
        return compute_four_core(frame,design,calibration,gains,1.)
class Imputed:
    def __init__(self,source,fusion=False):self.source=source;self.fusion=fusion
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        c=self.source.compute(frame,parameters,direct=direct) if self.fusion else self.source.compute(frame,design,parameters,direct=direct)
        vector=c.curvature_per_m[:,None]*np.column_stack((np.cos(c.direction_rad),np.sin(c.direction_rad)))
        interpolated=np.column_stack([np.interp(c.distance_m,c.distance_m[good],vector[good,j]) for j in range(2)])
        return replace(c,curvature_per_m=np.linalg.norm(interpolated,axis=1),direction_rad=np.arctan2(interpolated[:,1],interpolated[:,0]),diagnostics={**c.diagnostics,'imputed_wl_ids':bad})
result={}
for direct in (0,120,240):
    fusion=CoreDemoCurvature(service,s.design,s.parameters,('full','end15cm','end5cm'),direct)
    name,indices,design,_=fusion._regions[0];fusion._regions[0]=(name,indices,design,Channels())
    result[str(direct)]={}
    for name,adapter in [('single_full',Channels()),('single_full_imputed',Imputed(Channels())),('fusion_imputed',Imputed(fusion,True))]:
        s.curvature=adapter;curves,distance=s.reconstruct_shapes('circ',direct)
        m=s.evaluate(distance,{'circ':curves},direct).metrics
        result[str(direct)][name]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
        print('QUALITY_IMPUTATION_DIAGNOSTIC',direct,name,result[str(direct)][name],flush=True)
report={'metrics':result,'imputed_wl_ids':bad,'measured_points_used':int(good.sum()),'selection_rule':'maximum calibration curvature-vector RMSE over directions >0.3 per m','diagnostic_only':True,'requires_measurement_selection_review':True,'not_full_18_point_direct_inversion':True,'interp_num':256}
(run/'channel_level_weights_experimental/quality_imputation_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
