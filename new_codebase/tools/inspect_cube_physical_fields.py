"""Read-only physical-field diagnostics on the latest ten-level core candidate."""
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
root=Path(__file__).resolve().parents[1]
folder=root/'runs/11B1_Cube/pointwise_core_refined2_linear_levels10_kend05'
record=json.loads((folder/'full.json').read_text(encoding='utf-8'));blocks=record['parameters']
for b in blocks.values():
    if 'weights' in b:b['weights']={int(d):np.asarray(w) for d,w in b['weights'].items()}
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.curvature_interp='vector_linear';args.core_position_parameters=copy.deepcopy(blocks)
for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
configure_calibration_args(args);args.exp_directs=[0,120,240]
s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
layout=dict(blocks['full']['weight_layout']);layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
s.curvature=CurvatureService(replace(s.config.curvature,**layout));s.parameters=CalibrationParameters(blocks);s.load_data()
rows=[]
for direct in (0,120,240):
    for radius in (300,600,1000):
        f=s.frame(s.dataset.data['circ'][direct][radius]);c=s.curvature.compute(f,s.design,s.parameters,direct=direct,region='full')
        meta=blocks['full']['core_position_metadata']['calibration']
        offset=meta['core2_azimuth_deg']+meta['bend_side_offset_deg']+meta['direction_sign']*direct
        angle=c.direction_rad-np.deg2rad(offset);degrees=np.rad2deg(np.arctan2(np.sin(angle),np.cos(angle)))
        row={'direct':direct,'radius_mm':radius,'distance_m':f.distance_m.tolist(),'curvature_per_m':c.curvature_per_m.tolist(),'direction_deviation_deg':degrees.tolist()}
        calibration={**blocks['full']['core_position_metadata'],'sections':blocks['full']['core_position']}
        raw=compute_four_core(f,s.design,calibration,None,1.)
        residual=raw.diagnostics['corrected_strain_fit_rmse']*meta['strain_sensitivity_pm_per_ue']*1e6
        row['raw_four_core_fit_rmse_pm']=residual.tolist()
        print('RAW_FIT_RESIDUAL',direct,radius,np.round(residual,2).tolist(),flush=True)
        rows.append(row);print('PHYSICAL_FIELD',direct,radius,'direction_deg',np.round(degrees,2).tolist(),flush=True)
(folder/'physical_field_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
