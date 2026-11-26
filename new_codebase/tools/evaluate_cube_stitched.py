"""Evaluate the actual stitched four-core field on full calibration frames."""
import copy,json,pickle,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.core_regions import CoreDemoCurvature

root=Path(__file__).resolve().parents[1]
folder=root/'runs/11B1_Cube/segment_search'
count=int(sys.argv[1]) if len(sys.argv)>1 else 4
records={r:json.loads((folder/f'{count}seg_{r}.json').read_text(encoding='utf-8')) for r in ('full','end15cm','end5cm')}
blocks={r:records[r]['parameters'][r] for r in records}
for block in blocks.values():
    block['weights']={int(k):np.asarray(v) for k,v in block['weights'].items()}
args=parse_main_args(['--config','configs/11B1_Cube_core.yaml','--task','optim'])
args.optimization_region='full'
args.multi_region_calib=False
args.full_segmentation=records['full']['layout']
args.calibration_regions['full']['weight_group_ids']=np.repeat(np.arange(count),args.full_segmentation).tolist()
args.core_position_parameters=copy.deepcopy(blocks)
configure_calibration_args(args)
args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full')
session.parameters=CalibrationParameters(blocks,source='cube_segment_search')
dataset=session.load_data()
results={}
for direct in (0,120,240):
    service=CoreDemoCurvature(session.curvature,session.design,session.parameters,('full','end15cm','end5cm'),direct)
    curves={}
    for radius,values in dataset.data['circ'][direct].items():
        curvature=service.compute(session.frame(values),session.parameters,direct=direct)
        metadata=blocks['full']['core_position_metadata']['calibration']
        offset=metadata['core2_azimuth_deg']+metadata['bend_side_offset_deg']+metadata['direction_sign']*direct
        angles=curvature.direction_rad-np.deg2rad(offset)
        curvature=replace(curvature,direction_rad=np.arctan2(np.sin(angles),np.cos(angles)))
        prepared=session.reconstruction.prepare(curvature,session.design,np.array([1.,0.,0.]))
        shape=session.reconstruction.integrate(prepared)
        curves[radius]=shape.position_m
        session.cur_interp=prepared.curvature_per_m.copy()
    metrics=session.evaluate(shape.distance_m,{'circ':curves},direct).metrics
    results[str(direct)]={'point':float(metrics['avg_point_err']['calib']),'end':float(metrics['avg_end_err']['calib'])}
    print('STITCHED',count,direct,results[str(direct)],flush=True)
(folder/f'{count}seg_stitched.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
