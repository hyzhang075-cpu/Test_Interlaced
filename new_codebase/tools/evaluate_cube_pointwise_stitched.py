"""Validate actual full-grid fusion of independently calibrated pointwise cores."""
import argparse,copy,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.core_regions import CoreDemoCurvature
parser=argparse.ArgumentParser();parser.add_argument('--folder',default='runs/11B1_Cube/pointwise_core');parser.add_argument('--full-candidate');options=parser.parse_args()
folder=Path(options.folder);records={r:json.loads((folder/f'{r}.json').read_text(encoding='utf-8')) for r in ('full','end15cm','end5cm')}
if options.full_candidate:records['full']=json.loads(Path(options.full_candidate).read_text(encoding='utf-8'))
blocks={r:records[r]['parameters'][r] for r in records}
for block in blocks.values():block['weights']={int(k):np.asarray(v) for k,v in block['weights'].items()}
layout=blocks['full']['weight_layout']
args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
args.multi_region_calib=False;args.core_position_parameters=copy.deepcopy(blocks)
args.thresholds=layout['thresholds'];args.end_weights=layout['end_weights'];args.end_start=layout['end_start']
args.soft=layout['soft'];args.sharpness=layout['sharpness'];args.min_consecutive=layout['min_consecutive'];args.end5cm_independent_weight_score=layout['independent_weight_score']
args.calibration_regions['full']['weight_group_ids']=layout['weight_group_ids']
if folder.name.endswith('_unfiltered'):args.move_filter_for_vector=False;args.move_filter=1
configure_calibration_args(args);args.exp_directs=[0,120,240]
session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full');session.parameters=CalibrationParameters(blocks,source='pointwise_core_fusion_validation')
dataset=session.load_data();results={}
for direct in (0,120,240):
    service=CoreDemoCurvature(session.curvature,session.design,session.parameters,('full','end15cm','end5cm'),direct)
    curves={}
    for radius,values in dataset.data['circ'][direct].items():
        physical=service.compute(session.frame(values),session.parameters,direct=direct)
        metadata=blocks['full']['core_position_metadata']['calibration'];offset=metadata['core2_azimuth_deg']+metadata['bend_side_offset_deg']+metadata['direction_sign']*direct
        angles=physical.direction_rad-np.deg2rad(offset)
        physical=replace(physical,direction_rad=np.arctan2(np.sin(angles),np.cos(angles)))
        prepared=session.reconstruction.prepare(physical,session.design,np.array([1.,0.,0.]));shape=session.reconstruction.integrate(prepared)
        curves[radius]=shape.position_m;session.cur_interp=prepared.curvature_per_m.copy()
    metrics=session.evaluate(shape.distance_m,{'circ':curves},direct).metrics
    results[str(direct)]={'point':float(metrics['avg_point_err']['calib']),'end':float(metrics['avg_end_err']['calib'])}
    print('POINTWISE_STITCHED',direct,results[str(direct)],flush=True)
report={'metrics':results,'passed_full_one_percent':all(v['point']<.01 and v['end']<.01 for v in results.values()),'spatial_partition_search':False,'full_candidate':options.full_candidate}
destination=folder/('stitched.json' if not options.full_candidate else Path(options.full_candidate).stem+'_stitched.json')
destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
