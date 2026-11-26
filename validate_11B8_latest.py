import os,sys,pickle,json,copy
from pathlib import Path
from dataclasses import replace
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')]
os.chdir(root/'new_codebase')
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.core_regions import CoreDemoCurvature
out=Path('runs/11B8_core_point_refine_k1_end05')
results={}
for direct in (0,120,240):
 blocks={}
 for region in ('full','end15cm','end5cm'):
  with (out/f'{region}_{direct}.pkl').open('rb') as f:record=pickle.load(f)
  blocks[region]=record['parameters'].parameters_by_region[region]
 with (out/'shared_core_point_constrained_final.pkl').open('rb') as f:full=pickle.load(f)
 blocks['full']=full['parameters'].parameters_by_region['full']
 args=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
 args.multi_region_calib=False;args.optimization_region='full'
 args.calibration_regions['full']['weight_group_ids']=list(range(40))
 args.core_position_parameters=copy.deepcopy(blocks)
 configure_calibration_args(args);args.exp_directs=[0,120,240]
 cache=out/'full_dataset_cache.pkl'
 dataset=None
 if cache.exists():
  with cache.open('rb') as f:dataset=pickle.load(f)
 s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region='full',dataset=dataset);s.load_data()
 if dataset is None:
  with cache.open('wb') as f:pickle.dump(s.dataset,f)
 s.parameters=CalibrationParameters({'full':blocks['full']})
 c,d=s.reconstruct_shapes('circ',direct)
 single=s.evaluate(d,{'circ':c},direct).metrics
 print('FULL_VERIFIED',direct,single['avg_point_err']['calib'],single['avg_end_err']['calib'],flush=True)
 params=CalibrationParameters(blocks)
 assembler=CoreDemoCurvature(s.curvature,s.design,params,('full','end15cm','end5cm'),direct)
 curves={}
 for radius,signals in s.dataset.data['circ'][direct].items():
  physical=assembler.compute(s.frame(signals),params,direct=direct)
  core=blocks['full']['core_position_metadata']['calibration']
  angle=physical.direction_rad-np.deg2rad(core['core2_azimuth_deg']+core['bend_side_offset_deg']+core['direction_sign']*direct)
  physical=replace(physical,direction_rad=np.arctan2(np.sin(angle),np.cos(angle)))
  prepared=s.reconstruction.prepare(physical,s.design,np.array([1.,0.,0.]))
  shape=s.reconstruction.integrate(prepared);curves[radius]=shape.position_m;s.cur_interp=prepared.curvature_per_m
 m=s.evaluate(shape.distance_m,{'circ':curves},direct).metrics
 results[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
 print('STITCHED',direct,results[str(direct)],flush=True)
 (out/'stitched_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
