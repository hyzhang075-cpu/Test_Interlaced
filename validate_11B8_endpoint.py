import os,sys,json,pathlib,traceback
from dataclasses import replace
import numpy as np
root=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')]
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
from shape_sensing.curvature.core_regions import CoreDemoCurvature
base=root/'new_codebase';out=base/'runs/11B8_core_6seg'; results={}
def save(): (out/'assembled_validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
reports=[]
for p in (base/'runs/11B8_NeTi').glob('*-optim/optim.json'):
 d=json.loads(p.read_text(encoding='utf-8'))['result'][1]
 if '11B8_core_6seg_endpoint' in d['regions']['full']['configuration']['calibration_output_dir']:reports.append(d)
assert len(reports)==3
session=None
for d in sorted(reports,key=lambda d:d['direct']):
 direct=d['direct'];blocks=d['core_position_parameters'];cfg=d['regions']['full']['configuration'].copy()
 cfg.update(core_position_parameters=blocks,exp_directs=[0,120,240],folder_path=cfg['calibration_regions']['full']['folder_path'])
 if session is None:session=TaskSession(cfg,task='recon',region='full');session.load_data()
 params=CalibrationParameters(blocks)
 assembler=CoreDemoCurvature(session.curvature,session.design,params,('full','end15cm','end5cm'),direct)
 rs={'circ':{}}
 core=blocks['full']['core_position_metadata']['calibration']
 for radius,signals in session.dataset.data['circ'][direct].items():
  physical=assembler.compute(session.frame(signals),params,direct=direct)
  angle=physical.direction_rad-np.deg2rad(core['core2_azimuth_deg']+core['bend_side_offset_deg']+core['direction_sign']*direct)
  physical=replace(physical,direction_rad=np.arctan2(np.sin(angle),np.cos(angle)))
  prepared=session.reconstruction.prepare(physical,session.design,np.array([1.,0.,0.]))
  shape=session.reconstruction.integrate(prepared);rs['circ'][radius]=shape.position_m;session.cur_interp=prepared.curvature_per_m
 evaluated=session.evaluate(shape.distance_m,rs,direct)
 def clean(x):
  if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  return x
 results[str(direct)]={'metrics':clean(dict(evaluated.metrics)),'source_regions':list(assembler.source_regions),'interp_num':256,'segmentation':[6,6,7,7,7,7]}
 save();print(direct,json.dumps(results[str(direct)]['metrics']),flush=True)
