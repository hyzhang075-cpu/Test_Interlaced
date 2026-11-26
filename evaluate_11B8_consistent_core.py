import sys,os,pickle,copy
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')];os.chdir(root/'new_codebase')
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters
out=Path('runs/11B8_core_point_refine_k1_end05')
with (out/'shared_core_point_constrained_final.pkl').open('rb') as f:full=pickle.load(f)['parameters'].parameters_by_region['full']
for region,count in [('end15cm',21),('end5cm',7)]:
 with (out/f'{region}_240.pkl').open('rb') as f:b=copy.deepcopy(pickle.load(f)['parameters'].parameters_by_region[region])
 ids=b['core_position_metadata']['calibration']['wl_ids']
 b['core_position']=[copy.deepcopy(full['core_position'][i-1]) for i in ids]
 for section in b['core_position']:section['source_region']=region
 levels=len(b['weight_layout']['thresholds'])+1
 b['weights']={d:np.pad(np.asarray(full['weights'][d]).reshape(40,6)[np.asarray(ids)-1],((0,0),(0,levels-6)),mode='edge').reshape(-1) for d in (0,120,240)}
 a=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
 a.optimization_region=region;a.multi_region_calib=False;a.calibration_regions[region]['weight_group_ids']=list(range(count));a.core_position_parameters={region:b}
 configure_calibration_args(a);a.exp_directs=[0,120,240]
 cache=out/f'{region}_dataset_cache.pkl'
 dataset=None
 if cache.exists():
  with cache.open('rb') as f:dataset=pickle.load(f)
 s=TaskSession(module_parameters(a,optimizer_parameters),task='optim',region=region,dataset=dataset);s.load_data()
 if dataset is None:
  with cache.open('wb') as f:pickle.dump(s.dataset,f)
 metrics={}
 for d in (0,120,240):
  curves,dist=s.reconstruct_shapes('circ',d);m=s.evaluate(dist,{'circ':curves},d).metrics
  metrics[d]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
  print('CONSISTENT_CORE',region,d,metrics[d],flush=True)
 with (out/f'{region}_consistent_core.pkl').open('wb') as f:pickle.dump({'parameters':s.parameters,'metrics':metrics},f)
