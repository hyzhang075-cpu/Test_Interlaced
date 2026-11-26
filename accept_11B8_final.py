import os,sys,json,pickle,copy
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')];os.chdir(root/'new_codebase')
import numpy as np
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
out=Path('runs/11B8_core_point_refine_k1_end05')
with (out/'final_candidate.pkl').open('rb') as f:blocks=pickle.load(f)['blocks']
results={}
for region,count in [('full',40),('end15cm',21),('end5cm',7)]:
 a=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
 a.optimization_region=region;a.multi_region_calib=False;a.end15cm_point_calib=False;a.end5cm_point_calib=False;a.calibration_regions[region]['weight_group_ids']=list(range(count));a.core_position_parameters=copy.deepcopy(blocks)
 configure_calibration_args(a);a.exp_directs=[0,120,240]
 with (out/f'{region}_dataset_cache.pkl').open('rb') as f:dataset=pickle.load(f)
 s=TaskSession(module_parameters(a,optimizer_parameters),task='optim',region=region,dataset=dataset)
 results[region]={}
 for d in (0,120,240):
  curves,dist=s.reconstruct_shapes('circ',d);m=s.evaluate(dist,{'circ':curves},d).metrics
  results[region][str(d)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
  print('FINAL',region,d,results[region][str(d)],flush=True)
# Reuse the independently exercised full-grid assembly validator with final blocks.
source=(root/'validate_11B8_latest.py').read_text(encoding='utf-8')
start=source.index(' blocks={}');end=source.index(' args=parse_main_args',start)
source=source[:start]+" with (out/'final_candidate.pkl').open('rb') as f:blocks=pickle.load(f)['blocks']\n"+source[end:]
source=source.replace("out/'stitched_validation.json'", "out/'final_stitched_validation.json'")
exec(compile(source,'final_stitched_validator','exec'),{'__file__':str(root/'validate_11B8_latest.py'),'__name__':'final_stitched_validator'})
results['stitched']=json.loads((out/'final_stitched_validation.json').read_text())
passed=all(v[k]<(.01 if r in ('full','stitched') else .1) for r,ds in results.items() for v in ds.values() for k in ('point','end'))
report={'passed':passed,'interp_num':256,'method':'core_position','k_point':1.,'k_end':.5,'start_mm':177.,'end_mm':8.,'metrics':results,'validation_scope':'provided calibration data; independent reconstruction of saved parameters'}
(out/'final_acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
def native(x):
 if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [native(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
(out/'11B8_validated_core_parameters.json').write_text(json.dumps(native({'core_position_parameters':blocks,'acceptance':report}),ensure_ascii=False,indent=2),encoding='utf-8')
print('ACCEPTANCE_PASSED',passed,flush=True)
