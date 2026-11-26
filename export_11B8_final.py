import os,sys,json,pickle
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')]
import yaml
out=root/'new_codebase/runs/11B8_core_point_refine_k1_end05'
accepted=json.loads((out/'11B8_validated_core_parameters.json').read_text(encoding='utf-8'))
config=yaml.safe_load((root/'new_codebase/configs/11B8_CoreUnsegmented.yaml').read_text(encoding='utf-8'))
config.update(k_point=1.,k_end=.5,interp_num=256,end15cm_point_calib=False,end5cm_point_calib=False,core_position_calib=True)
config['zero_reference_path']=None
for region,count in [('full',40),('end15cm',21),('end5cm',7)]:config['calibration_regions'][region]['weight_group_ids']=list(range(count))
package={'package_type':'shape_sensing_algorithm_package','schema_version':1,'sensor_id':'11B8_NeTi','configuration':config,'core_position_parameters':accepted['core_position_parameters'],'calibration':{},'validation':{'verified':True,'deployment_ready':False,'scope':'provided calibration dataset','acceptance':accepted['acceptance'],'geometry_checks':json.loads((out/'final_geometry_checks.json').read_text())},'multi_region':{'region_order':['full','end15cm','end5cm'],'directions':[0,120,240]}}
path=out/'11B8_validated_algorithm.json';path.write_text(json.dumps(package,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
os.chdir(root/'new_codebase')
from shape_sensing.cli import parse_main_args,optimizer_parameters,configure_calibration_args
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.artifacts.packages import load_algorithm_package
p=load_algorithm_package(path)
with (out/'full_dataset_cache.pkl').open('rb') as f:data=pickle.load(f)
for direct in (0,120,240):
 args=parse_main_args(['--sensor_params',str(path),'--task','recon','--direct',str(direct)])
 configure_calibration_args(args)
 s=TaskSession(module_parameters(args,optimizer_parameters),task='recon',region='full',dataset=data)
 curves,dist=s.reconstruct_shapes('circ',direct);m=s.evaluate(dist,{'circ':curves},direct).metrics
 expected=accepted['acceptance']['metrics']['full'][str(direct)]
 assert abs(m['avg_point_err']['calib']-expected['point'])<1e-12
 assert abs(m['avg_end_err']['calib']-expected['end'])<1e-12
 print('PACKAGE_ROUNDTRIP',direct,'passed',flush=True)
print(path,flush=True)
