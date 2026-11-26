import os,sys,json,pathlib,subprocess,time
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root/'.codex_deps'))
import yaml
base=root/'new_codebase';out=base/'runs/11B8_core_unsegmented';out.mkdir(parents=True,exist_ok=True)
cfg=yaml.safe_load((base/'configs/11B8_Core6SegEndpoint.yaml').read_text(encoding='utf-8'))
cfg['full_segmentation']=[40]
cfg['calibration_output_dir']=str(out/'calibration_results')
for r,p in cfg['calibration_regions'].items():
 p.pop('segmentation',None)
cfgpath=base/'configs/11B8_CoreUnsegmented.yaml'
cfgpath.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
env=dict(os.environ,PYTHONPATH=str(root/'.codex_deps'),MPLBACKEND='Agg',PYTHONIOENCODING='utf-8')
initial=base/'runs/11B8_NeTi/20261125-114033-447422-ff847b782c-init'
status={'stage':'optimizing','mode':'fixed core positions; unsegmented regional weights','completed':[]}
for direct in (0,120,240):
 status['direction']=direct
 (out/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 with (out/f'optim_{direct}.log').open('w',encoding='utf-8') as log:
  rc=subprocess.run([sys.executable,'-u','main.py','--config',str(cfgpath),'--task','optim','--multi_region_calib','True','--resume',str(initial),'--direct',str(direct)],cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 status['completed'].append({'direction':direct,'returncode':rc})
 if rc:status['stage']='failed';break
else:status['stage']='requires_validation'
(out/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
