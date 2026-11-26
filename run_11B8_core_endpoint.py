import os,sys,time,json,subprocess,pathlib
root=pathlib.Path(__file__).resolve().parent
base=root/'new_codebase'; out=base/'runs/11B8_core_6seg_endpoint'; out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(root/'.codex_deps'),MPLBACKEND='Agg',PYTHONIOENCODING='utf-8')
initial=base/'runs/11B8_NeTi/20261125-114033-447422-ff847b782c-init'
status={'stage':'waiting_for_initialization','interpolation':256,'segmentation':[6,6,7,7,7,7],'completed':[]}
def save():
 status['updated']=time.strftime('%Y-%m-%d %H:%M:%S'); (out/'progress.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
save()
while True:
 m=json.loads((initial/'manifest.json').read_text(encoding='utf-8'))
 if m['status']=='succeeded':break
 if m['status']!='running':status.update(stage='failed',reason=m);save();sys.exit(1)
 time.sleep(5)
for direct in [0,120,240]:
 status.update(stage='optimizing',direction=direct);save()
 with (out/f'optim_{direct}.log').open('w',encoding='utf-8') as log:
  result=subprocess.run([sys.executable,'-u','main.py','--config','configs/11B8_Core6SegEndpoint.yaml','--task','optim','--multi_region_calib','True','--resume',str(initial),'--direct',str(direct)],cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT)
 status['completed'].append({'direction':direct,'returncode':result.returncode});save()
 if result.returncode:status.update(stage='failed');save();sys.exit(result.returncode)
status.update(stage='first_candidate_finished_requires_assembled_validation');save()
