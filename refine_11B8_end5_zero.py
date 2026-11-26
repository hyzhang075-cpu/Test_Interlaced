import os,sys,json,pickle,copy
from pathlib import Path
from dataclasses import replace
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'.codex_deps'),str(root/'new_codebase')]
os.environ['MPLBACKEND']='Agg'
os.chdir(root/'new_codebase')
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.optimization._kernel import OptimizationKernel
out=Path('runs/11B8_core_point_refine_k1_end05');out.mkdir(exist_ok=True)
seed=json.loads(Path('runs/11B8_NeTi/20261125-114033-447422-ff847b782c-init/initialization.json').read_text(encoding='utf-8'))['result']['core_position_parameters']
def monitored(self,**kw):
 f=kw['fun']; n=0;best=float('inf')
 def measured(x):
  nonlocal n,best
  v=f(x);n+=1
  if v<best:best=float(v);np.savez(out/f'best_{self.region}_{self.direct}.npz',x=x,loss=best)
  if n%100==0:print(f'LIVE {self.region} {self.direct} eval={n} best={best:.9f}',flush=True)
  return v
 kw['fun']=measured;kw['options']={'maxiter':100,'maxfun':3000,'ftol':1e-10}
 return minimize(**kw)
OptimizationKernel._minimize=monitored
for region,count in [('end5cm',7)]:
 args=parse_main_args(['--config','configs/11B8_CoreUnsegmented.yaml','--task','optim'])
 args.optimization_region=region;args.multi_region_calib=False;args.end15cm_point_calib=False;args.end5cm_point_calib=False
 args.max_iter=3;args.method='L-BFGS-B';args.k_point=1.;args.k_end=0.5;args.interp_num=256
 args.calibration_regions[region]['weight_group_ids']=list(range(count))
 args.core_position_parameters=copy.deepcopy(seed)
 for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
 configure_calibration_args(args);args.exp_directs=[0,120,240]
 s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region=region);s.load_data()
 for direct in (0,):
  s.config=replace(s.config,optimization=replace(s.config.optimization,direct=direct))
  print(f'START {region} {direct}',flush=True)
  checkpoint=out/f'{region}_{direct}.pkl'
  if checkpoint.exists() and region=='full':
   with checkpoint.open('rb') as f:s.parameters=pickle.load(f)['parameters']
   print(f'RESUMED {region} {direct}',flush=True)
   continue
  warm=out/f'best_{region}_{direct}.npz'
  if warm.exists():
   x=np.load(warm)['x']
   w=s.parameters.parameters_by_region[region]['weights'][direct]
   if x.shape==w.shape:w[:]=x
  s.optim_coeff()
  curves,dist=s.reconstruct_shapes('circ',direct)
  m=s.evaluate(dist,{'circ':curves},direct).metrics
  print(f'RESULT {region} {direct} point={m["avg_point_err"]["calib"]} end={m["avg_end_err"]["calib"]}',flush=True)
  with (out/f'{region}_{direct}.pkl').open('wb') as f:pickle.dump({'parameters':s.parameters,'metrics':m},f)
print('BATCH_COMPLETE',flush=True)
