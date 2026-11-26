import sys,pickle,numpy as np,copy
from pathlib import Path
sys.path[:0]=['.codex_deps','new_codebase']
from shape_sensing.initialization.four_core import central_enclosing_circle
out=Path('new_codebase/runs/11B8_core_point_refine_k1_end05')
def load(n):return pickle.load(open(out/n,'rb'))
joint=load('end5_joint_best.pkl');blocks=copy.deepcopy(joint['parameters'].parameters_by_region)
fullrec=load('shared_core_point_constrained_final.pkl');x=fullrec['result'].x
D=np.array(blocks['full']['core_position_metadata']['scope']['global_sample_dist_m'])
factors=np.exp(-np.interp(D,np.linspace(D[0],D[-1],40),x[40:]))
for region in ('full','end15cm'):
 b=blocks[region];levels=len(b['weight_layout']['thresholds'])+1
 for i,section in enumerate(b['core_position']):
  factor=factors[section['wl_id']-1]
  for core in section['cores'].values():core['r_um']*=factor
  for d in (0,120,240):
   w=np.asarray(b['weights'][d]).reshape(-1,levels).copy();w[i]*=factor;b['weights'][d]=w.reshape(-1)
for region,b in blocks.items():
 mins=[];rads=[]
 for section in b['core_position']:
  points=np.array([[c['r_um']*np.sin(c['alpha_rad'] or 0),c['r_um']*np.cos(c['alpha_rad'] or 0)] for c in section['cores'].values()])
  gap=np.linalg.norm(points[:,None]-points[None,:],axis=2)[np.triu_indices(4,1)].min();center,radius=central_enclosing_circle(points)
  mins.append(gap);rads.append(radius)
  section.update(envelope_center_uv_um=center.tolist(),envelope_containing_radius_um=radius,minimum_pairwise_distance_um=gap)
 print(region,'min_gap',min(mins),'max_central_envelope',max(rads),flush=True)
with (out/'final_candidate.pkl').open('wb') as f:pickle.dump({'blocks':blocks,'source_joint_metrics':joint['metrics']},f)
