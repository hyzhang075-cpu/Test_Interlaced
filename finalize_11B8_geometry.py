import pickle,sys,numpy as np
from pathlib import Path
sys.path[:0]=['.codex_deps','new_codebase']
from shape_sensing.initialization.four_core import central_enclosing_circle,enclosing_circle
out=Path('new_codebase/runs/11B8_core_point_refine_k1_end05')
r=pickle.load(open(out/'physical_final_candidate.pkl','rb'));checks={}
for region,b in r['blocks'].items():
 gaps=[];radii=[];order=[]
 for section in b['core_position']:
  p=np.array([[c['r_um']*np.sin(c['alpha_rad'] or 0),c['r_um']*np.cos(c['alpha_rad'] or 0)] for c in section['cores'].values()])
  gap=np.linalg.norm(p[:,None]-p[None,:],axis=2)[np.triu_indices(4,1)].min();center,radius=central_enclosing_circle(p);mc,mr=enclosing_circle(p)
  distances=np.linalg.norm(p-center,axis=1)
  gaps.append(gap);radii.append(radius);order.append(bool(np.all(distances[0]<=distances[1:]+1e-5)))
  section.update(minimum_enclosing_center_uv_um=mc.tolist(),minimum_enclosing_radius_um=float(mr),envelope_center_uv_um=center.tolist(),envelope_containing_radius_um=float(radius),minimum_pairwise_distance_um=float(gap),centroid_error_um=float(np.linalg.norm(p.mean(axis=0))))
  for key in ('fit_rmse_pm','fit_rmse_by_channel_pm','calibration_common_response_pm','constraint_solver'):
   if key in section:section['initialization_'+key]=section.pop(key)
  section['refinement']='joint core geometry and response weights; geometry checked after refinement'
 checks[region]={'minimum_pairwise_um':float(min(gaps)),'maximum_containing_radius_um':float(max(radii)),'center_order_valid':all(order),'passed':min(gaps)>=100-1e-5 and max(radii)<=125+1e-5 and all(order)}
 print(region,checks[region],flush=True)
assert all(v['passed'] for v in checks.values())
r['geometry_checks']=checks
pickle.dump(r,open(out/'final_candidate.pkl','wb'))
import json
(out/'final_geometry_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
