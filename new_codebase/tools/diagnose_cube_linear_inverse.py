"""Unconstrained inverse regression diagnostic, not an accepted physical model."""
import json,itertools
import numpy as np
import inspect_cube_physical_fields as baseline
from shape_sensing.contracts import CurvatureResult
from shape_sensing.initialization.four_core import enclosing_circle
s=baseline.s;meta=baseline.blocks['full']['core_position_metadata']['calibration']
response=[];target=[]
for direct in (0,120,240):
    angle=np.deg2rad(meta['core2_azimuth_deg']+meta['bend_side_offset_deg']+meta['direction_sign']*direct)
    for radius,values in s.dataset.data['circ'][direct].items():
        frame=s.frame(values);x=frame.signals/(meta['strain_sensitivity_pm_per_ue']*meta['strain_projection_sign'])
        response.append(x-x.mean(axis=1,keepdims=True))
        target.append(np.zeros(2) if radius==0 else np.array([np.cos(angle),np.sin(angle)])*1000/radius)
response=np.asarray(response);target=np.asarray(target)
coefficient=np.stack([np.linalg.lstsq(response[:,i,:],target,rcond=None)[0] for i in range(response.shape[1])])
geometry=[]
for wl,c in enumerate(coefficient,1):
    points=np.linalg.pinv(c.T)*1e6;center,radius=enclosing_circle(points)
    separation=min(np.linalg.norm(points[a]-points[b]) for a,b in itertools.combinations(range(4),2))
    geometry.append({'wl_id':wl,'effective_positions_uv_um':points.tolist(),'minimum_pair_um':float(separation),'envelope_radius_um':float(radius),'physical_size_feasible':bool(separation>=100-1e-5 and radius<=125+1e-5)})
class LinearInverse:
    def compute(self,frame,design,parameters,**kwargs):
        x=frame.signals/(meta['strain_sensitivity_pm_per_ue']*meta['strain_projection_sign']);x=x-x.mean(axis=1,keepdims=True)
        vector=np.einsum('pc,pcj->pj',x,coefficient)
        return CurvatureResult(frame.distance_m,np.linalg.norm(vector,axis=1),np.arctan2(vector[:,1],vector[:,0]),frame.source_indices)
s.curvature=LinearInverse();metrics={}
for direct in (0,120,240):
    curves,distance=s.reconstruct_shapes('circ',direct);m=s.evaluate(distance,{'circ':curves},direct).metrics
    metrics[str(direct)]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    print('LINEAR_INVERSE_DIAGNOSTIC',direct,metrics[str(direct)],flush=True)
report={'metrics':metrics,'geometry':geometry,'diagnostic_only':True,'all_physical_sizes_feasible':all(g['physical_size_feasible'] for g in geometry)}
(baseline.folder/'unconstrained_linear_inverse_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('LINEAR_INVERSE_PHYSICAL_FEASIBILITY',report['all_physical_sizes_feasible'],flush=True)
