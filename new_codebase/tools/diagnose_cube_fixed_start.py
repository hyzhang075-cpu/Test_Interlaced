"""Compare an unconfirmed fixture reference; diagnostic only, no refit or adoption."""
import json
import numpy as np
import check_cube_full_cases as baseline
s=baseline.s;rows=[]
for direct in (0,120,240):
    curves,distance=s.reconstruct_shapes('circ',direct)
    along=distance-distance[0];length=along[-1]
    for gap in (0.,.114):
        points=[];ends=[]
        for radius,position in curves.items():
            ideal=np.zeros_like(position)
            if radius==0:ideal[:,0]=along
            else:
                r=radius/1000.;arc=np.maximum(along-gap,0.)
                ideal[:,0]=np.minimum(along,gap)+r*np.sin(arc/r)
                ideal[:,1]=r*(1-np.cos(arc/r))
            error=np.linalg.norm(position-ideal,axis=1)/length
            points.append(float(error.mean()));ends.append(float(error[-1]))
        row={'direct':direct,'fixed_straight_start_m':gap,'point':float(np.mean(points)),'end':float(np.mean(ends)),'diagnostic_only':True,'parameters_refitted':False}
        rows.append(row);print('FIXTURE_REFERENCE_DIAGNOSTIC',row,flush=True)
(baseline.folder/'fixed_start_reference_diagnostic.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
