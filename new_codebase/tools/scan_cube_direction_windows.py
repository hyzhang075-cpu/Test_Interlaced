"""Compare production direction smoothing on fixed latest physical-core gains."""
import json
from dataclasses import replace
import check_cube_full_cases as baseline
from shape_sensing.reconstruction.service import ReconstructionService
s=baseline.s;rows=[]
for method in ('vector_pchip','vector_linear'):
    for window in (3,5,7,9,13,17):
        config=replace(s.config.reconstruction,interp_num=256,move_filter=window,move_filter_for_vector=True,curvature_interp=method)
        s.reconstruction=ReconstructionService(config)
        for direct in (0,120,240):
            curves,distance=s.reconstruct_shapes('circ',direct)
            m=s.evaluate(distance,{'circ':curves},direct).metrics
            row={'method':method,'window':window,'direct':direct,'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
            rows.append(row);print('DIRECTION_WINDOW',row,flush=True)
(baseline.folder/'direction_window_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
