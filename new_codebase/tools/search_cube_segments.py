"""Search segmented four-core weights; never fits twist slopes."""
import copy
import json
from pathlib import Path
import numpy as np
import os
from shape_sensing.cli import parse_main_args, configure_calibration_args, optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession

ROOT = Path(__file__).resolve().parents[1]
seed = json.loads((ROOT/'runs/11B1_Cube/20261125-172521-655152-8faf677802-init/initialization.json').read_text(encoding='utf-8'))['result']['core_position_parameters']
point_groups = os.environ.get('CUBE_POINT_GROUPS') == '1'
custom_layout = json.loads(os.environ['CUBE_LAYOUT']) if os.environ.get('CUBE_LAYOUT') else None
output = ROOT/'runs/11B1_Cube'/('layout_'+'_'.join(map(str,custom_layout)) if custom_layout else 'point_group_search' if point_groups else 'segment_search')
custom_bounds = json.loads(os.environ['CUBE_BOUNDS']) if os.environ.get('CUBE_BOUNDS') else None
warm_folder=output
if custom_bounds:
    output=Path(str(output)+'_bounds_'+'_'.join(map(str,custom_bounds)))
output.mkdir(exist_ok=True)
from scipy.optimize import minimize
from shape_sensing.optimization._kernel import OptimizationKernel
def monitored_minimize(self, **kwargs):
    original = kwargs['fun']
    evaluations = 0
    best = float('inf')
    def measured(x):
        nonlocal evaluations, best
        value = original(x)
        evaluations += 1
        if value < best:
            best = float(value)
            np.savez(output/f'active_{self.region}_{self.direct}.npz', x=x,loss=best)
        if evaluations % 100 == 0:
            print(f'LIVE region={self.region} direct={self.direct} evaluations={evaluations} best_loss={best:.9f}',flush=True)
        return value
    kwargs['fun'] = measured
    kwargs['options'] = {'maxiter':100,'maxfun':4000,'ftol':1e-9}
    return minimize(**kwargs)
OptimizationKernel._minimize = monitored_minimize
layouts = [[7,7,2,2], [3,4,7,2,2], [3,2,2,7,2,2], [3,2,2,3,4,2,2]]
if point_groups:
    layouts = [layouts[-1]]
if custom_layout:
    layouts = [custom_layout]
datasets = {}
for layout in layouts:
    for region in ('full','end15cm','end5cm'):
        if point_groups and region != 'end5cm':
            continue
        # Regional supports must consist of complete full segments.
        boundary = 7 if region == 'end15cm' else 14 if region == 'end5cm' else 0
        if boundary not in np.r_[0,np.cumsum(layout)]:
            continue
        local = layout[list(np.r_[0,np.cumsum(layout)]).index(boundary):]
        reusable = None
        for candidate in sorted(output.glob(f'*seg_{region}.json')):
            previous = json.loads(candidate.read_text(encoding='utf-8'))
            old = previous['layout']
            if boundary in np.r_[0,np.cumsum(old)]:
                old_local = old[list(np.r_[0,np.cumsum(old)]).index(boundary):]
                if old_local == local:
                    reusable = previous
                    break
        if reusable is not None:
            reusable['layout'] = layout
            (output/f'{len(layout)}seg_{region}.json').write_text(json.dumps(reusable,indent=2),encoding='utf-8')
            print('REUSED identical regional segmentation',layout,region,reusable['metrics'],flush=True)
            continue
        args = parse_main_args(['--config','configs/11B1_Cube_core.yaml','--task','optim'])
        args.full_segmentation = layout
        args.optimization_region = region
        args.multi_region_calib = False
        args.end5cm_independent_weights = False
        args.end5cm_independent_weight_score = False
        args.end5cm_coverage_aware_weights = False
        args.end15cm_point_calib = False
        args.end5cm_point_calib = False
        args.method = 'L-BFGS-B'
        if custom_bounds:
            args.weight_bounds=custom_bounds
        args.calibration_regions[region]['segmentation'] = local
        args.calibration_regions[region]['weight_group_ids'] = np.repeat(np.arange(len(local)),local).tolist()
        if point_groups:
            args.calibration_regions[region]['weight_group_ids'] = list(range(sum(local)))
        args.core_position_parameters = copy.deepcopy(seed)
        for block in args.core_position_parameters.values():
            block.pop('weights', None)
            block.pop('weight_layout', None)
        configure_calibration_args(args)
        args.exp_directs = [0,120,240]
        values = module_parameters(args, optimizer_parameters)
        session = TaskSession(values, task='optim',region=region,dataset=datasets.get(region))
        datasets[region] = session.load_data()
        if custom_layout:
            old_path=ROOT/f'runs/11B1_Cube/segment_search/7seg_{region}.json'
            old_record=json.loads(old_path.read_text(encoding='utf-8'))
            old_block=old_record['parameters'][region]
            old_groups=np.asarray(old_block['weight_layout']['weight_group_ids'])
            new_groups=np.asarray(args._resolved_weight_group_ids)
            levels=len(args.thresholds)+1
            current=session.parameters.parameters_by_region[region]
            for direction,weights in old_block['weights'].items():
                per_point=np.asarray(weights).reshape(-1,levels)[old_groups]
                current['weights'][int(direction)]=np.asarray([per_point[new_groups==g].mean(axis=0) for g in range(new_groups.max()+1)]).reshape(-1)
            if custom_bounds:
                import pickle
                for direction in (0,120,240):
                    warm=warm_folder/f'{len(layout)}seg_{region}_{direction}.pkl'
                    if warm.exists():
                        with warm.open('rb') as stream:
                            previous=pickle.load(stream)
                        current['weights'][direction]=previous['parameters'].parameters_by_region[region]['weights'][direction].copy()
        results = {}
        for direct in (0,120,240):
            import pickle
            saved = output/f'{len(layout)}seg_{region}_{direct}.pkl'
            if saved.exists():
                with saved.open('rb') as checkpoint:
                    previous = pickle.load(checkpoint)
                session.parameters = previous['parameters']
                results.update(previous['metrics'])
                print('RESUMED',layout,region,direct,results[str(direct)],flush=True)
                continue
            from dataclasses import replace
            session.config = replace(session.config, optimization=replace(session.config.optimization,direct=direct))
            print(f'SEARCH segments={layout} region={region} direction={direct}',flush=True)
            session.optim_coeff()
            curves,dist = session.reconstruct_shapes('circ',direct)
            metrics=session.evaluate(dist,{'circ':curves},direct).metrics
            results[str(direct)]={'point':float(metrics['avg_point_err']['calib']),'end':float(metrics['avg_end_err']['calib'])}
            print('RESULT',region,direct,results[str(direct)],flush=True)
            import pickle
            with (output/f'{len(layout)}seg_{region}_{direct}.pkl').open('wb') as checkpoint:
                pickle.dump({'layout':layout,'metrics':results,'parameters':session.parameters},checkpoint)
        def native(x):
            if isinstance(x,np.ndarray): return x.tolist()
            if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)): return [native(v) for v in x]
            return x
        (output/f'{len(layout)}seg_{region}.json').write_text(json.dumps(native({'layout':layout,'metrics':results,'parameters':session.parameters.parameters_by_region}),indent=2),encoding='utf-8')
