"""Per-real-WL four-core gain calibration with no spatial partition search."""
import copy,json,pickle
from dataclasses import asdict
from pathlib import Path
from dataclasses import replace
import numpy as np
import os
from scipy.optimize import minimize,lsq_linear
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.optimization._kernel import OptimizationKernel
root=Path(__file__).resolve().parents[1]
target_levels=int(os.environ.get('CUBE_GAIN_LEVELS','0'))
output=root/'runs/11B1_Cube'/('pointwise_core' if not target_levels else f'pointwise_core_levels{target_levels}')
unfiltered=os.environ.get('CUBE_VECTOR_FILTER')=='off'
if unfiltered:output=Path(str(output)+'_unfiltered')
core_seed_path=os.environ.get('CUBE_CORE_SEED')
core_seed=json.loads(Path(core_seed_path).read_text(encoding='utf-8')) if core_seed_path else None
if core_seed:output=Path(str(output)+'_physical_seed')
balanced=os.environ.get('CUBE_BALANCE_TARGETS')=='1'
warm_output=output
if balanced:output=Path(str(output)+'_balanced')
if os.environ.get('CUBE_GAIN_OUTPUT'):output=root/os.environ['CUBE_GAIN_OUTPUT']
output.mkdir(exist_ok=True)
seed=json.loads((root/'runs/11B1_Cube/20261125-172521-655152-8faf677802-init/initialization.json').read_text(encoding='utf-8'))['result']['core_position_parameters']
def native(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [native(v) for v in x]
    return x
def monitored(self,**kwargs):
    fun=kwargs['fun']; count=0; best=float('inf')
    def measured(x):
        nonlocal count,best
        value=fun(x); count+=1
        if balanced:
            point=float(self.avg_point_err['calib']);end=float(self.avg_end_err['calib'])
            value=max(point,end)+.02*(point+end)
        if value<best:
            best=float(value)
            np.savez(output/f'active_{self.region}_{self.direct}.npz',x=x,loss=best,
                     point=float(self.avg_point_err['calib']),
                     end=float(self.avg_end_err['calib']))
        if count%100==0:
            print('POINT_GAIN',self.region,self.direct,'evaluations',count,'best_loss',best,flush=True)
        return value
    kwargs['fun']=measured
    kwargs['options']={'maxiter':100,'maxfun':5000,'ftol':1e-9}
    return minimize(**kwargs)
OptimizationKernel._minimize=monitored
for region,count in [('full',18),('end15cm',11),('end5cm',4)]:
    if os.environ.get('CUBE_ONLY_REGION') and region!=os.environ['CUBE_ONLY_REGION']:continue
    args=parse_main_args(['--config','configs/11B1_Cube_core.yaml','--task','optim'])
    args.multi_region_calib=False
    args.optimization_region=region
    args.full_segmentation=None
    args.segmentation=None
    for profile in args.calibration_regions.values(): profile['segmentation']=None
    args.calibration_regions[region]['weight_group_ids']=list(range(count))
    args.end5cm_independent_weights=False
    args.end5cm_independent_weight_score=False
    args.end5cm_coverage_aware_weights=False
    args.end15cm_point_calib=False
    args.end5cm_point_calib=False
    args.weight_bounds=[.01,12.]
    if core_seed:args.weight_bounds=[.004,30.]
    args.method='L-BFGS-B'
    if os.environ.get('CUBE_MIN_LOSS_DIFF'):args.min_loss_diff=float(os.environ['CUBE_MIN_LOSS_DIFF'])
    if os.environ.get('CUBE_INTERPOLATION'):args.curvature_interp=os.environ['CUBE_INTERPOLATION']
    if unfiltered:
        args.move_filter_for_vector=False
        args.move_filter=1
    args.core_position_parameters=copy.deepcopy(seed)
    if core_seed and region in core_seed['parameters']:
        args.core_position_parameters[region]=copy.deepcopy(core_seed['parameters'][region])
    for block in args.core_position_parameters.values():
        block.pop('weights',None); block.pop('weight_layout',None)
    configure_calibration_args(args)
    args.exp_directs=[0,120,240]
    session=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region=region)
    session.load_data()
    zero_references={}
    if os.environ.get('CUBE_SUBTRACT_ZERO')=='1':
        signals=copy.deepcopy(session.dataset.data)
        for direct,states in signals['circ'].items():
            zero=np.array(states[0],copy=True);zero_references[str(direct)]=zero.tolist()
            for radius in states:states[radius]=states[radius]-zero
        session.dataset=replace(session.dataset,data=signals)
        (output/f'{region}_zero_references.json').write_text(json.dumps({'preprocessed_signal_baseline_by_direction':zero_references,'region':region,'source':'measured straight state; subtract before physical inversion'},indent=2),encoding='utf-8')
    old=json.loads((root/f'runs/11B1_Cube/segment_search/7seg_{region}.json').read_text(encoding='utf-8'))['parameters'][region]
    levels=len(args.thresholds)+1
    for direct,weights in old['weights'].items():
        session.parameters.parameters_by_region[region]['weights'][int(direct)]=np.asarray(weights).reshape(-1,levels)[old['weight_layout']['weight_group_ids']].reshape(-1)
    if core_seed and region in core_seed['parameters']:
        for direct,weights in core_seed['parameters'][region]['weights'].items():
            session.parameters.parameters_by_region[region]['weights'][int(direct)]=np.asarray(weights)
        layout=copy.deepcopy(core_seed['parameters'][region]['weight_layout'])
        settings=dict(layout);settings['end5cm_independent_weight_score']=settings.pop('independent_weight_score')
        session.config=replace(session.config,curvature=replace(session.config.curvature,**settings))
        session.curvature=CurvatureService(session.config.curvature)
        session.parameters.parameters_by_region[region]['weight_layout']=layout
    if balanced:
        for direct in (0,120,240):
            warm=warm_output/f'active_{region}_{direct}.npz'
            if warm.exists():
                with np.load(warm) as active:
                    session.parameters.parameters_by_region[region]['weights'][direct]=active['x'].copy()
    if unfiltered and not core_seed:
        for direct in (0,120,240):
            warm=root/f'runs/11B1_Cube/pointwise_core/{region}_{direct}.pkl'
            if warm.exists():
                with warm.open('rb') as stream:previous=pickle.load(stream)
                session.parameters.parameters_by_region[region]['weights'][direct]=previous['parameters'].parameters_by_region[region]['weights'][direct].copy()
    if target_levels:
        old_curvature=session.curvature
        frames=[]
        source=copy.deepcopy(session.parameters.parameters_by_region[region]['weights'])
        for direct in (0,120,240):
            saved=root/f'runs/11B1_Cube/pointwise_core/{region}_{direct}.pkl'
            if saved.exists() and not core_seed:
                with saved.open('rb') as stream: previous=pickle.load(stream)
                source[direct]=previous['parameters'].parameters_by_region[region]['weights'][direct].copy()
            for radius,values in session.dataset.data['circ'][direct].items():
                _,magnitude,_,_=session.curvature.base(session.frame(values),session.design,session.parameters,direct=direct,region=region)
                gain=session.curvature._effective_core_weights(magnitude,np.asarray(source[direct]))
                frames.append((direct,radius,magnitude,gain))
        pool=np.concatenate([m for d,r,m,g in frames if r>0])
        thresholds=np.unique(np.quantile(pool,np.linspace(.03,.97,target_levels-1))).tolist()
        config=replace(session.config.curvature,thresholds=thresholds)
        session.config=replace(session.config,curvature=config)
        session.curvature=CurvatureService(config)
        block=session.parameters.parameters_by_region[region]
        for direct in (0,120,240):
            rows=[(m,g) for d,r,m,g in frames if d==direct]
            masses=np.asarray([session.curvature.allocate(session.curvature.scores(m)) for m,g in rows])
            gains=np.asarray([g for m,g in rows])
            block['weights'][direct]=np.asarray([lsq_linear(masses[:,:,point],gains[:,point],bounds=(.01,12.)).x for point in range(count)]).reshape(-1)
        block['weight_layout']=session._weight_layout()
        print('CURVATURE_LEVELS',region,len(thresholds)+1,'thresholds',thresholds,flush=True)
    results={}
    for direct in ([int(d) for d in os.environ['CUBE_DIRECTIONS'].split(',')] if os.environ.get('CUBE_DIRECTIONS') else (0,120,240)):
        checkpoint=output/f'{region}_{direct}.pkl'
        if checkpoint.exists():
            with checkpoint.open('rb') as stream: saved=pickle.load(stream)
            session.parameters=saved['parameters']; results.update(saved['metrics'])
            print('RESUMED_POINT_GAIN',region,direct,results[str(direct)],flush=True)
            continue
        if target_levels and not core_seed:
            warm=root/f'runs/11B1_Cube/pointwise_core/{region}_{direct}.pkl'
            if warm.exists():
                with warm.open('rb') as stream:previous=pickle.load(stream)
                reference=previous['parameters'].parameters_by_region[region]['weights'][direct]
                rows=[(m,old_curvature._effective_core_weights(m,np.asarray(reference))) for d,r,m,g in frames if d==direct]
                masses=np.asarray([session.curvature.allocate(session.curvature.scores(m)) for m,g in rows]);gains=np.asarray([g for m,g in rows])
                session.parameters.parameters_by_region[region]['weights'][direct]=np.asarray([lsq_linear(masses[:,:,point],gains[:,point],bounds=(.01,12.)).x for point in range(count)]).reshape(-1)
        session.config=replace(session.config,optimization=replace(session.config.optimization,direct=direct))
        print('START_POINT_GAIN',region,direct,'real_points',count,'spatial_partitions',False,flush=True)
        session.optim_coeff()
        curves,distance=session.reconstruct_shapes('circ',direct)
        metrics=session.evaluate(distance,{'circ':curves},direct).metrics
        results[str(direct)]={'point':float(metrics['avg_point_err']['calib']),'end':float(metrics['avg_end_err']['calib'])}
        print('RESULT_POINT_GAIN',region,direct,results[str(direct)],flush=True)
        with checkpoint.open('wb') as stream: pickle.dump({'parameters':session.parameters,'metrics':results},stream)
    (output/f'{region}.json').write_text(json.dumps(native({'metrics':results,'parameters':session.parameters.parameters_by_region,'spatial_partition_search':False,'core_seed':core_seed_path,'objective':'max(point,end)+0.02*(point+end)' if balanced else 'configured_geometry_loss','reconstruction':asdict(session.config.reconstruction),'optimization':asdict(session.config.optimization)}),indent=2),encoding='utf-8')
