"""Refine physical-core gains for real full-grid fusion, retaining local gates."""
import copy,json,os
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
from shape_sensing.contracts import CalibrationParameters,PreparedFrame
from shape_sensing.curvature.service import CurvatureService
from shape_sensing.curvature.core_regions import CoreDemoCurvature

root=Path(__file__).resolve().parents[1]
source=root/os.environ.get('CUBE_FUSION_SOURCE','runs/11B1_Cube/pointwise_core_validation')
out=root/os.environ.get('CUBE_FUSION_OUTPUT','runs/11B1_Cube/pointwise_core_fusion_refinement');out.mkdir(exist_ok=True)
regions=('full','end15cm','end5cm')
blocks={r:json.loads((source/f'{r}.json').read_text(encoding='utf-8'))['parameters'][r] for r in regions}
for b in blocks.values():b['weights']={int(d):np.asarray(w) for d,w in b['weights'].items()}
base=copy.deepcopy(blocks);parameters=CalibrationParameters(blocks,source='physical_core_fusion_gain_refinement')
sessions={};fields={};original={};fractions={}
low_only=os.environ.get('CUBE_FUSION_LOW_LEVELS')=='1'
constrained=os.environ.get('CUBE_FUSION_CONSTRAINED')=='1'
def acceptance_slack(result):
    return np.asarray([(0.00999 if r in ('full','single_full') else .09999)-v for r,m in result.items() for v in m.values()])
def low_fraction(frame,design,service,region,direct):
    weights=np.asarray(base[region]['weights'][direct])
    levels=len(service.config.thresholds)+1
    mask=np.r_[0.,service.config.thresholds]<3.5
    selected=(weights.reshape(-1,levels)*np.where(mask,1.,1e-30)[None,:]).reshape(-1)
    _,magnitude,_,_=service.base(frame,design,parameters,direct=direct,region=region)
    return service._effective_core_weights(magnitude,selected)/service._effective_core_weights(magnitude,weights)
for region in regions:
    args=parse_main_args(['--config','configs/11B1_Cube_pointwise.yaml','--task','optim'])
    args.multi_region_calib=False;args.optimization_region=region
    if os.environ.get('CUBE_INTERPOLATION'):args.curvature_interp=os.environ['CUBE_INTERPOLATION']
    args.core_position_parameters=copy.deepcopy(blocks)
    for b in args.core_position_parameters.values():b.pop('weights',None);b.pop('weight_layout',None)
    configure_calibration_args(args);args.exp_directs=[0,120,240]
    s=TaskSession(module_parameters(args,optimizer_parameters),task='optim',region=region)
    layout=dict(blocks[region]['weight_layout'])
    layout['end5cm_independent_weight_score']=layout.pop('independent_weight_score')
    s.config=replace(s.config,curvature=replace(s.config.curvature,**layout))
    s.curvature=CurvatureService(s.config.curvature);s.parameters=parameters;s.load_data()
    sessions[region]=s;original[region]=s.curvature;fields[region]={};fractions[region]={}
    for direct in (0,120,240):
        service=CoreDemoCurvature(s.curvature,s.design,parameters,regions,direct) if region=='full' else s.curvature
        for radius,values in s.dataset.data['circ'][direct].items():
            frame=s.frame(values)
            value=service.compute(frame,parameters,direct=direct) if region=='full' else service.compute(frame,s.design,parameters,direct=direct,region=region)
            fields[region][(direct,frame.signals.tobytes())]=value
            if low_only:
                fraction=np.ones(len(frame.distance_m))
                if region=='full':
                    for name,indices,local,regional in service._regions:
                        probe=PreparedFrame.from_design(local,frame.signals[indices],frame.channel_ids)
                        fraction[indices]=low_fraction(probe,local,regional,name,direct)
                else:fraction=low_fraction(frame,s.design,service,region,direct)
                fractions[region][(direct,frame.signals.tobytes())]=fraction

joint_head=os.environ.get('CUBE_FUSION_JOINT_HEAD')=='1'
all_full=joint_head and os.environ.get('CUBE_FUSION_ALL_FULL')=='1'
if joint_head:
    s=object.__new__(TaskSession)
    s.__dict__.update(sessions['full'].__dict__)
    sessions['single_full']=s
    original['single_full']=original['full'];fields['single_full']={};fractions['single_full']={}
    for direct in (0,120,240):
        for radius,values in s.dataset.data['circ'][direct].items():
            frame=s.frame(values)
            fields['single_full'][(direct,frame.signals.tobytes())]=original['full'].compute(frame,s.design,parameters,direct=direct,region='full')
            if low_only:fractions['single_full'][(direct,frame.signals.tobytes())]=low_fraction(frame,s.design,original['full'],'full',direct)
scales={r:np.ones(len(s.design.source_indices)) for r,s in sessions.items()}
class Cached:
    def __init__(self,region):self.region=region
    def compute(self,frame,design,parameters,*,direct=0,**kwargs):
        value=fields[self.region][(direct,frame.signals.tobytes())]
        scale=scales[self.region]
        if low_only:scale=1+fractions[self.region][(direct,frame.signals.tobytes())]*(scale-1)
        return replace(value,curvature_per_m=value.curvature_per_m*scale)
for region,s in sessions.items():s.curvature=Cached(region)
def metrics(direct):
    result={}
    for region,s in sessions.items():
        curves,distance=s.reconstruct_shapes('circ',direct)
        m=s.evaluate(distance,{'circ':curves},direct).metrics
        result[region]={'point':float(m['avg_point_err']['calib']),'end':float(m['avg_end_err']['calib'])}
    return result
results={}
for direct in (0,120,240):
    size=29 if all_full else (18 if joint_head else 11)
    calls=0;best=float('inf');best_x=np.zeros(size);best_rank=(float('inf'),)*3;metric_cache={}
    def evaluate(x):
        global calls,best,best_x,best_rank
        key=np.asarray(x).tobytes()
        if key in metric_cache:return metric_cache[key][0]
        factor=np.exp(x)
        head=factor[:7] if joint_head else np.ones(7)
        if joint_head:
            scales['single_full']=factor[:18] if all_full else np.r_[head,np.ones(11)]
            factor=factor[18:] if all_full else factor[7:]
        scales['full']=np.r_[head,factor]
        scales['end15cm']=np.r_[factor[:7],np.ones(4)]
        scales['end5cm']=factor[7:]
        result=metrics(direct);full=np.array(list(result['full'].values()))
        local=np.array([v for r in ('end15cm','end5cm') for v in result[r].values()])
        # User-selected geometry loss; acceptance checks each error separately.
        loss=float(sum(m['point']+.5*m['end'] for m in result.values()))
        metric_cache[key]=(loss,copy.deepcopy(result))
        if len(metric_cache)>1024:del metric_cache[next(iter(metric_cache))]
        violation=np.maximum(-acceptance_slack(result),0)
        rank=(float(violation.max()),float(violation.sum()),loss) if constrained else (loss,0.,0.)
        calls+=1
        if rank<best_rank:
            best=loss;best_x=x.copy();best_rank=rank
            np.savez(out/f'active_{direct}.npz',x=x,loss=loss)
        if calls%100==0:print('FUSION_GAIN',direct,calls,'best',best,'metrics',result,flush=True)
        return loss
    evaluate(np.zeros(size))
    if constrained:
        def gates(x):
            evaluate(x)
            return acceptance_slack(metric_cache[np.asarray(x).tobytes()][1])
        opt=minimize(evaluate,np.zeros(size),method='SLSQP',bounds=[(-.5,.5)]*size,constraints=[{'type':'ineq','fun':gates}],options={'maxiter':120,'ftol':1e-10})
    else:
        opt=minimize(evaluate,np.zeros(size),method='L-BFGS-B',bounds=[(-.5,.5)]*size,options={'maxiter':120,'ftol':1e-11,'maxfun':3000})
    metric_cache.clear()
    evaluate(best_x);results[str(direct)]=metrics(direct)
    factor=np.exp(best_x)
    updates=[]
    if joint_head:
        updates.append(('full',factor[:18] if all_full else np.r_[factor[:7],np.ones(11)]))
        factor=factor[18:] if all_full else factor[7:]
    updates.extend([('end15cm',np.r_[factor[:7],np.ones(4)]),('end5cm',factor[7:])])
    for region,multipliers in updates:
        b=blocks[region];levels=len(b['weight_layout']['thresholds'])+1
        mask=np.r_[0.,b['weight_layout']['thresholds']]<3.5
        adjustment=1+(multipliers[:,None]-1)*(mask[None,:] if low_only else np.ones((1,levels)))
        b['weights'][direct]=(base[region]['weights'][direct].reshape(len(multipliers),levels)*adjustment).reshape(-1)
    print('FUSION_GAIN_RESULT',direct,results[str(direct)],flush=True)

# Independently replay the final parameters through the production inverse/fusion.
actual={}
for region,s in sessions.items():s.curvature=original[region]
for direct in (0,120,240):
    s=sessions['full'];fusion=CoreDemoCurvature(original['full'],s.design,parameters,regions,direct)
    class ProductionFusion:
        def compute(self,frame,design,parameters,*,direct=0,**kwargs):return fusion.compute(frame,parameters,direct=direct)
    s.curvature=ProductionFusion();actual[str(direct)]=metrics(direct)
    for region in sessions:
        for key in ('point','end'):
            assert abs(actual[str(direct)][region][key]-results[str(direct)][region][key])<1e-8
    print('FUSION_GAIN_VERIFIED',direct,actual[str(direct)],flush=True)
def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    return x
report={'metrics':actual,'parameters':blocks,'spatial_partition_search':False,'source':str(source),'k_point':1.0,'k_end':0.5,'interp_num':256,'acceptance_passed':all(m['full']['point']<.01 and m['full']['end']<.01 and all(m[r]['point']<.1 and m[r]['end']<.1 for r in ('end15cm','end5cm')) for m in actual.values())}
if joint_head:
    report['acceptance_passed']=report['acceptance_passed'] and all(m['single_full']['point']<.01 and m['single_full']['end']<.01 for m in actual.values())
report['joint_full_head_gains']=joint_head
report['curvature_interp']=sessions['full'].config.reconstruction.curvature_interp
(out/'fusion_gain_refinement.json').write_text(json.dumps(native(report),indent=2),encoding='utf-8')
