from types import SimpleNamespace
from pathlib import Path
from shape_sensing.context import LegacyContext
from shape_sensing.pipelines import build_context, execute_task

class Engine:
    curvature_interp='hermite'
    calib_shapes=['circ']
    def reconstruct_shapes(self, **kwargs): return ([[0,0,0],[.1,0,0]],[0.,.1])
    def _compute_error(self, interp_dist, rs):
        self.avg_point_err={'calib':.2}; self.avg_end_err={'calib':.1}; self.avg_pitch_err=0.; self.avg_radius_err=0.
    def _print_errors(self): self.printed=True

class Vis:
    def __init__(self,recon): self.engine=recon; self.shown=False
    def show_reconstruction(self,**kwargs): self.shown=True

def args(task,**overrides):
    values=dict(task=task,use_stored_zero=True,weights=None,slopes=None,calibration_output_dir='runs',
        direct=0,rot_ang=0.,ang_step=.1,ang_range=1.,
        total_scale=1.,imu=False,imu_com='COM1',imu_maxlen=10,demo_recorder=False,unwrap_threshold=.5,
        align_threshold=.2,direction_blend_enabled=False,direction_reference_direct=0,direction_min_curvature=.5,
        direction_min_consecutive=3,direction_consistency_threshold=.5,direction_smooth_alpha=.2,
        demo_kalman_enabled=False,demo_kalman_Rk=.1,demo_kalman_Bk=.1,anchor_constraint_enabled=False,
        anchor_translation_enabled=False,anchor_rotation_enabled=False,anchor_offset_m=0.)
    values.update(overrides); return SimpleNamespace(**values)

def deps(engine_factory=lambda **kw: Engine(), artifact_root=Path(".pytest_cache/task-pipeline-runs")):
    captured={}
    def runner(jobs, factory, output_root): captured.update(jobs=jobs,factory=factory,output_root=output_root); return 'batch-result'
    return dict(optimiser_factory=engine_factory,visualizer_factory=Vis,app_factory=lambda argv:None,
        configure_args=lambda ns:'calibration',parameter_builder=lambda ns:{},prepare_jobs=lambda ns,fn:['job'],
        run_jobs=runner,compute_error_ang=lambda **kw:'init-result',optim_ang=lambda **kw:'ang-result',
        project_root=Path('project'),artifact_root=artifact_root)

def test_build_context_creates_isolated_task_engine(tmp_path):
    engines=[]
    def factory(**kw):
        engine=Engine(); engines.append(engine); return engine
    d=deps(factory,tmp_path)
    one=build_context(args('init'),optimiser_factory=d['optimiser_factory'],parameter_builder=d['parameter_builder'],configure_args=d['configure_args'])
    two=build_context(args('init'),optimiser_factory=d['optimiser_factory'],parameter_builder=d['parameter_builder'],configure_args=d['configure_args'])
    assert one.engine is not two.engine and len(engines)==2

def test_optim_and_score_use_batch_path(tmp_path):
    d=deps(artifact_root=tmp_path); assert execute_task(args('optim'),**d)=='batch-result'
    assert d['prepare_jobs'](None,None)==['job']
    assert execute_task(args('score'),**d)=='batch-result'

def test_init_ang_recon_use_adapters_and_preserve_results(tmp_path):
    d=deps(artifact_root=tmp_path)
    assert execute_task(args('init'),**d)=='init-result'
    assert execute_task(args('ang'),**d)=='ang-result'
    metrics=execute_task(args('recon'),**d)
    assert metrics['avg_point_err']=={'calib':.2}


def test_batch_resume_restores_each_region_not_last_engine(tmp_path):
    import json
    from shape_sensing.artifacts.runs import create_run
    profiles={name:{'thresholds':[1.], 'segmentation':[2], 'weights':None,'slopes':None} for name in ('full','end15cm','end5cm')}
    source=create_run('S','optim',{'sensor_id':'S','optimization_region':'full','multi_region_calib':True,
        'optimization_regions':['full','end15cm','end5cm'],'calibration_regions':profiles},tmp_path/'source')
    report={'direct':0,'regions':{name:{'status':'complete','parameters':{
        'weights':{'0':[float(i+1)]},'slopes':{'0':[[float(i+1)]]},
        'global_phases_set':(i+1)/10,'rot_ang':i+1}} for i,name in enumerate(profiles)}}
    source.write_result({'result':['results.json',report]},'optim'); source.finish('succeeded')
    captured={}
    def prepare(ns,configure): captured['args']=ns; return ['batch']
    d=deps(artifact_root=tmp_path/'continued'); d['prepare_jobs']=prepare
    ns=args('optim',sensor_id='S',multi_region_calib=True,optimization_region='full',
        optimization_regions=['full','end15cm','end5cm'],calibration_regions=profiles,resume=str(source.path))
    execute_task(ns,**d)
    assert ns.weights is None and ns.slopes is None
    assert ns.calibration_regions['full']['weights'][0]==[1.]
    assert ns.calibration_regions['end15cm']['weights'][0]==[2.]
    assert ns.calibration_regions['end5cm']['weights'][0]==[3.]
    assert [ns.calibration_regions[r]['global_phases_set'] for r in profiles]==[.1,.2,.3]
    assert [ns.calibration_regions[r]['rot_ang'] for r in profiles]==[1,2,3]

def test_single_region_resume_requires_requested_direction(tmp_path):
    import pytest
    from shape_sensing.artifacts.runs import create_run
    run=create_run('S','optim',{'sensor_id':'S','optimization_region':'full'},tmp_path)
    report={'direct':0,'regions':{'full':{'status':'complete','parameters':{
        'weights':{'0':[1.]},'slopes':{'0':[[1.]]}}}}}
    run.write_result({'result':['r',report]},'optim'); run.finish('succeeded')
    ns=args('optim',sensor_id='S',direct=120,resume=str(run.path),optimization_region='full')
    with pytest.raises(ValueError,match='direct=120'):
        execute_task(ns,**deps(artifact_root=tmp_path/'continued'))
