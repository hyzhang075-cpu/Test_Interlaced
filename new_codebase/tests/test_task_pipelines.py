from types import SimpleNamespace
from pathlib import Path
from shape_sensing.pipelines import tasks
from shape_sensing.pipelines import build_session, execute_task

class Engine:
    def __init__(self, parameters=None, task='recon', region='full'):
        from types import SimpleNamespace
        self.parameters=SimpleNamespace(parameters_by_region={region:{'rot_ang':0.}})
        self.region=region
        self.config=SimpleNamespace(optimization=SimpleNamespace(k_end=0.,k_pitch=0.))
        self.dataset=SimpleNamespace(radii=[0])
        self.initialization_result=None
        self.weights=None; self.slopes=None; self.global_phases_set=0.
    calib_shapes=['circ']
    def initialize(self, verbose=False): self.initialized=True
    def load_data(self): self.loaded=True
    def reconstruct_shapes(self, kind, direct): return ([[0,0,0],[.1,0,0]],[0.,.1])
    def evaluate(self, distance, curves, direct):
        return SimpleNamespace(metrics={'avg_point_err':{'calib':.2},'avg_end_err':{'calib':.1},'avg_pitch_err':0.},details={'ideals':None})

class Vis:
    def __init__(self,recon=None): self.engine=recon; self.shown=False
    def init_visualise(self,direct): pass
    def show_reconstruction(self,*args,**kwargs): self.shown=True

def args(task,**overrides):
    values=dict(task=task,use_stored_zero=True,weights=None,slopes=None,calibration_output_dir='runs',
        direct=0,rot_ang=0.,ang_step=.1,ang_range=1.,
        total_scale=1.,imu=False,imu_com='COM1',imu_maxlen=10,demo_recorder=False,unwrap_threshold=.5,
        align_threshold=.2,direction_blend_enabled=False,direction_reference_direct=0,direction_min_curvature=.5,
        direction_min_consecutive=3,direction_consistency_threshold=.5,direction_smooth_alpha=.2,
        demo_kalman_enabled=False,demo_kalman_Rk=.1,demo_kalman_Bk=.1,anchor_constraint_enabled=False,
        anchor_translation_enabled=False,anchor_rotation_enabled=False,anchor_offset_m=0.)
    values.update(overrides); return SimpleNamespace(**values)

def deps(monkeypatch, engine_factory=lambda **kw: Engine(), artifact_root=Path(".pytest_cache/task-pipeline-runs")):
    captured={}
    def runner(jobs, *, session_factory, output_root): captured.update(jobs=jobs,factory=session_factory,output_root=output_root); return 'batch-result'
    def build_parameters(ns): return vars(ns)
    def configure(ns): return 'calibration'
    monkeypatch.setattr(tasks, 'TaskSession', lambda parameters, **options: Engine(parameters, **options))
    monkeypatch.setattr(tasks, '_visualization', lambda session, **kwargs: Vis(session))
    return dict(optimiser_factory=engine_factory,visualizer_factory=Vis,app_factory=lambda argv:None,
        configure_args=configure,parameter_builder=build_parameters,prepare_jobs=lambda ns,fn:[(args('optim',optimization_region='full',weights={0:[1.]},slopes={0:[[0.]]},global_phases_set=0.),{})],
        run_jobs=runner,compute_error_ang=lambda **kw:'init-result',optim_ang=lambda **kw:'ang-result',
        project_root=Path('project'),artifact_root=artifact_root)

def test_build_session_creates_isolated_task_engine(tmp_path, monkeypatch):
    """两次任务装配返回各自的会话和区域参数映射。"""
    engines=[]
    def factory(**kw):
        engine=Engine(); engines.append(engine); return engine
    d=deps(monkeypatch,factory,tmp_path)
    one=build_session(args('init'),parameter_builder=d['parameter_builder'],configure_args=d['configure_args'])
    two=build_session(args('init'),parameter_builder=d['parameter_builder'],configure_args=d['configure_args'])
    assert one is not two
    assert one.parameters is not two.parameters

def test_optim_and_score_use_batch_path(tmp_path,monkeypatch):
    d=deps(monkeypatch,artifact_root=tmp_path); assert execute_task(args('optim'),**d)=='batch-result'
    assert len(d['prepare_jobs'](None,None)[0])==2
    assert execute_task(args('score'),**d)=='batch-result'

def test_init_ang_recon_use_adapters_and_preserve_results(tmp_path,monkeypatch):
    d=deps(monkeypatch,artifact_root=tmp_path)
    init=execute_task(args('init'),**d)
    assert init['errors']=={0:.2}
    ang=execute_task(args('ang',ang_range=0.),**d)
    assert ang['rot_ang']==0. and ang['errors']=={0:.2}
    metrics=execute_task(args('recon'),**d)
    assert metrics['avg_point_err']=={'calib':.2}


def test_batch_resume_restores_each_region_not_last_engine(tmp_path,monkeypatch):
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
    def prepare(ns,configure):
        captured['args']=ns
        return [(args('optim', optimization_region=region, **ns.calibration_regions[region]), {})
                for region in ns.optimization_regions]
    d=deps(monkeypatch,artifact_root=tmp_path/'continued'); d['prepare_jobs']=prepare
    ns=args('optim',sensor_id='S',multi_region_calib=True,optimization_region='full',
        optimization_regions=['full','end15cm','end5cm'],calibration_regions=profiles,resume=str(source.path))
    execute_task(ns,**d)
    assert ns.weights is None and ns.slopes is None
    assert ns.calibration_regions['full']['weights'][0]==[1.]
    assert ns.calibration_regions['end15cm']['weights'][0]==[2.]
    assert ns.calibration_regions['end5cm']['weights'][0]==[3.]
    assert [ns.calibration_regions[r]['global_phases_set'] for r in profiles]==[.1,.2,.3]
    assert [ns.calibration_regions[r]['rot_ang'] for r in profiles]==[1,2,3]

def test_single_region_resume_requires_requested_direction(tmp_path,monkeypatch):
    import pytest
    from shape_sensing.artifacts.runs import create_run
    run=create_run('S','optim',{'sensor_id':'S','optimization_region':'full'},tmp_path)
    report={'direct':0,'regions':{'full':{'status':'complete','parameters':{
        'weights':{'0':[1.]},'slopes':{'0':[[1.]]}}}}}
    run.write_result({'result':['r',report]},'optim'); run.finish('succeeded')
    ns=args('optim',sensor_id='S',direct=120,resume=str(run.path),optimization_region='full')
    with pytest.raises(ValueError,match='direct=120'):
        execute_task(ns,**deps(monkeypatch,artifact_root=tmp_path/'continued'))


def test_batch_initialization_seeds_every_region_and_direction(tmp_path, monkeypatch):
    """一次 init 保存各区权重和分段斜率，三个方向 optim 分别恢复匹配初值。"""
    from copy import deepcopy
    import json
    import pytest
    from shape_sensing.artifacts.packages import export_package

    regions = ['full', 'end15cm', 'end5cm']
    directions = (0, 120, 240)
    profiles = {region: {'segmentation': [i + 1], 'weights': None, 'slopes': None}
                for i, region in enumerate(regions)}

    class InitialSession(Engine):
        def initialize(self, verbose=False):
            """生成不同区域、不同方向的可区分初值，模拟已完成的拟合。"""
            i = regions.index(self.region)
            self.dataset = SimpleNamespace(radii=directions)
            self.weights = {d: [i + 1 + d / 120] for d in directions}
            self.slopes = {d: [[i + 1 + d / 120] * (i + 1)] for d in directions}
            self.global_phases_set = i / 10

    common = dict(sensor_id='S', multi_region_calib=True, optimization_region='full',
                  optimization_regions=regions, exp_directs=list(directions))
    d = deps(monkeypatch, artifact_root=tmp_path)
    monkeypatch.setattr(tasks, 'TaskSession', InitialSession)
    displayed = []
    class BatchPlot(Vis):
        def show_reconstruction(self, direct, curves, distance, *, showanime, ideals):
            """记录方向及是否弹出交互窗口，绘图本身由专门测试覆盖。"""
            displayed.append((self.engine.region, direct, showanime))
    monkeypatch.setattr(tasks, '_visualization', lambda session, **kwargs: BatchPlot(session))

    def prepare(ns, configure):
        """按区域返回已解析的分段和初值，模拟预检后的 job 列表。"""
        if not ns.multi_region_calib:
            ns.segmentation = ns.calibration_regions[ns.optimization_region]['segmentation']
            return [(ns, {})]
        return [(args(ns.task, optimization_region=region, direct=ns.direct, **ns.calibration_regions[region]), {})
                for region in ns.optimization_regions]

    d['prepare_jobs'] = prepare
    init = execute_task(args('init', calibration_regions=deepcopy(profiles), **common), **d)
    assert list(init['regions']) == regions
    assert displayed == [(region, direct, False) for region in regions for direct in directions]
    source, = (tmp_path / 'S').glob('*-init')
    snapshot = json.loads((source / 'initialization.json').read_text(encoding='utf-8'))
    assert 'global_phases_set' not in snapshot
    package = export_package([source], tmp_path / 'package.json')
    assert set(package['initialization']) == set(regions)

    def run(jobs, **kwargs):
        """校验优化真正收到的区域、方向和分段初值。"""
        assert len(jobs) == 3
        for i, (job, _) in enumerate(jobs):
            assert job.optimization_region == regions[i]
            expected = i + 1 + job.direct / 120
            assert job.weights[job.direct] == [expected]
            assert job.slopes[job.direct] == [[expected] * (i + 1)]
            assert job.global_phases_set == i / 10

    d['run_jobs'] = run
    for direction in directions:
        execute_task(args('optim', direct=direction, resume=str(source),
                          calibration_regions=deepcopy(profiles), **common), **d)

    wrong_profiles = deepcopy(profiles)
    wrong_profiles['end5cm']['segmentation'] = [1, 2]
    with pytest.raises(ValueError, match='end5cm.*segmentation'):
        execute_task(args('optim', resume=str(source), calibration_regions=wrong_profiles, **common), **d)


    def run_single(jobs, **kwargs):
        """多区域快照可续算指定区域，不要求批量模式。"""
        (job, _), = jobs
        assert job.optimization_region == 'end15cm'
        assert job.weights[240] == [4.]
        assert job.slopes[240] == [[4., 4.]]
        assert job.global_phases_set == .1

    d['run_jobs'] = run_single
    single = dict(common, multi_region_calib=False, optimization_region='end15cm')
    execute_task(args('optim', direct=240, resume=str(source),
                      calibration_regions=deepcopy(profiles), **single), **d)


    d['artifact_root'] = tmp_path / 'single'
    single_init = execute_task(args('init', calibration_regions=deepcopy(profiles), **single), **d)
    assert 'regions' not in single_init and single_init['global_phases_set'] == .1
    single_source, = (d['artifact_root'] / 'S').glob('*-init')
    execute_task(args('optim', direct=240, resume=str(single_source),
                      calibration_regions=deepcopy(profiles), **single), **d)
    with pytest.raises(ValueError, match='单区域初始化快照'):
        execute_task(args('optim', resume=str(single_source), calibration_regions=deepcopy(profiles),
                          **dict(common, optimization_region='end15cm')), **d)
