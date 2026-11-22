"""新库独立性验收：真实实现归属、安装隔离和禁止旧模块依赖。"""
import ast
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'sensor','initialiser','reconstructor','optimiser','visualisor','demonstrator',
             'proto_demonstrator','IMU','main','calibration_batch','calibration_regions',
             'calibration_phase_reference','entrance_anchor','zero_reference','utils','tools'}


def test_standalone_package_has_no_imports_from_old_repository_modules():
    bad=[]
    for path in (ROOT/'shape_sensing').rglob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                imports=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.level==0:
                imports=[node.module or '']
            else: continue
            for name in imports:
                if name.split('.')[0] in FORBIDDEN:
                    bad.append(f'{path.relative_to(ROOT)}:{node.lineno}: {name}')
    assert not bad, '\n'.join(bad)


def test_numerical_implementations_live_in_their_responsible_modules():
    from shape_sensing.input_preprocessing.compat import Sensor
    from shape_sensing.initialization.compat import Initialiser
    from shape_sensing.reconstruction.compat import Reconstructor
    from shape_sensing.optimization.compat import Optimiser
    from shape_sensing.reconstruction.integration import ShapeIntegrator
    cases=[(Sensor.read_data_FBG,'input_preprocessing'),
           (Initialiser.rot_seg_fit,'initialization'),
           (Initialiser._calculate_curvature_and_angle,'curvature'),
           (Reconstructor._Strain_2_Cur,'curvature'),
           (Reconstructor._assemble_multiregion_parameters,'curvature'),
           (Reconstructor._parallel_frame,'reconstruction'),
           (Optimiser.optim_coeff,'optimization'),
           (Optimiser._compute_error,'evaluation'),
           (Optimiser._construct_ideal_circ,'evaluation'),
           (Reconstructor._construct_ideal_magnitude_for_recon,'evaluation')]
    for function,owner in cases:
        path=Path(inspect.getsourcefile(function)).resolve()
        assert ROOT/'shape_sensing'/owner in path.parents, (function,path,owner)
    assert ShapeIntegrator.__module__.startswith('shape_sensing.reconstruction')


def test_installed_wheel_computes_without_old_files(tmp_path):
    """离线构建并安装 wheel，禁用旧模块后实际计算，按 UTF-8 校验中文帮助。"""
    wheels=tmp_path/'wheels'; wheels.mkdir()
    build=subprocess.run([sys.executable,'-m','pip','wheel',str(ROOT),'--no-deps',
                          '--no-build-isolation','--wheel-dir',str(wheels)],
                         cwd=tmp_path,capture_output=True,text=True,timeout=180)
    assert build.returncode==0, build.stdout+'\n'+build.stderr
    paths=list(wheels.glob('*.whl')); assert len(paths)==1
    # 重命名后的发布物只包含compat入口，防止旧build残留再次进入wheel。
    import zipfile
    renamed = {'input_preprocessing': 'sensor', 'initialization': 'fitting',
               'reconstruction': 'engine', 'optimization': 'engine',
               'runtime': 'demonstrator'}
    with zipfile.ZipFile(paths[0]) as archive:
        members = set(archive.namelist())
    for module, old_name in renamed.items():
        assert f'shape_sensing/{module}/compat.py' in members
        assert f'shape_sensing/{module}/{old_name}.py' not in members
    installed=tmp_path/'installed'
    install=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
                            '--target',str(installed),str(paths[0])],
                           cwd=tmp_path,capture_output=True,text=True,timeout=120)
    assert install.returncode==0, install.stdout+'\n'+install.stderr
    script=r'''
import sys, importlib.abc, pathlib, runpy
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
blocked={'sensor','initialiser','reconstructor','optimiser','visualisor','demonstrator',
'proto_demonstrator','IMU','main','calibration_batch','calibration_regions',
'calibration_phase_reference','entrance_anchor','zero_reference','utils','tools'}
class DenyLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Old repository import: '+fullname)
sys.meta_path.insert(0,DenyLegacy())
import numpy as np
from shape_sensing.initialization.compat import Initialiser
from shape_sensing.optimization.compat import Optimiser
from shape_sensing.reconstruction import ShapeIntegrator
from shape_sensing.configuration import parameter_catalog
assert len(parameter_catalog())>=110
angles=np.tile(np.arange(3)*2*np.pi/3,(6,1)).T
strain=-.0001*10*np.cos(angles-.35)
obj=Initialiser.__new__(Initialiser); obj.R=.0001; obj.num_channels_recon=3
magnitude,direction=obj._calculate_curvature_and_angle(strain,angles)
np.testing.assert_allclose(magnitude,10.,atol=1e-12)
np.testing.assert_allclose(direction,.35,atol=1e-12)
distance=np.linspace(0.,.04,6)
integrator=ShapeIntegrator(frame_rule='TNN')
position,tangent,y,z=integrator.integrate(magnitude,np.asarray(direction),distance,np.array([1.,0.,0.]))
assert position.shape==(6,3) and np.isfinite(position).all()
assert np.linalg.norm(position[-1]-position[0])>0
ideal=Optimiser.__new__(Optimiser)._construct_ideal_shape('circ',distance,.04,.05,0.)
expected=np.column_stack((.05*np.sin(distance/.05),.05*(1-np.cos(distance/.05)),np.zeros(6)))
np.testing.assert_allclose(ideal,expected,atol=1e-12)
from shape_sensing.common.paths import resource_path
assert resource_path('resources', 'sensor_template.yaml').is_file()
assert pathlib.Path(sys.argv[1]) in pathlib.Path(sys.modules['shape_sensing'].__file__).parents
print('INSTALLED_NUMERICAL_COMPUTATION_OK')
sys.modules['PyQt5']=None; sys.modules['vtk']=None
sys.argv=['shape_sensing','--help']
runpy.run_module('shape_sensing',run_name='__main__')
'''
    run=subprocess.run([sys.executable,'-I','-X','utf8','-c',script,str(installed)],cwd=tmp_path,
                       capture_output=True,encoding='utf-8',timeout=60)
    assert run.returncode==0,run.stdout+'\n'+run.stderr
    assert 'INSTALLED_NUMERICAL_COMPUTATION_OK' in run.stdout
    assert 'usage:' in run.stdout.lower()


def test_public_create_engine_uses_validated_namespace_and_preserves_context(tmp_path, monkeypatch):
    """程序化入口复用真实配置解析，在构造数据对象前截获参数。"""
    from types import SimpleNamespace
    import yaml
    from shape_sensing.pipelines import session
    from shape_sensing.api import create_engine
    config={'sensor_id':'API-CHECK','R':.0001,'start_value':0.,'end_value':10.,
            'samp_dist_preset':[0.,20.,110.,130.,190.,290.,390.,490.],
            'full_segmentation':[2,2,2,2],
            'calibration_regions':{'full':{'length_m':None,'folder_path':'data',
                'radii':{'circ':[0,30]},'thresholds':[1.],
                'end_weights':False,'end_start':-1,'weights':None,'slopes':None}}}
    path=tmp_path/'sensor.yaml'; path.write_text(yaml.safe_dump(config),encoding='utf-8')
    received={}
    class FakeSession(SimpleNamespace):
        def initialize(self): self.initialized = True
    def factory(parameters, **session_options):
        received.update(parameters)
        received['session_options'] = session_options
        return FakeSession(calibration_context=None)
    monkeypatch.setattr(session,'TaskSession',factory)
    engine=create_engine(path,task='init',max_iter=3,full_segmentation=[2,2,2,2])
    assert received['sensor_id']=='API-CHECK' and received['max_iter']==3
    assert engine.calibration_context['optimization_region']=='full'


def test_installed_usage_paths_follow_working_directory_not_package_directory(tmp_path, monkeypatch):
    """数据和运行结果不得默认写入安装目录/site-packages。"""
    import shape_sensing.cli as cli
    import shape_sensing.pipelines as pipelines
    monkeypatch.chdir(tmp_path)
    args=cli.parse_main_args(['--task','init','--calibration_data_root','calibration',
                              '--folder_path','data'])
    cli.configure_calibration_args(args)
    assert Path(args.folder_path)==tmp_path/'calibration'/'data'
    captured={}
    def execute(args,**kwargs): captured.update(kwargs)
    monkeypatch.setattr(pipelines,'execute_task',execute)
    monkeypatch.setattr(cli,'parse_main_args',lambda: args)
    cli.main()
    assert Path(captured['project_root'])==tmp_path
