"""程序化 API 的显式覆盖必须保留 CLI/续算同等优先级。"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _config(tmp_path, task=None, profile_seeds=True):
    data={"sensor_id":"API-OVERRIDE","R":0.0001,"start_value":0.0,"end_value":10.0,
          "samp_dist_preset":[0.0,20.0,110.0,130.0,190.0,290.0,390.0,490.0],
          "full_segmentation":[2,2,2,2],"global_phases_set":0.25,
          "calibration_regions":{"full":{"length_m":None,"folder_path":"data",
              "radii":{"circ":[0,30]},"thresholds":[1.0],"end_weights":False,
              "end_start":-1,"weights":{"invalid-profile-seed":1} if profile_seeds else None,
              "slopes":{"invalid-profile-seed":1} if profile_seeds else None}}}
    if task is not None: data['task']=task
    path=tmp_path/'sensor.yaml'
    path.write_text(yaml.safe_dump(data),encoding='utf-8')
    return path


def _capture_configure(monkeypatch):
    import shape_sensing.cli as cli
    from shape_sensing.pipelines import session
    from shape_sensing.api import create_engine
    captured={}
    real_configure=cli.configure_calibration_args
    def capture(args):
        context=real_configure(args)
        captured['args']=args
        captured['context']=context
        return context
    monkeypatch.setattr(cli,'configure_calibration_args',capture)
    class FakeSession(SimpleNamespace):
        def initialize(self): self.initialized = True
    monkeypatch.setattr(session,'TaskSession',lambda parameters, **kw: FakeSession(**parameters, **kw))
    return create_engine,captured


def test_none_seed_overrides_and_phase_are_marked_explicit(tmp_path,monkeypatch):
    path=_config(tmp_path,task='init')
    create_engine,captured=_capture_configure(monkeypatch)
    engine=create_engine(path,weights=None,slopes=None,global_phases_set=0.0)
    args=captured['args']
    assert args.weights is None and args.slopes is None
    assert args.global_phases_set==0.0
    assert {'weights','slopes','global_phases_set'} <= set(args._explicit_cli_parameters)
    assert {'weights','slopes'} <= set(args._explicit_seed_parameters)
    assert engine.calibration_context is captured['context']


def test_explicit_region_scope_is_not_collapsed_to_default_direct(tmp_path,monkeypatch):
    path=_config(tmp_path,profile_seeds=False)
    create_engine,captured=_capture_configure(monkeypatch)
    create_engine(path,task='recon',exp_directs=[0,120,240],global_phases_set=0.8)
    args=captured['args']
    assert args.exp_directs==[0,120,240]
    assert args.global_phases_set==0.8
    assert {'exp_directs','global_phases_set'} <= set(args._explicit_cli_parameters)


def test_resume_is_rejected_by_engine_factory_instead_of_silently_ignored(tmp_path):
    from shape_sensing.api import create_engine
    with pytest.raises(ValueError,match='resume.*task pipeline'):
        create_engine(_config(tmp_path),resume='saved-run')


def test_relative_prototype_model_path_uses_working_directory_only(monkeypatch):
    from shape_sensing.runtime import prototype
    calls=[]
    monkeypatch.setattr(prototype.os.path,'exists',lambda path: calls.append(path) or False)
    instance=prototype.Demonstrator.__new__(prototype.Demonstrator)
    instance.body_model_path='model.stl'
    instance.add_model()
    assert calls==[str(Path.cwd()/'model.stl')]
