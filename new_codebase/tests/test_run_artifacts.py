from pathlib import Path
import json
import numpy as np
import pytest
from shape_sensing.artifacts.runs import create_run, load_resume

def test_runs_are_unique_snapshot_and_status(tmp_path):
    config={'sensor_id':'s','R':.1,'weights':None}
    a=create_run('s','init',config,tmp_path); b=create_run('s','init',config,tmp_path)
    assert a.path!=b.path
    a.write_result({'weights':np.array([1.,2.]),'slopes':[[3.]]},'initialization')
    config['R']=9
    assert json.loads((a.path/'config_resolved.json').read_text())['R']==.1
    a.finish('succeeded')
    manifest=json.loads((a.path/'manifest.json').read_text())
    assert manifest['status']=='succeeded'
    assert set(manifest['code'])=={'commit','branch','working_tree_dirty'}
    assert manifest['code']['commit']

def test_resume_checks_sensor_geometry_and_requires_slope_pair(tmp_path):
    run=create_run('s','optim',{'sensor_id':'s','R':.1,'optimization_region':'full'},tmp_path)
    run.write_result({'weights':{'0':[1]},'slopes':{'0':[[2]]}},'optim')
    assert load_resume(run.path,{'sensor_id':'s','R':.1,'optimization_region':'full'})['weights']
    with pytest.raises(ValueError,match='R'):
        load_resume(run.path,{'sensor_id':'s','R':.2,'optimization_region':'full'})
    run.write_result({'weights':{'0':[1]}},'optim')
    with pytest.raises(ValueError,match='slopes'):
        load_resume(run.path,{'sensor_id':'s','R':.1,'optimization_region':'full'})

def test_resume_requires_explicit_result_path(tmp_path):
    with pytest.raises(FileNotFoundError): load_resume(tmp_path/'absent.json',{})


def test_resume_rejects_effective_top_level_threshold_and_full_layout_overrides(tmp_path):
    config={'sensor_id':'s','optimization_region':'full','thresholds':[1.,2.],
            'full_segmentation':[2,2], 'calibration_regions':{'full':{'thresholds':[1.,2.], 'segmentation':[2,2]}}}
    run=create_run('s','optim',config,tmp_path)
    run.write_result({'weights':{0:[1]},'slopes':{0:[[1]]}},'optim')
    run.finish('succeeded')
    for changed in ({'thresholds':[9.,10.]},{'full_segmentation':[1,3]}):
        candidate=dict(config); candidate.update(changed)
        with pytest.raises(ValueError,match='resume 区域配置不兼容'):
            load_resume(run.path,candidate)
