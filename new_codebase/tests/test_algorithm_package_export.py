from pathlib import Path
import json
import pytest
from shape_sensing.artifacts.runs import create_run
from shape_sensing.artifacts.packages import export_package, load_algorithm_package

def test_export_uses_run_snapshots_and_marks_unverified(tmp_path):
    config={'sensor_id':'S','R':.001,'samp_dist_preset':[1,2],'direct':0,'optimization_region':'full','max_iter':7}
    run=create_run('S','optim',config,tmp_path/'runs')
    report={'direct':0,'regions':{'full':{'status':'complete','parameters':{'weights':{'0':[1.,2.]},'slopes':{'0':[[1.]]}}}}}
    run.write_result({'result':['results.json',report]},'optim')
    run.finish('succeeded')
    output=tmp_path/'package.json'
    package=export_package([run.path],output)
    loaded=load_algorithm_package(output)
    assert loaded['calibration']['full']['0']['weights']=={0:[1.,2.]}
    assert loaded['configuration']['max_iter']==7
    assert loaded['validation']=={'deployment_ready':False,'verified':False}

def test_export_rejects_cross_sensor_and_missing_slope(tmp_path):
    paths=[]
    for sensor in ('A','B'):
        run=create_run(sensor,'optim',{'sensor_id':sensor,'R':.1,'direct':0},tmp_path)
        run.write_result({'result':['r',{'direct':0,'regions':{'full':{'status':'complete','parameters':{'weights':{'0':[1]},'slopes':{'0':[[1]]}}}}}]},'optim')
        run.finish('succeeded')
        paths.append(run.path)
    with pytest.raises(ValueError,match='不同 sensor_id'): export_package(paths,tmp_path/'no.json')
    one=create_run('A','optim',{'sensor_id':'A','direct':0},tmp_path)
    one.write_result({'result':['r',{'direct':0,'regions':{'full':{'status':'complete','parameters':{'weights':{'0':[1]}}}}}]},'optim')
    one.finish('succeeded')
    with pytest.raises(ValueError,match='缺少 slopes'): export_package([one.path],tmp_path/'bad.json')


def test_main_loads_new_package_for_selected_direction(tmp_path):
    from main import parse_main_args
    config={'sensor_id':'S','R':.001,'samp_dist_preset':[1,2],'direct':0}
    run=create_run('S','optim',config,tmp_path/'runs')
    report={'direct':0,'regions':{'full':{'status':'complete','parameters':{'weights':{'0':[1.,2.]},'slopes':{'0':[[1.]]}}}}}
    run.write_result({'result':['r',report]},'optim')
    run.finish('succeeded')
    path=tmp_path/'package.json'; export_package([run.path],path)
    args=parse_main_args(['--task','recon','--sensor_params',str(path)])
    assert args.sensor_id=='S' and args.weights[0]==[1.,2.] and args.slopes[0]==[[1.]]



def test_regions_and_directions_do_not_overwrite_and_duplicate_conflicts(tmp_path):
    paths=[]
    for region,direct in [('full',0),('end15cm',0),('full',120)]:
        cfg={'sensor_id':'S','R':.1,'samp_dist_preset':[1,2],'direct':direct,'optimization_region':region}
        run=create_run('S','optim',cfg,tmp_path/region/str(direct))
        report={'direct':direct,'regions':{region:{'status':'complete','parameters':{'weights':{str(direct):[float(direct+1)]},'slopes':{str(direct):[[2.]]}}}}}
        run.write_result({'result':['r',report]},'optim'); run.finish('succeeded'); paths.append(run.path)
    package=export_package(paths,tmp_path/'multi.json')
    assert set(package['calibration'])=={'full','end15cm'}
    assert set(package['calibration']['full'])=={'0','120'}
    with pytest.raises(ValueError,match='重复结果'): export_package([paths[0],paths[0]],tmp_path/'dup.json')


def test_export_carries_zero_reference_and_init_is_not_final(tmp_path):
    zero=tmp_path/'zero.json'; zero.write_text('{"version":1}',encoding='utf-8')
    run=create_run('S','init',{'sensor_id':'S','R':.1,'zero_reference_path':str(zero),'optimization_region':'full'},tmp_path/'runs')
    run.write_result({'weights':None,'slopes':[[0.]],'global_phases_set':0.0},'initialization'); run.finish('succeeded')
    target=tmp_path/'published.json'; package=export_package([run.path],target)
    assert package['calibration']=={} and package['scope'][0]['final_calibration'] is False
    assert Path(target.parent/package['configuration']['zero_reference_path']).read_text(encoding='utf-8')=='{"version":1}'
    loaded=load_algorithm_package(target)
    assert loaded['configuration']['zero_reference_path']==str((target.parent/package['configuration']['zero_reference_path']).resolve())


def test_four_core_package_selects_the_optimized_direction_from_each_run(tmp_path):
    from main import parse_main_args

    paths = []
    directions = (0, 120, 240)
    for direct in directions:
        config = dict(sensor_id='S', core_position_calib=True, direct=direct)
        run = create_run('S', 'optim', config, tmp_path / 'runs')
        block = dict(core_position=[{'wl_id': 1}, {'wl_id': 2}],
                     core_position_metadata=dict(schema_version=6,
                                                 calibration=dict(directions_deg=list(directions))),
                     weight_layout=dict(thresholds=[1.0]),
                     weights={d: [0.5, 0.6] if d == direct else [1.0, 1.0] for d in directions})
        result = dict(core_position_parameters={'full': block},
                      reports={'full': {'stages': {'weights': {'status': 'converge', 'final_loss': 0.1}}}})
        run.write_result({'result': result}, 'optim')
        run.finish('succeeded')
        paths.append(run.path)
    output = tmp_path / 'package.json'
    with pytest.raises(ValueError, match='缺少优化方向'):
        export_package(paths[:1], output)
    package = export_package(paths, output)
    expected = {str(d): [0.5, 0.6] for d in directions}
    assert package['core_position_parameters']['full']['weights'] == expected
    args = parse_main_args(['--task', 'recon', '--sensor_params', str(output), '--direct', '120'])
    assert args.core_position_calib
    assert args.core_position_parameters['full']['weights'] == expected
