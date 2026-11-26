"""Build a versioned candidate package from measured-data calibration results."""
import json
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1];folder=root/'runs/11B1_Cube/all18_affine_core_candidate'
record=json.loads((folder/'candidate.json').read_text(encoding='utf-8'))
assert record['all_gates_passed'] and record['all_18_measurements_retained'] and not record['measurement_imputation']
config=yaml.safe_load((root/'configs/11B1_Cube_pointwise.yaml').read_text(encoding='utf-8'))
for key in ('task','schema_version','resume','sensor_params','config'):config.pop(key,None)
config.update(core_position_calib=True,interp_num=256,curvature_interp='vector_linear',k_point=1.,k_end=.5,
    thresholds=None,end_weights=None,end_start=None,full_segmentation=[],segmentation=None,
    ideal_len=None,straight_trim=None,folder_path=None,radii=None,multi_region_calib=True,
    optimization_region='full',end5cm_independent_weights=False,end5cm_independent_weight_score=False)
blocks=record['parameters'];calibration={}
for region,block in blocks.items():
    layout=block['weight_layout'];profile=config['calibration_regions'][region]
    profile.update(thresholds=layout['thresholds'],end_weights=layout['end_weights'],end_start=layout['end_start'],weight_group_ids=layout['weight_group_ids'])
    if region!='full':profile['segmentation']=None
    calibration[region]={}
    for direct in (0,120,240):
        metric=record['metrics'][str(direct)]['single_full' if region=='full' else region]
        calibration[region][str(direct)]={'errors':metric,'objective':{'k_point':1.,'k_end':.5},'weight_kind':'per_wl_level_channel' if 'channel_weights' in block else 'per_wl_level_scalar'}
    assert 'quality_imputation' not in block
package={'package_type':'shape_sensing_algorithm_package','schema_version':2,'required_runtime_features':['core_channel_level_weights','core_signal_bias_pm'],
    'sensor_id':config['sensor_id'],'configuration':config,'core_position_parameters':blocks,'calibration':calibration,
    'multi_region':{'region_order':['full','end15cm','end5cm'],'directions':[0,120,240],'deployment_ready':False},
    'validation':{'verified':False,'calibration_thresholds_passed':True,'production_replay_pending':True,'all_18_measurements_retained':True,'measurement_imputation':False},
    'provenance':{'source_candidate':str(folder/'candidate.json'),'calibration_scripts':['tools/fit_cube_channel_bias_levels.py','tools/polish_cube_all18_candidate.py'],'data_root':config['calibration_data_root']},'resources':[]}
destination=folder/'algorithm_package_candidate.json'
destination.write_text(json.dumps(package,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(destination)
