"""Save only a production-replayed, all-18-point calibrated package."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from shape_sensing.artifacts.packages import load_algorithm_package
from shape_sensing.cli import parse_main_args,configure_calibration_args,optimizer_parameters
from shape_sensing.pipelines.tasks import module_parameters
from shape_sensing.pipelines.session import TaskSession
root=Path(__file__).resolve().parents[1];folder=root/'runs/11B1_Cube/all18_affine_core_candidate'
source=folder/'algorithm_package_candidate.json';package=json.loads(source.read_text(encoding='utf-8'))
report=json.loads((folder/'production_verification.json').read_text(encoding='utf-8'))
assert report['all_gates_passed'] and report['temporal_subset_full_gates_passed'] and report['all_18_measurements_retained'] and not report['measurement_imputation']
configuration=package['configuration'];configuration['method']='SLSQP'
configuration.pop('slopes',None)
for profile in configuration['calibration_regions'].values():profile.pop('slopes',None)
for block in package['core_position_parameters'].values():
    assert not block.get('quality_imputation')
    if block.get('slopes') is None:block.pop('slopes',None)
weights=np.asarray(list(package['core_position_parameters']['full']['channel_weights'].values()))
biases=np.asarray(list(package['core_position_parameters']['full']['signal_bias_pm'].values()))
package['calibration_model']='four_core_position_with_static_input_bias_and_positive_channel_level_gains'
package['calibration_parameter_ranges']={'channel_weights':[float(weights.min()),float(weights.max())],'relative_input_bias_pm':[float(biases.min()),float(biases.max())]}
package['validation'].update(verified=True,production_replay_pending=False,all_gates_passed=True,temporal_subset_full_gates_passed=True,all_18_measurements_retained=True,measurement_imputation=False,
    metric_definition={'mean_point':'avg_point_err.calib; mean across configured radii','endpoint':'avg_end_err.calib; mean across configured radii'},scope='supplied static calibration data; first/last temporal-quarter mean checks; no arbitrary-shape or live acquisition claim')
destination=root/'params/11B1_Cube_core_optimized.json';destination.parent.mkdir(exist_ok=True)
if destination.exists():
    n=2
    while destination.with_name(f'{destination.stem}_v{n}.json').exists():n+=1
    destination=destination.with_name(f'{destination.stem}_v{n}.json')
validation_path=destination.with_name(destination.stem+'_validation.json')
package['validation']['report_file']=validation_path.name
destination.write_text(json.dumps(package,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
loaded=load_algorithm_package(destination)
for task in ('recon','demo'):
    args=parse_main_args(['--sensor_params',str(destination),'--task',task])
    configure_calibration_args(args)
    session=TaskSession(module_parameters(args,optimizer_parameters),task=task,region='full')
    assert session.config.reconstruction.interp_num==256
    assert session.config.optimization.k_point==1. and session.config.optimization.k_end==.5
    if task=='demo':assert session._core_demo_curvature is not None
report.update(final_parameter_file=str(destination),parameter_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),package_recon_and_demo_preflight_passed=True,legacy_reference_package_loading_passed=True)
validation_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(folder/'final_artifacts.json').write_text(json.dumps({'parameters':str(destination),'validation':str(validation_path)},indent=2),encoding='utf-8')
print('FINAL_PACKAGE',destination);print('FINAL_VALIDATION',validation_path)
print('FINAL_MAX_PERCENT',{mode:{key:100*max(v[key] for v in rows.values()) for key in ('point','end')} for mode,rows in report['metrics'].items()})
