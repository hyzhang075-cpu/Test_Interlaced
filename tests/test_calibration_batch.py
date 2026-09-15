"""Batch calibration isolation, persistence, and actual gating diagnostics."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser_args(*extra):
    import main
    assert hasattr(main, 'build_argument_parser'), 'missing portable parser builder'
    return main.build_argument_parser().parse_args(['--task', 'optim', *extra])


def runner():
    assert importlib.util.find_spec('calibration_batch'), 'missing calibration batch runner'
    import calibration_batch
    return calibration_batch


def test_parser_contains_sensor_region_configuration_and_accepts_none():
    args = parser_args('--weights', 'None', '--slopes', 'None', '--direct', '0')
    assert args.weights is None and args.slopes is None
    assert args.multi_region_calib is False
    assert args.optimization_regions == ['full', 'end15cm', 'end5cm']
    assert args.calibration_regions['end5cm']['length_m'] == .05
    assert args.full_segmentation == [2, 3, 5, 2, 3, 3, 2, 2]
    assert args.calibration_data_root == '0908-06B6'


def test_parser_parses_numeric_optimizer_bounds_from_cli():
    args = parser_args('--weight_bounds', '(0.4, 3.0)', '--slope_upper_bound', '80')
    assert args.weight_bounds == (0.4, 3.0)
    assert args.slope_upper_bound == 80.0


def test_full_segmentation_cli_override_does_not_conflict_with_parser_default():
    import main
    args = parser_args('--task', 'init', '--segmentation', '[2,3,5,2,3,7]')
    context = main.configure_calibration_args(args)
    assert args.segmentation == [2,3,5,2,3,7]
    assert context['segmentation'] == [2,3,5,2,3,7]


def test_two_explicit_full_segment_options_must_agree():
    import main
    args = parser_args('--task', 'init', '--full_segmentation', '[2,3,5,2,3,3,4]',
                       '--segmentation', '[2,3,5,2,3,7]')
    with pytest.raises(ValueError, match='不一致'):
        main.configure_calibration_args(args)


def test_all_regions_prepared_independently_without_mutating_parser_defaults():
    import main
    args = parser_args('--multi_region_calib', 'True')
    jobs = runner().prepare_calibration_jobs(args, main.configure_calibration_args)
    assert [j.optimization_region for j, _ in jobs] == ['full', 'end15cm', 'end5cm']
    assert [j.segmentation for j, _ in jobs] == [[2,3,5,2,3,3,2,2], [3,2,2], [2,2]]
    assert [j.radii['circ'][-1] for j, _ in jobs] == [200, 30, 15]
    assert args.folder_path is None and args.straight_trim is None
    assert args.calibration_regions['end5cm']['segmentation'] == [2, 2]


def test_batch_rejects_one_folder_override_for_multiple_lengths():
    import main
    args = parser_args('--multi_region_calib', 'True', '--folder_path', 'wrong-for-all')
    with pytest.raises(ValueError, match='folder_path'):
        runner().prepare_calibration_jobs(args, main.configure_calibration_args)


def test_explicit_none_resets_region_seed_but_default_uses_it():
    import main
    args = parser_args('--task', 'init', '--multi_region_calib', 'True', '--optimization_region', 'end5cm')
    level_count = len(args.calibration_regions['end5cm']['thresholds']) + 1
    seeds = {d: [1.] * level_count for d in [0, 120, 240]}
    args.calibration_regions['end5cm']['weights'] = seeds
    main.configure_calibration_args(args)
    assert args.weights[0].tolist() == [1.] * level_count
    reset = parser_args('--task', 'init', '--multi_region_calib', 'True', '--optimization_region', 'end5cm', '--weights', 'None')
    reset.calibration_regions['end5cm']['weights'] = seeds
    main.configure_calibration_args(reset)
    assert reset.weights is None


def test_batch_saves_completed_regions_before_later_failure(tmp_path):
    import main
    args = parser_args('--multi_region_calib', 'True', '--calibration_output_dir', str(tmp_path))
    jobs = runner().prepare_calibration_jobs(args, main.configure_calibration_args)
    seen = []
    class Optim:
        def __init__(self, **params):
            self.direct = params['direct']
            self.weights = {0: np.ones(6)}
            self.slopes = {0: np.ones((6, len(params['segmentation'])))}
            self.global_phases_set = .3
            self.rot_ang = 0
            self.avg_point_err = {'calib': .01}
            self.avg_end_err = {'calib': .02}
            self.avg_pitch_err = self.avg_radius_err = 0
            self.point_errors_avg = {'circ': {200: .01}}
            self.end_errors = {'circ': {200: .02}}
            self.k_point, self.k_end, self.k_pitch, self.k_radius = 1, .3, 1, 1
            self.weight_bounds = [(0.6, 5.0)]
            self.slope_upper_bound = 50.0
        def optim_coeff(self, verbose=True):
            region = self.calibration_context['optimization_region']
            seen.append(region)
            if region == 'end15cm':
                raise RuntimeError('second region failed')
            return {'stages': {'slopes': {'status': 'converge'}}}
    with pytest.raises(RuntimeError, match='second region'):
        runner().run_calibration_jobs(jobs, Optim, output_root=tmp_path)
    output_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert output_dir.name.startswith('06B6_NeTi_v2_centraltube-')
    assert '-direct0-' in output_dir.name
    report = json.loads(next(tmp_path.rglob('results.json')).read_text(encoding='utf-8'))
    assert seen == ['full', 'end15cm']
    assert report['regions']['full']['status'] == 'complete'
    assert report['regions']['full']['parameters']['weights']['0'] == [1.] * 6
    assert report['regions']['full']['errors']['normal_weighted_loss'] == pytest.approx(.016)
    assert report['regions']['end15cm']['status'] == 'failed'
    assert report['regions']['end5cm']['status'] == 'pending'


def test_parameter_diagnostics_report_bounds_without_calling_them_divergence():
    diagnostics = runner().diagnose_parameter_bounds(
        weights=np.array([0.6, 1.0, 5.0]),
        slopes=np.array([[-75.0, 50.0], [10.0, -75.0]]),
        weight_bounds=(0.6, 5.0),
        slope_upper_bound=50.0,
    )
    assert diagnostics['weights'] == {
        'configured_bounds': [0.6, 5.0],
        'lower_bound_hits': 1,
        'upper_bound_hits': 1,
        'total': 3,
    }
    assert diagnostics['slopes']['configured_abs_upper_bound'] == 50.0
    assert diagnostics['slopes']['effective_abs_upper_bound'] == 75.0
    assert diagnostics['slopes']['configured_bound_exceedances'] == 2
    assert diagnostics['slopes']['effective_bound_hits'] == 2


def test_score_report_uses_base_magnitude_and_actual_edge_masses():
    from reconstructor import Reconstructor
    from types import SimpleNamespace
    r = Reconstructor.__new__(Reconstructor)
    r.total_points = 4
    r.samp_dist = np.array([.007, .018, .029, .040])
    r.num_channels_recon = 3
    r.R = .0001
    r.active_angles = None
    r.interlaced_grating = False
    r.global_phases_set = 0.
    r.base_segments = [(0,2), (2,4)]
    r.slopes = {0: np.zeros((6, 2))}
    r.discont = np.pi
    r.turn_threshold = 2
    r.thresholds = [.8, 1.2, 2, 3, 4]
    r.num_levels = 6
    r.sharpness = 4.8
    r.min_consecutive = 3
    r.soft = True
    r.direct = 0
    r.calib_shapes = ['circ']
    r.radii = {0: {'circ': [50]}}
    strain = np.tile(-20*r.R*np.cos(np.arange(3)*2*np.pi/3), (4, 1))
    r.processed_Data = {'circ': {0: {50: strain}}}
    r.global_point_ids = np.array([19,20,21,22])
    report = runner().diagnose_score_distribution(r)
    sample = report['samples'][0]
    assert sample['score_magnitude'] == pytest.approx([20]*4)
    assert np.array(sample['masses'])[-1] == pytest.approx([2/3, 1, 1, 2/3])
    assert sample['global_point_ids'] == [19,20,21,22]
    assert sample['theoretical_curvature_m_inv'] == 20
    assert report['sigmoid_10_90_width_m_inv'] == pytest.approx(.91551024056)
