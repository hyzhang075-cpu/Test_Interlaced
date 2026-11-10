# Algorithm package implementation progress

## Task 0 — baseline and execution context
- Branch: `AlgorithmPackage-08B5`; baseline commit: `e27b245` (per design).
- Existing untracked experiment directories `09B7/` and `1016-09B7-自研（1+3）22点86cmFBG+镍钛管全末段定标测试/` were present before work and are left untouched.
- Baseline: 330 passed, 2 failed in 27.96s. Existing failures are the project-relative package fixture outside its configured root and the vector filter test expecting `False` while parser default is `True`. Preserve parser behavior and path validation.
- Project execution is through the authorized local branch; system Python path per handoff is `D:\anaconda3\python.exe`.
- Read the design and implementation plan. Current architecture is legacy-heavy: `main.py` imports Qt/VTK-era classes at module load; no `shape_sensing/` package exists yet.

## Task 1 — configuration, template, and CLI compatibility
- Added a generated lightweight catalog for all 110 parser destinations, including help, default, type and choices; the template includes all 108 non-control values and their Chinese help comments, with all 9 choice sets documented. `task`, `--config`, and `--sensor_params` are documented as entry controls.
- Added schema_version=1 YAML loading with duplicate-key, unknown-field, type, choice, nested calibration-region, and design-required validation. Preserves false/zero/null where allowed, rejects null for required numeric fields, and resolves calibration root relative to YAML then region folders relative to that root.
- Added `--config` and `--resume` parser controls. Precedence is parser defaults < YAML < explicit CLI. Rejects simultaneous config and legacy parameter package, duplicate CLI parameters, and excludes the new controls from Optimiser kwargs. Existing parser helper definitions and parser JSON package compatibility remain in place.
- Corrected the vector-filter help text to match the actual `True` default; runtime default remains unchanged.
- Tests: `tests/test_sensor_configuration.py`: 8 passed. Combined configuration + package + vector-filter regression: 54 passed, 2 failed. Both failures match the recorded baseline: external fixture path outside configured calibration root, and stale test expecting vector filter default False while parser behavior is True. Log: `.pytest_cache/algorithm-package-task1-compat.log`.
- Full-template integration test fills sensor_id/R/samp_dist_preset, loads via `--config`, resolves calibration region paths, and calls `configure_calibration_args` successfully.

## Task 2 — public algorithm adapters
- Corrected adapter targets against legacy implementations: FiberTecc/Hippo FBG and LUNA OFDR reads, interrogator-specific receivers, `_process_centreData`, `_reconstruct_from_interpolated_curvature`, `_construct_ideal_shape`, and `_compute_error` snapshots.
- Added array/result contracts and Chinese interface documentation for units, dimensions, state and side effects. GUI/runtime construction remains delayed to explicit calls.
- Independent acceptance: input routing, numerical integration allclose, ideal curve allclose, store_true YAML validation, explicit zero phase and no-GUI imports pass.

## Task 3 — task pipelines
- `main.main()` delegates to `shape_sensing.pipelines.execute_task`; init/ang/optim/score/recon/demo routes retain injected legacy factories and helper callbacks, including `main.Optimiser` monkeypatch compatibility.
- Batch region flow remains `prepare_calibration_jobs/run_calibration_jobs`; recon uses reconstruction/evaluation adapters; demo closes runtime in `finally`.
- Pipeline/regression batch: 80 passed before later package work; current full-suite evidence is recorded in the final log.

## Task 4 — run snapshots and resume
- Each dispatched task creates a UUID-unique run, stores merged resolved Namespace, source provenance, result state and succeeded/failed status. Init/optim stage payloads include engine initial/final values; batch report is retained intact.
- Resume requires its own `config_resolved.json`, checks sensor/geometry and selected-region layout while allowing moved data roots, restores saved parameters rather than the error summary, normalizes direction keys, and requires paired weights/slopes.
- Run artifact and resume regression: 18 passed; source commit/branch/dirty metadata is now included.

## Task 5 — export and compatibility
- Added `export_algorithm_package.py` and package loader. Export reads only successful run snapshots and actual region/direct reports; duplicate combinations and incompatible sensor design fail. Init-only packages are marked non-final.
- Packages include region/direct weights, slopes, phase/rotation provenance, runtime code provenance, validation=false, and copied zero reference resource when configured. Single-region and complete three-region/three-direction packages load through `main --sensor_params`; incomplete directions/regions fail.
- Full multi-region synthetic acceptance normalizes through legacy `Reconstructor._normalise_multiregion_parameters` and verifies owned point coverage.

## Task 6 — docs and final validation
- Added `docs/algorithm-packaging.md`, README entry and `.gitignore` rule for `/runs/`. Corrected help/template unit note for `interlaced_offset` to radians (default unchanged) and documented the `phase=0` compatibility correction.
- No real hardware or production data optimization was run. Final full-suite result is recorded below.

## Final verification evidence (2026-11-10)
- Full suite: **365 passed** in 36.99s; final log `.pytest_cache/algorithm-package-final.log`.
- Resume review regressions cover batch per-region restore without last-engine fallback, selected direction checks, top-level thresholds/full segmentation overrides, and weights-only resume slope provenance; targeted suite passed 19 tests.
- `main.py --help`: exit 0. `tools/export_algorithm_package.py --help`: exit 0. `compileall`: exit 0. `git diff --check`: exit 0 after preserving original mixed line endings on unchanged lines and using LF for edited lines.
- Runtime source provenance (commit/branch/dirty) is captured in each run manifest and inherited by exported packages.
- Branch remains `AlgorithmPackage-08B5`; no commit or push. Experiment data directories remain untouched and untracked. No hardware or production optimization was run.

- Final scoped follow-up: batch resume now applies each validated region's rot_ang during configuration unless the CLI explicitly supplies rot_ang; regression covers phases 0.1/0.2/0.3 and rotations 1/2/3. Targeted verification: 12 passed in 7.12s; git diff --check is clean.
