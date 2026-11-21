# Standalone eight-module migration progress
Plan: docs/superpowers/plans/2026-11-10-standalone-eight-module-migration.md
Baseline: 23e180a on AlgorithmPackage-08B5

## Source preservation inventory
Original source files were read and AST-indexed before any shim conversion. SHA-256 inventory:
- `sensor.py` — 32e253d1d35b0c8e9a00d578ba00b95a07d9b801e2b3b2cdb8222923dcfad52d
- `initialiser.py` — 67e9a8d10f3ffa01849bb4a176e889f41077c806675f08bb409bfbcdd4d63d5e
- `reconstructor.py` — 4ee02304f92c60fc2072893efc55a11c5e61600beb1b017418551be9fdaeaa24
- `optimiser.py` — 6f649c8a7c34bee86985c9753bfec84a47cb30b4e81aceec5d6c995cee859423
- `visualisor.py` — 433230a8642da63f29231a37058748cd21fc8ac478b627d6ec382e105843a688
- `demonstrator.py` — 47334b71a8be97be00694f161bf8dec730966827e58f8a496a12b39c72f241b1
- `proto_demonstrator.py` — 80c120f39fc9e72c0e2231512bdc48ad69eeedce352a475239e8201640cfeb8c
- `IMU.py` — 85d069af18ac342391c901e40242f28e1250c8e2a92ca19406b42ab873a485c8
- `entrance_anchor.py` — 34bddd68f8706108fb8d7e6330d1b32832250ca52db87cfe7e6f54f97cbc4f4c
- `main.py` — 7a365399cb5a478a8fee522bbabbe24c723ef9fc679b83ed8af7b9f7cbbfbd7d
- `calibration_batch.py` — f024f0fc347a011525c784fc7e4f7e7053f034f04adbbf2f304c9e4b4806287d
- `calibration_regions.py` — 16bdd7e54069ec0c237836b7905734c17ba59b28293b07ba473f711d5ba85a9a
- `calibration_phase_reference.py` — 0175144fa7eb27c2d9c8c47d06b9177861855557fe171612cdb47dc96cf544f8
- `tools/utils.py` — ccc5156504e61a85ca18551b80962c93552ca01e1de7bb7660449adaf7d4b647
- `tools/zero_reference.py` — 7c3ca23e20bf4dbca42d0240bfe66e77fbcd281d185a01641421bf86a8cda968
- `tools/end5_weight_coverage.py` — 0b757376d198a9f1d742650aad6cc6bdf80e7c1145b5187dd2827442e6be364b
- `tools/sensor_parameter_package.py` — e8f54cbb3f775c29de991d08c4adaeff87aa1986c595062829f119ab3e70c02c

## Status
- Completed: source implementations were migrated into all planned package areas: Sensor/zero-reference, initialization/phase reference, curvature geometry/model, optimization/batch/coverage, reconstruction/integration, evaluation/ideal curvature/metrics, visualization/VTK view, runtime/prototype/IMU/anchor, CLI, legacy package compatibility, and shared utilities/paths.
- Numerical methods were transferred from source text rather than `ast.unparse`; mixins have no constructors and preserve the legacy inheritance/super order. Core function/source ownership is checked by the acceptance suite. The independent source review recorded zero missing method comments and identical core method ASTs apart from package import paths.
- Root modules and legacy `tools` modules are reverse compatibility aliases; the direct `tools/sensor_parameter_package.py` script shim alone restores the repository parent on `sys.path` before importing the package.
- Public context facades retain their baseline dispatch, snapshot, and engine-state behavior. The additive `ShapeIntegrator` works without Sensor/data files; TNN and TNB outputs match the saved legacy arrays (TNB fixture uses `TNB_threshold=0.0`).
- Added `pyproject.toml`, `python -m shape_sensing`, `shape-sensing` console entry, API factory, and packaged parameter catalog/template. Relative runtime data/output roots use the working directory; YAML-relative paths remain anchored by the configuration loader; packaged resources remain package-relative.
- Validation: **370 passed** in the complete pytest suite (57.13s); includes five standalone acceptance checks (including working-directory path behavior) for forbidden imports, implementation ownership, isolated offline wheel execution with real calculations and CLI help, and public API validation/context preservation. Legacy script/module entrypoint checks: 14 passed with the standalone checks. `git diff --check` exits 0; Git emitted only expected LF-to-CRLF notices for repository autocrlf settings.
- Workspace state is intentionally uncommitted and unpushed for main-window review. Original `09B7/` and long-named experiment data directories remain untouched and untracked.


## Public API/path review follow-up (2026-11-10)
- Fixed `create_engine(config, **overrides)` provenance: all programmatic options are added to `_explicit_cli_parameters`; `weights`/`slopes` also update `_explicit_seed_parameters`; `full_segmentation` sets `_full_segmentation_explicit`. This keeps explicit `None`, region scope, phase, and seed overrides from being replaced by YAML profiles.
- `create_engine` does not execute a task pipeline, so an explicit or configured `resume` now raises a clear `ValueError` instead of being silently recorded without loading a snapshot.
- Prototype relative STL paths now resolve from the process working directory; absolute paths are used as supplied. The package installation directory is no longer considered.
- Non-script legacy module shims no longer call a nonexistent `main()` when run as scripts; supported CLI shim guards remain.
- Added `tests/test_public_api_overrides.py` with four checks covering explicit empty seeds against populated YAML seeds, explicit `exp_directs`, phase precedence, resume rejection, and prototype path resolution (four tests total). Follow-up targeted validation: standalone/API/tool entrypoint suites **18 passed** in 31.92s. The earlier complete suite was **370 passed** before this isolated P2 follow-up; main-window full verification remains requested. No commit or push.

## Main-window final acceptance
- After the final API/prototype/shim corrections, independently reran the complete suite: **374 passed in 45.94s**.
- Built the source ZIP from 51 allowlisted files, extracted outside the repository, built/installed its wheel offline, and passed real numerical operations with legacy imports blocked; module and console CLI help also passed. Archive and wheel contents match the final source tree.
- TNN/TNB outputs match the pre-migration reference elementwise; all four arrays in each branch have maximum absolute difference 0.
- Independent reviewer re-ran both API override reproductions and closed the finding. See docs/standalone-acceptance.md for artifact checksums and validation scope.
- Main-window acceptance is complete; changes are ready for local branch commit. No remote push or experiment-data modification.
- Final inherited-whitespace cleanup preserved executable AST and comment text; staged diff check passes. Rebuilt/reinstalled final artifacts outside the repository and independently repeated the complete suite: **374 passed in 45.94s**.
