# Project layout and migration

## Project boundaries

- `new_codebase/` is the maintained standalone algorithm project. Its package, command line, configuration, tools, tests, requirements, and documentation live together here.
- `legacy_codebase/` contains the former top-level compatibility modules and the historical tools. Legacy callers can use the new package after installing `new_codebase` in editable mode from its own directory.
- The repository root is reserved for Git metadata, shared editor settings, and a short README directing users to either project.
- Experiment inputs, parameter sets, algorithm packages, runs, and calibration results are migrated as independent project data. Never share these directories through imports or symlinks.

## Entry points and imports

From `new_codebase/`, run `python -m shape_sensing`, `python shape_sensing/main.py`, or `python main.py`. Each entry loads the CLI from this project. The installed `shape-sensing` command is also defined by `pyproject.toml`. New-project code and tools must not add `legacy_codebase` to `sys.path` or import from it.

## Independent tools and tests

`new_codebase/tools/` is a complete copy of the former tools directory. Tools there must import the canonical `shape_sensing` package and other tools within this project. Tests under `new_codebase/tests/` are run with `new_codebase/` as their root and use only its package, tools, and fixtures.

## Data path audit

When data migration is complete, record every copied or relocated data/config path in `MIGRATION_DATA.json`. Configuration references are project-relative where possible. Keep the legacy and new project datasets separate and verify both projects can locate their own inputs without falling back to the repository root.

## Acceptance order

1. Check that the root contains only repository metadata/settings and its README.
2. Run each new-project entry point from the project root and from the package directory; confirm imports resolve under `new_codebase/`.
3. Run the standalone tests from `new_codebase/` and verify the copied tools do not resolve legacy modules.
4. Review `MIGRATION_DATA.json`, then confirm data and parameter references resolve independently in both project directories.
