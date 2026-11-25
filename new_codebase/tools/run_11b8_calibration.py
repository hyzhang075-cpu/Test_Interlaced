"""Run three weight optimizations from a completed 11B8 init and export JSON."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--init-run', type=Path, required=True)
    args = parser.parse_args()
    seed = args.init_run.resolve()
    manifest = json.loads((seed / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'succeeded' or manifest['task'] != 'init':
        raise ValueError('Need a successful initialization run')
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(ROOT / '.runtime_deps') + os.pathsep + env.get('PYTHONPATH', '')
    runs = []
    run_root = ROOT / 'runs' / '11B8_CoreCalib'
    for direction in (0, 120, 240):
        before = set(run_root.iterdir())
        log = ROOT / f'11B8-optim-{direction}.log'
        print(f'Optimizing direction {direction}; log: {log}', flush=True)
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-m', 'shape_sensing', '--config',
                'configs/11B8_CoreCalib.yaml', '--task', 'optim', '--multi_region_calib',
                'True', '--direct', str(direction), '--resume', str(seed)],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        created = set(run_root.iterdir()) - before
        if len(created) != 1:
            raise ValueError(f'Expected one new optimization run, got {created}')
        run = created.pop()
        state = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        if state['status'] != 'succeeded':
            raise ValueError(f'Optimization did not succeed: {run}')
        runs.append(str(run))
    output = ROOT / 'packages' / '11B8_CoreCalib-v1.json'
    subprocess.run([sys.executable, 'tools/export_algorithm_package.py', '--runs',
                    *runs, '--output', str(output)], cwd=ROOT, env=env, check=True)
    print(f'Exported: {output}', flush=True)


if __name__ == '__main__':
    main()
