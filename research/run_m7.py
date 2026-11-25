"""Print the fixed M7 matrix, or execute it sequentially with --execute after GPU recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_HASH = 'd0d90421470e49ac5c3828e5486ade87da326907a33f994d725b5fc18d47f85a'
DATASET_HASH = 'f7d96a5015010969051af29532cbc0d2203b0cc852703fb0b5dec0a43c704fd5'


def trial_specs():
    return [dict(mode=mode, split=split, max_episodes=count, seed=seed, seed_mode='episode', timeout=35.)
            for seed in (0, 1) for split, count in (('train', 3), ('val', 2))
            for mode in ('hybrid', 'hybrid-corridor')]


def completion_evidence(folder, returncode):
    log = folder / 'process.log'
    summary = folder / 'summary.json'
    text = log.read_text(errors='replace')
    data = json.loads(summary.read_text())
    if (returncode != 0 or 'AUTOPICKER_ROLLOUT_COMPLETE' not in text
            or 'Traceback (most recent call last)' in text or 'AUTOPICKER_ROLLOUT_FAILED' in text
            or not data.get('completed') or data.get('fatal_error')):
        raise ValueError(f'Process did not complete cleanly: {folder}')
    return {'schema': 'autopicker-process-completion-v1', 'returncode': returncode,
            'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
            'summary_sha256': hashlib.sha256(summary.read_bytes()).hexdigest()}


def record_completion(folder, returncode):
    evidence = completion_evidence(folder, returncode)
    with (folder / 'completion.json').open('x') as stream:
        json.dump(evidence, stream, indent=2)


def verify_completion(folder):
    receipt = json.loads((folder / 'completion.json').read_text())
    if receipt != completion_evidence(folder, receipt['returncode']):
        raise ValueError(f'Completion evidence changed: {folder}')


def verify_resume(folder, spec, sources):
    verify_completion(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    indices = [1, 2, 4] if spec['split'] == 'train' else [0, 5]
    expected = {key: spec[key] for key in ('mode', 'split', 'seed', 'seed_mode')}
    expected.update(completed=True, requested=len(indices), attempted=len(indices),
                    dataset_sha256=DATASET_HASH, checkpoint_sha256=CHECKPOINT_HASH)
    if (any(summary.get(key) != value for key, value in expected.items())
            or [r['source_episode_index'] for r in summary['results']] != indices):
        raise ValueError(f'Refusing to overwrite or skip incomplete/different run: {folder}')
    for ordinal in range(len(indices)):
        directory = folder / f'episode_{ordinal:04d}'
        path = directory / ('failure.json' if (directory / 'failure.json').exists() else 'episode.json')
        if json.loads(path.read_text())['metadata']['source_sha256'] != sources:
            raise ValueError(f'Refusing to mix implementation versions: {folder}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--isaac-python', type=Path,
                        default=Path('/home/oem/Downloads/IsaacSim/_build/linux-x86_64/release/python.sh'))
    args = parser.parse_args()
    specs = trial_specs()
    if not args.execute:
        print(json.dumps({'planned_attempts': 20, 'groups': specs}, indent=2))
        return
    python = ROOT / 'work/autopicker_train_env/bin/python'
    # Fail before creating any trial output when CUDA is unavailable.
    probe = subprocess.run([str(python), '-c', 'import torch; torch.cuda.init(); print(torch.cuda.get_device_name(0))'],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
    if probe.returncode:
        raise SystemExit('CUDA preflight failed; no trial started.\n' + probe.stderr)
    from autopicker_dp.run import SOURCE_FILES, SOURCE_HASHES
    sources = dict(SOURCE_HASHES)
    for path, expected in ((ROOT / 'runs/autopicker/m5_train_sample_long_v1/checkpoint.pt', CHECKPOINT_HASH),
                           (ROOT / 'datasets/autopicker/lowdim_v2/manifest.json', DATASET_HASH)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen M7 input changed: {path}')
    for spec in specs:
        current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_FILES}
        if current != sources:
            raise ValueError('Implementation changed during M7 experiment')
        name = f"m7_{spec['mode']}_{spec['split']}_s{spec['seed']}"
        folder = ROOT / 'runs/autopicker' / name
        if folder.exists():
            verify_resume(folder, spec, sources)
            print('REUSE_COMPLETED', name, flush=True)
            continue
        command = [str(args.isaac_python), str(ROOT / 'research/autopicker_dp/rollout.py'),
                   '--dataset', str(ROOT / 'datasets/autopicker/lowdim_v2'), '--output', str(folder),
                   '--checkpoint', str(ROOT / 'runs/autopicker/m5_train_sample_long_v1/checkpoint.pt'),
                   '--inference-python', str(python), '--upstream', str(ROOT / 'work/diffusion_policy_upstream'),
                   '--headless']
        for key, value in spec.items():
            command += ['--' + key.replace('_', '-'), str(value)]
        log = ROOT / 'work' / f'{name}.log'
        print('START', name, flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if not folder.exists():
            raise RuntimeError(f'Run failed; preserve outputs and inspect {log}')
        shutil.copyfile(log, folder / 'process.log')
        record_completion(folder, result.returncode)
        verify_resume(folder, spec, sources)
        print('COMPLETE', name, flush=True)
    subprocess.run([str(python), str(ROOT / 'research/tests/validate_m7.py'), '--output',
                    str(ROOT / 'docs/research/2026-11-25-m7-validation.json')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
