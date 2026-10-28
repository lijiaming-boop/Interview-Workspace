"""Run the existing evaluator unchanged; compare batch vs fresh processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

TASK = 'pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='/mnt/workspace/openvla-oft')
    p.add_argument('--checkpoint', default='/mnt/workspace/openvla-oft-ckpts/openvla-7b-oft-finetuned-libero-spatial')
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    if args.repeats < 2:
        p.error('use at least two repeats')
    repo = Path(args.repo).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    evaluator = repo / 'experiments/robot/libero/run_libero_eval.py'
    env = dict(os.environ)
    env.update(PYTHONPATH=f'{repo}:/mnt/workspace/LIBERO:' + env.get('PYTHONPATH', ''),
               MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', TOKENIZERS_PARALLELISM='false',
               TF_CPP_MIN_LOG_LEVEL='2', PYTHONUNBUFFERED='1')
    help_run = subprocess.run([sys.executable, str(evaluator), '--help'], cwd=repo,
                              env=env, capture_output=True, text=True)
    (out / 'help.txt').write_text(help_run.stdout + help_run.stderr)
    if help_run.returncode:
        raise RuntimeError('evaluator import failed; inspect help.txt')
    supports_targets = '--target_episodes' in help_run.stdout
    if not supports_targets and '--initial_state_offset' not in help_run.stdout:
        raise RuntimeError('No supported state selection option')
    files = [evaluator, repo / 'experiments/robot/openvla_utils.py']
    files += list(Path(args.checkpoint).glob('*checkpoint.pt'))
    files += [Path(args.checkpoint) / 'dataset_statistics.json']
    manifest = {'checkpoint': args.checkpoint, 'seed': 7, 'files': {}}
    for f in files:
        if f.is_file():
            with f.open('rb') as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            manifest['files'][str(f)] = h.hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    records = []
    for rep in range(args.repeats):
        # Every invocation starts Python/model/environment afresh. Batch retains
        # whatever within-process reset behavior the user's evaluator implements.
        for mode, states in [('batch', list(range(50))), ('single29', [29]), ('single32', [32])]:
            tag = f'{mode}-repeat{rep + 1}'
            folder = out / tag
            folder.mkdir()
            cmd = [sys.executable, str(evaluator), '--model_family', 'openvla',
                   '--pretrained_checkpoint', args.checkpoint, '--task_suite_name', 'libero_spatial',
                   '--use_l1_regression', 'True', '--use_diffusion', 'False', '--use_film', 'False',
                   '--num_images_in_input', '2', '--use_proprio', 'True', '--center_crop', 'True',
                   '--num_open_loop_steps', '8', '--num_steps_wait', '10', '--seed', '7',
                   '--num_trials_per_task', str(len(states)), '--local_log_dir', str(folder),
                   '--run_id_note', tag]
            if supports_targets:
                cmd += ['--target_episodes', ','.join(f'4:{s}' for s in states)]
            else:
                cmd += ['--initial_state_offset', str(states[0])]
            (folder / 'command.json').write_text(json.dumps(cmd, indent=2))
            run_env = dict(env, GRASP_TRACE_PATH=str(folder / 'grasp.json'))
            print(f'RUN {tag}; log={folder / "console.log"}', flush=True)
            with (folder / 'console.log').open('w') as log:
                code = subprocess.run(cmd, cwd=repo, env=run_env, stdout=log, stderr=subprocess.STDOUT).returncode
            text = (folder / 'console.log').read_text(errors='replace')
            results = re.findall(r'\bSuccess:\s*(True|False)', text)
            descriptions = re.findall(r'\bTask:\s*([^\r\n]+)', text)
            if code or 'Episode error' in text or len(results) != len(states):
                raise RuntimeError(f'{tag}: invalid run (exit={code}, results={len(results)}); inspect log')
            if not descriptions or any(TASK not in d for d in descriptions):
                raise RuntimeError(f'{tag}: wrong/missing task description; inspect task filter')
            row = {'mode': mode, 'repeat': rep + 1, 'states': states,
                   'success': [v == 'True' for v in results], 'log': str(folder / 'console.log'),
                   'index_source': 'requested sequence + validated result count (not per-step trace)'}
            records.append(row)
            (out / 'results.json').write_text(json.dumps(records, indent=2))
            print(f'{tag}: {sum(row["success"])}/{len(states)}; failures=' +
                  str([s for s, ok in zip(states, row['success']) if not ok]), flush=True)
    for state in (29, 32):
        for rep in range(1, args.repeats + 1):
            batch = next(r for r in records if r['mode'] == 'batch' and r['repeat'] == rep)
            single = next(r for r in records if r['mode'] == f'single{state}' and r['repeat'] == rep)
            print(f'state={state} repeat={rep}: batch={batch["success"][state]} single={single["success"][0]}')
    print('This comparison detects context sensitivity; it does not isolate RNG from environment reset.')


if __name__ == '__main__':
    main()
