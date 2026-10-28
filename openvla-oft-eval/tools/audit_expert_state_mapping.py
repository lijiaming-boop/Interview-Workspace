"""Check numerical matches, never assume demo ID equals benchmark state ID."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tolerance', type=float, default=1e-6)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(4)
    expected_file = Path(suite.get_task_demonstration(4)).name
    if Path(args.hdf5).name != expected_file:
        raise ValueError(f'task4 expects {expected_file}, got {Path(args.hdf5).name}')
    states = np.asarray(suite.get_task_init_states(4), dtype=np.float64)
    report = {'task': str(task), 'expected_hdf5': expected_file,
              'benchmark_shape': list(states.shape), 'demos': [], 'target_matches': {}}
    with h5py.File(args.hdf5, 'r') as f:
        names = sorted(f['data'], key=lambda n: int(n.split('_')[-1]))
        first = []
        for name in names:
            d = f['data'][name]
            first.append(np.asarray(d['states'][0], dtype=np.float64))
            actions = np.asarray(d['actions'])
            close = np.flatnonzero((actions[1:, -1] > 0) & (actions[:-1, -1] < 0)) + 1
            row = {'demo': name, 'length': len(actions),
                   'close_transitions': close.tolist(),
                   'window_45_91_exists': len(actions) >= 91,
                   'close_transitions_in_45_91': close[(close >= 45) & (close < 91)].tolist(),
                   'terminal_reward': float(np.asarray(d['rewards'])[-1])}
            report['demos'].append(row)
        first = np.asarray(first)
    if states.ndim != 2 or first.ndim != 2 or states.shape[1] != first.shape[1]:
        report['mapping_status'] = 'INCONCLUSIVE: incompatible state shapes'
    else:
        # Full flattened simulator state, including time. No arbitrary component
        # is dropped to manufacture a match. A mismatch is not proof of wrong demo.
        for s in (29, 32):
            distances = np.max(np.abs(first - states[s]), axis=1)
            order = np.argsort(distances)
            exact = [names[i] for i in order if distances[i] <= args.tolerance]
            report['target_matches'][str(s)] = {
                'within_tolerance': exact,
                'unique_numerical_match': len(exact) == 1,
                'nearest': [{'demo': names[i], 'max_abs_diff': float(distances[i])} for i in order[:5]],
                'same_id_matches': f'demo_{s}' in exact}
        report['mapping_status'] = 'NUMERICAL_CHECK_ONLY: verify XML/controller and replay before labeling corrections'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'demos'}, indent=2))
    print('demo_29 / demo_32 window checks:')
    for row in report['demos']:
        if row['demo'] in ('demo_29', 'demo_32'):
            print(row)
    print('A close command does not prove contact or lift; inspect replay for phase labels.')


if __name__ == '__main__':
    main()
