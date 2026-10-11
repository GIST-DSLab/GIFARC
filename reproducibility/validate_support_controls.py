"""Execute only the three inspected frozen support-code pairs offline.

Reuses the actual structure validator and seeds 0-2 behavior validator, then
compares both programs with gold outputs on all 29 supplied stored inputs.
Each task runs in a child Python process with a 20-second bound. This finite
check is empirical and is not program equivalence or semantic validation.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

INPUT = Path(__file__).resolve().parent / 'summaries' / 'finite-support-check'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_inputs():
    manifest = json.loads((INPUT / 'input-manifest.json').read_text(encoding='utf-8'))
    if digest(INPUT / 'common.py') != manifest['common_sha256']:
        raise ValueError('Configured common helper differs from the inspected input')
    for support in manifest['supports']:
        path = INPUT / support['file']
        if digest(path) != support['sha256']:
            raise ValueError('A support file differs from the inspected input')
        record = json.loads(path.read_text(encoding='utf-8'))
        for field in ('solution_code', 'neutral_code'):
            tree = ast.parse(record[field])
            if not all(isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) for n in tree.body):
                raise ValueError('Unexpected support top-level statement')
            for n in ast.walk(tree):
                if isinstance(n, ast.Import) and not all(a.name in ('numpy', 'random') for a in n.names):
                    raise ValueError('Unexpected support import')
                if isinstance(n, ast.ImportFrom) and n.module != 'common':
                    raise ValueError('Unexpected support from-import')
                if isinstance(n, ast.Call):
                    call = ast.unparse(n.func)
                    if call in ('eval', 'exec', 'compile', 'open', '__import__') or call.startswith(('os.', 'subprocess.', 'socket.', 'requests.', 'shutil.')):
                        raise ValueError('Unexpected support operation')
    return manifest


def worker(task_id, output):
    sys.dont_write_bytecode = True
    import numpy as np
    import support_code_validator as validator
    record = json.loads((INPUT / f'support-{task_id}.json').read_text(encoding='utf-8'))
    structure, structure_reason = validator.validate_code_structure(record['solution_code'], record['neutral_code'])
    behavior, behavior_reason = validator.validate_behavior(original_code=record['solution_code'], neutralized_code=record['neutral_code'], common_module_dir=INPUT, task_id=record['legacy_task_id'])
    original = validator._load_module_from_source(record['solution_code'], f'original_{task_id}', INPUT)
    neutral = validator._load_module_from_source(record['neutral_code'], f'neutral_{task_id}', INPUT)
    rows = []
    for split in ('train_pairs', 'test_pairs'):
        for index, pair in enumerate(record[split]):
            x, gold = np.array(pair['input']), np.array(pair['output'])
            left, right = np.array(original.main(x.copy())), np.array(neutral.main(x.copy()))
            rows.append({'task_id': task_id, 'split': split, 'index': index,
                         'original_equals_neutral': bool(np.array_equal(left, right)),
                         'original_equals_gold': bool(np.array_equal(left, gold)),
                         'neutral_equals_gold': bool(np.array_equal(right, gold)),
                         'input_sha256': hashlib.sha256(json.dumps(x.tolist(), separators=(',', ':')).encode()).hexdigest(),
                         'gold_sha256': hashlib.sha256(json.dumps(gold.tolist(), separators=(',', ':')).encode()).hexdigest()})
    result = {'task_id': task_id, 'structure_valid': structure, 'structure_reason': structure_reason,
              'seeded_behavior_valid': behavior, 'seeded_behavior_reason': behavior_reason,
              'validator_seeds': [0, 1, 2], 'stored_pairs': rows, 'numpy_version': np.__version__}
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker-task', type=int, choices=(838, 878, 902), help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = inspect_inputs()
    if args.worker_task:
        worker(args.worker_task, args.output_dir / f'result-{args.worker_task}.json')
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for support in manifest['supports']:
        task_id = support['task_id']
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--output-dir', str(args.output_dir), '--worker-task', str(task_id)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        if completed.returncode:
            raise RuntimeError(f'Support {task_id} failed: {completed.stderr}')
        results.append(json.loads((args.output_dir / f'result-{task_id}.json').read_text(encoding='utf-8')))
    pairs = [row for result in results for row in result['stored_pairs']]
    if len(pairs) != 29 or not all(r['structure_valid'] and r['seeded_behavior_valid'] for r in results) or not all(r[k] for r in pairs for k in ('original_equals_neutral', 'original_equals_gold', 'neutral_equals_gold')):
        raise ValueError('Finite validation differs from the included evidence')
    report = {'tasks': len(results), 'structure_passes': sum(r['structure_valid'] for r in results),
              'seeded_behavior_passes': sum(r['seeded_behavior_valid'] for r in results), 'validator_seeds': [0, 1, 2],
              'stored_pairs': len(pairs), 'original_neutral_equal': sum(r['original_equals_neutral'] for r in pairs),
              'original_gold_equal': sum(r['original_equals_gold'] for r in pairs), 'neutral_gold_equal': sum(r['neutral_equals_gold'] for r in pairs),
              'subprocess_timeout_seconds': 20,
              'limitations': json.loads((INPUT / 'recorded-report.json').read_text(encoding='utf-8'))['limits']}
    (args.output_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
