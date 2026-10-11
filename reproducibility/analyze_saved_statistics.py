"""Reproduce task-paired bootstrap analyses from the released saved metrics.

NumPy is the only nonstandard dependency. No task programs, model backend,
embedding client, or network requests are used. Public task IDs are one-based.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
from paired_statistics import (
    MODELS, CONDITIONS, RAW_METRICS, CACHED_METRICS, digest, finite,
    reproduce, analyze_pairs, write_csv, self_test,
)

DEFAULT_INPUT = Path(__file__).resolve().parent / 'summaries' / 'paired-statistics'


def load_records(path):
    records = {}
    with path.open(newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            key = (row['model'], row['condition'], int(row['task_id']), int(row['run']))
            if key in records:
                raise ValueError(f'Duplicate metric record: {key}')
            metrics = {m: (float(row[m]) if row[m] else None) for m in RAW_METRICS + CACHED_METRICS}
            if any(v is not None and not finite(v) for v in metrics.values()):
                raise ValueError(f'Nonfinite metric: {key}')
            for metric in ('MWR', 'exact', 'GC', 'FG'):
                if finite(metrics[metric]) and not 0 <= metrics[metric] <= 1:
                    raise ValueError(f'Out-of-range proportion: {key}/{metric}')
            flags = {}
            for field in ('execution_error_flag', 'parse_error_flag'):
                if row[field] not in ('True', 'False'):
                    raise ValueError(f'Invalid flag: {key}/{field}')
                flags[field] = row[field] == 'True'
            records[key] = {'metrics': metrics, 'execution_error': flags['execution_error_flag'],
                            'parse_error': flags['parse_error_flag']}
    tasks = sorted({key[2] for key in records})
    expected = {(m, c, t, r) for m in MODELS for c in CONDITIONS for t in tasks for r in (1, 2, 3)}
    if set(records) != expected or len(tasks) != 100 or min(tasks) < 1:
        raise ValueError('Expected complete saved grid: 100 canonical targets × 3 runs × 2 models × 7 conditions')
    return records, tasks


def compare_csv(actual, expected):
    with actual.open(newline='', encoding='utf-8') as a, expected.open(newline='', encoding='utf-8') as b:
        left, right = list(csv.DictReader(a)), list(csv.DictReader(b))
    if len(left) != len(right):
        raise ValueError(f'Row count differs: {actual.name}')
    for index, (a, b) in enumerate(zip(left, right), 1):
        if a.keys() != b.keys():
            raise ValueError(f'Columns differ: {actual.name}')
        for key, value in a.items():
            if value == b[key]:
                continue
            try:
                equal = math.isclose(float(value), float(b[key]), rel_tol=1e-12, abs_tol=1e-12)
            except ValueError:
                equal = False
            if not equal:
                raise ValueError(f'Numerical/string mismatch: {actual.name} row {index}/{key}')
    return len(left)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--verify-included', action='store_true', help='Compare regenerated CSV fields to all included reference outputs')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.verify_included and (args.samples, args.seed) != (50000, 20261008):
        parser.error('--verify-included requires the original 50000 draws and seed base 20261008')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.input_dir / 'normalized-records.csv'
    reference = json.loads((args.input_dir / 'analysis-manifest.json').read_text(encoding='utf-8'))
    if digest(source) != reference['output_sha256']['normalized-records.csv']:
        raise ValueError('Normalized input hash differs from the included saved-evidence manifest')
    records, tasks = load_records(source)
    run_rows, aggregates, audits, table8 = reproduce(records, tasks, args.samples, args.seed)
    pairs, task_means, comparisons = analyze_pairs(records, tasks, args.samples, args.seed)
    outputs = {
        'valid-run-summary.csv': run_rows,
        'original-aggregate-summary.csv': aggregates,
        'published-rounding-audit.csv': audits,
        'table8-condition-confidence-intervals.csv': table8,
        'paired-task-run-differences.csv': pairs,
        'paired-task-means.csv': task_means,
        'paired-confidence-intervals.csv': comparisons,
    }
    for name, rows in outputs.items():
        write_csv(args.output_dir / name, rows)
    core = [r for r in comparisons if r['left_condition'] == 'rich' and r['right_condition'] == 'neutral' and r['metric'] in ('MWR', 'AAS', 'margin')]
    solving = [r for r in comparisons if r['comparison_type'] == 'matched_wording' and r['metric'] in ('exact', 'GC', 'FG')]
    if len(core) != 6 or len(solving) != 18 or not all(r['rounding_matches'] for r in audits):
        raise ValueError('Core contrasts, solving contrasts, or original table rounding differ')
    write_csv(args.output_dir / 'core-confidence-intervals.csv', core)
    write_csv(args.output_dir / 'solving-confidence-intervals.csv', solving)
    verified = {name: compare_csv(args.output_dir / name, args.input_dir / name) for name in outputs} if args.verify_included else {}
    manifest = {
        'analysis': 'offline_saved_results_task_bootstrap',
        'normalized_records': len(records), 'target_tasks': len(tasks),
        'bootstrap_draws': args.samples, 'bootstrap_seed_base': args.seed,
        'interval_method': reference['interval_method'], 'primary_estimand': reference['primary_estimand'],
        'scope': reference['scope'], 'caveats': reference['caveats'],
        'python': platform.python_version(), 'numpy': np.__version__,
        'input_sha256': digest(source), 'adapter_sha256': digest(Path(__file__)),
        'core_contrasts': len(core), 'matched_solving_contrasts': len(solving),
        'all_comparisons': len(comparisons), 'original_table_statistics': len(audits),
        'verified_included_csv_rows': verified,
        'output_sha256': {p.name: digest(p) for p in sorted(args.output_dir.glob('*.csv'))},
    }
    (args.output_dir / 'analysis-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('normalized_records', 'target_tasks', 'core_contrasts', 'matched_solving_contrasts', 'all_comparisons', 'original_table_statistics', 'verified_included_csv_rows')}, indent=2))


if __name__ == '__main__':
    main()
