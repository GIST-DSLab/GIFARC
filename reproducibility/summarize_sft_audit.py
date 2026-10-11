"""Recompute current tokenizer audit counts from included numeric row evidence.

This needs only Python's standard library and performs no tokenizer downloads,
model inference, or training. Counts refer to reconstructed sequence retention
at 6144 tokens; historical executed truncation counts remain unknown.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

INPUT = Path(__file__).resolve().parent / 'summaries' / 'sft-sequence-audit'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reference = json.loads((args.input_dir / 'current-pinned-tokenizer-reconstruction.json').read_text(encoding='utf-8'))
    summaries = []
    checks = {'raw_sequence_over_6144': ('raw_whole_over_6144', 'True'),
              'reconstruction_input_shortened': ('reconstruction_input_shortened', 'True'),
              'reconstruction_incomplete_target_text': ('reconstruction_complete_target_text_preserved', 'False'),
              'reconstruction_incomplete_rendered_gold_grid': ('reconstruction_complete_rendered_gold_grid_preserved', 'False'),
              'reconstruction_incomplete_gold_grid_fence': ('reconstruction_complete_gold_grid_fence_preserved', 'False')}
    for group in reference['summary']:
        path = args.input_dir / group['csv']
        if hashlib.sha256(path.read_bytes()).hexdigest() != group['csv_sha256']:
            raise ValueError('Included row-evidence hash mismatch')
        with path.open(newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 1000 or any(int(r['reconstruction_array_positions']) != 6144 for r in rows):
            raise ValueError('Expected 1000 reconstructed sequences of exactly 6144 positions')
        summary = {k: group[k] for k in ('tokenizer', 'seed', 'dataset')}
        summary['rows'] = len(rows)
        for field, (column, value) in checks.items():
            summary[field] = sum(r[column] == value for r in rows)
            if summary[field] != group[field]:
                raise ValueError(f'Count mismatch: {summary}/{field}')
        summaries.append(summary)
    report = {'scope': reference['scope'], 'sequence_limit': 6144,
              'historical_executed_truncation_counts': None, 'label_masking_population_audit': False,
              'summary': summaries, 'verified_rows': sum(r['rows'] for r in summaries)}
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as file:
            file.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
