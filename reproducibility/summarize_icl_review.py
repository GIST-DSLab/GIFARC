#!/usr/bin/env python3
"""Count the two recorded rule labels in the fixed human-reviewed error sample."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=root / "summaries/human-reviewed-errors.jsonl")
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r['response_id'] for r in rows}) != len(rows):raise ValueError("Duplicate response_id")
    counts={"declared_rule_contradiction_only":0,"rule_output_inconsistency_only":0,"both":0,"neither":0}
    for row in rows:
        if row['annotation_status'] != 'human_reviewed_sample':raise ValueError("Expected confirmed sample judgments")
        labels=set(row['human_labels']);a='declared_rule_contradiction' in labels;b='rule_output_inconsistency' in labels
        key='both' if a and b else 'declared_rule_contradiction_only' if a else 'rule_output_inconsistency_only' if b else 'neither'
        counts[key]+=1
    print(json.dumps({"reviewed_error_responses":len(rows),"exclusive_rule_counts":counts,
                      "declared_rule_contradiction":counts['declared_rule_contradiction_only']+counts['both'],
                      "rule_output_inconsistency":counts['rule_output_inconsistency_only']+counts['both'],
                      "scope":"Only the two recorded rule labels; absence of visual labels is not evidence of no visual error."},indent=2))


if __name__ == '__main__':main()
