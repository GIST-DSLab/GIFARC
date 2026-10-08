#!/usr/bin/env python3
"""Reproduce cached target-filtered solving summaries using supplied released scores.

Reads the included compact saved metrics; no predictions or task programs run.
Original supports stay fixed. Null metrics retain the historical non-null policy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from cached_aggregation import aggregate, make_subsets, write_csv
from filter_alignment import read_scores


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Released alignment_scores.jsonl")
    parser.add_argument("--metrics", type=Path, default=root / "summaries/saved-case-metrics.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory")
    scores = {k: r["score"] for k, r in read_scores(args.scores).items() if r["score"] is not None}
    groups = defaultdict(list)
    for line in args.metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        # The original scientific helper uses zero-based IDs; release files use canonical IDs.
        row["task_id"] -= 1
        groups[row.pop("repeat")].append(row)
    runs = [(Path(repeat), rows) for repeat, rows in sorted(groups.items())]
    targets = {row["task_id"] for rows in groups.values() for row in rows}
    if len(runs) != 3 or len(targets) != 100:
        raise ValueError("Expected the original three repeats and 100 targets")
    subsets = make_subsets(targets, scores)
    per_run, summary = aggregate(runs, subsets)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per-run.csv", per_run)
    write_csv(args.output_dir / "summary.csv", summary)
    print(json.dumps({"target_tasks":len(targets), "scored_targets":len(targets & {k-1 for k in scores}),
                      "repeats":len(runs), "summary_rows":len(summary), "solver_calls":0}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
