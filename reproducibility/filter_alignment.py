#!/usr/bin/env python3
"""Filter released GIFARC records by stored alignment score, entirely offline.

Keep scores >= tau. Null scores are excluded. Source code is never executed.
Selected JSONL lines are written verbatim, preserving original score precision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def validate_alignment(alignment, task_id):
    if not isinstance(alignment, dict):
        raise ValueError(f"Missing alignment object for task {task_id}")
    score, status, reason = (alignment.get(k) for k in ("score", "status", "unscored_reason"))
    if score is None:
        if status != "unscored" or not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Unscored task {task_id} needs its verified reason")
    elif type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError(f"Invalid cosine score for task {task_id}")
    elif status != "scored" or reason is not None:
        raise ValueError(f"Invalid scored status/reason for task {task_id}")
    return score


def read_scores(path):
    indexed = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if type(task_id) is not int or not 1 <= task_id <= 1614 or task_id in indexed:
            raise ValueError(f"Invalid/duplicate canonical task_id at score line {line_number}")
        alignment = row.get("alignment", row)
        validate_alignment(alignment, task_id)
        source_hash = row.get("source_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError(f"Missing source_sha256 for task {task_id}")
        indexed[task_id] = {**row, **alignment}
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Released data/data.jsonl")
    parser.add_argument("--scores", type=Path, help="Optional data/alignment_scores.jsonl integrity check")
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New filtered JSONL file")
    args = parser.parse_args()
    if not math.isfinite(args.tau) or not -1 <= args.tau <= 1:
        parser.error("--tau must be a finite cosine threshold in [-1, 1]")
    if args.output.exists():
        parser.error("--output already exists; choose a new file")
    scores = read_scores(args.scores) if args.scores else None
    selected, seen, scored, unscored = [], set(), 0, 0
    for line_number, line in enumerate(args.data.read_bytes().splitlines(keepends=True), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if type(task_id) is not int or not 1 <= task_id <= 1614 or task_id in seen:
            raise ValueError(f"Invalid/duplicate canonical task_id at data line {line_number}")
        seen.add(task_id)
        score = validate_alignment(row.get("alignment"), task_id)
        if scores is not None:
            if task_id not in scores:
                raise ValueError(f"Missing sidecar task {task_id}")
            expected = scores[task_id]
            if any(row["alignment"].get(k) != expected[k] for k in ("score", "status", "unscored_reason")):
                raise ValueError(f"Sidecar alignment differs for task {task_id}")
            if hashlib.sha256(row["source"].encode("utf-8")).hexdigest() != expected["source_sha256"]:
                raise ValueError(f"Source hash differs for task {task_id}")
        if score is None:
            unscored += 1
        else:
            scored += 1
            if score >= args.tau:
                selected.append(line if line.endswith(b"\n") else line + b"\n")
    if scores is not None and set(scores) != seen:
        raise ValueError("Data and score sidecar canonical task IDs differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.writelines(selected)
    print(json.dumps({"threshold": args.tau, "input_tasks": len(seen), "scored_tasks": scored,
                      "unscored_excluded": unscored, "retained_tasks": len(selected),
                      "policy": "score >= tau; null excluded; selected original lines preserved"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
