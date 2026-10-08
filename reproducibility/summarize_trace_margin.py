#!/usr/bin/env python3
"""Reproduce Appendix B task-weighted rule-description alignment margin.

The trace_* filename is legacy; scores concern emitted inferred_rule descriptions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from trace_margin import _aggregate_per_task, _summarize_conditions


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=root / "summaries/trace-margin-per-case.csv")
    parser.add_argument("--output",type=Path,required=True,help="New summary CSV")
    args=parser.parse_args()
    if args.output.exists():parser.error("--output already exists")
    rows=list(csv.DictReader(args.input.open(encoding="utf-8",newline="")))
    fields=("own_similarity","mean_other_similarity","margin_delta","own_reference_rank","top1","top5","reciprocal_rank")
    for row in rows:
        row["task_id"]=int(row["task_id"])
        for key in fields:row[key]=float(row[key])
    per_task=_aggregate_per_task(rows);summary=_summarize_conditions(per_task)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps({"valid_case_scores":len(rows),"task_condition_groups":len(per_task),"conditions":len(summary)}))


if __name__ == "__main__":main()
