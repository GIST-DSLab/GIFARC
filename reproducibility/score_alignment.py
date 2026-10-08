#!/usr/bin/env python3
"""Compute original reverse-caption cosine from supplied saved embedding vectors.

Offline only: no caption/embedding request and no generated task execution.
Input JSONL rows: task_id, caption_embedding, description_embedding.
The vectors must correspond to the documented first-three-pair GPT-5.4 caption
and original sketch, embedded with text-embedding-3-large. Their provenance is
supplied by the user; this CLI does not establish that provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Released data.jsonl for canonical identity/source hashes")
    parser.add_argument("--output", type=Path, required=True, help="New cosine-score JSONL")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new file")
    from alignment_protocol import cosine
    data = {}
    for line in args.data.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        record = json.loads(line); task_id = record["task_id"]
        if type(task_id) is not int or not 1 <= task_id <= 1614 or task_id in data:
            raise ValueError("Invalid/duplicate data task_id")
        data[task_id] = record
    results, seen = [], set()
    for line in args.vectors.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line); task_id = row["task_id"]
        if type(task_id) is not int or not 1 <= task_id <= 1614 or task_id in seen:
            raise ValueError("Invalid/duplicate canonical task_id")
        seen.add(task_id)
        if task_id not in data: raise ValueError(f"Unknown canonical task {task_id}")
        vectors = [row["caption_embedding"], row["description_embedding"]]
        if any(not isinstance(v, list) or len(v) != 3072 or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in vectors):
            raise ValueError(f"Task {task_id} needs two finite 3072-dimensional vectors")
        score = cosine(*vectors)
        if not math.isfinite(score):
            raise ValueError(f"Zero-norm vector for task {task_id}")
        results.append({"task_id":task_id,
                        "source_sha256":hashlib.sha256(data[task_id]["source"].encode("utf-8")).hexdigest(),
                        "alignment":{"score":score, "status":"scored", "unscored_reason":None},
                        "reverse_caption":row.get("reverse_caption")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in results: stream.write(json.dumps(row) + "\n")
    print(json.dumps({"scored_tasks":len(results), "network_calls":0}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))
