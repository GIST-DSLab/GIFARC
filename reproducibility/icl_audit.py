"""Actual paired eligibility, fixed random sampling and model-label counts.
This code aggregates supplied model labels; it does not classify responses.
"""
from __future__ import annotations
from collections import Counter
import random
ANALYSIS_ID = "icl-paired-confidence-audit-20261007-v1"

def high(response):
    o, t = response["observable"], response["triage"]
    return bool(o["parsed_ok"] and t["confidence"] == "high" and "undetermined" not in t["labels"])


def select_audit(bundle, sample_size=30, seed=20261007):
    common = [p for p in bundle["pairs"] if all(r["observable"]["parsed_ok"] and r["observable"]["shape_correct"] for r in p["responses"])]
    eligible = [p for p in common if {r["condition"] for r in p["responses"]} == {"base", "full"} and all(high(r) for r in p["responses"])]
    pool = sorted(r["response_id"] for p in eligible for r in p["responses"])
    if not 0 < sample_size <= len(pool): raise ValueError("audit size exceeds eligible response pool")
    sampled = sorted(random.Random(seed).sample(pool, sample_size))
    return common, eligible, sampled


def summarize(bundle, common, eligible, sampled):
    responses = [r for p in bundle["pairs"] for r in p["responses"]]
    counts = {}
    for condition in ("base", "full"):
        rows = [r for p in eligible for r in p["responses"] if r["condition"] == condition]
        labels = Counter(l for r in rows for l in r["triage"]["labels"])
        counts[condition] = {"denominator": len(rows), "labels": dict(labels), "percent": {l: 100 * n / len(rows) for l, n in labels.items()}, "sampled_responses": sum(r["response_id"] in sampled for r in rows)}
    return {"analysis_id": ANALYSIS_ID, "dataset_id": bundle["dataset_id"], "source_manifest_sha256": bundle["source_manifest_sha256"], "total_responses": len(responses), "usable_responses": sum(r["observable"]["parsed_ok"] for r in responses), "global_high_confidence_without_undetermined": sum(high(r) for r in responses), "common_shape_pairs": len(common), "selected_pairs": len(eligible), "selected_responses": 2 * len(eligible), "unique_evaluation_tasks": len({(p["task_id"], p["test_index"]) for p in eligible}), "unique_source_task_files": len({p["task_id"] for p in eligible}), "eligible_coverage_of_common_shape_pairs": len(eligible) / len(common), "by_condition": counts, "sampled_response_ids": sampled, "annotation_status": "model_only", "human_audit_status": "pending_export", "human_audit_completed": None, "labels_nonexclusive": True, "inference_limit": "Descriptive labels conditional on common shape correctness and both conditions having high model confidence. Not all-task error prevalence or a causal explanation."}
