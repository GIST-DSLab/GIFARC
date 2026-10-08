"""Original Appendix B aggregation over available runs within each task."""
from __future__ import annotations
from collections import defaultdict
import math
import statistics
from typing import Any

def _aggregate_per_task(per_case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_case_rows:
        groups[(row["model_name"], row["condition"], row["task_id"])].append(row)
    output: list[dict[str, Any]] = []
    for (model_name, condition, task_id), rows in sorted(groups.items()):
        condition_label = rows[0]["condition_label"]
        output.append(
            {
                "model_name": model_name,
                "condition": condition,
                "condition_label": condition_label,
                "task_id": task_id,
                "run_count": len(rows),
                "own_similarity_mean": _mean_key(rows, "own_similarity"),
                "mean_other_similarity_mean": _mean_key(rows, "mean_other_similarity"),
                "margin_delta_mean": _mean_key(rows, "margin_delta"),
                "own_reference_rank_mean": _mean_key(rows, "own_reference_rank"),
                "top1_rate": _mean_key(rows, "top1"),
                "top5_rate": _mean_key(rows, "top5"),
                "mrr": _mean_key(rows, "reciprocal_rank"),
            }
        )
    return output


def _summarize_conditions(per_task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_task_rows:
        groups[(row["model_name"], row["condition"])].append(row)
    output: list[dict[str, Any]] = []
    for (model_name, condition), rows in sorted(groups.items()):
        margins = [float(row["margin_delta_mean"]) for row in rows]
        output.append(
            {
                "model_name": model_name,
                "condition": condition,
                "condition_label": rows[0]["condition_label"],
                "n": len(rows),
                "run_count_mean": _mean_key(rows, "run_count"),
                "own_similarity_mean": _mean_key(rows, "own_similarity_mean"),
                "mean_other_similarity_mean": _mean_key(rows, "mean_other_similarity_mean"),
                "margin_delta_mean": _mean_values(margins),
                "margin_delta_std": _sample_std(margins),
            }
        )
    return output


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    return _mean_values([row[key] for row in rows if isinstance(row.get(key), (int, float))])


def _mean_values(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))
