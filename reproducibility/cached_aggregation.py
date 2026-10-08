"""Original cached-results aggregation; internal task IDs are zero-based.
The release adapter converts canonical task IDs at this boundary only.
"""
from __future__ import annotations
import csv
import report_metrics as reporter
THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65)
METRICS = ("exact", "GC", "FG", "AAS", "MWR")
MODELS = ("gpt-5.4", "gpt-5.4-mini")
CONDITIONS = ("rich", "neutral", "neutral A+G", "A+G", "C+G", "G only", "neutral C+G")

def make_subsets(targets: set[int], scores: dict[int, float]):
    scored = {task for task in targets if task + 1 in scores}
    subsets = [("original_all_100", None, targets), (f"scored_all_{len(scored)}", None, scored)]
    subsets.extend(
        (f"score_ge_{threshold:.2f}", threshold,
         {task for task in scored if scores[task + 1] >= threshold})
        for threshold in THRESHOLDS
    )
    return subsets


def aggregate(runs, subsets):
    per_run, summary = [], []
    for label, threshold, retained in subsets:
        for model in MODELS:
            for condition in CONDITIONS:
                repeated = []
                for run, rows in runs:
                    selected = [
                        row for row in rows if row["task_id"] in retained
                        and row["model_name"] == model and row["condition"] == condition
                    ]
                    result = {
                        "subset": label, "threshold": threshold, "run": run.name,
                        "model": model, "condition": condition, "retained_tasks": len(retained),
                        "execution_error_n": sum(r["execution_error"] for r in selected),
                        "parse_error_n": sum(r["parse_error"] for r in selected),
                    }
                    for metric in METRICS:
                        result[f"{metric}_valid_n"] = sum(
                            isinstance(row.get(metric), (int, float)) for row in selected
                        )
                        result[f"{metric}_cached_valid_mean"] = reporter._mean(selected, metric)
                    per_run.append(result)
                    repeated.append(result)
                result = {
                    "subset": label, "threshold": threshold, "model": model,
                    "condition": condition, "retained_tasks": len(retained), "runs": len(runs),
                    "planned_rows": len(retained) * len(runs),
                    "execution_error_n": sum(r["execution_error_n"] for r in repeated),
                    "parse_error_n": sum(r["parse_error_n"] for r in repeated),
                }
                for metric in METRICS:
                    means = [r[f"{metric}_cached_valid_mean"] for r in repeated]
                    means = [x for x in means if x is not None]
                    result[f"{metric}_valid_n_total"] = sum(r[f"{metric}_valid_n"] for r in repeated)
                    result[f"{metric}_run_mean"] = reporter._mean_values(means)
                    result[f"{metric}_run_sample_std"] = reporter._sample_std(means)
                summary.append(result)
    return per_run, summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
