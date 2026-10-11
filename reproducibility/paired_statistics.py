"""Scientific definitions copied from the saved-result statistical analysis.

The original numeric definitions are unchanged. Release loading and CLI live in
analyze_saved_statistics.py; inputs contain only existing saved metrics.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
import numpy as np
MODELS = ["gpt-5.4", "gpt-5.4-mini"]
CONDITIONS = ["rich", "neutral", "A+G", "neutral A+G", "C+G", "neutral C+G", "G only"]
TRACE_CONDITIONS = ["rich", "neutral", "neutral A+G"]
RAW_METRICS = ["MWR", "AAS", "exact", "GC", "FG"]
CACHED_METRICS = ["own", "other", "margin"]
EXPECTED_TABLE3 = {
    ("gpt-5.4", "rich"): [.082, .001, .452, .001],
    ("gpt-5.4", "neutral"): [.073, .002, .444, .005],
    ("gpt-5.4", "neutral A+G"): [.079, .002, .440, .001],
    ("gpt-5.4-mini", "rich"): [.098, .004, .452, .004],
    ("gpt-5.4-mini", "neutral"): [.090, .006, .446, .004],
    ("gpt-5.4-mini", "neutral A+G"): [.093, .002, .436, .004],
}
EXPECTED_TABLE8 = {
    ("gpt-5.4", "rich"): [.4527, .4000, .0527],
    ("gpt-5.4", "neutral"): [.4452, .3959, .0493],
    ("gpt-5.4", "neutral A+G"): [.4407, .3925, .0482],
    ("gpt-5.4-mini", "rich"): [.4522, .4125, .0397],
    ("gpt-5.4-mini", "neutral"): [.4462, .4084, .0377],
    ("gpt-5.4-mini", "neutral A+G"): [.4366, .4013, .0353],
}
EXPECTED_TABLE9 = {
    ("gpt-5.4", "rich"): [.772, .029, .919, .023, .898, .019],
    ("gpt-5.4", "neutral"): [.775, .021, .913, .026, .894, .016],
    ("gpt-5.4", "A+G"): [.783, .014, .924, .033, .896, .035],
    ("gpt-5.4", "neutral A+G"): [.752, .003, .920, .041, .897, .031],
    ("gpt-5.4", "C+G"): [.773, .007, .915, .028, .894, .018],
    ("gpt-5.4", "neutral C+G"): [.763, .021, .939, .015, .908, .009],
    ("gpt-5.4", "G only"): [.757, .020, .913, .031, .887, .021],
    ("gpt-5.4-mini", "rich"): [.154, .035, .386, .031, .306, .031],
    ("gpt-5.4-mini", "neutral"): [.177, .016, .388, .049, .289, .017],
    ("gpt-5.4-mini", "A+G"): [.137, .035, .367, .008, .273, .015],
    ("gpt-5.4-mini", "neutral A+G"): [.161, .009, .357, .054, .269, .035],
    ("gpt-5.4-mini", "C+G"): [.167, .021, .379, .021, .289, .029],
    ("gpt-5.4-mini", "neutral C+G"): [.147, .021, .394, .019, .298, .019],
    ("gpt-5.4-mini", "G only"): [.160, .010, .358, .018, .275, .027],
}

def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(base, label):
    return (base + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")) % (2**63)


def bootstrap_ci(values, samples, seed, run_equal=False):
    """Resample equal-weight task clusters; optional 3-run-balanced estimand.

    For the primary estimand, values are task-level mean paired differences. This
    is algebraically identical to resampling all available paired records by task
    and re-averaging within each selected task. Runs are never resampled separately.
    The sensitivity estimand uses an n_task x 3 matrix of paired differences.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, samples, 1000):
        ids = rng.integers(0, n, size=(min(1000, samples - start), n))
        selected = values[ids]
        if run_equal:
            valid = np.isfinite(selected)
            counts = valid.sum(axis=1)
            sums = np.where(valid, selected, 0.0).sum(axis=1)
            # Our real groups have >=80 tasks per run. Fail rather than silently
            # replacing an undefined run mean in a bootstrap replicate.
            if np.any(counts == 0):
                raise ValueError("A bootstrap draw has no valid pairs in a run")
            draws.append((sums / counts).mean(axis=1))
        else:
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError("Primary bootstrap requires finite task means")
            draws.append(selected.mean(axis=1))
    lo, hi = np.quantile(np.concatenate(draws), [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def collapse_pairs(records, task_ids, model, left, right, metric):
    pairs, tasks = [], []
    for task in task_ids:
        left_values, right_values, common = [], [], []
        matrix = [float("nan")] * 3
        for run in (1, 2, 3):
            a = records[(model, left, task, run)]["metrics"].get(metric)
            b = records[(model, right, task, run)]["metrics"].get(metric)
            if finite(a):
                left_values.append(float(a))
            if finite(b):
                right_values.append(float(b))
            if finite(a) and finite(b):
                row = {"task_id": task, "run": run, "left": float(a),
                       "right": float(b), "difference": float(a) - float(b)}
                pairs.append(row)
                common.append(row)
                matrix[run - 1] = row["difference"]
        tasks.append({
            "task_id": task,
            "paired_runs": len(common),
            "left_valid_runs": len(left_values),
            "right_valid_runs": len(right_values),
            "left_paired_mean": mean(r["left"] for r in common),
            "right_paired_mean": mean(r["right"] for r in common),
            "paired_difference": mean(r["difference"] for r in common),
            "separate_available_difference": (mean(left_values) - mean(right_values))
                if left_values and right_values else None,
            "matrix": matrix,
        })
    return pairs, tasks


def missing_reason(case, metric):
    if case.get("execution_error"):
        return "execution_error_flag"
    if case.get("parse_error"):
        return "parse_error_flag"
    meta = case.get("metric_metadata") or {}
    if metric == "FG" and meta.get("foreground", {}).get("foreground_empty"):
        return "empty_foreground"
    if metric in ("MWR", "AAS", "own", "other", "margin") and not case.get("parsed_inferred_rule", ""):
        return "empty_rule"
    return "metric_unavailable_no_error_flag"


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reproduce(records, tasks, samples, base_seed):
    runs, aggregates, audits, table8 = [], [], [], []
    for model in MODELS:
        for condition in CONDITIONS:
            for metric in RAW_METRICS + (CACHED_METRICS if condition in TRACE_CONDITIONS else []):
                run_means = []
                for run in (1, 2, 3):
                    cases = [records[(model, condition, t, run)] for t in tasks]
                    values = [v["metrics"].get(metric) for v in cases]
                    valid = [v for v in values if finite(v)]
                    run_means.append(mean(valid))
                    reasons = Counter(missing_reason(case, metric) for case, value in zip(cases, values) if not finite(value))
                    runs.append({"model": model, "condition": condition, "metric": metric, "run": run,
                                 "planned_n": len(tasks), "valid_n": len(valid), "missing_n": len(tasks) - len(valid),
                                 "valid_mean": mean(valid), "missing_reasons": json.dumps(reasons, sort_keys=True)})
                aggregates.append({"model": model, "condition": condition, "metric": metric,
                                   "mean_of_run_means": mean(run_means), "sample_sd_of_run_means": statistics.stdev(run_means),
                                   "valid_n": sum(r["valid_n"] for r in runs if r["model"] == model and
                                                  r["condition"] == condition and r["metric"] == metric)})
    index = {(r["model"], r["condition"], r["metric"]): r for r in aggregates}
    for table, expected, metrics in [(3, EXPECTED_TABLE3, ["MWR", "AAS"]),
                                      (9, EXPECTED_TABLE9, ["exact", "GC", "FG"])]:
        for (model, condition), vals in expected.items():
            for j, metric in enumerate(metrics):
                for k, statistic in enumerate(["mean_of_run_means", "sample_sd_of_run_means"]):
                    value = index[(model, condition, metric)][statistic]
                    audits.append({"table": table, "model": model, "condition": condition,
                                   "metric": metric, "statistic": statistic, "computed": value,
                                   "published": vals[2*j+k], "digits": 3,
                                   "rounding_matches": round(value, 3) == vals[2*j+k]})
    for (model, condition), vals in EXPECTED_TABLE8.items():
        for metric, published in zip(CACHED_METRICS, vals):
            means = [mean(v for r in (1, 2, 3) if finite(v := records[(model, condition, t, r)]["metrics"].get(metric))) for t in tasks]
            valid = [v for v in means if finite(v)]
            lo, hi = bootstrap_ci(valid, samples, stable_seed(base_seed, f"level/{model}/{condition}/{metric}"))
            row = {"model": model, "condition": condition, "metric": metric, "task_mean": mean(valid),
                   "n_tasks": len(valid), "task_sample_sd": statistics.stdev(valid),
                   "ci_low": lo, "ci_high": hi}
            table8.append(row)
            audits.append({"table": 8, "model": model, "condition": condition, "metric": metric,
                           "statistic": "task_mean", "computed": row["task_mean"], "published": published,
                           "digits": 4, "rounding_matches": round(row["task_mean"], 4) == published})
    return runs, aggregates, audits, table8


def analyze_pairs(records, task_ids, samples, base_seed):
    comparisons = [("rich", "neutral", ["MWR", "AAS", "own", "other", "margin"], "matched_wording"),
                   ("rich", "neutral", ["exact", "GC", "FG"], "matched_wording"),
                   ("A+G", "neutral A+G", ["exact", "GC", "FG"], "matched_wording"),
                   ("C+G", "neutral C+G", ["exact", "GC", "FG"], "matched_wording"),
                   ("rich", "neutral A+G", ["MWR", "AAS", "margin"], "wording_and_code_change")]
    all_pairs, all_tasks, results = [], [], []
    for model in MODELS:
        for left, right, metrics, kind in comparisons:
            for metric in metrics:
                pairs, tasks = collapse_pairs(records, task_ids, model, left, right, metric)
                identity = {"model": model, "left_condition": left, "right_condition": right,
                            "metric": metric, "comparison_type": kind}
                all_pairs.extend(dict(identity, **p) for p in pairs)
                all_tasks.extend(dict(identity, **{k:v for k,v in t.items() if k != "matrix"}) for t in tasks)
                common = [t for t in tasks if t["paired_runs"] > 0]
                strict = [t for t in tasks if t["paired_runs"] == 3]
                separate = [t for t in tasks if finite(t["separate_available_difference"])]
                label = f"{model}/{left}/{right}/{metric}"
                seed = stable_seed(base_seed, label)
                primary = [t["paired_difference"] for t in common]
                lo, hi = bootstrap_ci(primary, samples, seed)
                strict_lo, strict_hi = bootstrap_ci([t["paired_difference"] for t in strict], samples, stable_seed(base_seed, label + "/strict"))
                separate_lo, separate_hi = bootstrap_ci([t["separate_available_difference"] for t in separate], samples,
                                                         stable_seed(base_seed, label + "/separate"))
                restricted_lo, restricted_hi = bootstrap_ci([t["separate_available_difference"] for t in common], samples,
                                                           stable_seed(base_seed, label + "/separate-common"))
                matrix = np.array([t["matrix"] for t in common], dtype=float)
                run_equal = float(np.nanmean(matrix, axis=0).mean())
                run_lo, run_hi = bootstrap_ci(matrix, samples, stable_seed(base_seed, label + "/run-equal"), run_equal=True)
                counts = Counter(t["paired_runs"] for t in tasks)
                results.append(dict(identity,
                    planned_tasks=len(task_ids), paired_tasks=len(common), paired_task_runs=len(pairs),
                    pairs_run1=sum(p["run"] == 1 for p in pairs),
                    pairs_run2=sum(p["run"] == 2 for p in pairs),
                    pairs_run3=sum(p["run"] == 3 for p in pairs),
                    tasks_0_pairs=counts[0], tasks_1_pair=counts[1], tasks_2_pairs=counts[2], tasks_3_pairs=counts[3],
                    left_task_mean=mean(t["left_paired_mean"] for t in common),
                    right_task_mean=mean(t["right_paired_mean"] for t in common),
                    paired_mean_difference=mean(primary), ci_low=lo, ci_high=hi,
                    ci_includes_zero=(lo <= 0 <= hi),
                    complete3_tasks=len(strict), complete3_difference=mean(t["paired_difference"] for t in strict),
                    complete3_ci_low=strict_lo, complete3_ci_high=strict_hi,
                    separate_available_tasks=len(separate),
                    separate_available_difference=mean(t["separate_available_difference"] for t in separate),
                    separate_available_ci_low=separate_lo, separate_available_ci_high=separate_hi,
                    separate_on_common_tasks_difference=mean(t["separate_available_difference"] for t in common),
                    separate_on_common_tasks_ci_low=restricted_lo, separate_on_common_tasks_ci_high=restricted_hi,
                    run_equal_paired_difference=run_equal, run_equal_ci_low=run_lo, run_equal_ci_high=run_hi,
                    bootstrap_seed=seed))
    return all_pairs, all_tasks, results


def self_test():
    # Unequal runs demonstrate why record-pooling and separate available-run means
    # must not silently replace the paired, equal-task estimator.
    records = {}
    for task in (1, 2):
        for run in (1, 2, 3):
            a = {1: [1, 0, None], 2: [1, 1, 1]}[task][run-1]
            b = {1: [0, None, 1], 2: [0, 0, 0]}[task][run-1]
            records[("m", "a", task, run)] = {"metrics": {"v": a}}
            records[("m", "b", task, run)] = {"metrics": {"v": b}}
    pairs, tasks = collapse_pairs(records, [1, 2], "m", "a", "b", "v")
    assert len(pairs) == 4 and [t["paired_runs"] for t in tasks] == [1, 3]
    assert mean(t["paired_difference"] for t in tasks) == 1
    assert mean(t["separate_available_difference"] for t in tasks) == .5
    # Equal task weighting differs from equal record weighting.
    values = [0., 1.]
    assert mean(values) == .5 and mean([0., 1., 1., 1.]) == .75
    assert bootstrap_ci([.2]*10, 1000, 10) == (.2, .2)
    assert bootstrap_ci(values, 1000, 10) == bootstrap_ci(values, 1000, 10)
    assert all(math.isclose(v, .2, abs_tol=1e-14) for v in bootstrap_ci([[.2]*3]*10, 1000, 10, True))
    # True zero is finite; missing is not. Errors are not implicitly zeroed here.
    assert finite(0) and not finite(None) and not finite(float("nan"))
    print("PASS: pairing, missingness, task weights, clustered CI, deterministic seed, true zeros", flush=True)
