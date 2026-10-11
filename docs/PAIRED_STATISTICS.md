# Task-paired statistics from saved runs

Run from the Software directory with NumPy available:

```bash
python reproducibility/analyze_saved_statistics.py --help
python reproducibility/analyze_saved_statistics.py --self-test
python reproducibility/analyze_saved_statistics.py \
  --output-dir results/paired_statistics --verify-included
```

The output directory must be new. The default analysis uses 50,000 draws and seed base `20261008`. The scientific functions in `reproducibility/paired_statistics.py` are copied unchanged from the actual saved-result analysis; source-file and definition hashes are recorded in `reproducibility/source-provenance.json`. The release adapter changes input loading and public identifiers. It performs no model calls, embedding calls, or task-program execution.

## Inputs and estimator

`reproducibility/summaries/paired-statistics/normalized-records.csv` contains 4,200 saved records: 100 selected targets × three recorded runs × two models × seven conditions. It contains numeric metrics and technical failure flags only. Public `task_id` values are canonical one-based IDs, with original zero-based analysis IDs shifted by one. `source-case-hashes.csv` binds each public model/condition/task/run key to the original saved-case SHA-256 without exposing internal run directories or responses. The raw cases are not bundled, so their hashes establish provenance rather than permit regeneration of unavailable responses.

For each contrast and metric, a pair exists only when both values are finite for the **same task and same recorded run**. True numeric zeros are retained; missing values are excluded rather than set to zero. Differences are left condition minus right condition. Within each task, valid paired run differences are averaged; the primary estimate is the equal-weight mean of these task means. A task with no valid paired run is excluded from that contrast. Metric-specific valid denominators and missingness reasons are retained in `valid-run-summary.csv`.

Bootstrap replicates resample whole tasks with replacement, then average the selected task means. Recorded runs are never resampled separately. The 95% intervals use the 2.5th and 97.5th percentiles with NumPy's linear quantile interpolation. Each comparison receives a stable seed calculated from the base plus the first eight bytes of its SHA-256 label, modulo `2**63`. The reference run used Python 3.12.14 and NumPy 2.3.5. Pairing recorded run indices does not assert shared random draws. Some recorded run-1 cases were retained from an earlier saved run, as stated in the sanitized analysis manifest.

The full output includes 34 comparisons, strict-three-pair estimates, separate available-run estimates, and run-equal sensitivity estimates. Six core intervals compare Full Description (`rich`) with Metaphor-Removed (`neutral`) for Usage Ratio (`MWR`), original saved Similarity (`AAS`), and task-specific cached similarity margin (`margin`). All 18 matched solving comparisons cover exact match, grid-cell accuracy (`GC`), and foreground accuracy (`FG`) for `rich` vs `neutral`, `A+G` vs `neutral A+G`, and `C+G` vs `neutral C+G`, across both models. The `rich` vs `neutral A+G` comparisons change both wording and code availability and are separately labeled.

## Core reference values

| Model | Metric | Paired difference | Pointwise 95% CI | Paired tasks / task-runs |
|---|---|---:|---|---:|
| GPT-5.4 | Usage Ratio | 0.008533 | [0.003926, 0.013339] | 99 / 282 |
| GPT-5.4 | Similarity | 0.008160 | [0.001712, 0.014457] | 99 / 282 |
| GPT-5.4 | Cached margin | 0.003558 | [-0.000822, 0.007792] | 99 / 282 |
| GPT-5.4-mini | Usage Ratio | 0.008010 | [0.002586, 0.013365] | 98 / 289 |
| GPT-5.4-mini | Similarity | 0.005396 | [-0.002540, 0.013160] | 98 / 289 |
| GPT-5.4-mini | Cached margin | 0.001444 | [-0.004045, 0.006817] | 98 / 289 |

These results support more metaphorical language and mixed alignment evidence in emitted rule descriptions. The margin is cached own-task cosine minus mean other-task cosine. Cached `own` and originally saved `AAS` differ slightly per case and are preserved as distinct measures.

## Outputs and interpretation

The adapter reproduces the included `paired-task-run-differences.csv`, `paired-task-means.csv`, `paired-confidence-intervals.csv`, `valid-run-summary.csv`, `original-aggregate-summary.csv`, `published-rounding-audit.csv`, and `table8-condition-confidence-intervals.csv`. It additionally writes convenient six-row `core-confidence-intervals.csv` and 18-row `solving-confidence-intervals.csv` files. `--verify-included` compares every regenerated field to its included reference, with `1e-12` numeric tolerance, and checks all 126 printed statistics from the original Tables 3/8/9, which correspond to Tables 3/8/10 in the final manuscript. The original table means and run SDs retain their original estimands; they are not silently replaced by paired task means. `summarize_trace_margin.py` remains available to reproduce the earlier cached-margin summary.

Intervals are pointwise, conditional on these saved executions and available responses. They do not provide simultaneous multiple-comparison guarantees, equivalence tests, or evidence of zero effect when an interval includes zero. The selected 100 targets are not a probability sample of all GIFARC tasks. Pair exclusion does not resolve informative missingness. The analysis does not quantify uncertainty over future runs, internal reasoning, or downstream transfer.
