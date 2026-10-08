# ICL error analysis and reviewed sample

The ARC-AGI-2 ICL analysis reads existing Base and Full Description responses from GPT-5.4-mini. It covers 120 source task files, 167 test queries, three repeats, and two conditions: 1,002 saved responses. Technical failures and missing predictions are distinguished from semantic output errors. No solver or classifier was rerun for this supplement.

The actual preliminary classifier was **GPT-6.1 Sol through Codex with ultra reasoning effort**. Its configured alias was `gpt-6.1-sol`; an immutable serving snapshot was not recovered. This was in-session model classification of saved task packets, not a separate judge API experiment. Packets included training examples, test input, reference output, predicted output, and the complete declared rule. Dimensions, cell/color counts, equality, and overlap differences were computed observations; semantic labels were model proposals. Categorical confidence is uncalibrated.

## Label taxonomy

These are the original nine interface labels. Multiple labels are allowed; `no_observed_error` is used alone. `undetermined` may accompany an observed error if other subtypes remain unclear.

| Key | Definition |
| --- | --- |
| `format_runtime` | A recorded API, timeout, parse, or execution failure prevents a valid prediction. |
| `shape` | Prediction and reference differ in grid height or width; this concerns the entire output grid. |
| `object_selection` | A required object/part is omitted, or an object/part that should not be processed is selected. |
| `localization` | A matched object/part is moved, aligned, or placed at incorrect coordinates. |
| `color` | A matched object/background has an incorrect color, distinguished from differences due only to motion or missing objects. |
| `declared_rule_contradiction` | Applying the model's declared rule conflicts with the training input/output examples. |
| `rule_output_inconsistency` | Applying the declared rule to the test input differs from the model's saved prediction. |
| `undetermined` | Evidence is insufficient to assign some or all error subtypes. |
| `no_observed_error` | The prediction matches the reference and no output error is observed within the reviewed scope. |

The semantic labels describe saved outputs and declared rules. They do not reveal hidden reasoning mechanisms or establish renderer causality. `icl_audit.py` copies the actual paired selection, sampling, and model-label aggregation functions; it consumes supplied classifications and does not generate semantic labels.

## Paired cohort and fixed random sample

The common shape-correct subset contains **111 response pairs** across the three repeats. A pair is one test query in one repeat with one Base and one Full response; it is not necessarily a unique task. Of these, **79 pairs** have parsed, shape-correct predictions in both conditions, and both classifications are `high` confidence without `undetermined`. This gives **158 responses**, 79 per condition, covering 44 distinct test queries from 35 source task files. It is a conditional diagnostic subset, not all-task error prevalence.

The original procedure uniformly samples **30 responses** without replacement from sorted eligible response IDs using seed **20261007**. Four are exact matches checked mechanically against the reference. The remaining **26 incorrect responses** have confirmed human error judgments. The 26 were not separately sampled. `icl-confidence-selection.json` preserves the selected pair IDs and the fixed 30 sampled response IDs; `icl-model-label-counts.csv` and `icl-model-label-summary.json` keep the cohort's nonexclusive model-only counts and denominators.

Humans could inspect training input/output pairs, test input, reference output, prediction, and the declared rule; saved response text and summaries were also available in the original interface. Decisions recorded chosen labels, confidence, and written evidence. Model-proposal viewing was available and one confirmed sampled review records proposal exposure before the first decision, so independence or full blinding is not claimed.

## Released 26-response sample

`human-reviewed-errors.jsonl` is a compact anonymous export of only these 26 confirmed sampled error judgments. It includes task/test/repeat/condition identifiers, grids, the saved declared rule, source hash, recorded human labels/confidence/evidence, and corresponding model labels. It excludes reviewer identity, timestamps, internal paths, API identifiers, full prompts, and raw response logs. The complete remaining prediction population is not bundled.

The human records contain the following two rule labels:

| Recorded rule category | Responses |
| --- | ---: |
| `declared_rule_contradiction` only | 12 |
| `rule_output_inconsistency` only | 3 |
| Both | 11 |
| Total reviewed incorrect responses | 26 |

The nonexclusive counts are 23 for `declared_rule_contradiction` and 14 for `rule_output_inconsistency`, with an intersection of 11. Here “only” refers to these two reported categories; absence of other visual labels does not establish that no visual error occurred. The nine-label interface was available, and this supplement does not infer that human review was originally restricted to only two labels.

```bash
python reproducibility/summarize_icl_review.py
```

`human-review-summary.json` records the fixed-sample counts and limitations. Unreviewed classifications remain **model-only**. This selective human sample audit is not whole-cohort human annotation, judge-accuracy estimation, or confidence calibration.
