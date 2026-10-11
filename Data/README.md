# GIFARC Data Supplement (Anonymous)

This archive contains the processed GIFARC dataset used for the anonymous ARR/EMNLP submission.
It is intended to be uploaded through the ARR **Data** field.

## Contents

- `data/data.jsonl`: all 1,614 processed GIFARC tasks, with canonical IDs and alignment metadata, one task per line.
- `data/alignment_scores.jsonl`: per-task source hashes, scores/status/reasons and anonymous reverse captions.
- `alignment_report.json`: scoring protocol, coverage, threshold counts and anonymous unscored-attempt evidence.
- `task_metadata.jsonl` and `task_metadata.csv`: optional per-task dimensions and exact alignment metadata, joined by canonical ID and source SHA-256.
- `task_metadata_schema.json`: field definitions for the optional metadata.
- `subset_ids.json` and `subset_ids/`: 11 optional selections as sorted canonical ID lists and equivalent text files.
- `schema.json`: description of the main records, sidecars, selectors and grid representation.
- `statistics.json`: basic dataset statistics and checksum for `data/data.jsonl`.
- `dataset_card.md`: short dataset card for reviewers.
- `DATA_LICENSE.md`: data release note.
- `checksums.txt`: SHA-256 checksums for every release file except the checksum list itself.

## Dataset summary

GIFARC contains 1,614 processed ARC-style tasks with 17,010 input-output grid pairs.
Each task contains executable Python transformation code, input-output grid pairs, and generation support artifacts used to construct the task.

The released archive does **not** contain original GIF files or source GIF URLs.
The data consists of synthesized ARC-style grids, task descriptions embedded in code comments, and executable solution code.

## JSONL format

Each line is a JSON object with the following fields:

- `source`: executable Python code for the generated ARC-style task.
- `examples`: input-output grid pairs. Each grid is a 2D integer array with values from 0 to 9.
- `seeds`: generation support artifacts inherited from the synthesis pipeline. Despite the field name, this does **not** mean random seeds. In the released data, each task typically contains four retrieved ARC-style seed example references (e.g., `.py` filenames) and one generated task sketch containing concepts and description.

## Anonymity note

This archive has been prepared for anonymous peer review. It should not contain author names, institution names, personal emails, non-anonymous repository links, original GIF files, or source GIF URLs.

## Alignment metadata update

The original `source`, `examples`, and `seeds` values and task order are preserved. Each task now also has an integer `task_id` and an `alignment` object. Canonical experiment IDs were joined by unique source-code SHA-256 and exact equality of the original task fields; IDs are not JSONL line numbers.

`alignment` contains `score` (a finite cosine similarity float at stored precision, or `null`), `status` (`scored` or `unscored`), and `unscored_reason` (`null` for scored tasks). The sidecar contains the same alignment object, a UTF-8 `source_sha256`, and the anonymous `reverse_caption` used in scoring, or `null` if absent.

Scores are available for 1,608 of 1,614 tasks. Canonical IDs 18, 260, 536, 719, 781 and 1310 remain unscored, with reason `reverse_caption_max_output_tokens`: both attempts for each task reached 16,384 output tokens without returning a caption. Their score is unknown and is never replaced by zero. This is an execution limitation, not a judgment about task quality or difficulty.

Scoring uses GPT-5.4 to reverse-caption the first three valid stored input-output pairs in their original order, without the sketch or executable code. The score is cosine similarity between embeddings of that caption and the original sketch description, using `text-embedding-3-large` with 3,072 dimensions. The protocol and verified coverage are in `alignment_report.json`.

| Minimum score | Retained corpus tasks | Also all stored grids ≤30×30 | Unscored tasks excluded |
| --- | ---: | ---: | ---: |
| 0.45 | 987 | 469 | 6 |
| 0.50 | 622 | 285 | 6 |
| 0.55 | 319 | 140 | 6 |
| 0.60 | 129 | 61 | 6 |
| 0.65 | 39 | 21 | 6 |

These corpus counts apply to all 1,614 released tasks. Human validation is conditional on the separate 140-task binary-agreement subset: 103 aligned and 37 misaligned, with 60 disagreements among 200 annotated tasks excluded. The existing binary mapping treats `y`/`y?` as aligned and `n`/`n?` as misaligned. Before filtering, 103/140 (73.6%) consensus tasks are aligned. At threshold 0.45, 83/105 retained consensus tasks are aligned (79.0% precision), 5.5 percentage points above that base rate; recall is 83/103 (80.6%). These results validate the score only on the consensus subset; they are not a random full-corpus quality estimate or adjudicated ground truth. The release does not certify all task quality, training readiness, or filtered-training gains. Exact denominators and the other thresholds are in `statistics.json`. No single filtering threshold is prescribed.

## Construct a filtered subset

From the extracted archive directory, choose a threshold and retain only scored tasks meeting it:

```python
import json

threshold = 0.50
with open("data/data.jsonl", encoding="utf-8") as source, open("filtered.jsonl", "w", encoding="utf-8") as target:
    for line in source:
        task = json.loads(line)
        alignment = task["alignment"]
        if alignment["status"] == "scored" and alignment["score"] >= threshold:
            target.write(line)
```

At 0.50 this writes 622 tasks. The complete archive retains all 1,614 tasks, including the six unscored ones. Verify file integrity with `sha256sum -c checksums.txt` from the extracted archive directory.

## Resource and interpretation scope

GIFARC is a GIF-derived resource for analogy-guided task framing, with observable task annotations, executable code and grid demonstrations. It supports empirical comparisons of model-generated rule descriptions and solving performance.

The annotations and reverse captions are observable artifacts, not hidden model reasoning traces. Alignment scores and rule-description similarity do not establish causal mechanisms or guarantee solving or transfer gains.

## Grid sizes and generation-record limits

The full corpus retains all stored grid sizes. A task meets ARC's 30x30 bound only when every stored input and output grid has at most 30 rows and 30 columns. This holds for 798 tasks; the other 816 contain at least one larger grid. Across 17,010 pairs, 3,160 input grids (18.6%) and 5,578 output grids (32.8%) exceed the bound. Input maxima are 196 rows, 510 columns and 12,500 cells; output maxima are 10,933 rows, 3,105 columns and 283,024 cells. Each maximum is computed separately and may correspond to a different grid. Size compatibility is separate from execution checks and alignment-score filtering. See `statistics.json`.

Historical generation materials lack complete rejected-candidate and API-attempt records. Filter-specific task rejection rates and actual API retry counts cannot be recovered; input, pair and program failure counts do not identify rejected-task counts.


## Optional dimension metadata and exact selectors

The full release retains all 1,614 tasks and imposes no new 30×30 requirement. Users can optionally select the 798 tasks whose **every stored input and output** has height ≤30 and width ≤30, apply an alignment threshold, or use both conditions. `all_grids_within_30` does not guarantee the dimensions of new `generate_input()` results or enforce generator behavior. Size compatibility does not certify alignment or task quality.

`task_metadata.jsonl` and its CSV equivalent contain 1,614 rows in ascending canonical `task_id` order. Metadata order can differ from `data/data.jsonl`: join by `task_id`, which is not a line number, and check `source_sha256` against the exact UTF-8 source string and `data/alignment_scores.jsonl`. Rows include the stored pair count, input/output maxima, oversized input/output counts and exact alignment score/status/reason. Maximum height, width and cells are computed independently, so multiplying maximum height by maximum width does not recover the maximum cell count. CSV empty cells mean JSON null and CSV booleans are `true`/`false`.

`subset_ids.json` contains all 11 sorted, exact selections: `all_grids_within_30`, five `alignment_score_ge_0_XX` selectors and five `all_grids_within_30_and_alignment_score_ge_0_XX` intersections. Equivalent one-ID-per-line files are in `subset_ids/`. The bounded-only selection is independent of score status; all score-based selectors exclude the six unknown/null scores. The table above lists each intersection count.

From the extracted archive directory, this example writes the optional intersection at 0.50 while preserving main-record order and each selected JSONL line:

```python
import hashlib
import json

with open("task_metadata.jsonl", encoding="utf-8") as source:
    metadata = {row["task_id"]: row for row in map(json.loads, source)}
with open("subset_ids.json", encoding="utf-8") as source:
    ids = set(json.load(source)["all_grids_within_30_and_alignment_score_ge_0_50"])

count = 0
with open("data/data.jsonl", encoding="utf-8") as source, open("selected.jsonl", "w", encoding="utf-8") as target:
    for line in source:
        task = json.loads(line)
        info = metadata[task["task_id"]]
        assert hashlib.sha256(task["source"].encode("utf-8")).hexdigest() == info["source_sha256"]
        assert task["alignment"]["score"] == info["alignment_score"]
        assert task["alignment"]["status"] == info["alignment_status"]
        if task["task_id"] in ids:
            target.write(line)
            count += 1
assert count == 285
```

Editing or regenerating examples requires a new dimension audit. The original main dataset, score sidecar, source/examples/seeds values, scores/status and record order remain byte-identical to the preceding release. This addendum does not provide a new quality evaluation or completed filtered-training experiment.
