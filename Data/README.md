# GIFARC Data Supplement (Anonymous)

This archive contains the processed GIFARC dataset used for the anonymous ARR/EMNLP submission.
It is intended to be uploaded through the ARR **Data** field.

## Contents

- `data/data.jsonl`: all 1,614 processed GIFARC tasks, with canonical IDs and alignment metadata, one task per line.
- `data/alignment_scores.jsonl`: per-task source hashes, scores/status/reasons and anonymous reverse captions.
- `alignment_report.json`: scoring protocol, coverage, threshold counts and anonymous unscored-attempt evidence.
- `schema.json`: description of the top-level fields and grid representation.
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

| Minimum score | Retained corpus tasks | Unscored tasks excluded |
| --- | ---: | ---: |
| 0.45 | 987 | 6 |
| 0.50 | 622 | 6 |
| 0.55 | 319 | 6 |
| 0.60 | 129 | 6 |
| 0.65 | 39 | 6 |

These counts apply to all 1,614 released tasks. Precision and recall in the paper are computed on the separate 140-task human-consensus subset (103 aligned and 37 misaligned); they are reproduced in `statistics.json`. The automatic scores do not establish that all tasks were human verified. No single filtering threshold is prescribed.

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
