# Dataset Card: GIFARC (Anonymous Review Version)

## Dataset Description

GIFARC is a GIF-derived ARC-style resource for studying analogy-guided task framing.
The dataset converts short dynamic visual events into synthetic ARC-style grid-transformation tasks.
Each task contains input-output grid examples and executable Python code implementing the latent transformation.

## Intended Use

This release is intended for research on ARC-style reasoning, analogy-guided task framing, synthetic data generation, and evaluation of reasoning models.
It is not intended as a replacement for human-authored ARC tasks.

## Data Format

The main file is `data/data.jsonl`. Each line is one JSON object with executable task code and input-output examples.
Grids are represented as 2D arrays of integer color values from 0 to 9.

## Source Data and Redistribution

The pipeline used G-rated GIFs as sources for deriving visual abstractions during dataset construction.
This anonymous data archive does not redistribute original GIF files or source GIF URLs.

## Limitations

The analogy and task-generation process may introduce noise, subjective interpretations, or imperfect alignment between the original visual abstraction and the final grid task.
The dataset should therefore be used as a resource for studying analogy-guided framing rather than as a collection of uniquely correct analogical interpretations.

## Alignment Metadata and Coverage

This version preserves the original 1,614 tasks and 17,010 input-output pairs and adds a verified canonical experiment `task_id` and an `alignment` object to every task. The sidecar `data/alignment_scores.jsonl` supplies the same alignment fields, the SHA-256 of the exact UTF-8 source code, and the anonymous reverse caption used for scoring. No original task content was replaced.

The automatic alignment score compares a grid-only reverse caption of the first three valid stored input-output pairs against the original task sketch description through embedding cosine similarity. The captioning model does not receive the executable code or sketch. Finite scores are available for 1,608 tasks (99.6%). Six tasks have `score: null`, `status: unscored`, and `unscored_reason: reverse_caption_max_output_tokens`, because both reverse-caption attempts reached a 16,384-token output limit without a caption. Null scores represent unknown alignment and are excluded from thresholded subsets.

Thresholds 0.45, 0.50, 0.55, 0.60 and 0.65 retain 987, 622, 319, 129 and 39 tasks, respectively, using score >= threshold. No single threshold is prescribed.

## Human Validation Scope

The score was validated only on a 140-task binary-agreement subset with 103 aligned and 37 misaligned tasks; 60 disagreements among 200 annotated tasks were excluded. The existing binary mapping treats `y`/`y?` as aligned and `n`/`n?` as misaligned. Before filtering, 103/140 (73.6%) consensus tasks are aligned. At threshold 0.45, 83/105 retained consensus tasks are aligned (79.0% precision), a 5.5 percentage-point increase over that base rate, with recall 83/103 (80.6%). Exact counts and other thresholds are in `statistics.json`. These are conditional consensus-subset results, not a random full-corpus quality estimate or adjudicated ground truth. The release does not certify all task quality. Scores are proxies for sketch-to-grid alignment and do not certify training readiness, uniquely correct analogy, or filtered-training effectiveness. The six unscored tasks' execution limitation does not establish their quality or difficulty.

## Observable Artifacts and Interpretation Scope

GIFARC is a GIF-derived resource for analogy-guided task framing, with observable task annotations, executable code and grid demonstrations. It supports empirical comparisons of model-generated rule descriptions and solving performance.

The annotations and reverse captions are observable artifacts, not hidden model reasoning traces. Alignment scores and rule-description similarity do not establish causal mechanisms or guarantee solving or transfer gains.

## Grid-Size Compatibility

The corpus includes grids beyond ARC's 30x30 bound. All stored input and output grids are within that bound for 798 tasks; 816 tasks contain at least one larger grid. Among 17,010 pairs, 3,160 input grids and 5,578 output grids exceed 30 rows or columns. This compatibility criterion is independent of alignment-score thresholds and execution checks; size maxima are listed in `statistics.json`.

## Historical Generation Records

Historical generation materials lack complete rejected-candidate and API-attempt records. Filter-specific task rejection rates and actual API retry counts cannot be recovered; input, pair and program failure counts do not identify rejected-task counts.


## Optional Stored-Grid Metadata and Selections

The original `data/data.jsonl` and `data/alignment_scores.jsonl` remain byte-identical to the preceding release, preserving original task payloads, scores/status and order. `task_metadata.jsonl` and its CSV equivalent add 1,614 rows of optional dimensions, stored-pair counts and exact alignment metadata. Join by canonical `task_id` and verify the exact UTF-8 `source_sha256`; ID order is not the original JSONL order. Field definitions are in `task_metadata_schema.json`. Input/output height, width and cell maxima are independent and may come from different stored grids.

`subset_ids.json` and 11 matching text files in `subset_ids/` expose optional user selections. The five inclusive alignment thresholds retain 987/622/319/129/39 tasks; intersecting each with the 798-task stored-grid size selection retains 469/285/140/61/21 tasks. Null scores remain unknown and are excluded from score selections. The dimension-only selector is independent of score status.

No 30×30 requirement is imposed on the full corpus, and no primary alignment threshold is prescribed. Dimension flags cover every stored example only; they do not constrain or guarantee newly generated examples. Editing or regenerating examples requires a new audit. This addendum does not provide a new quality evaluation or completed filtered-training experiment.
