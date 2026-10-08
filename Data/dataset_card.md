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

The score was validated against a 140-task consensus subset with 103 aligned and 37 misaligned tasks. The reported precision and recall apply to that subset, not to human verification of the full corpus. Scores are proxies for sketch-to-grid alignment and do not certify training readiness, uniquely correct analogy, or filtered-training effectiveness. The six unscored tasks' execution limitation does not establish their quality or difficulty.

## Observable Artifacts and Interpretation Scope

GIFARC is a GIF-derived resource for analogy-guided task framing, with observable task annotations, executable code and grid demonstrations. It supports empirical comparisons of model-generated rule descriptions and solving performance.

The annotations and reverse captions are observable artifacts, not hidden model reasoning traces. Alignment scores and rule-description similarity do not establish causal mechanisms or guarantee solving or transfer gains.

## Grid-Size Compatibility

The corpus includes grids beyond ARC's 30x30 bound. All stored input and output grids are within that bound for 798 tasks; 816 tasks contain at least one larger grid. Among 17,010 pairs, 3,160 input grids and 5,578 output grids exceed 30 rows or columns. This compatibility criterion is independent of alignment-score thresholds and execution checks; size maxima are listed in `statistics.json`.

## Historical Generation Records

Historical generation materials lack complete rejected-candidate and API-attempt records. Filter-specific task rejection rates and actual API retry counts cannot be recovered; input, pair and program failure counts do not identify rejected-task counts.
