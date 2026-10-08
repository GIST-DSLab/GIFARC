# Reproducing the released analyses

The `reproducibility/` directory reuses definitions from the actual alignment and Section 5 analysis sources. `source-provenance.json` records the original repository-relative filenames, source-file hashes, copied symbols, and definition hashes. The copied definitions are unchanged; imports and CLI adapters are adapted to the anonymous release. No new model or embedding calls were made to prepare this supplement.

## Reverse-caption alignment

The actual method first validates stored input/output pairs, then uses the **first three valid pairs in stored order**, or all available pairs if fewer than three. It provides grids only to GPT-5.4; the task sketch and executable code are withheld. The prompt requests a 2–4 sentence transformation description using colors by their roles, with a strict JSON schema containing a single `description` field.

Source configuration verifies model snapshot **`gpt-5.4-2026-03-05`**, **medium** reasoning effort, and **16,384** maximum output tokens. These settings are taken from the original request builder and full-corpus preflight, rather than inferred from a model name. Captions and original sketch descriptions are embedded using **`text-embedding-3-large`**, with **3,072** dimensions in the full-corpus run. The stored score is their cosine similarity. The concepts are parsed separately from the sketch description and are not concatenated to the embedded sketch.

The full-corpus runner disables SDK transport retries (`max_retries=0`) and permits one explicit retry only after one failed caption attempt, preserving the prompt, model, effort, and output limit. Both attempts for each of the six unscored tasks reached the 16,384-token output limit. The release contains 1,608 finite scores and six explicit unscored records out of 1,614 tasks. It does not substitute zero for these failures.

`alignment_protocol.py` contains the actual grid normalizer, stored-pair selector, prompt, strict request-body recipe, caption parser, cosine, and retry eligibility helper. These definitions make no request by themselves. To recompute cosine from existing vectors:

```bash
pip install -r reproducibility/requirements.txt
python reproducibility/score_alignment.py --help
python reproducibility/score_alignment.py \
  --data Data/data/data.jsonl --vectors saved_vectors.jsonl \
  --output results/recomputed_scores.jsonl
```

The input JSONL must contain canonical `task_id`, `caption_embedding`, and `description_embedding`, with two 3,072-dimensional vectors per row and optionally the matching `reverse_caption` text. `--data` binds recomputed scores to canonical task identities and source hashes; only tasks with supplied vectors appear in its output. The CLI performs no network calls and does not validate how the supplied caption or vectors were obtained. The separate Data sidecar includes the stored reverse captions and exact scores; original embedding vectors are not bundled. API-backed re-execution would require the documented original prompt and model settings plus independent service access and authorization.

## Filtering and task identity

The Data supplement adds the stored alignment object to each original task row and supplies `data/alignment_scores.jsonl`. Each sidecar row binds its canonical `task_id` and nested `alignment` object (full-precision score/status/reason) to the UTF-8 SHA-256 of its task `source`; it also contains the stored `reverse_caption`, or null when unavailable. Task identities are independent of JSONL line order. Some legacy evaluation code used zero-based IDs; only the adapters convert those IDs. Public task IDs and support IDs are canonical one-based integers.

`filter_alignment.py` checks status/reason consistency, duplicate IDs, finite cosine values, and, when `--scores` is supplied, the sidecar equality and source hashes. It retains scores **greater than or equal to** the requested threshold and excludes all null scores. Selected JSONL lines are copied verbatim without changing source, pairs, seeds, or decimal score precision. Output files must be new.

The reported thresholds 0.45, 0.50, 0.55, 0.60, and 0.65 retain 987, 622, 319, 129, and 39 tasks, respectively, and are sensitivity settings. The existing solving runs and SFT checkpoints were not created by applying this later alignment filter.

The 1,614 tasks contain 17,010 stored input/output pairs. Every stored input and output is at most 30 rows and 30 columns for 798 tasks; the remaining 816 tasks have at least one larger grid. There are 3,160 input grids (18.6%) and 5,578 output grids (32.8%) beyond this bound. These stored-grid statistics do not establish bounds for new examples sampled from task programs.

## Saved solving metrics

`saved-case-metrics.jsonl` contains 4,200 compact records: 100 original targets × three repeats × two models × seven conditions. It includes saved metric values, technical failure flags, and source hashes, with no full prompts, raw responses, API IDs, or internal file paths. `cached-solving-summary.csv` and `cached-solving-per-run.csv` contain the original cached reaggregation, including metric-specific valid denominators.

The copied `cached_aggregation.py` and `report_metrics.py` retain the original policy: average non-null metric values within each run, then report the equally weighted mean and sample standard deviation of the three run means. Null metrics are excluded, not silently changed to zeros or new failure scores. Run labels are anonymized as `r0`, `r1`, and `r2`. `reaggregate_results.py` recomputes these summaries against the released scores; this is target-filtered reaggregation on existing predictions with unchanged supports, rather than a new full-corpus solving evaluation.

`grid_metrics.py` preserves the original exact, GC, and FG definitions. Exact is grid equality. GC is the fraction of matching cells and is zero for a shape mismatch. FG evaluates cells where gold differs from the input when those shapes match; when gold and input shapes differ it evaluates all gold cells. A prediction/gold shape mismatch gives FG zero; an empty difference foreground gives one. Cached missing metrics still follow the aggregation policy above.

## Rule-description measures and Appendix B alignment margin

Section 5.1 analyzes the emitted `inferred_rule` field. Usage Ratio is the proportion of detected word tokens tagged as metaphorical by the metaphor detector, with a denominator of at least one. Similarity is the cosine similarity between an emitted rule description and its reference GIFARC task frame, embedded with `text-embedding-3-large`. In the saved metric files, the legacy key `MWR` denotes Usage Ratio and `AAS` denotes Similarity. These names do not imply observation of internal reasoning. The preserved CSV `display_condition` values `Analogy-Removed` and `Only Analogy-Removed` correspond to the current paper labels Metaphor-Removed (`neutral`) and Only Metaphor-Removed (`neutral A+G`), respectively.

`trace-margin-per-case.csv` retains compact numeric alignment scores for these emitted rule descriptions. Existing `trace_margin.py`, `summarize_trace_margin.py`, and `trace-*` filenames are legacy names retained for compatibility. For each saved inferred rule, Own Similarity compares it with its task's reference frame; Other Similarity averages similarity to the other 99 task reference frames; Margin is Own minus Other. The rule descriptions and references use `text-embedding-3-large`.

Appendix B averages available valid runs **within each task**, then averages tasks with at least one valid run. This differs from the run-level averaging used by Table 3 and the cached solving summaries. `trace_margin.py` copies the actual Appendix B aggregation functions. `summarize_trace_margin.py` reproduces the included `trace-margin-summary.csv`. `trace-valid-denominators.json` provides the available case counts, task counts, and counts by run:

| Model | Condition | Valid scores / 300 | Tasks with a valid run / 100 |
| --- | --- | ---: | ---: |
| GPT-5.4 | Full Description (`rich`) | 288 | 99 |
| GPT-5.4 | Metaphor-Removed (`neutral`) | 287 | 100 |
| GPT-5.4 | Only Metaphor-Removed (`neutral A+G`) | 287 | 99 |
| GPT-5.4-mini | Full Description (`rich`) | 292 | 98 |
| GPT-5.4-mini | Metaphor-Removed (`neutral`) | 291 | 98 |
| GPT-5.4-mini | Only Metaphor-Removed (`neutral A+G`) | 292 | 99 |

The modest differences characterize alignment of generated rule descriptions with reference task frames. They do not establish improved analogical reasoning quality or downstream task solving.

## Supports and SFT scope

`paper-settings.json` summarizes source-verified stage settings, actual saved-run configurations, and explicitly marked paper-reported SFT details. The fixed support tasks are canonical IDs **838, 902, and 878** (legacy zero-based IDs 837, 901, and 877). Their scores are 0.526157702277487, 0.47414481203078224, and 0.5012805125575244. At thresholds 0.45/0.50/0.55/0.60/0.65, respectively, 3/2/0/0/0 supports pass. `support-alignment-overlap.json` records this retrospective audit; the supports remain unchanged in all supplied reaggregations.

The reported SFT uses 1,000 randomly sampled tasks per seed after standard preprocessing, four LoRA epochs, and seeds 444/555/666, **without alignment filtering**. Inputs contain three input/output demonstrations from the same task and a test input. Targets jointly contain the analogical task frame, executable solution code, and gold test-output grid, in that order; grids use space-separated color names and newline-separated rows. Both compared data sources retain 1,000 tasks per seed to preserve matched training sizes. The transfer gains therefore do not isolate the contribution of task framing.

`sft-training-alignment-overlap.json` records retained-file membership relative to the thresholds. This audit does not certify exact checkpoint-to-training-file hash lineage or tokenizer-stage retention. The paper reports a maximum sequence length of 6,144 tokens; the historical per-model tokenizer settings and counts of truncated examples are not verified by this artifact. A filtered SFT effect would require new training. The paper reports ARC-AGI-1 exact-match improvements up to 6.4% and ARC-AGI-2 GC/FG improvements with exact match remaining zero; ICL results vary across metrics.

## Limits and provenance

The offline commands reproduce released scores' filtering and existing numerical analyses. They do not rerun the original crawl, generate the entire paper corpus, retrain checkpoints, reproduce paid captions/embeddings, or certify immutable serving snapshots for in-session ICL annotation. See [ICL_ERROR_ANALYSIS.md](ICL_ERROR_ANALYSIS.md) for the reviewed sample and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing/provenance gaps.
