# Current pinned-tokenizer SFT sequence audit

The included audit reconstructs sequence building at the paper's **6,144-token** limit using the retained training source and current pinned official tokenizers. It performs no training or model inference. Historical executed tokenizer revisions and historical truncation counts remain unknown.

## Reproduce the included counts offline

Run from the Software directory:

```bash
python reproducibility/summarize_sft_audit.py --help
python reproducibility/summarize_sft_audit.py --output results/sft_sequence_audit.json
```

Only Python's standard library is needed. The command verifies row-evidence hashes, recomputes counts from all 12,000 numeric audit rows, and checks 1,000 arrays of length 6,144 for every tokenizer/dataset/seed group. The evidence is in `reproducibility/summaries/sft-sequence-audit/`. Internal source record names and raw training messages are not included. The manifests retain original saved-training-file hashes without private locations or URLs.

| Current tokenizer | Seed 444 | Seed 555 | Seed 666 |
|---|---:|---:|---:|
| NeMo: incomplete rendered GIFARC gold grid | 91 / 1,000 | 84 / 1,000 | 77 / 1,000 |
| Qwen: incomplete rendered GIFARC gold grid | 90 / 1,000 | 82 / 1,000 | 76 / 1,000 |
| Either tokenizer: incomplete rendered ARC-AGI-2 gold grid | 0 / 1,000 | 0 / 1,000 | 0 / 1,000 |

These are **incomplete retained grid targets after reconstruction**, not counts of raw sequences exceeding the limit. For example, NeMo raw GIFARC sequences above 6,144 numbered 177/163/157, and Qwen numbered 176/162/157; input shortening preserved some of their complete targets. The row evidence reports raw lengths, input shortening, complete target text, complete rendered grid, complete code fence, markers, and array/nonpad positions separately. ARC base targets lack GIFARC's grid header by design; header absence is not treated as an incomplete grid.

## Tokenizers, source, and replay method

The recorded runtime is **Transformers 4.53.2** and **tokenizers 0.21.2**. The tokenizer manifest records public official URLs, exact revisions, file sizes, and SHA-256 hashes:

- [nvidia/Mistral-NeMo-Minitron-8B-Base](https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base/tree/cc94637b669b62c4829b1e0c3b9074fecd883b74), revision `cc94637b669b62c4829b1e0c3b9074fecd883b74`.
- [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base/tree/0b0bd3732e2c374d483664439ea334928b65f304), revision `0b0bd3732e2c374d483664439ea334928b65f304`.

The retained `sft_train.py` SHA-256 is `e7c13eb98ed4a68ed0bb1c11275affaf7ebf5f4286a2a02a9bb4d596df288ff1`. `reproducibility/sft_sequence_logic.py` preserves `convert_to_sft_format` and the unchanged sequence-building statements from the nested `prepare_arc_dataset.tokenize_function` (original statement lines 387–515). The nested function is lifted to module scope, keeping its name; a result return replaces tensor conversion and label masking. This adapter supports the saved messages-format files; the legacy source/examples branches are outside its scope.

For a user who already possesses the matching saved training file and locally downloaded pinned tokenizer files, the full current sequence replay is:

```bash
python reproducibility/replay_sft_sequences.py --help
python reproducibility/replay_sft_sequences.py \
  --data local_saved_gifarc_full_seed444_1000.jsonl \
  --tokenizer-dir local_tokenizers/nemo --alias nemo \
  --seed 444 --dataset gifarc_full \
  --output results/nemo_gifarc_seed444_replay.csv
```

The adapter enforces the matching saved input hash, pinned tokenizer-file hashes, recorded runtime versions, and 1,000 messages-format records. It loads tokenizers with `local_files_only=True` and `trust_remote_code=False`; it never downloads tokenizers or model weights. Model weights are unnecessary. Repeat with `--alias qwen3_14b`, seeds 555/666, or `--dataset arc_agi2_base` and the matching saved inputs. Tokenizer files themselves and raw saved training records are not bundled, so full tokenization cannot be rerun using only this supplement; the included numerical summaries and count checks remain reproducible offline.

The retained sequence code serializes system/user text, the `Assistant:` marker, and the gold answer. For GIFARC full targets, the answer contains analogy, solution code, and output grid in that order. The original code first reserves up to 70% of the limit for input when necessary, then constructs, pads, or truncates the combined token sequence. The audit decodes the resulting IDs and separately checks the tokenizer-normalized complete answer, space-separated color-name gold grid, and its complete code fence. It does not equate raw over-limit status with target loss.

`current-replay-verification.json` records six representative comparisons against the complete unmodified original preprocessing function: a normal record, input shortening with a complete target, and target truncation for each tokenizer. Reconstructed IDs exactly matched the full original function on those records, including original tensor/label processing. This population audit did not replay label masking. It does not recover checkpoint lineage, prove historical token retention, validate semantic preservation, or estimate the effect of alternative preprocessing. No prospective sample-selection candidates are included as experimental results.
