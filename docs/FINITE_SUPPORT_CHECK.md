# Finite validation of the three frozen support-code pairs

Run from the Software directory with NumPy available:

```bash
python reproducibility/validate_support_controls.py --help
python reproducibility/validate_support_controls.py \
  --output-dir results/finite_support_check
```

The output directory must be new. This command executes only the inspected support pairs for canonical task IDs **838, 878, and 902** (original zero-based IDs 837, 877, and 901). It does not execute the full dataset. `reproducibility/summaries/finite-support-check/` includes the original and rewritten code, all 29 stored input/gold-output pairs, the configured `common.py`, input hashes, safety inspection, and the recorded report. Raw preprocessing metadata, source URLs, local filenames, and review flags are omitted. The code and grids are preserved.

`reproducibility/support_code_validator.py` contains the actual unchanged `validate_code_structure`, `validate_behavior`, and module-loading/helper definitions. Source and copied-definition hashes are in `source-provenance.json`. The configured common helper is copied byte-for-byte from the retained validation environment. The module-loader executes Python source, so the adapter first requires the exact hashes of the three packaged inputs and common helper; it is not a general-purpose runner for arbitrary programs. Static inspection of these six code bodies found only `common`, NumPy, and random imports and no file, network, process, or dynamic-execution calls. These inspections apply only to the shipped support pairs.

The original structure validator compares imports, top-level assignment targets, and top-level function names/signature argument counts. The original behavior validator resets Python and NumPy random seeds to **0, 1, and 2**, regenerates each input under each program, and checks input and output equality. The adapter then runs both `main` functions on each supplied stored input and checks original = rewritten = gold. Each support runs in a separate Python process with a 20-second bound.

The recorded and reproduced results are **3/3 structure passes**, **3/3 seeded behavior passes**, and **29/29 stored inputs** with identical original, rewritten, and gold outputs, with no stored execution errors. The reference check used Python 3.12.14 and NumPy 2.3.5. Each input/gold hash can be inspected in `stored-pair-checks.csv`.

These finite checks are empirical validation of the inspected support programs. They are not a proof of equivalence for arbitrary inputs, a full alpha-equivalent AST proof, independent validation of the rewritten descriptions' semantics, or certification of every historical preprocessing call or model-serving snapshot.
