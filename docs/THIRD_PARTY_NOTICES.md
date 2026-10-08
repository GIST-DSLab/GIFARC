# Third-Party Notices

GIFARC builds on public ARC-style puzzle-generation ideas and seed-program conventions from prior work, including BARC. The seed programs under `src/seeds/` are used as examples for prompt construction and generated-code validation.

Some seed filenames preserve upstream identifiers, for example `25d487eb_Kevin.py` and `264363fd_Kevin.py`. These names are kept for reproducibility and provenance. They are not intended to identify the authors of this anonymous submission.

External benchmark datasets and public dependencies should be obtained from their original sources when reproducing experiments. Do not vendor nested third-party repositories, private logs, model checkpoints, or local machine paths into this repository.

## Reused evaluation source

The added `reproducibility/` helpers are copied from the actual GIFARC alignment and solving-analysis workspaces. `source-provenance.json` records repository-relative filenames and hashes; the release does not distribute the internal workspaces or their raw outputs. Compact saved metrics and the 26-response human-review export are derived scientific artifacts rather than third-party source checkouts.

The retained generation supplement does not include a complete per-seed-program upstream licensing inventory or a pinned upstream BARC revision. Its filenames preserve provenance identifiers but do not establish each seed program's complete license lineage. The local source snapshots used for the added helpers do not supply a separate upstream copyright/licensing ledger. These provenance gaps are documented rather than replaced with a new claim of independently verified upstream rights.

External ARC benchmark data, raw GIF files, model weights, provider services, and dependencies retain their own terms. The repository MIT notice is a software notice and does not grant rights to those materials. Benchmark references needed for the small reviewed sample are scientific evaluation examples; the original benchmark's terms apply. No full benchmark checkout is redistributed by this supplement.
