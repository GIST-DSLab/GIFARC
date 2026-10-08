# GIFARC generation pipeline

Place user-provided GIF files under `data/GIF/`, configure the provider credentials locally as described in [SETUP.md](SETUP.md), and run the cells in `description_executor.ipynb`. Outputs are written under `results/`. Use a new results directory when restarting or reusing previous outputs so cached intermediate files remain distinguishable.

The preserved generation pipeline renders GIF descriptions, uses task sketches and seed programs to generate executable transformations, and validates generated input/output pairs. The raw GIF corpus and original crawl metadata used for the paper are omitted from the anonymous review artifact, so the original crawling stage cannot be exactly reproduced from this bundle.

The original `src/generate_problems.py` CLI defaults to `--total_timeout 30` seconds per problem-generation run. Its `generate_problem` function also defaults to 30 seconds; individual generation and transformation helper calls retain their original one-second defaults. The notebook explicitly runs `src/generate_problems.py` with **`--total_timeout 300`**. This release does not change those defaults or the executable generation code.

Alignment scoring is a separate check of stored pairs against their original description. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the actual reverse-caption method and the offline filter supplied with the reviewed dataset. The scores are an alignment proxy, not a proof of task validity or a guarantee of solver improvement.
