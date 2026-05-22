# Repository Guidelines

## Project Structure & Module Organization

GIFARC is a Python/Jupyter pipeline for generating ARC-style puzzles from GIF analogies. Core scripts live in `src/`: `generate_descriptions.py`, `generate_problems.py`, `execution.py`, and `generate_visualization_html.py`. Reusable helpers are under `src/GIFARC_utils/` and `src/utility/`; prompt templates are in `src/prompts/`; seed puzzle programs are in `src/seeds/`. Input GIFs belong in `data/GIF/`. Generated artifacts, metadata, and batch lists are written under `results/`; logs go under `loggings/`. Documentation is in `docs/`, and README assets are in `images/`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt && pip install -r requirements-dev.txt`: install runtime and notebook/pipeline dependencies.
- `docker compose up -d`: build and run the dev container. Jupyter is exposed through the compose/devcontainer setup on host port `8997`.
- `python src/GIFARC_data_batch/data_batch_generation.py src/all_gifs_metadata.csv -m 300 -c id -o results/batch_list`: regenerate batch-list files from metadata.
- `python src/generate_descriptions.py --help` and `python src/generate_problems.py --help`: inspect CLI options before launching model-backed generation.
- `python -m py_compile <changed-file.py>`: quick syntax check for changed Python files.

## Coding Style & Naming Conventions

Use Python 3.11-compatible code and 4-space indentation. Keep modules and functions in `snake_case`; classes use `PascalCase`. Prefer existing utility modules over duplicating parsing, LLM, cache, or ARC-grid helpers. Prompt files should remain Markdown with names that match the experiment variant, for example `[float-variable]system_prompt_code.md`.

## Testing Guidelines

There is no dedicated `tests/` suite in this snapshot. Validate changes with targeted syntax checks, small local runs, and relevant notebook cells. For generation changes, use a small `--samples` value and write to a temporary `--outdir` under `results/` before larger jobs. Check JSONL, metadata CSVs, and `loggings/error_desc/` for silent failures.

## Commit & Pull Request Guidelines

Recent history uses short, imperative messages such as `Update README.md`, `Revise GIFARC citation in README`, and occasional `chore : ...` commits. Keep commits focused on one pipeline, docs, or data concern. PRs should describe the affected stage, list commands or notebook sections run, mention API keys or model settings, and include screenshots or sample paths when outputs change.

## Security & Configuration Tips

Store provider credentials only in `.env`; do not commit API keys, cache databases, or large result dumps unless they are intentional dataset artifacts. `OPENAI_API_KEY` is the default key used by `src/utility/llm.py`, with optional provider-specific keys supported there.
