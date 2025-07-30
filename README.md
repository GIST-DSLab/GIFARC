 
# GIFARC 🌟  
**Synthetic Dataset for Leveraging Human‑Intuitive Analogies to Elevate AI Reasoning**

[![Paper](https://img.shields.io/badge/NeurIPS%202025-Under%20Review-orange.svg)](https://neurips.cc/) 
[![Hugging Face Datasets](https://img.shields.io/badge/HF%20Datasets-gif_arc-ff69b4.svg)](https://huggingface.co/datasets/DumDev/gif_arc) 
[![License](https://img.shields.io/github/license/DumDev/gifarc.svg)](LICENSE) 
[![Build](https://img.shields.io/github/actions/workflow/status/DumDev/gifarc/tests.yml?label=tests)](https://github.com/DumDev/gifarc/actions)

> **GIFARC** distills the magic of everyday analogies into a rigorously curated, GIF‑centric benchmark that pushes large language models (LLMs) beyond pattern matching toward **genuine reasoning**.

---

## ✨ TL;DR

* **4 880** animated analogy puzzles across **16** visual–conceptual categories  
* Pair‑wise **ground‑truth mappings** + rich **textual rationales** for supervised or in‑context use  
* **Plug‑and‑play generation pipeline** – extend or remix new analogy families in a few lines of Python  
* **Friendly Hugging Face dataset** & interactive **web demo** for instant exploration
 

## Table of Contents
1. [Quick Start](#rocket-quick-start)
2. [Dataset Card](#open_file_folder-dataset-card)
3. [Pipeline Overview](#factory-pipeline-overview)
4. [Results](#bar_chart-results)
5. [Project Structure](#file_cabinet-project-structure)
6. [Citing GIFARC](#bookmark_tabs-citing-gifarc)
7. [Contributing](#handshake-contributing)
8. [Acknowledgements](#sparkles-acknowledgements)
9. [License](#scroll-license)

---

## :rocket: Quick Start

### 1. Install

```bash
git clone https://github.com/DumDev/gifarc.git
cd gifarc
pip install -r requirements.txt        # poetry support coming soon!
````

### 2. Pull the Dataset

```python
from datasets import load_dataset
ds = load_dataset("DumDev/gif_arc")
print(ds["train"][0])
```

### 3. Generate Your Own Split

```bash
python src/generate_dataset.py \
  --config configs/sample.yaml \
  --output_dir data/custom_split
```

### 4. Try the Web Demo

```bash
# launches http://localhost:8501
streamlit run demo/app.py
```

---

## \:open\_file\_folder: Dataset Card

| Split | #Tasks | #Unique GIFs | Avg. Frames |   Size |
| ----- | -----: | -----------: | ----------: | -----: |
| Train |  3 904 |        2 110 |        22.3 | 3.6 GB |
| Val   |    488 |          276 |        22.1 | 0.5 GB |
| Test  |    488 |          280 |        22.6 | 0.5 GB |

*Every task packages →* `{"gif_a": ..., "gif_b": ..., "analogy_text": ..., "mapping": {...}, "rationale": ...}`

See the full [🤗 dataset card](https://huggingface.co/datasets/DumDev/gif_arc) for licensing, intended use, and data statements.

---

## \:factory: Pipeline Overview

```mermaid
flowchart TD
    A[Input GIF pool (GIPHY API)] --> B{Concept Pair Sampler}
    B --> C[Frame‑level<br/>Visual Filters]
    C --> D[Analogy Composer<br/>(text + mapping)]
    D --> E[Quality Gate ✓]
    E --> F(Output<br/>GIFARC JSONL)
```

* **Modular & extensible** – swap your own samplers, filters, or analogy templates
* **Stateless workers** enable easy scaling on multi‑GPU clusters (see `docker/`).
* All intermediate artifacts are cached for reproducibility.

Detailed instructions live in **[GENERATION.md](GENERATION.md)**.

---

## \:bar\_chart: Results

> *To be released with camera‑ready!*

| Model     | Split | Accuracy ↑ | Spearman ρ ↑ | Human Gap ↓ |
| --------- | :---: | ---------: | -----------: | ----------: |
| GPT‑4o    |  Test |          — |            — |           — |
| **Human** |  Test |  **100 %** |     **1.00** |           — |

Stay tuned – we’re running the final sweep and will publish checkpoints & logs here.

---

## \:file\_cabinet: Project Structure

```
gifarc/
├─ configs/           # YAML recipes for generation runs
├─ data/              # (git‑ignored) – raw & processed datasets
├─ demo/              # Streamlit visualization
├─ docker/            # GPU‑ready images & compose files
├─ scripts/           # helper CLIs (download_gifs, sanity_check, ...)
├─ src/
│  ├─ gifarc/         # core library
│  └─ evaluation/     # metrics & leaderboards
└─ tests/             # unit & integration tests
```

---

## \:bookmark\_tabs: Citing GIFARC

```bibtex
@misc{gifarc2025,
  title   = {GIFARC: Synthetic Dataset for Leveraging Human‑Intuitive Analogies to Elevate AI Reasoning},
  author  = {Kim, Hyunseok and Lee, Jisoo and Park, Minho},
  year    = {2025},
  note    = {Under review at NeurIPS Datasets & Benchmarks 2025},
  url     = {https://github.com/DumDev/gifarc}
}
```

---

## \:handshake: Contributing

Pull requests are welcome! Please:

1. Create a new branch from `main`.
2. Add tests for new features (`pytest -q`).
3. Run `pre-commit run --all-files`.
4. Open a PR and describe your changes clearly.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## \:sparkles: Acknowledgements

* **GIPHY** for powering the GIF search API
* **BARC** – our generation pipeline stands on the shoulders of this excellent project
* GIFARC wouldn’t be possible without the open‑source community and our amazing reviewers.

---

## \:scroll: License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

```

Feel free to tweak badges, stats, or any placeholders (`TBW`) once results and acceptance details are finalized. Happy publishing!
 
