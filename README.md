# GIFARC  
**Synthetic Dataset for Leveraging Human‑Intuitive Analogies to Elevate AI Reasoning**

<img width="1536" height="1024" alt="image" src="./images/logo.png" /> 


[![Paper](https://img.shields.io/badge/NeurIPS%202025-Under%20Review-orange.svg)](https://neurips.cc/) 
[![Hugging Face Datasets](https://img.shields.io/badge/HF%20Datasets-gif_arc-ff69b4.svg)](https://huggingface.co/datasets/DumDev/gif_arc) 
[![License](https://img.shields.io/github/license/DumDev/gifarc.svg)](#) 
[![Build](https://img.shields.io/github/actions/workflow/status/DumDev/gifarc/tests.yml?label=tests)](https://github.com/)

> By embedding robust human-intuitive analogies into ARC-style tasks, GIFARC guides AI agents to evaluate the task analogically before engaging in brute-force pattern search, thus efficiently reducing problem complexity and build a more concise and human-understandable solution.

---

## TL;DR

* **10k** ARC style puzzles made from GIF with analogy.  
* Pair‑wise **ground‑truth mappings** + rich **textual rationales** for supervised or in‑context use.  
* **Easy Play generation pipeline** - extend or remix new analogy families with gif in a few minutes.
* **Friendly Hugging Face dataset** & interactive **web demo** for instant exploration.
 

## Table of Contents
1. [Quick Start](#rocket-quick-start)  
2. [Dataset Card](#open_file_folder-dataset-card)  
3. [Pipeline Overview](#factory-pipeline-overview)  
4. [Project Structure](#file_cabinet-project-structure)
5. [Citing GIFARC](#bookmark_tabs-citing-gifarc)
6. [Acknowledgements](#sparkles-acknowledgements)
7. [License](#scroll-license)

---

## Quick Start

### 1. Install

We highly command to using docker. To setting with docker check [SETUP.md](docs/SETUP.md).  

```bash
git clone <GIT_url>
cd gifarc
pip install -r requirements.txt
pip install -r requirements-dev.txt      
````

### 2. Pull the Dataset

```python
from datasets import load_dataset
ds = load_dataset("DumDev/gif_arc")
```

### 3. Generate Your Own GIFARC

Once your Set up is down, open `description_executor.ipynb` and run the code here.

### 4. Check the Web Demo

[GIFARC Web Demo](https://gifarc.vercel.app).  

---

## Dataset Card

| Split | #Tasks | #Unique GIFs |    Size |
| ----- | -----: | -----------: |  -----: |
| Train |  10,000 |      10,000   |   < 24 MB |


Every task packages →    
```
{
  "source": "<source code>", # python code string
  "examples": [
      [<input_grid_1>,<output_grid_1>], # pair 1
      [<input_grid_2>,<output_grid_2>], # pair 2
      ...
    ], 
  "seeds": [
      "<file_name_1>",
      "<file_name_2>",
      ...,
      "<file_name_N>",
      "<Concept_and_description>"
    ], 
  "url": "<minified_url>"
}
```
See the full [dataset card](https://huggingface.co/datasets/DumDev/gif_arc) for licensing, intended use, and data statements.

---

## Pipeline Overview

<img width="2048" height="893" alt="image" src="./images/pipeline.png" />


* **Modular & Easy generation** – After put GIF in data/GIF, just click all run button at `description_executor.ipynb` to generate Your own data! 
* **Stable environment setting** enable easy set up with docker and devcontainer.  
* All intermediate artifacts are cached for reproducibility.

Detailed instructions live in **[GENERATION.md](docs/GENERATION.md)**.

---

## \:file\_cabinet: Project Structure

```
./GIFARC
├── data
│   └── GIF
├── description_executor.ipynb # use this to execute
├── docker-compose.yml
├── docs
│   ├── EXPERIMENTS.md
│   ├── GENERATION.md
│   ├── project_directory_tree.txt
│   └── SETUP.md
├── loggings
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── results # this will generate automatically
└── src
    ├── execution.py
    ├── experiments.py
    ├── generate_descriptions.py
    ├── generate_problems.py
    ├── generate_visualization_html.py
    ├── GIFARC_data_batch
    ├── GIFARC_utils
    ├── misc
    ├── parse_batch_description_samples.py
    ├── prompts
    ├── seeds
    ├── utility
    └── visualize_problems.py
```

---

## Citing GIFARC

```bibtex
@misc{gifarc2025,
  title   = {GIFARC: Synthetic Dataset for Leveraging Human-Intuitive Analogies to Elevate AI Reasoning},
  author  = { Anonymous },
  year    = {2025},
  note    = {Under review at NeurIPS Datasets & Benchmarks 2025},
  url     = {}
}
```

---

## Acknowledgements

* **GIPHY** for powering the GIF search API.  
* **BARC** – our generation pipeline stands on the shoulders of this excellent project.  
* GIFARC wouldn’t be possible without the open‑source community and our amazing reviewers.

---

## License

Distributed under the **MIT License**.

Feel free to tweak badges, stats, or any placeholders (`TBW`) once results and acceptance details are finalized. Happy publishing!