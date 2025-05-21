## Requirements & Setup

First, clone this git.

```clone
git clone <GIT_URL>
```

To install requirements, move to the cloned directory and run `pip install`:

```setup
pip install -r requirements.txt
```

Next, we need several files from [BARC](https://github.com/xu3kev/BARC) project. Check you are in the project root directory and clone BARC files with git.

```import
cd <PROJECT_ROOT_DIRECTORY>
git clone https://github.com/xu3kev/BARC.git
```

From the cloned codes, move following listed directory and files to the project root directory.

```
BARC/
  |  execution.py
  |  parse_batch_description_samples.py
  |  prompt.py
  └─ seeds/
```

The folder setup is done. Directory structure should look like following. Full project directory tree is written in [project_directory_tree.txt](./project_directory_tree.txt). 

```
│  .gitignore
│  all_gifs_metadata.csv
│  data_generation_script.sh
│  data_generation_script_o4-mini.sh
│  description_executor.ipynb
│  error_prob_code_format.json
│  execution.ipynb
│  execution.py
│  experiments.py
│  generate_code.py
│  generate_code_origin.py
│  generate_descriptions.py
│  generate_descriptions_serial.py
│  generate_problems.py
│  llm.py
│  parse_batch_description_samples.py
│  pip.txt
│  prompt.py
│  prompt_utils.py
│  README.md
│  requirements-dev.txt
│  requirements.txt
│  tester.ipynb
│  tree.txt
│  unified_gif_seed_pipeline.py
│  utils.py
│  utils_gif.py
│  visualize_problems.py
│  
├─GIFARC_data_batch/
│  ├─data_batchs/ 
│  └─uuid_batchs/
|
├─GIFARC_utils/
├─prompts/
├─scripts/
└─seeds/
    ├─ConceptARC/
    └─input_sandbox/
```
