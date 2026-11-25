# Supervised fine-tuning and evaluation

This directory contains the GIFARC training, dataset-preparation, and evaluation scripts. The training script accepts a prepared SFT JSONL and writes its final model and tokenizer to the requested output directory. The evaluator defaults to transduction, which scores predicted output grids.

Use an environment with PyTorch, Transformers, PEFT, Datasets, Accelerate, tqdm, NumPy, tiktoken, and Weights & Biases. Transformers uses the cache configured in your environment.

## Train and evaluate

Run these commands from the repository root with paths to your own prepared inputs.

~~~bash
python reproducibility/sft/sft_train.py \
  --model_name nvidia/Mistral-NeMo-Minitron-8B-Base \
  --ground_truth_data_path /path/to/training.jsonl \
  --output_dir ./models/nemo-gifarc-seed777

python reproducibility/sft/sft_eval.py \
  --model_name nvidia/Mistral-NeMo-Minitron-8B-Base \
  --model_path ./models/nemo-gifarc-seed777 \
  --input_data_path /path/to/eval-input.jsonl \
  --ground_truth_data_path /path/to/eval-targets.jsonl \
  --output_path ./evaluation.jsonl \
  --temperature 0
~~~

## Evaluation inputs and results

Give each input and its target the same *problem_id* and *test_index*. Every input needs one target. The evaluator validates the files before loading the model. For raw source/examples records, the last example supplies the test input, and an omitted test index defaults to 0. Message-format input and raw input can share one file.

Transduction scores the predicted grid directly. It accepts the documented output marker or a standalone grid of space-separated color names. A wrong or unparsable prediction counts as incorrect. A runtime failure stops the run, and results already written remain available.

To resume, run the same command with the same output path. The evaluator validates the saved rows, removes a truncated final line if present, and runs the missing keys. It rejects duplicate keys, extra result keys, corrupt complete lines, and mismatched run settings where those settings are recorded. Results from older versions may lack run settings, so use the original model, data, and arguments when resuming them. If an old raw result has a null test index, use a new output path to regenerate it with a valid key. A successful run reports coverage and accuracy across all expected keys, including rows written by previous runs.

The historical Bash launchers expect their original dataset layouts. Set `GIFARC_TRAIN_OUTPUT_DIR` for the training output root or `GIFARC_CHECKPOINT_ROOT` for the evaluation checkpoint root. Both roots default to `./models` relative to the launch directory.

The optional `induction` and `both` modes execute generated Python in the evaluation process.
